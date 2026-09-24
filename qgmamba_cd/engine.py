from __future__ import annotations

import copy
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .checkpoint import initialize_from_checkpoint, load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .data import DatasetBundle
from .distributed import (
    DistributedContext,
    broadcast_module_state,
    reduce_sum,
    resolve_amp_dtype,
    unwrap_model,
)
from .evaluation import sweep_thresholds
from .losses import FullLoss, estimate_pos_weight
from .model import ModelEMA, QGMambaDiffCD


def create_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for parameter in unwrap_model(model).encoder.parameters():
        parameter.requires_grad = trainable


def build_layerwise_parameter_groups(model: QGMambaDiffCD, config: ExperimentConfig):
    """Apply progressively smaller learning rates toward the encoder input layers, with the head at the configured learning rate. Returns a list of parameter groups and a list of maximum learning rates for each group."""
    encoder_named = list(model.encoder.named_parameters())
    layer_indices = []
    for name, _ in encoder_named:
        match = re.search(r"(?:layers|stages)\.(\d+)", name)
        if match:
            layer_indices.append(int(match.group(1)) + 1)
    maximum_depth = max(layer_indices, default=1) + 1
    grouped: dict[int, list[torch.nn.Parameter]] = {}
    encoder_ids: set[int] = set()
    for name, parameter in encoder_named:
        encoder_ids.add(id(parameter))
        match = re.search(r"(?:layers|stages)\.(\d+)", name)
        if "patch_embed" in name or "stem" in name:
            depth = 0
        elif match:
            depth = int(match.group(1)) + 1
        else:
            depth = maximum_depth
        grouped.setdefault(depth, []).append(parameter)
    parameter_groups = []
    maximum_lr_values = []
    for depth in sorted(grouped):
        scale = config.train.layerwise_lr_decay ** (maximum_depth - depth)
        learning_rate = (
            config.train.learning_rate * config.train.encoder_lr_multiplier * scale
        )
        parameter_groups.append({"params": grouped[depth], "lr": learning_rate})
        maximum_lr_values.append(learning_rate)
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    parameter_groups.append({"params": head_parameters, "lr": config.train.learning_rate})
    maximum_lr_values.append(config.train.learning_rate)
    return parameter_groups, maximum_lr_values


def _all_ranks_finite(loss: torch.Tensor, context: DistributedContext) -> bool:
    finite = torch.tensor(
        1 if torch.isfinite(loss).item() else 0,
        dtype=torch.int32,
        device=context.device,
    )
    if context.distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def train_one_epoch(
    model: torch.nn.Module,
    ema: ModelEMA,
    criterion: FullLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    bundle: DatasetBundle,
    config: ExperimentConfig,
    context: DistributedContext,
    epoch: int,
):
    model.train()
    running_loss = 0.0
    seen_batches = 0
    skipped_batches = 0
    progress = tqdm(
        bundle.train_loader,
        desc=f"Epoch {epoch}/{config.train.epochs}",
        disable=not context.is_main,
        leave=False,
    )
    amp_enabled = config.train.amp and context.device.type == "cuda"
    amp_dtype = resolve_amp_dtype(config.train.amp_dtype, context.device)
    for batch_index, (image_a, image_b, mask) in enumerate(progress):
        image_a = image_a.to(context.device, non_blocking=True)
        image_b = image_b.to(context.device, non_blocking=True)
        mask = mask.to(context.device, non_blocking=True)
        if config.train.channels_last and context.device.type == "cuda":
            image_a = image_a.contiguous(memory_format=torch.channels_last)
            image_b = image_b.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(context.device.type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(image_a, image_b, mask)
            loss = criterion(*outputs, mask)
        if not _all_ranks_finite(loss, context):
            skipped_batches += 1
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step = (epoch - 1) * len(bundle.train_loader) + batch_index + 1
        if global_step % config.train.ema_update_every == 0:
            global_batch = config.train.batch_size * context.world_size
            decay_per_step = config.train.ema_decay ** (
                global_batch / config.train.ema_reference_batch_size
            )
            effective_decay = decay_per_step ** config.train.ema_update_every
            ema.update(unwrap_model(model), decay=effective_decay)
        running_loss += float(loss.detach().item())
        seen_batches += 1
        if context.is_main:
            progress.set_postfix(loss=f"{loss.item():.4f}")
    statistics = torch.tensor(
        [running_loss, seen_batches, skipped_batches], dtype=torch.float64, device=context.device
    )
    statistics = reduce_sum(statistics).cpu().tolist()
    global_loss, global_seen, global_skipped = statistics
    if global_seen == 0:
        raise FloatingPointError("All training batches were non-finite across all ranks")
    return global_loss / global_seen, int(global_skipped)


def build_training_objects(config: ExperimentConfig, bundle: DatasetBundle, context: DistributedContext):
    model_config = copy.deepcopy(config.model)
    if config.train.resume:
        # A strict resume contains the complete encoder. Weight-only
        # initialization may target a different encoder, so it must retain its
        # configured pretrained initialization before compatible head weights
        # are restored.
        model_config.pretrained = False
    model = QGMambaDiffCD(model_config).to(context.device)
    model.use_channels_last = config.train.channels_last
    if config.train.sync_batchnorm and context.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if config.train.channels_last and context.device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if config.train.init_checkpoint:
        checkpoint, source, missing, unexpected = initialize_from_checkpoint(
            config.train.init_checkpoint,
            model,
            prefer_ema=config.train.init_from_ema,
        )
        if context.is_main:
            print(
                f"Initialized from {config.train.init_checkpoint} ({source}, "
                f"epoch={checkpoint.get('epoch', 'unknown')}, "
                f"missing_new={len(missing)}, unexpected={len(unexpected)})"
            )
            if unexpected:
                print("Unexpected checkpoint keys:", unexpected[:20])
    ema = ModelEMA(model, decay=config.train.ema_decay)
    parameter_groups, maximum_lr_values = build_layerwise_parameter_groups(model, config)
    optimizer_options = {
        "lr": config.train.learning_rate,
        "weight_decay": config.train.weight_decay,
    }
    if config.train.fused_optimizer and context.device.type == "cuda":
        optimizer_options["fused"] = True
    try:
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_options)
    except TypeError:
        optimizer_options.pop("fused", None)
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_options)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=maximum_lr_values,
        epochs=config.train.epochs,
        steps_per_epoch=len(bundle.train_loader),
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1000.0,
        anneal_strategy="cos",
    )
    scaler = create_grad_scaler(
        config.train.amp
        and context.device.type == "cuda"
        and resolve_amp_dtype(config.train.amp_dtype, context.device) == torch.float16
    )
    start_epoch = 1
    best_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5
    history = {
        "epoch": [],
        "train_loss": [],
        "val_f1": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": [],
        "val_accuracy": [],
        "val_threshold": [],
        "learning_rate": [],
        "skipped_nonfinite": [],
    }
    if config.train.resume:
        checkpoint = load_checkpoint(
            config.train.resume, model, ema, optimizer, scheduler, scaler, strict=True
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_f1 = float(checkpoint.get("best_f1", checkpoint.get("val_metrics", {}).get("f1", -1.0)))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", -1)))
        best_threshold = float(checkpoint.get("threshold", 0.5))
        history = checkpoint.get("history", history)
    if context.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            output_device=context.local_rank if context.device.type == "cuda" else None,
            find_unused_parameters=config.train.find_unused_parameters,
            broadcast_buffers=True,
        )
    return model, ema, optimizer, scheduler, scaler, start_epoch, best_f1, best_epoch, best_threshold, history


def fit(config: ExperimentConfig, bundle: DatasetBundle, context: DistributedContext) -> None:
    (
        model,
        ema,
        optimizer,
        scheduler,
        scaler,
        start_epoch,
        best_f1,
        best_epoch,
        best_threshold,
        history,
    ) = build_training_objects(config, bundle, context)
    pos_weight = config.train.pos_weight
    if pos_weight is None:
        pos_weight = estimate_pos_weight(bundle.train_paths, read_mask=bundle.readers.read_mask)
    criterion = FullLoss(config.loss, pos_weight).to(context.device)
    if start_epoch <= config.train.freeze_encoder_epochs:
        set_encoder_trainable(model, False)
    output_dir = Path(config.train.output_dir)
    top_checkpoints: list[tuple[float, Path]] = []
    for path in output_dir.glob("top_epoch*_f1_*.pt"):
        match = re.search(r"_f1_([0-9.]+)\.pt$", path.name)
        if match:
            top_checkpoints.append((float(match.group(1)), path))
    top_checkpoints.sort(key=lambda item: item[0], reverse=True)
    top_checkpoints = top_checkpoints[: config.train.keep_top_k]
    patience_counter = 0
    started = time.time()
    last_validation = None

    for epoch in range(start_epoch, config.train.epochs + 1):
        if bundle.train_sampler is not None:
            bundle.train_sampler.set_epoch(epoch)
        if epoch == config.train.freeze_encoder_epochs + 1 and config.train.freeze_encoder_epochs > 0:
            set_encoder_trainable(model, True)
            if context.is_main:
                print("Encoder unfrozen")
        train_loss, skipped = train_one_epoch(
            model,
            ema,
            criterion,
            optimizer,
            scheduler,
            scaler,
            bundle,
            config,
            context,
            epoch,
        )
        should_validate = (
            epoch == 1
            or epoch % config.train.validation_interval == 0
            or epoch == config.train.epochs
            or epoch == config.train.minimum_f1_epoch
        )
        validation = None
        threshold = best_threshold
        improved = False
        if should_validate:
            # DDP broadcasts training BatchNorm buffers, but each rank's final
            # local update can differ. Evaluate an identical EMA on every shard.
            use_ema = epoch > config.train.ema_warmup_epochs
            evaluation_model = ema.ema if use_ema else unwrap_model(model)
            evaluation_weights = "ema" if use_ema else "model"
            broadcast_module_state(evaluation_model)
            threshold, validation, _ = sweep_thresholds(
                evaluation_model,
                bundle.val_dataset,
                config.eval.thresholds,
                config.eval,
                context,
                config.train.amp,
                config.train.amp_dtype,
                config.train.channels_last,
            )
            last_validation = validation
            improved = validation["f1"] > best_f1
            if improved:
                best_f1 = validation["f1"]
                best_epoch = epoch
                best_threshold = threshold
                patience_counter = 0
            else:
                patience_counter += 1
        learning_rate = optimizer.param_groups[-1]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_f1"].append(validation["f1"] if validation else float("nan"))
        history["val_iou"].append(validation["iou"] if validation else float("nan"))
        history["val_precision"].append(validation["precision"] if validation else float("nan"))
        history["val_recall"].append(validation["recall"] if validation else float("nan"))
        history["val_accuracy"].append(validation["accuracy"] if validation else float("nan"))
        history["val_threshold"].append(threshold)
        history["learning_rate"].append(learning_rate)
        history["skipped_nonfinite"].append(skipped)
        if context.is_main:
            if validation:
                print(
                    f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                    f"F1={validation['f1']:.4f} IoU={validation['iou']:.4f} "
                    f"P={validation['precision']:.4f} R={validation['recall']:.4f} "
                    f"threshold={threshold:.2f} weights={evaluation_weights} "
                    f"lr={learning_rate:.3e} skipped={skipped}"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | loss={train_loss:.4f} | validation=skipped "
                    f"lr={learning_rate:.3e} skipped={skipped}"
                )
            checkpoint_state = {
                "epoch": epoch,
                "model_state": unwrap_model(model).state_dict(),
                "ema_state": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "threshold": best_threshold,
                "val_metrics": last_validation,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "history": history,
                "eval_weights": evaluation_weights if should_validate else (
                    "ema" if epoch > config.train.ema_warmup_epochs else "model"
                ),
            }
            if config.train.save_every_epoch:
                save_checkpoint(output_dir / "last.pt", checkpoint_state)
            if improved:
                save_checkpoint(output_dir / "best.pt", checkpoint_state)
                print(f"Saved new best checkpoint: {output_dir / 'best.pt'}")
            if validation and (
                len(top_checkpoints) < config.train.keep_top_k
                or validation["f1"] > top_checkpoints[-1][0]
            ):
                top_path = output_dir / f"top_epoch{epoch:03d}_f1_{validation['f1']:.5f}.pt"
                save_checkpoint(top_path, checkpoint_state)
                top_checkpoints.append((validation["f1"], top_path))
                top_checkpoints.sort(key=lambda item: item[0], reverse=True)
                while len(top_checkpoints) > config.train.keep_top_k:
                    _, obsolete = top_checkpoints.pop()
                    if obsolete.exists():
                        obsolete.unlink()
        below_minimum = (
            should_validate
            and config.train.minimum_f1_epoch is not None
            and epoch >= config.train.minimum_f1_epoch
            and best_f1 < config.train.minimum_f1
        )
        stop = should_validate and (
            patience_counter >= config.train.patience or below_minimum
        )
        stop_tensor = torch.tensor(int(stop), device=context.device)
        if context.distributed:
            dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item():
            if context.is_main:
                reason = (
                    f"minimum F1 {config.train.minimum_f1:.4f} not reached"
                    if below_minimum
                    else "patience exhausted"
                )
                print(f"Early stopping at epoch {epoch}: {reason}; best epoch={best_epoch}")
            break
    if context.is_main:
        elapsed_minutes = (time.time() - started) / 60.0
        print(
            f"Training complete | best F1={best_f1:.4f} epoch={best_epoch} "
            f"threshold={best_threshold:.2f} elapsed={elapsed_minutes:.1f} min"
        )
