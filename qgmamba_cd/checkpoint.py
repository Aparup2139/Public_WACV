from __future__ import annotations

import os
from pathlib import Path

import torch

from .model import ModelEMA


def _torch_load(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    replacements = (
        (".spatial_att.", ".spatial_attention."),
        (".gate_conv.", ".gate."),
        ("decoder.head_feat.", "decoder.head_features."),
        ("decoder.cond_proj.", "decoder.condition_projection."),
        ("diffusion.unet.t_embed.", "diffusion.unet.time_embedding."),
        ("diffusion.unet.inc.", "diffusion.unet.input_block."),
        ("diffusion.unet.down1.", "diffusion.unet.down_1."),
        ("diffusion.unet.db1.", "diffusion.unet.block_1."),
        ("diffusion.unet.down2.", "diffusion.unet.down_2."),
        ("diffusion.unet.db2.", "diffusion.unet.block_2."),
        ("diffusion.unet.mid.", "diffusion.unet.middle."),
        ("diffusion.unet.up2.", "diffusion.unet.up_2."),
        ("diffusion.unet.ub2.", "diffusion.unet.up_block_2."),
        ("diffusion.unet.up1.", "diffusion.unet.up_1."),
        ("diffusion.unet.ub1.", "diffusion.unet.up_block_1."),
        ("diffusion.unet.out_conv.", "diffusion.unet.output."),
        (".t_proj.", ".time_projection."),
        ("diffusion.refine_proj.", "diffusion.refine_projection."),
        ("diffusion.sqrt_one_minus_ab", "diffusion.sqrt_one_minus_alpha_bar"),
        ("tgate3.", "temporal_gate3."),
        ("tgate4.", "temporal_gate4."),
        ("aux_head.", "auxiliary_head."),
    )
    normalized = {}
    for key, value in state_dict.items():
        if key.endswith("total_ops") or key.endswith("total_params"):
            continue
        for old, new in replacements:
            key = key.replace(old, new)
        normalized[key] = value
    return normalized


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    ema: ModelEMA | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
    strict: bool = True,
):
    checkpoint = _torch_load(path, map_location=next(model.parameters()).device)
    raw_state = checkpoint.get("model_state", checkpoint.get("model", checkpoint))
    model.load_state_dict(normalize_state_dict(raw_state), strict=strict)
    if ema is not None:
        ema_state = checkpoint.get("ema_state", raw_state)
        ema.load_state_dict(normalize_state_dict(ema_state))
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def initialize_from_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    prefer_ema: bool = True,
):
    """Load model weights only while intentionally resetting training state."""
    checkpoint = _torch_load(path, map_location=next(model.parameters()).device)
    if prefer_ema and checkpoint.get("ema_state") is not None:
        raw_state = checkpoint["ema_state"]
        source = "ema_state"
    else:
        raw_state = checkpoint.get("model_state", checkpoint.get("model", checkpoint))
        source = "model_state"
    normalized = normalize_state_dict(raw_state)
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in normalized.items():
        if key not in current:
            skipped.append(f"{key} (not present)")
        elif current[key].shape != value.shape:
            skipped.append(
                f"{key} (checkpoint {tuple(value.shape)} != model {tuple(current[key].shape)})"
            )
        else:
            compatible[key] = value
    incompatible = model.load_state_dict(compatible, strict=False)
    unexpected = list(incompatible.unexpected_keys) + skipped
    return checkpoint, source, list(incompatible.missing_keys), unexpected


def save_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
