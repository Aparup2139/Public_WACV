from __future__ import annotations

from pathlib import Path

import torch

from .checkpoint import normalize_state_dict
from .config import ExperimentConfig, apply_overrides, load_config
from .model import QGMambaDiffCD

"""Factory functions for building QG-Mamba-Diff models and loading checkpoints."""
def _torch_load(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_model(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    overrides: dict[str, dict] | None = None,
    device: torch.device | None = None,
    force_ddim: bool = True,
    strict: bool = True,
) -> tuple[QGMambaDiffCD, ExperimentConfig, dict]:
    """Build a QGMambaDiffCD from a YAML config and, optionally, load a checkpoint's weights.

    Checkpoint loading reproduces the pattern hand-rolled in all three source notebooks: the
    checkpoint's own `eval_weights` field ("ema" or "model") -- not a caller-supplied flag --
    decides which state dict is the one to evaluate with; `normalize_state_dict` renames legacy
    submodule keys; the load is `strict=True` by default so a config/checkpoint architecture
    mismatch fails loudly. This is deliberately not `checkpoint.initialize_from_checkpoint`,
    which is a different, permissive (`strict=False`, shape-compatible-only) mechanism built for
    `engine.build_training_objects`'s cross-architecture warm-start use case.

    `overrides` applies a dataset's `config.yaml` data/train/eval sections on top of the base
    model-architecture YAML -- without this, those per-dataset tuned values never reach the
    ExperimentConfig object actually used by build_datasets()/fit().
    """
    config = load_config(config_path)
    if overrides:
        apply_overrides(config, overrides)
    if force_ddim:
        config.model.use_ddim_at_inference = True
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QGMambaDiffCD(config.model).to(device)
    if force_ddim and not model.diffusion.use_ddim:
        raise RuntimeError("Requested DDIM inference but the constructed model's diffusion refiner is inactive")

    checkpoint: dict = {}
    if checkpoint_path is not None:
        checkpoint = _torch_load(checkpoint_path, map_location=device)
        wrapper_keys = ("model_state", "state_dict", "ema_state")
        if not isinstance(checkpoint, dict) or not any(key in checkpoint for key in wrapper_keys):
            raw_state = checkpoint  # bare state dict, no wrapper
        elif checkpoint.get("eval_weights", "model") == "ema" and checkpoint.get("ema_state") is not None:
            raw_state = checkpoint["ema_state"]
        else:
            raw_state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(normalize_state_dict(raw_state), strict=strict)
    model.eval()
    return model, config, checkpoint
