from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "./data/LEVIR-CD256"
    image_size: int = 224
    train_multiplier: int = 1
    subset_fraction: float = 1.0
    change_focus_prob: float = 0.6
    min_change_ratio: float = 0.005
    max_crop_tries: int = 25
    temporal_swap_prob: float = 0.0
    independent_photometric_prob: float = 0.7
    shadow_prob: float = 0.15
    compression_prob: float = 0.15
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class ModelConfig:
    encoder_name: str = "swin_tiny_patch4_window7_224"
    pretrained: bool = True
    input_size: int = 224
    allow_encoder_fallback: bool = True
    diffusion_timesteps: int = 100
    diffusion_infer_steps: int = 5
    diffusion_start_timestep: int = 50
    diffusion_noise_scale: float = 0.0
    diffusion_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    use_ddim_at_inference: bool = False
    diffusion_enabled: bool = True
    corrected_diffusion_enabled: bool = False
    symmetric_temporal_enabled: bool = False
    edge_fusion_enabled: bool = False
    scan_chunk_size: int = 16
    refinement_enabled: bool = False
    refinement_channels: int = 64


@dataclass
class LossConfig:
    bce: float = 0.4
    focal: float = 0.0
    dice: float = 0.3
    boundary: float = 0.1
    diffusion: float = 0.1
    orthogonal: float = 0.0
    # Paper-style OFD regularizers (BMD-CD Eq. 4). Defaults of 0.0 keep every
    # pre-existing config and checkpoint byte-identical in behavior.
    orthogonal_gram: float = 0.0
    variance: float = 0.0
    variance_margin: float = 1.0
    coarse: float = 0.25
    auxiliary: float = 0.1
    hard: float = 0.5
    iou: float = 0.0
    tversky: float = 0.0
    tversky_alpha: float = 0.4
    tversky_beta: float = 0.6
    lovasz: float = 0.2
    ohem: float = 0.0
    ohem_fraction: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 50
    batch_size: int = 24
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    ema_decay: float = 0.999
    pos_weight: float | None = None  # overrides estimate_pos_weight() when set; some datasets (e.g. Valais) need a value outside its [1, 4] clip range
    freeze_encoder_epochs: int = 3
    patience: int = 8
    amp: bool = True
    amp_dtype: str = "float16"
    grad_clip: float = 1.0
    channels_last: bool = False
    allow_tf32: bool = True
    fused_optimizer: bool = True
    encoder_lr_multiplier: float = 1.0
    layerwise_lr_decay: float = 0.8
    ema_update_every: int = 1
    ema_reference_batch_size: int = 24
    ema_warmup_epochs: int = 0
    validation_interval: int = 1
    sync_batchnorm: bool = False
    find_unused_parameters: bool = True
    output_dir: str = "./outputs/levir_qgmamba"
    resume: str | None = None
    init_checkpoint: str | None = None
    init_from_ema: bool = True
    save_every_epoch: bool = True
    keep_top_k: int = 5
    minimum_f1_epoch: int | None = None
    minimum_f1: float | None = None


@dataclass
class EvalConfig:
    tile_size: int = 224
    tile_stride: int = 224
    final_stride: int = 112
    batch_tiles: int = 4
    tta: bool = False
    final_tta: bool = True
    d4_tta: bool = False
    thresholds: list[float] = field(
        default_factory=lambda: [round(0.20 + 0.05 * i, 2) for i in range(15)]
    )
    quick_val_images: int | None = None
    postprocess: bool = False
    min_area: int = 64
    morph_kernel: int = 3


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    valid = set(instance.__dataclass_fields__)
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown configuration keys for {type(instance).__name__}: {sorted(unknown)}")
    for key, value in values.items():
        setattr(instance, key, value)
    return instance


def apply_overrides(config: ExperimentConfig, overrides: dict[str, dict]) -> ExperimentConfig:
    """Merge dataset-specific `config.yaml` override sections onto an already-loaded config.

    Reuses `_merge_dataclass`'s per-field "unknown key" validation, but does NOT re-run
    `load_config`'s cross-field validation block (e.g. `model.input_size == data.image_size`).
    Only pass overrides already known to be internally consistent.
    """
    allowed = {"data", "model", "loss", "train", "eval"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    for section, values in overrides.items():
        _merge_dataclass(getattr(config, section), values or {})
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    allowed = {"data", "model", "loss", "train", "eval"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")
    cfg = ExperimentConfig()
    for section in allowed:
        if section in raw:
            _merge_dataclass(getattr(cfg, section), raw[section] or {})
    if not 0.0 < cfg.data.subset_fraction <= 1.0:
        raise ValueError("data.subset_fraction must be in (0, 1]")
    if cfg.train.batch_size < 1:
        raise ValueError("train.batch_size is the per-GPU batch size and must be positive")
    if cfg.train.amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("train.amp_dtype must be 'float16' or 'bfloat16'")
    if cfg.train.ema_update_every < 1 or cfg.train.validation_interval < 1:
        raise ValueError("EMA and validation intervals must be positive")
    if not 0.0 <= cfg.data.temporal_swap_prob <= 1.0:
        raise ValueError("data.temporal_swap_prob must be in [0, 1]")
    for name in ("independent_photometric_prob", "shadow_prob", "compression_prob"):
        if not 0.0 <= getattr(cfg.data, name) <= 1.0:
            raise ValueError(f"data.{name} must be in [0, 1]")
    if cfg.model.input_size != cfg.data.image_size:
        raise ValueError("model.input_size and data.image_size must match")
    if len(cfg.model.diffusion_channels) != 3:
        raise ValueError("model.diffusion_channels must contain exactly three integers")
    if cfg.model.scan_chunk_size < 1:
        raise ValueError("model.scan_chunk_size must be positive")
    if not 0 <= cfg.model.diffusion_start_timestep < cfg.model.diffusion_timesteps:
        raise ValueError("model.diffusion_start_timestep must be within the diffusion schedule")
    if not 0.0 <= cfg.model.diffusion_noise_scale <= 1.0:
        raise ValueError("model.diffusion_noise_scale must be in [0, 1]")
    if not 0.0 < cfg.loss.ohem_fraction <= 1.0:
        raise ValueError("loss.ohem_fraction must be in (0, 1]")
    if cfg.train.resume and cfg.train.init_checkpoint:
        raise ValueError("Use either train.resume or train.init_checkpoint, not both")
    if not 0.0 < cfg.train.layerwise_lr_decay <= 1.0:
        raise ValueError("train.layerwise_lr_decay must be in (0, 1]")
    if (cfg.train.minimum_f1_epoch is None) != (cfg.train.minimum_f1 is None):
        raise ValueError("train.minimum_f1_epoch and train.minimum_f1 must be set together")
    if cfg.train.keep_top_k < 1:
        raise ValueError("train.keep_top_k must be positive")
    return cfg
