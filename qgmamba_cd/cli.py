from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig, load_config

DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "configs" / "levir_final_config.yaml")

"""For Client code, we provide a common parser and config override functions to reduce boilerplate."""
def common_parser(description: str, checkpoint: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", default=None, help="Override data.root")
    parser.add_argument("--output-dir", default=None, help="Override train.output_dir")
    parser.add_argument("--seed", type=int, default=None, help="Override train.seed")
    if checkpoint:
        parser.add_argument("--checkpoint", required=True)
    return parser


def apply_common_overrides(config: ExperimentConfig, arguments) -> ExperimentConfig:
    if arguments.data_root:
        config.data.root = arguments.data_root
    if arguments.output_dir:
        config.train.output_dir = arguments.output_dir
    if getattr(arguments, "seed", None) is not None:
        config.train.seed = arguments.seed
    return config


def config_from_arguments(arguments) -> ExperimentConfig:
    return apply_common_overrides(load_config(arguments.config), arguments)
