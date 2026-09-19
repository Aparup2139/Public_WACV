# scripts/build_experiments_notebook.py
"""Regenerates experiments.ipynb. Run after editing any CELLS entry below.

Usage: python scripts/build_experiments_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent

MARKDOWN_INTRO = """\
# QGMamba-Diff-CD — Multi-Dataset Experiments

Set `DATASET` in the next cell to `"s2looking"`, `"valais_bmd"`, or `"b_flair"` and run all cells.
Every other cell is dataset-agnostic; only the imported dataset module and its `config.yaml` change.
"""

CELL_SELECT_DATASET = """\
DATASET = "s2looking"  # one of: "s2looking", "valais_bmd", "b_flair"
"""

CELL_IMPORTS = """\
import importlib
import os
from pathlib import Path

import torch
import yaml

from qgmamba_cd.data import build_datasets
from qgmamba_cd.distributed import DistributedContext, configure_cuda, seed_everything
from qgmamba_cd.engine import fit
from qgmamba_cd.evaluation import evaluate_threshold, sweep_thresholds
from qgmamba_cd.factory import build_model
from qgmamba_cd.threshold import pick_threshold_plateau
from qgmamba_cd.viz import qg_show
from qgmamba_cd.benchmark import profile_model
"""

CELL_LOAD_DATASET_MODULE = """\
dataset_module = importlib.import_module(DATASET)
with open(Path(DATASET) / "config.yaml", "r", encoding="utf-8") as handle:
    ds_config = yaml.safe_load(handle)

def _expand(value):
    return os.path.expandvars(value) if isinstance(value, str) else value

paths = {key: _expand(value) for key, value in ds_config["paths"].items()}
print(f"Loaded dataset module '{DATASET}' with paths:")
for key, value in paths.items():
    print(f"  {key}: {value}")
"""

CELL_BUILD_MODEL = """\
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configure_cuda(True)
seed_everything(42)
CONTEXT = DistributedContext(rank=0, local_rank=0, world_size=1, device=DEVICE)

# data/train/eval sections of <dataset>/config.yaml are dataset-tuned overrides (crop-focus
# probability, pos_weight, eval tiling, ...) that must be merged onto the base model-architecture
# config here -- they are otherwise never applied to the ExperimentConfig object used below.
overrides = {key: ds_config[key] for key in ("data", "train", "eval") if key in ds_config}

checkpoint_path = paths.get("checkpoint_path") or paths.get("transfer_checkpoint_path")
model, config, checkpoint = build_model(
    paths["model_config_path"],
    checkpoint_path=checkpoint_path,
    overrides=overrides,
    device=DEVICE,
)
threshold = float(checkpoint.get("threshold", 0.5))
print(f"Model built. Checkpoint threshold: {threshold}")
"""

CELL_BUILD_DATA = """\
config.data.root = paths["data_root"]  # data_root is a path, not a hyperparameter -- set directly, not via config.yaml's "data" override section

index_kwargs = {}
if DATASET == "valais_bmd" and paths.get("pruned_names_pickle"):
    from valais_bmd.dataset import load_pruned_names
    pruned = load_pruned_names(paths["pruned_names_pickle"])
    index_kwargs = {"pruned_names": pruned}

def index_builder(root, split):
    if index_kwargs:
        return dataset_module.build_index(root, split, **index_kwargs)
    try:
        return dataset_module.build_index(root, split)
    except TypeError:
        return dataset_module.build_index(root)  # b_flair: flat layout, no split argument

bundle = build_datasets(
    config.data,
    CONTEXT,
    seed=42,
    batch_size=config.train.batch_size,
    index_builder=index_builder,
    readers=dataset_module.READERS,
)
print(f"train={len(bundle.train_dataset)} val={len(bundle.val_dataset)} test={len(bundle.test_dataset)}")
"""

CELL_TRAIN_OR_EVAL = """\
# Zero-shot evaluation at the checkpoint's stored threshold (always safe to run).
# config.eval already carries this dataset's config.yaml eval-section overrides (merged in the
# previous cell via apply_overrides) -- reuse it directly rather than re-parsing ds_config here.
zero_shot_metrics = evaluate_threshold(
    model, bundle.test_dataset, threshold, config.eval, CONTEXT, amp=True, final=True
)
print("Zero-shot metrics:", zero_shot_metrics)

# Uncomment to fine-tune on this dataset using the shared training loop:
# config.train.output_dir = paths["output_dir"]
# fit(config, bundle, CONTEXT)
"""

CELL_VISUALIZE = """\
figure = qg_show(model, bundle.test_dataset, threshold, DEVICE, n=3, title=f"{DATASET} zero-shot", prefer_changed=True)
"""

CELL_BENCHMARK = """\
efficiency = profile_model(model, input_size=config.model.input_size, device=DEVICE)
print(efficiency)
"""

CELLS = [
    ("markdown", MARKDOWN_INTRO),
    ("code", CELL_SELECT_DATASET),
    ("code", CELL_IMPORTS),
    ("code", CELL_LOAD_DATASET_MODULE),
    ("code", CELL_BUILD_MODEL),
    ("code", CELL_BUILD_DATA),
    ("code", CELL_TRAIN_OR_EVAL),
    ("code", CELL_VISUALIZE),
    ("code", CELL_BENCHMARK),
]


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    for cell_type, source in CELLS:
        if cell_type == "markdown":
            notebook.cells.append(nbf.v4.new_markdown_cell(source))
        else:
            notebook.cells.append(nbf.v4.new_code_cell(source))
    return notebook


if __name__ == "__main__":
    notebook = build_notebook()
    output_path = REPO_ROOT / "experiments.ipynb"
    nbf.write(notebook, output_path)
    print(f"Wrote {output_path}")
