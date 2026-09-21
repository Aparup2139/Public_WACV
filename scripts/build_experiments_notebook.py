# scripts/build_experiments_notebook.py
"""Regenerates experiments.ipynb. Run after editing any CELLS entry below.

Usage: python scripts/build_experiments_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent

MARKDOWN_INTRO = """\
# QGMamba-Diff-CD — Configurable Multi-Dataset Inference

Choose one profile in the next cell and run all cells. Dataset downloads, paths, readers, model
configuration, checkpoint, tiling, threshold policy, and test indexing are selected together.

| `DATASET` | Preserved inference protocol |
|---|---|
| `"levir_cd"` | LEVIR-CD256 split layout; bundled QGMamba accuracy checkpoint; checkpoint threshold or validation argmax. |
| `"s2looking"` | Split layout; RGB reader; `{0,255}` masks; checkpoint threshold or validation argmax. |
| `"valais_bmd"` | 2017/2023 pairs; `{0,1}` masks; curated test filter; checkpoint threshold or validation plateau. |
| `"b_flair"` | Flat test-only layout; 5-band TIFF reduced to RGB; zero-shot inference with checkpoint threshold only. |

For LEVIR-CD, set `DATA_ROOT` (or `LEVIR_DATA_ROOT`) to a directory containing `A/`, `B/`,
`label/`, and `list/{train,val,test}.txt`. Other datasets can be downloaded automatically.
Checkpoints are loaded from a repository path or the profile's configured Hugging Face model repository.
"""

CELL_SELECT_DATASET = """\
# ---------------------------- USER CONFIGURATION ----------------------------
DATASET = "levir_cd"  # choose: "levir_cd", "s2looking", "valais_bmd", or "b_flair"
DATA_ROOT = None  # LEVIR-CD path; None uses LEVIR_DATA_ROOT, then data/levir_cd
CHECKPOINT_PATH = None  # compatible repo-relative/absolute checkpoint; None uses the profile default
AUTO_DOWNLOAD_DATA = True
SELECT_THRESHOLD_ON_VALIDATION = False  # LEVIR/S2Looking/ValaisCD only; forbidden for b-FLAIR
VISUALIZE_SAMPLES = 3

DATASET_PROFILES = {
    "levir_cd": {
        "task": "LEVIR-CD supervised test inference",
        "module": "qgmamba_cd.data",
        "dataset_config": "config/levir_cd_accuracy_4x40gb.yaml",
        "data_root": "data/levir_cd",
        "data_env": "LEVIR_DATA_ROOT",
        "checkpoint": "levir_qgmamba_accuracy_best.pt.zip",
        "checkpoint_pickle": "best.pt/data.pkl",
        "threshold_policy": "checkpoint; optional validation argmax",
    },
    "s2looking": {
        "task": "S2Looking supervised test inference",
        "module": "s2looking",
        "dataset_config": "s2looking/config.yaml",
        "data_env": "S2LOOKING_DATA_ROOT",
        "checkpoint": "s2looking/qgmamba_s2looking_best.pt",
        "threshold_policy": "checkpoint; optional validation argmax",
    },
    "valais_bmd": {
        "task": "ValaisCD curated test inference",
        "module": "valais_bmd",
        "dataset_config": "valais_bmd/config.yaml",
        "data_env": "VALAIS_DATA_ROOT",
        "checkpoint": "valais_bmd/checkpoints/best.pt",
        "threshold_policy": "checkpoint; optional validation plateau",
    },
    "b_flair": {
        "task": "b-FLAIR zero-shot test inference",
        "module": "b_flair",
        "dataset_config": "b_flair/config.yaml",
        "data_env": "BFLAIR_DATA_ROOT",
        "checkpoint": "s2looking/qgmamba_s2looking_best.pt",
        "threshold_policy": "checkpoint only; no test-label tuning",
    },
}
if DATASET not in DATASET_PROFILES:
    raise ValueError(f"Unknown DATASET={DATASET!r}; choose one of {tuple(DATASET_PROFILES)}")
DATASET_PROFILES[DATASET]
"""

CELL_IMPORTS = """\
import importlib
import os
import shutil
import zipfile
from pathlib import Path

import torch
import yaml

from qgmamba_cd.data import DEFAULT_READERS, build_datasets
from qgmamba_cd.artifacts import ensure_checkpoint
from qgmamba_cd.distributed import DistributedContext, configure_cuda, seed_everything
from qgmamba_cd.evaluation import evaluate_threshold, sweep_thresholds
from qgmamba_cd.factory import build_model
from qgmamba_cd.threshold import pick_threshold_plateau
from qgmamba_cd.viz import qg_show
from qgmamba_cd.benchmark import profile_model
"""

CELL_LOAD_DATASET_MODULE = """\
REPO_ROOT = Path.cwd().resolve()
if not (REPO_ROOT / "qgmamba_cd").is_dir():
    raise RuntimeError("Start Jupyter from the cloned Public_WACV repository root.")

profile = DATASET_PROFILES[DATASET]
dataset_module = importlib.import_module(profile["module"])
dataset_config_path = REPO_ROOT / profile["dataset_config"]
with open(dataset_config_path, "r", encoding="utf-8") as handle:
    ds_config = yaml.safe_load(handle)

def _expand(value):
    return os.path.expandvars(value) if isinstance(value, str) else value

paths = {key: _expand(value) for key, value in ds_config.get("paths", {}).items()}
paths.setdefault("data_root", profile.get("data_root"))
paths.setdefault("checkpoint_path", profile["checkpoint"])
paths.setdefault("model_config_path", profile["dataset_config"])
if os.environ.get(profile["data_env"]):
    paths["data_root"] = os.environ[profile["data_env"]]
if DATA_ROOT is not None:
    paths["data_root"] = str(DATA_ROOT)
if CHECKPOINT_PATH is not None:
    paths["checkpoint_path"] = str(CHECKPOINT_PATH)
    # An explicit local checkpoint always wins over a configured remote artifact.
    paths["checkpoint_hf_repo"] = None
    paths["checkpoint_hf_filename"] = None

data_root = Path(paths["data_root"])
if not data_root.is_absolute():
    data_root = (REPO_ROOT / data_root).resolve()

checkpoint_path = Path(paths["checkpoint_path"])
if not checkpoint_path.is_absolute():
    checkpoint_path = REPO_ROOT / checkpoint_path
checkpoint_path = ensure_checkpoint(
    checkpoint_path,
    repo_id=paths.get("checkpoint_hf_repo"),
    filename=paths.get("checkpoint_hf_filename"),
    revision=paths.get("checkpoint_hf_revision"),
)

# The supplied LEVIR artifact is a split PyTorch archive: tensor records are in the .zip and
# its pickle metadata is the adjacent best.pt/data.pkl. Assemble a loadable cached .pt once.
if DATASET == "levir_cd" and checkpoint_path.suffix.lower() == ".zip":
    pickle_path = REPO_ROOT / profile["checkpoint_pickle"]
    if not pickle_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint metadata sidecar: {pickle_path}")
    prepared_path = REPO_ROOT / "data" / ".checkpoints" / checkpoint_path.stem
    source_mtime = max(checkpoint_path.stat().st_mtime, pickle_path.stat().st_mtime)
    if not prepared_path.is_file() or prepared_path.stat().st_mtime < source_mtime:
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = prepared_path.with_suffix(prepared_path.suffix + ".tmp")
        shutil.copyfile(checkpoint_path, temporary_path)
        with zipfile.ZipFile(temporary_path, mode="a", compression=zipfile.ZIP_STORED) as archive:
            roots = {name.split("/", 1)[0] for name in archive.namelist() if "/" in name}
            if len(roots) != 1:
                raise RuntimeError(f"Unexpected checkpoint archive layout: {sorted(roots)}")
            archive.write(pickle_path, arcname=f"{roots.pop()}/data.pkl")
        temporary_path.replace(prepared_path)
    checkpoint_path = prepared_path
paths["checkpoint_path"] = str(checkpoint_path)

prepare_data_root = getattr(dataset_module, "prepare_data_root", None)
if prepare_data_root is not None:
    data_root = prepare_data_root(data_root, download=AUTO_DOWNLOAD_DATA)
elif not data_root.is_dir():
    raise FileNotFoundError(
        f"LEVIR-CD root not found: {data_root}. Set DATA_ROOT in the first cell or "
        f"the {profile['data_env']} environment variable."
    )
paths["data_root"] = str(data_root)

model_config_path = Path(paths["model_config_path"])
if not model_config_path.is_absolute():
    model_config_path = REPO_ROOT / model_config_path
paths["model_config_path"] = str(model_config_path)

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

checkpoint_path = paths["checkpoint_path"]
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

pruned_names = None
if DATASET == "valais_bmd":
    from valais_bmd.dataset import find_pruned_names_pickle, load_pruned_names
    pruned_path = find_pruned_names_pickle(paths["data_root"], "test")
    if pruned_path:
        pruned_names = load_pruned_names(pruned_path)
        print(f"Using curated test list: {pruned_path}")

if DATASET == "b_flair":
    print("b-FLAIR protocol: flat test-only index, 5-band TIFF -> RGB, no threshold tuning.")
elif DATASET == "valais_bmd":
    print("ValaisCD protocol: 2017/2023 pairs, binary {0,1} masks, curated test filter.")
elif DATASET == "levir_cd":
    print("LEVIR-CD protocol: split A/B/label layout, binary masks, bundled QGMamba weights.")
else:
    print("S2Looking protocol: split Image1/Image2/label layout and {0,255} masks.")

def index_builder(root, split):
    if DATASET == "levir_cd":
        return dataset_module.parse_split(root, split)
    if DATASET == "b_flair":
        return dataset_module.build_index(root)  # flat layout, no split argument
    if DATASET == "valais_bmd":
        # pruned_names is a TEST-split-specific artifact (test_imgs_filtered_5000_2.pkl);
        # applying it to train/val would filter out nearly every triplet.
        return dataset_module.build_index(root, split, pruned_names=pruned_names if split == "test" else None)
    return dataset_module.build_index(root, split)

bundle = build_datasets(
    config.data,
    CONTEXT,
    seed=42,
    batch_size=config.train.batch_size,
    index_builder=index_builder,
    readers=getattr(dataset_module, "READERS", DEFAULT_READERS),
)
print(f"train={len(bundle.train_dataset)} val={len(bundle.val_dataset)} test={len(bundle.test_dataset)}")
"""

CELL_THRESHOLD_SELECTION = """\
# Dataset-specific threshold selection happens before final test inference.
if SELECT_THRESHOLD_ON_VALIDATION:
    if DATASET == "b_flair":
        raise ValueError(
            "b-FLAIR is test-only. Tuning on it would leak test labels; use the checkpoint threshold."
        )
    threshold_grid = ds_config.get("threshold_grid", {})
    grid_spec = threshold_grid.get("final")
    if not grid_spec:
        raise ValueError(f"No validation threshold grid configured for {DATASET}")
    start, stop, step = grid_spec
    candidate_thresholds = [
        round(start + step * i, 4)
        for i in range(int(round((stop - start) / step)) + 1)
    ]
    _, _, threshold_rows = sweep_thresholds(
        model, bundle.val_dataset, candidate_thresholds, config.eval, CONTEXT, amp=True
    )
    if threshold_grid.get("selection_strategy") == "plateau":
        threshold = float(pick_threshold_plateau(threshold_rows))
        selection_method = "validation plateau"
    else:
        threshold = float(max(threshold_rows, key=lambda row: row["f1"])["threshold"])
        selection_method = "validation F1 argmax"
else:
    selection_method = "checkpoint metadata"

print(f"Operating threshold: {threshold:.4f} ({selection_method})")
"""

CELL_INFERENCE = """\
# Final inference uses the selected profile's held-out test index and eval/tiling configuration.
inference_metrics = evaluate_threshold(
    model, bundle.test_dataset, threshold, config.eval, CONTEXT, amp=True, final=True
)
print(f"{DATASET} inference metrics:", inference_metrics)
"""

CELL_VISUALIZE = """\
figure = qg_show(
    model, bundle.test_dataset, threshold, DEVICE, n=VISUALIZE_SAMPLES,
    title=f"{DATASET} inference", prefer_changed=True,
    tile=config.eval.tile_size, stride=config.eval.tile_stride,
)
"""

CELL_BENCHMARK = """\
efficiency = profile_model(model, input_size=config.model.input_size, device=DEVICE)
print(efficiency)
"""

CELLS = [
    ("markdown", MARKDOWN_INTRO),
    ("code", CELL_SELECT_DATASET),
    ("markdown", "## Setup\n\nResolve the selected profile, dataset source, checkpoint, and model."),
    ("code", CELL_IMPORTS),
    ("code", CELL_LOAD_DATASET_MODULE),
    ("code", CELL_BUILD_MODEL),
    ("markdown", "## Data\n\nBuild the dataset-specific index and readers without changing its protocol."),
    ("code", CELL_BUILD_DATA),
    ("markdown", "## Inference\n\nApply the permitted threshold policy before evaluating held-out test data."),
    ("code", CELL_THRESHOLD_SELECTION),
    ("code", CELL_INFERENCE),
    ("markdown", "## Qualitative results\n\nInspect representative changed samples with the same tiling settings."),
    ("code", CELL_VISUALIZE),
    ("markdown", "## Efficiency\n\nProfile the selected model architecture on the active device."),
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
