# Multi-Dataset Notebook Modularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three standalone, duplicated-logic notebooks (`b-flair-test-zero-shot-s2looking (1).ipynb`, `unet-s2looking_best (1) (5).ipynb`, `valais-bmd-cd (4).ipynb`) with one common `experiments.ipynb` that runs the QGMamba-Diff-CD pipeline against any of the three datasets by swapping a single dataset-name variable, pulling each dataset's unique code/paths/checkpoints from its own root-level folder.

**Architecture:** `qgmamba_cd/` already contains a complete, dataset-agnostic model/train/eval/checkpoint pipeline (`model.py`, `engine.fit`, `evaluation.py`, `checkpoint.py`). It has exactly one dataset-specific seam — how `(image_a, image_b, mask)` triples are located on disk and read into arrays — and today that seam is hardcoded to LEVIR-CD's `root/{A,B,label}` + `root/list/{split}.txt` layout (`data.py::parse_split`, `_read_rgb`, `_read_mask`). None of the three notebooks ever plugged into this seam: each reimplemented its own dataset class, training loop, and checkpoint loader from scratch, which is why they diverged and why bugs like "mask threshold `>127` silently zeroing Valais's `{0,1}` labels" had to be independently rediscovered per notebook. This plan widens that one seam (an injectable `index_builder` + `Readers` pair) instead of duplicating the rest of the pipeline, gives each dataset a thin folder holding only its true unique code (~30-100 lines) plus its own `config.yaml` and `checkpoints/`, and drives all three from one notebook that only ever changes which folder it imports.

**Tech Stack:** Python, PyTorch, OpenCV, tifffile, Albumentations, pandas/numpy, pytest, nbformat (for programmatic notebook generation).

**Spec:** This plan file is self-contained; it was derived from reading `qgmamba_cd/{data,engine,evaluation,checkpoint,losses,config}.py` directly and from full-notebook analyses of all three source notebooks (summarized in the "Source notebook findings" section below). There is no separate spec document.

## Source notebook findings (reference — do not re-derive, cite this section)

| Aspect | S2Looking (from `unet-s2looking_best (1) (5).ipynb`, cells 7-12) | Valais BMD (`valais-bmd-cd (4).ipynb`) | b-FLAIR (`b-flair-test-zero-shot-s2looking (1).ipynb`) |
|---|---|---|---|
| Directory layout | `<root>/<split>/{Image1,Image2,label}/*` | `<root>/<split>/{2017,2023,labels}/*`, filtered by pruned pickle `test_imgs_filtered_{version}_2.pkl` | flat `t1/`, `t2/`, `annotations/` + `metadata.json`, no splits |
| Image reader | `cv2.imread` → BGR2RGB (3-band PNG) | `cv2.imread` → BGR2RGB (3-band) | `tifffile.imread` (5-band TIFF: RGB+IR+Elevation), keep first 3 bands only |
| Mask encoding | `{0,255}`, threshold `> 127` | `{0,1}`, threshold `> 0` (a `>127` threshold silently zeros every label — guarded in-notebook with an assert) | `{0,255}`, threshold `> 0` (works for both 0/1 and 0/255) |
| Unique per-dataset dataset class | `S2LookingDataset` | `ValaisCDDataset` + `ValaisPairDataset` adapter | `BFlairCDDataset` + `BFlairPairDataset` adapter (plus dead aliases `ValaisCDDataset = S2LookingDataset = BFlairCDDataset`) |
| Model build | `build_qgmamba()`: `load_config(yaml)` → force `use_ddim_at_inference=True` → `QGMambaDiffCD(cfg.model)` — duplicated near-verbatim in all 3 notebooks | same | same |
| Checkpoint load | hand-rolled ~30-40 line block duplicating `qgmamba_cd.checkpoint.normalize_state_dict` + `eval_weights` switch, which **already exists** as `load_checkpoint`/`initialize_from_checkpoint` in the package but was never called | same hand-rolled pattern, plus cross-dataset transfer-reset logic (drop epoch/optimizer/history when initializing from the S2Looking checkpoint) | same hand-rolled pattern |
| Training loop | hand-rolled ~150 line loop (AdamW + layerwise groups + OneCycleLR + EMA + bf16/fp16 autocast + top-K checkpoints) duplicating `qgmamba_cd.engine.fit`, which **already exists** but was never called | same hand-rolled pattern (`nn.DataParallel`-wrapped for 2×T4) | none (zero-shot eval only) |
| Threshold selection | plain argmax over a grid via `sweep_thresholds` (already generic/reusable) | `pick_threshold_plateau`: median of the contiguous run of thresholds within 1% of best F1 (new logic, not in package) — needed because Valais positives are extremely sparse (~0.18% of pixels) | plain argmax (reused) |
| Visualization | `qg_show`/`denorm` (4-panel: pre/post/GT/pred) | same, `prefer_changed=True` bias | same |
| Efficiency benchmarking | duplicated FLOPs/latency/memory profiling cell (fully dataset-agnostic) | same duplicated cell | not present |
| Diagnostics | broken/dead cell (undefined `_`, never fixed) attempting size-binned recall | working size-binned component recall (`cv2.connectedComponentsWithStats`), Valais-specific bin edges | none |
| Env | PACE HPC absolute paths (`/storage/home/hcoda1/...`) | Kaggle absolute paths (`/kaggle/input/...`, `/kaggle/working/...`) | local Windows paths (`D:\Public_WACV\...`) |

## Global Constraints

- Do not change `qgmamba_cd`'s existing public behavior for LEVIR-CD: `parse_split`, `_read_rgb`, `_read_mask`, `LEVIRPatchDataset`, `LEVIRFullDataset`, `build_datasets`, `fit` must keep working unmodified for existing LEVIR configs/checkpoints (add optional parameters with defaults that reproduce current behavior; never remove or rename without a backward-compatible alias).
- Every new per-dataset module goes in a root-level folder named after the dataset (`s2looking/`, `valais_bmd/`, `b_flair/`) — not nested under a shared `datasets/` wrapper — per explicit user instruction.
- No new third-party dependencies beyond what the notebooks already use (`tifffile`, `albumentations`, `nbformat` for notebook generation). `nbformat` ships with Jupyter; check it's already importable before adding it to any requirements file.
- Checkpoint files (`*.pt`) and generated results (`*.json`, `*.csv`) are binary/large — they go in each dataset folder's `checkpoints/` and `results/` subfolders and must be `.gitignore`d, never committed.
- **Per-dataset `config.yaml` overrides (`data`/`train`/`eval` sections) must actually be applied to the `ExperimentConfig` object, not just sit in the YAML file.** `qgmamba_cd.config.load_config` only ever reads the base model-architecture YAML (e.g. `config/levir_cd_accuracy_4x40gb.yaml`); the per-dataset `config.yaml` files this plan creates are a separate, smaller schema (`paths` + override sections) that the common notebook must explicitly merge onto the loaded `ExperimentConfig` — see the `apply_overrides` addition in Task 3. Skipping this would make every dataset-tuned value in `s2looking/config.yaml`/`valais_bmd/config.yaml` (crop-focus probability, pos_weight, eval tiling) silently inert.
- Follow the label1/label2 YAGNI note: the current model/loss pipeline is single-channel binary change detection; none of the three notebooks' new/demolished sub-masks (`label1`/`label2` for S2Looking, duplicate placeholders for Valais/b-FLAIR) feed the model or loss. Do not port them — only the single "any change" mask is needed. If multi-class change typing becomes a real requirement later, extend `SplitPaths`/`Readers` then.
- **`estimate_pos_weight`'s exact formula must not change.** The real implementation (`qgmamba_cd/losses.py:201`) is `sqrt((total - positives) / (positives + 1e-6))` clipped to `[1.0, 4.0]` — not a plain ratio. Any task touching this function may only swap the `cv2.imread`/`>127` lines for an injected reader; the return-statement math is copied verbatim.
- **All three source notebooks hardcode `pos_weight` rather than computing it**: S2Looking used `QG_POS_WEIGHT=3.0`, Valais used `QG_POS_WEIGHT=5.0` (which exceeds `estimate_pos_weight`'s own `4.0` clip ceiling — the automatic estimator cannot reproduce Valais's tuned value at all). `qgmamba_cd.config.TrainConfig` has no `pos_weight` field today and `engine.fit()` always calls `estimate_pos_weight(...)` unconditionally (`engine.py:260`) — an override hook must be added (Task 2) and each dataset's `config.yaml` must set it explicitly, or fine-tuning Valais/S2Looking under the shared `fit()` would silently use different loss weighting than what the original notebooks tuned and reported results for.
- **Checkpoint loading must reproduce the notebooks' exact semantics, not `qgmamba_cd.checkpoint.initialize_from_checkpoint`.** All three notebooks: (1) read the checkpoint's own `eval_weights` field (`"ema"` or `"model"`) to decide which state dict to load — not a caller-supplied `prefer_ema` flag; (2) run it through `normalize_state_dict`; (3) load with `strict=True`, so any architecture/config mismatch fails loudly (this is explicitly relied on in the Valais notebook to catch the S2Looking-checkpoint/Valais-config architecture incompatibility). `initialize_from_checkpoint` in the existing package instead takes an externally-supplied `prefer_ema` bool and loads with shape-compatible `strict=False` filtering — a different, more permissive mechanism built for `engine.build_training_objects`'s cross-architecture warm-start use case. Task 3 must not substitute one for the other.

---

## File Structure

```
qgmamba_cd/
  data.py            MODIFY — add Readers, make index_builder/readers injectable, generalize mask threshold
  losses.py          MODIFY — estimate_pos_weight takes an injectable read_mask (formula unchanged)
  config.py          MODIFY — add TrainConfig.pos_weight override field + apply_overrides() helper
  engine.py          MODIFY — fit() threads bundle.readers through to estimate_pos_weight and respects train.pos_weight
  factory.py         CREATE — build_model(config_path, checkpoint_path=None, overrides=None, ...): the build_qgmamba() dedup, with notebook-faithful (eval_weights-driven, strict=True) checkpoint loading
  viz.py             CREATE — denorm(), qg_show()
  benchmark.py       CREATE — profile_model(): FLOPs/latency/memory dedup
  threshold.py       CREATE — pick_threshold_plateau() (reconstructed — verify against the real notebook cell per Task 6)

s2looking/
  __init__.py        CREATE
  config.yaml        CREATE — paths + ExperimentConfig overrides + s2looking-only extras
  dataset.py         CREATE — build_index(root, split) -> SplitPaths (Image1/Image2/label layout)
  checkpoints/       CREATE (empty dir + .gitkeep)
  results/           CREATE (empty dir + .gitkeep)

valais_bmd/
  __init__.py        CREATE
  config.yaml        CREATE
  dataset.py         CREATE — pruned-pkl aware build_index() (2017/2023/labels layout)
  diagnostics.py     CREATE — component_recall_by_size() (Valais-specific bin edges)
  checkpoints/       CREATE (empty dir + .gitkeep)
  results/           CREATE (empty dir + .gitkeep)

b_flair/
  __init__.py        CREATE
  config.yaml        CREATE
  dataset.py         CREATE — build_index() (flat t1/t2/annotations layout) + tifffile Readers
  checkpoints/       CREATE (empty dir + .gitkeep)
  results/           CREATE (empty dir + .gitkeep)

tests/
  test_data_readers.py       CREATE — Readers/index_builder injection, mask-threshold regression (the exact bug class already seen)
  test_dataset_indexes.py    CREATE — each dataset's build_index() against a tmp fixture tree
  test_factory.py            CREATE — build_model() smoke test
  test_threshold.py          CREATE — pick_threshold_plateau() unit tests

scripts/
  build_experiments_notebook.py   CREATE — generates experiments.ipynb via nbformat (source of truth, not hand-edited JSON)

experiments.ipynb    CREATE (generated, committed)
.gitignore           CREATE — checkpoints/, results/, __pycache__, .pytest_cache
```

---

### Task 1: Generalize `qgmamba_cd/data.py` behind an injectable index-builder + readers seam

**Files:**
- Modify: `qgmamba_cd/data.py`
- Test: `tests/test_data_readers.py`

**Interfaces:**
- Produces: `Readers` (frozen dataclass: `read_image: Callable[[Path], np.ndarray]`, `read_mask: Callable[[Path], np.ndarray]`), `DEFAULT_READERS: Readers`, `make_grayscale_mask_reader(threshold: int) -> Callable[[Path], np.ndarray]`, `read_rgb_cv2(path: Path) -> np.ndarray` (renamed-and-exported `_read_rgb`), `IndexBuilder = Callable[[str | Path, str], SplitPaths]`, `PairedPatchDataset` (renamed `LEVIRPatchDataset`, now takes `readers: Readers = DEFAULT_READERS`), `PairedFullDataset` (renamed `LEVIRFullDataset`, same), `LEVIRPatchDataset = PairedPatchDataset` and `LEVIRFullDataset = PairedFullDataset` (backward-compat aliases), `DatasetBundle` (add `readers: Readers` field), `build_datasets(cfg, context, seed, batch_size, index_builder: IndexBuilder = parse_split, readers: Readers = DEFAULT_READERS) -> DatasetBundle`.
- Consumes: nothing new from other tasks — this is the foundation task, do it first.

- [ ] **Step 1: Write the failing regression test for the mask-threshold bug class**

```python
# tests/test_data_readers.py
import numpy as np
import cv2
from pathlib import Path

from qgmamba_cd.data import make_grayscale_mask_reader, read_rgb_cv2, DEFAULT_READERS


def test_grayscale_mask_reader_default_threshold_matches_legacy_127(tmp_path):
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), np.array([[0, 100, 200, 255]], dtype=np.uint8))
    reader = make_grayscale_mask_reader(threshold=127)
    result = reader(mask_path)
    assert result.tolist() == [[0, 0, 1, 1]]


def test_grayscale_mask_reader_zero_threshold_for_binary_zero_one_masks(tmp_path):
    # ValaisCD-style masks are {0, 1}, not {0, 255}; threshold=127 would zero everything.
    mask_path = tmp_path / "mask01.png"
    cv2.imwrite(str(mask_path), np.array([[0, 1, 1, 0]], dtype=np.uint8))
    reader = make_grayscale_mask_reader(threshold=0)
    result = reader(mask_path)
    assert result.sum() == 2  # would be 0 with the old hardcoded >127 threshold


def test_default_readers_read_mask_reproduces_legacy_behavior(tmp_path):
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), np.array([[0, 200]], dtype=np.uint8))
    assert DEFAULT_READERS.read_mask(mask_path).tolist() == [[0, 1]]


def test_read_rgb_cv2_converts_bgr_to_rgb(tmp_path):
    image_path = tmp_path / "img.png"
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[0, 0] = (10, 20, 30)  # BGR
    cv2.imwrite(str(image_path), bgr)
    rgb = read_rgb_cv2(image_path)
    assert rgb[0, 0].tolist() == [30, 20, 10]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_readers.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_grayscale_mask_reader' from 'qgmamba_cd.data'`

- [ ] **Step 3: Implement the generalized readers/index-builder seam**

In `qgmamba_cd/data.py`, replace the module-level `_read_rgb`/`_read_mask` functions and thread injectable readers through both dataset classes and `build_datasets`:

```python
from dataclasses import dataclass, field
from typing import Callable

@dataclass(frozen=True)
class Readers:
    read_image: Callable[[Path], np.ndarray]
    read_mask: Callable[[Path], np.ndarray]


def read_rgb_cv2(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def make_grayscale_mask_reader(threshold: int = 127) -> Callable[[Path], np.ndarray]:
    def _read(path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"OpenCV could not read mask: {path}")
        return (mask > threshold).astype(np.uint8)
    return _read


DEFAULT_READERS = Readers(read_image=read_rgb_cv2, read_mask=make_grayscale_mask_reader(127))

IndexBuilder = Callable[["str | Path", str], "SplitPaths"]
```

Keep `_read_rgb = read_rgb_cv2` and `_read_mask = DEFAULT_READERS.read_mask` as private aliases so any other internal call sites in this file keep working unchanged.

Rename `LEVIRPatchDataset` → `PairedPatchDataset` and `LEVIRFullDataset` → `PairedFullDataset`. Add `readers: Readers = DEFAULT_READERS` as a keyword-only constructor argument on both, and replace every internal `_read_rgb(...)`/`_read_mask(...)` call inside `__getitem__` with `self.readers.read_image(...)`/`self.readers.read_mask(...)`. At the end of the class definitions add:

```python
LEVIRPatchDataset = PairedPatchDataset
LEVIRFullDataset = PairedFullDataset
```

Update `DatasetBundle` to carry the readers used to build it:

```python
@dataclass
class DatasetBundle:
    train_dataset: PairedPatchDataset
    train_loader: DataLoader
    train_sampler: DistributedSampler | None
    val_dataset: PairedFullDataset
    test_dataset: PairedFullDataset
    train_paths: SplitPaths
    readers: Readers
```

Update `build_datasets` signature and body:

```python
def build_datasets(
    cfg: DataConfig,
    context: DistributedContext,
    seed: int,
    batch_size: int,
    index_builder: IndexBuilder = parse_split,
    readers: Readers = DEFAULT_READERS,
) -> DatasetBundle:
    train_transform, eval_transform = build_transforms(cfg)
    train_paths = subset_split(index_builder(cfg.root, "train"), cfg.subset_fraction)
    val_paths = subset_split(index_builder(cfg.root, "val"), cfg.subset_fraction)
    test_paths = subset_split(index_builder(cfg.root, "test"), cfg.subset_fraction)
    train_dataset = PairedPatchDataset(
        train_paths,
        patch_size=cfg.image_size,
        transform=train_transform,
        epoch_multiplier=cfg.train_multiplier,
        change_focus_prob=cfg.change_focus_prob,
        min_change_ratio=cfg.min_change_ratio,
        max_tries=cfg.max_crop_tries,
        temporal_swap_prob=cfg.temporal_swap_prob,
        readers=readers,
    )
    # ... (sampler/loader construction unchanged) ...
    return DatasetBundle(
        train_dataset,
        loader,
        sampler,
        PairedFullDataset(val_paths, eval_transform, readers=readers),
        PairedFullDataset(test_paths, eval_transform, readers=readers),
        train_paths,
        readers,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_readers.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Regression-check nothing else in the repo references the old names incompatibly**

Run: `grep -rn "LEVIRPatchDataset\|LEVIRFullDataset\|_read_rgb\|_read_mask" qgmamba_cd config` (PowerShell: `Select-String -Path qgmamba_cd\*.py,config\*.py -Pattern "LEVIRPatchDataset|LEVIRFullDataset|_read_rgb|_read_mask"`)
Expected: only the aliases/definitions inside `data.py` itself, plus any call sites in `engine.py`/`losses.py` handled in Tasks 2-3.

- [ ] **Step 6: Commit**

```bash
git add qgmamba_cd/data.py tests/test_data_readers.py
git commit -m "refactor(data): make image/mask readers and split index-building injectable

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Generalize `estimate_pos_weight`'s mask reading and add a `pos_weight` override hook

**Files:**
- Modify: `qgmamba_cd/losses.py:188-201`
- Modify: `qgmamba_cd/config.py:76-107` (`TrainConfig`)
- Modify: `qgmamba_cd/engine.py:260`
- Test: `tests/test_data_readers.py` (append)

**Interfaces:**
- Consumes: `Readers` from Task 1 (`qgmamba_cd.data.Readers`, `DEFAULT_READERS`).
- Produces: `estimate_pos_weight(paths: SplitPaths, max_files: int = 200, read_mask: Callable[[Path], np.ndarray] | None = None) -> float` (same clipped-sqrt formula, injectable reader only), `TrainConfig.pos_weight: float | None = None`.

**The exact existing formula (do not change):** `float(np.clip(np.sqrt((total - positives) / (positives + 1.0e-6)), 1.0, 4.0))`. This is a real constraint, not a style note: Valais's notebook hardcoded `pos_weight=5.0`, which is *outside* this function's `[1.0, 4.0]` clip range — the automatic estimator can never reproduce it, which is exactly why an override hook (not just an injectable reader) is required.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_data_readers.py
import numpy as np
from qgmamba_cd.data import SplitPaths
from qgmamba_cd.losses import estimate_pos_weight


def test_estimate_pos_weight_uses_injected_read_mask_and_keeps_sqrt_clip_formula(tmp_path):
    mask_path = tmp_path / "m.png"
    mask_path.write_bytes(b"not a real image")  # never actually decoded by the fake reader below

    # 200 pixels total, 2 positive -> sqrt(198/2.000001) = sqrt(99) ~= 9.9497, clipped to 4.0
    paths = SplitPaths(a=[mask_path], b=[mask_path], labels=[mask_path] * 1, names=["m"])

    def fake_read_mask(path):
        mask = np.zeros((10, 20), dtype=np.uint8)
        mask[0, :2] = 1
        return mask

    weight = estimate_pos_weight(paths, read_mask=fake_read_mask)
    assert weight == 4.0  # clipped ceiling, proves the sqrt/clip formula, not a plain ratio, is in effect


def test_estimate_pos_weight_lower_clip_bound(tmp_path):
    mask_path = tmp_path / "m.png"
    mask_path.write_bytes(b"x")
    paths = SplitPaths(a=[mask_path], b=[mask_path], labels=[mask_path], names=["m"])

    def fake_read_mask(path):
        return np.ones((4, 4), dtype=np.uint8)  # all positive -> sqrt(0/16) = 0, clipped up to 1.0

    assert estimate_pos_weight(paths, read_mask=fake_read_mask) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_readers.py -v -k estimate_pos_weight`
Expected: FAIL (`TypeError: estimate_pos_weight() got an unexpected keyword argument 'read_mask'`)

- [ ] **Step 3: Implement, preserving the exact return formula**

```python
# qgmamba_cd/losses.py
def estimate_pos_weight(
    paths: "SplitPaths",
    max_files: int = 200,
    read_mask: "Callable[[Path], np.ndarray] | None" = None,
) -> float:
    if read_mask is None:
        from .data import DEFAULT_READERS
        read_mask = DEFAULT_READERS.read_mask
    step = max(1, len(paths) // max_files)
    positives = 0
    total = 0
    for path in paths.labels[::step]:
        mask = read_mask(path)
        positives += int(mask.sum())
        total += int(mask.size)
    return float(np.clip(np.sqrt((total - positives) / (positives + 1.0e-6)), 1.0, 4.0))
```

Only the loop body's `mask = read_mask(path)` line replaces the old `cv2.imread(...)` + `mask = mask > 127` pair. The `return` statement is byte-for-byte identical to the current implementation — do not "simplify" it to a plain ratio.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_readers.py -v`
Expected: PASS (all tests including the two new ones)

- [ ] **Step 5: Add the `pos_weight` override field to `TrainConfig`**

In `qgmamba_cd/config.py`, add one field to `TrainConfig` (anywhere in the dataclass body, e.g. after `ema_decay`):

```python
    pos_weight: float | None = None  # overrides estimate_pos_weight() when set; some datasets (e.g. Valais) need a value outside its [1, 4] clip range
```

No validation branch is needed in `load_config` — `None` is a valid default and any positive float is acceptable as-is.

- [ ] **Step 6: Thread `bundle.readers` and the override through `engine.fit()`**

In `qgmamba_cd/engine.py:260`, change:
```python
criterion = FullLoss(config.loss, estimate_pos_weight(bundle.train_paths)).to(context.device)
```
to:
```python
pos_weight = config.train.pos_weight
if pos_weight is None:
    pos_weight = estimate_pos_weight(bundle.train_paths, read_mask=bundle.readers.read_mask)
criterion = FullLoss(config.loss, pos_weight).to(context.device)
```

- [ ] **Step 7: Run the full existing test suite to check for regressions**

Run: `pytest -v`
Expected: PASS (no prior tests broken by the signature/field additions — both are purely additive with backward-compatible defaults)

- [ ] **Step 8: Commit**

```bash
git add qgmamba_cd/losses.py qgmamba_cd/config.py qgmamba_cd/engine.py tests/test_data_readers.py
git commit -m "refactor(losses): inject mask reader into estimate_pos_weight; add train.pos_weight override

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Extract `build_qgmamba()` duplication into `qgmamba_cd/factory.py`

**Files:**
- Create: `qgmamba_cd/factory.py`
- Modify: `qgmamba_cd/config.py` (add `apply_overrides`)
- Modify: `qgmamba_cd/__init__.py` (export `build_model`)
- Test: `tests/test_factory.py`

**Interfaces:**
- Consumes: `load_config` (`qgmamba_cd.config`), `QGMambaDiffCD` (`qgmamba_cd.model`), `normalize_state_dict` (`qgmamba_cd.checkpoint`) — **not** `initialize_from_checkpoint`/`load_checkpoint`, which implement a different (permissive, `strict=False`) loading policy meant for `engine.build_training_objects`'s cross-architecture warm-start case. Notebook-style zero-shot/fine-tune evaluation must fail loudly on a mismatch, not silently drop keys — see the Global Constraints note above.
- Produces: `qgmamba_cd.config.apply_overrides(config: ExperimentConfig, overrides: dict[str, dict]) -> ExperimentConfig` (merges `data`/`train`/`eval`/`loss`/`model` section dicts onto an already-loaded config, reusing the same per-field validation as `load_config`'s "unknown key" check — but it does **not** re-run `load_config`'s cross-field validation block such as `model.input_size == data.image_size`; only pass overrides already known to be internally consistent, exactly as each dataset's `config.yaml` values were derived from a working notebook run), `build_model(config_path: str | Path, checkpoint_path: str | Path | None = None, overrides: dict[str, dict] | None = None, device: torch.device | None = None, force_ddim: bool = True, strict: bool = True) -> tuple[QGMambaDiffCD, ExperimentConfig, dict]` — the `dict` is the raw loaded checkpoint (or `{}` if `checkpoint_path` is `None`), so callers can still read `threshold`/`best_f1`/etc. Every dataset notebook cell currently calling its own copy of `build_qgmamba()` + manual checkpoint block becomes one call to this, with identical checkpoint-selection semantics.

- [ ] **Step 1: Write the failing tests — including one that proves strict loading still fails loudly, and one that proves `eval_weights` (not a caller flag) drives ema-vs-model selection**

```python
# tests/test_factory.py
import copy

import torch
import yaml

from qgmamba_cd.factory import build_model


def _write_minimal_config(path):
    path.write_text(yaml.safe_dump({
        "model": {"input_size": 64, "pretrained": False, "encoder_name": "swin_tiny_patch4_window7_224.ms_in1k"},
        "data": {"image_size": 64},
    }))


def test_build_model_without_checkpoint_forces_ddim(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    _write_minimal_config(config_path)

    model, config, checkpoint = build_model(config_path, checkpoint_path=None, device=torch.device("cpu"))

    assert config.model.use_ddim_at_inference is True
    assert model.diffusion.use_ddim
    assert checkpoint == {}


def test_build_model_loads_model_state_when_eval_weights_is_model(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    _write_minimal_config(config_path)
    model, config, _ = build_model(config_path, device=torch.device("cpu"))
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state": model.state_dict(), "threshold": 0.42, "eval_weights": "model"}, ckpt_path)

    reloaded, _, checkpoint = build_model(config_path, checkpoint_path=ckpt_path, device=torch.device("cpu"))

    assert checkpoint["threshold"] == 0.42
    for key, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key])


def test_build_model_prefers_ema_state_when_checkpoint_says_eval_weights_is_ema(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    _write_minimal_config(config_path)
    model, _, _ = build_model(config_path, device=torch.device("cpu"))
    model_state = copy.deepcopy(model.state_dict())
    ema_state = {key: value + 1.0 for key, value in model_state.items()}  # deliberately different values
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state": model_state, "ema_state": ema_state, "eval_weights": "ema"}, ckpt_path)

    reloaded, _, _ = build_model(config_path, checkpoint_path=ckpt_path, device=torch.device("cpu"))

    for key, value in ema_state.items():
        assert torch.equal(value, reloaded.state_dict()[key])  # ema_state won, not model_state


def test_build_model_strict_load_raises_on_key_mismatch(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    _write_minimal_config(config_path)
    model, _, _ = build_model(config_path, device=torch.device("cpu"))
    bad_state = {"totally_unexpected_key": torch.zeros(1)}
    ckpt_path = tmp_path / "bad_ckpt.pt"
    torch.save({"model_state": bad_state}, ckpt_path)

    try:
        build_model(config_path, checkpoint_path=ckpt_path, device=torch.device("cpu"))
        assert False, "expected a strict state_dict load to raise"
    except RuntimeError:
        pass


def test_build_model_applies_dataset_overrides(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    _write_minimal_config(config_path)

    model, config, _ = build_model(
        config_path,
        device=torch.device("cpu"),
        overrides={"data": {"change_focus_prob": 0.55}, "train": {"pos_weight": 3.0}},
    )

    assert config.data.change_focus_prob == 0.55
    assert config.train.pos_weight == 3.0


def test_apply_overrides_rejects_unknown_section(tmp_path):
    from qgmamba_cd.config import apply_overrides, ExperimentConfig

    try:
        apply_overrides(ExperimentConfig(), {"not_a_real_section": {}})
        assert False, "expected a ValueError for an unknown top-level section"
    except ValueError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_factory.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'qgmamba_cd.factory'`)

- [ ] **Step 3: Add `apply_overrides` to `qgmamba_cd/config.py`**

Add this function next to `_merge_dataclass` (reusing it directly — same "unknown key" validation `load_config` already applies per-section):

```python
# qgmamba_cd/config.py
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
```

- [ ] **Step 4: Implement `build_model`, mirroring the notebooks' exact checkpoint-selection and strictness policy**

```python
# qgmamba_cd/factory.py
from __future__ import annotations

from pathlib import Path

import torch

from .checkpoint import normalize_state_dict
from .config import ExperimentConfig, apply_overrides, load_config
from .model import QGMambaDiffCD


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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_factory.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Export from package `__init__.py`**

```python
# qgmamba_cd/__init__.py
from .factory import build_model
from .config import apply_overrides
__all__ = ["ExperimentConfig", "QGMambaDiffCD", "load_config", "apply_overrides", "build_model"]
```

- [ ] **Step 7: Commit**

```bash
git add qgmamba_cd/factory.py qgmamba_cd/config.py qgmamba_cd/__init__.py tests/test_factory.py
git commit -m "feat(factory): add build_model()/apply_overrides() to replace per-notebook build_qgmamba() duplication

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Extract shared visualization into `qgmamba_cd/viz.py`

**Files:**
- Create: `qgmamba_cd/viz.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: `predict_probability_map` (`qgmamba_cd.evaluation`), `Readers`/`DatasetBundle`'s `PairedFullDataset` shape `(image_a, image_b, target, name)` (Task 1).
- Produces: `denorm(image: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray` (single 3-channel image, HWC, `[0,1]` float), `qg_show(model, dataset, threshold, device, n=3, title="", prefer_changed=True, tile=224, stride=224, tta=False, d4_tta=False, amp=True, amp_dtype="float16") -> matplotlib.figure.Figure`.

- [ ] **Step 1: Write the failing test (denorm only — qg_show is a plotting function, verified by a manual smoke run in Task 10, not unit-tested)**

```python
# tests/test_viz.py
import torch
from qgmamba_cd.viz import denorm
from qgmamba_cd.data import IMAGENET_MEAN, IMAGENET_STD


def test_denorm_inverts_normalization():
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    original = torch.full((3, 2, 2), 0.6)
    normalized = (original - mean) / std

    recovered = denorm(normalized)

    assert recovered.shape == (2, 2, 3)
    assert torch.allclose(torch.tensor(recovered), original.permute(1, 2, 0), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_viz.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'qgmamba_cd.viz'`)

- [ ] **Step 3: Implement**

```python
# qgmamba_cd/viz.py
from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD
from .evaluation import predict_probability_map


def denorm(image: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    mean_tensor = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=image.device).view(-1, 1, 1)
    restored = (image * std_tensor + mean_tensor).clamp(0, 1)
    return restored.permute(1, 2, 0).cpu().numpy()


def qg_show(
    model,
    dataset,
    threshold: float,
    device,
    n: int = 3,
    title: str = "",
    prefer_changed: bool = True,
    tile: int = 224,
    stride: int = 224,
    tta: bool = False,
    d4_tta: bool = False,
    amp: bool = True,
    amp_dtype: str = "float16",
):
    indices = list(range(len(dataset)))
    if prefer_changed:
        # ponytail: scans up to the whole dataset to find changed samples (O(n) reads+transforms).
        # The original notebooks avoided this cost with a precomputed change_stats CSV cache;
        # add a similar cache here if this becomes slow on a dataset with thousands of images.
        random.shuffle(indices)
        changed = []
        for i in indices:
            if dataset[i][2].sum() > 0:
                changed.append(i)
            if len(changed) >= n:
                break
        indices = changed if len(changed) >= n else indices
    chosen = random.sample(indices, k=min(n, len(indices)))

    figure, axes = plt.subplots(len(chosen), 4, figsize=(12, 3 * len(chosen)))
    axes = np.atleast_2d(axes)
    for row, index in enumerate(chosen):
        image_a, image_b, target, name = dataset[index]
        probability = predict_probability_map(
            model, image_a, image_b, device, tile, stride, 8, tta, amp, amp_dtype, False, d4_tta
        )
        prediction = (probability > threshold).astype(np.uint8)
        axes[row, 0].imshow(denorm(image_a))
        axes[row, 1].imshow(denorm(image_b))
        axes[row, 2].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        for col, header in enumerate(("pre", "post", "ground truth", "prediction")):
            axes[row, col].set_title(f"{name}: {header}" if col == 0 else header, fontsize=8)
            axes[row, col].axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    return figure
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_viz.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qgmamba_cd/viz.py tests/test_viz.py
git commit -m "feat(viz): extract shared denorm()/qg_show() visualization from notebooks

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Extract shared efficiency benchmarking into `qgmamba_cd/benchmark.py`

**Files:**
- Create: `qgmamba_cd/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `count_parameters(model: torch.nn.Module) -> dict` (total/trainable/buffers), `profile_model(model, input_size: int, device, batch_sizes=(1,4,8), repeats=20) -> dict` (params, latency per batch size, throughput, peak memory if CUDA). FLOPs counting is best-effort (native `torch.utils.flop_counter.FlopCounterMode`, falling back to `None` if unavailable) — do not add `fvcore`/`thop` as new required dependencies; only use them if already importable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark.py
import torch
from qgmamba_cd.benchmark import count_parameters, profile_model


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, a, b):
        return (self.linear(a.mean(dim=(2, 3))),)


def test_count_parameters():
    model = Tiny()
    counts = count_parameters(model)
    assert counts["total"] == counts["trainable"] == 4 * 4 + 4


def test_profile_model_runs_on_cpu():
    model = Tiny()
    result = profile_model(model, input_size=4, device=torch.device("cpu"), batch_sizes=(1, 2), repeats=2)
    assert set(result["latency_ms"].keys()) == {1, 2}
    assert result["params"]["total"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement**

```python
# qgmamba_cd/benchmark.py
from __future__ import annotations

import time

import torch


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return {"total": total, "trainable": trainable, "buffers": buffers}


def profile_model(
    model: torch.nn.Module,
    input_size: int,
    device: torch.device,
    batch_sizes: tuple[int, ...] = (1, 4, 8),
    repeats: int = 20,
    channels: int = 3,
) -> dict:
    model = model.to(device).eval()
    latency_ms: dict[int, float] = {}
    throughput_ips: dict[int, float] = {}
    for batch_size in batch_sizes:
        image_a = torch.randn(batch_size, channels, input_size, input_size, device=device)
        image_b = torch.randn(batch_size, channels, input_size, input_size, device=device)
        with torch.no_grad():
            for _ in range(3):  # warmup
                model(image_a, image_b)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repeats):
                model(image_a, image_b)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        per_call_ms = (elapsed / repeats) * 1000
        latency_ms[batch_size] = per_call_ms
        throughput_ips[batch_size] = batch_size / (per_call_ms / 1000)
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {
        "params": count_parameters(model),
        "latency_ms": latency_ms,
        "throughput_ips": throughput_ips,
        "peak_memory_mb": peak_memory_mb,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add qgmamba_cd/benchmark.py tests/test_benchmark.py
git commit -m "feat(benchmark): extract shared FLOPs/latency/memory profiling from notebooks

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Add `pick_threshold_plateau` to `qgmamba_cd/threshold.py`

**Files:**
- Create: `qgmamba_cd/threshold.py`
- Test: `tests/test_threshold.py`

**Interfaces:**
- Consumes: the `rows` list-of-dicts shape already returned by `qgmamba_cd.evaluation.sweep_thresholds` (each row: `{"threshold": float, "precision", "recall", "f1", "iou", "accuracy"}`).
- Produces: `pick_threshold_plateau(rows: list[dict], tolerance: float = 0.99) -> float`.

**⚠️ Verification required before trusting this implementation:** `pick_threshold_plateau` does not exist anywhere in `qgmamba_cd` today — the algorithm below is reconstructed from the Valais notebook-analysis description ("median of the contiguous run of thresholds within 1% of the best F1 that contains the argmax"), not copied from the notebook's literal source line-by-line (the analysis agent summarized behavior, it did not quote the exact cell). Before this task is considered done, open `valais-bmd-cd (4).ipynb` directly, find the real `pick_threshold_plateau`/threshold-selection cell, and diff its literal logic against the implementation in Step 3 below — in particular confirm: (a) `>=` vs `>` at the tolerance boundary, (b) whether tolerance is applied as `best_f1 * tolerance` or `best_f1 - tolerance`, (c) whether ties in the plateau are broken by median, mean, or "closest to argmax". If the real cell differs, update Step 3 to match the real cell exactly — this task's job is to port existing logic, not to invent a plausible-sounding replacement.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_threshold.py
from qgmamba_cd.threshold import pick_threshold_plateau


def _row(threshold, f1):
    return {"threshold": threshold, "f1": f1}


def test_plateau_picks_median_of_contiguous_near_best_run():
    rows = [
        _row(0.1, 0.50),
        _row(0.2, 0.90),
        _row(0.3, 0.91),
        _row(0.4, 0.92),  # best
        _row(0.5, 0.91),
        _row(0.6, 0.40),
    ]
    # Plateau containing the argmax (0.4) within 1% of 0.92: thresholds 0.2..0.5 all qualify
    # (0.90/0.92 = 0.978 < 0.99 tolerance -> excluded; 0.91/0.92 = 0.989 -> included)
    result = pick_threshold_plateau(rows, tolerance=0.99)
    assert result == 0.4  # median of [0.3, 0.4, 0.5]


def test_plateau_falls_back_to_argmax_for_single_point_peak():
    rows = [_row(0.1, 0.10), _row(0.2, 0.95), _row(0.3, 0.10)]
    assert pick_threshold_plateau(rows, tolerance=0.999) == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_threshold.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement**

```python
# qgmamba_cd/threshold.py
from __future__ import annotations

import statistics


def pick_threshold_plateau(rows: list[dict], tolerance: float = 0.99) -> float:
    sorted_rows = sorted(rows, key=lambda row: row["threshold"])
    best_f1 = max(row["f1"] for row in sorted_rows)
    best_index = max(range(len(sorted_rows)), key=lambda i: sorted_rows[i]["f1"])
    near_best = [row["f1"] >= best_f1 * tolerance for row in sorted_rows]

    start = best_index
    while start > 0 and near_best[start - 1]:
        start -= 1
    end = best_index
    while end < len(sorted_rows) - 1 and near_best[end + 1]:
        end += 1

    plateau_thresholds = [sorted_rows[i]["threshold"] for i in range(start, end + 1)]
    return float(statistics.median(plateau_thresholds))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_threshold.py -v`
Expected: PASS

- [ ] **Step 5: Diff against the real notebook cell (do not skip — see the ⚠️ note above)**

Open `valais-bmd-cd (4).ipynb` in Jupyter or `jupyter nbconvert --to script --stdout "valais-bmd-cd (4).ipynb"` and locate the actual threshold-plateau-selection cell. Compare it line-by-line against Step 3's implementation. If they differ in tolerance direction, boundary operator, or tie-break rule, edit `qgmamba_cd/threshold.py` to match the notebook exactly and add a regression test for the specific case that would have differed (e.g. a threshold exactly at the 1% boundary).

- [ ] **Step 6: Commit**

```bash
git add qgmamba_cd/threshold.py tests/test_threshold.py
git commit -m "feat(threshold): add pick_threshold_plateau for sparse-positive datasets

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Create `s2looking/` dataset folder

**Files:**
- Create: `s2looking/__init__.py`, `s2looking/config.yaml`, `s2looking/dataset.py`, `s2looking/checkpoints/.gitkeep`, `s2looking/results/.gitkeep`
- Test: `tests/test_dataset_indexes.py` (S2Looking section)

**Interfaces:**
- Consumes: `SplitPaths` (`qgmamba_cd.data`), `subset_split` (`qgmamba_cd.data`).
- Produces: `s2looking.dataset.build_index(root: str | Path, split: str) -> SplitPaths`, `s2looking.dataset.READERS: Readers` (re-exports `qgmamba_cd.data.DEFAULT_READERS` — S2Looking images are 3-band PNG read via cv2 and masks are `{0,255}`, both identical to the package default, per the source-notebook findings table above).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset_indexes.py
from pathlib import Path
import numpy as np
import cv2

from s2looking.dataset import build_index, READERS


def _make_pair(root: Path, split: str, name: str):
    for sub in ("Image1", "Image2", "label"):
        (root / split / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / split / "Image1" / f"{name}.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(root / split / "Image2" / f"{name}.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(root / split / "label" / f"{name}.png"), np.zeros((4, 4), dtype=np.uint8))


def test_build_index_finds_complete_triplets_only(tmp_path):
    _make_pair(tmp_path, "train", "0001")
    (tmp_path / "train" / "Image1" / "0002.png").parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp_path / "train" / "Image1" / "0002.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    # 0002 is missing Image2/label -> must be excluded

    result = build_index(tmp_path, "train")

    assert result.names == ["0001.png"]
    assert len(result) == 1


def test_readers_use_cv2_and_threshold_127(tmp_path):
    mask_path = tmp_path / "m.png"
    cv2.imwrite(str(mask_path), np.array([[0, 200]], dtype=np.uint8))
    assert READERS.read_mask(mask_path).tolist() == [[0, 1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_indexes.py -v -k s2looking or test_build_index_finds_complete_triplets_only`
Expected: FAIL (`ModuleNotFoundError: No module named 's2looking'`)

- [ ] **Step 3: Implement `s2looking/dataset.py`**

```python
# s2looking/dataset.py
from __future__ import annotations

from pathlib import Path

from qgmamba_cd.data import DEFAULT_READERS, SplitPaths

READERS = DEFAULT_READERS  # 3-band PNG via cv2, {0,255} masks -> threshold 127, both match package defaults

_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def build_index(root: str | Path, split: str) -> SplitPaths:
    root = Path(root)
    image1_dir = root / split / "Image1"
    image2_dir = root / split / "Image2"
    label_dir = root / split / "label"
    if not image1_dir.exists():
        raise FileNotFoundError(f"S2Looking split layout missing: {image1_dir}")

    a_paths, b_paths, label_paths, names = [], [], [], []
    for candidate in sorted(image1_dir.iterdir()):
        if candidate.suffix.lower() not in _EXTENSIONS:
            continue
        image2_path = image2_dir / candidate.name
        label_path = label_dir / candidate.name
        if image2_path.exists() and label_path.exists():
            a_paths.append(candidate)
            b_paths.append(image2_path)
            label_paths.append(label_path)
            names.append(candidate.name)
    if not names:
        raise ValueError(f"No complete S2Looking triplets found under {root / split}")
    return SplitPaths(a=a_paths, b=b_paths, labels=label_paths, names=names)
```

- [ ] **Step 4: Add `s2looking/__init__.py`**

```python
# s2looking/__init__.py
from .dataset import build_index, READERS

__all__ = ["build_index", "READERS"]
```

- [ ] **Step 5: Write `s2looking/config.yaml`**

Port the S2Looking hyperparameters from `unet-s2looking_best (1) (5).ipynb` cells 2/7/9 (crop-nonempty prob 0.55, `pos_weight`-driving loss config, epochs 140, patience 18) on top of the existing `config/levir_cd_accuracy_4x40gb.yaml` model architecture (swin encoder config already validated against the shipped `qgmamba_s2looking_best_11.pt` checkpoint):

```yaml
# s2looking/config.yaml
# Paths are placeholders — override per machine via env var or notebook cell, never commit real absolute paths.
paths:
  data_root: "${S2LOOKING_DATA_ROOT}"        # expects <root>/{train,val,test}/{Image1,Image2,label}
  checkpoint_path: "s2looking/checkpoints/qgmamba_s2looking_best_11.pt"
  model_config_path: "config/levir_cd_accuracy_4x40gb.yaml"
  output_dir: "s2looking/results"

data:
  image_size: 256
  train_multiplier: 4        # matches the notebook's CROPS_PER_IMAGE=4 (train_df replicated 4x)
  change_focus_prob: 0.55    # matches the notebook's CROP_NONEMPTY_P=0.55 -- do NOT leave this at the LEVIR base config's 0.0
  num_workers: 8
  pin_memory: true

train:
  epochs: 140
  patience: 18
  ema_decay: 0.999
  pos_weight: 3.0             # notebook's QG_POS_WEIGHT was a hardcoded constant, not computed -- requires Task 2's override hook

eval:
  tile_size: 256
  tile_stride: 256
  final_stride: 128
  d4_tta: true

threshold_grid:
  final: [0.40, 0.775, 0.015]   # [start, stop, step] -> matches the 25-point final-protocol grid
  quick: [0.44, 0.80, 0.03]
```

Note in a comment at the top of the file that `paths.data_root` must be supplied via the `S2LOOKING_DATA_ROOT` environment variable or overridden in the notebook — never hardcode the PACE HPC path from the original notebook. `data.change_focus_prob` and `train.pos_weight` are copied from the notebook's tuned constants (`CROP_NONEMPTY_P`, `QG_POS_WEIGHT`), not from `config/levir_cd_accuracy_4x40gb.yaml`'s defaults — verify both against the actual notebook cells (cells 2 and 9 of `unet-s2looking_best (1) (5).ipynb`) before training, the same way Task 6 verifies `pick_threshold_plateau`.

- [ ] **Step 6: Create empty tracked directories**

```bash
mkdir -p s2looking/checkpoints s2looking/results
touch s2looking/checkpoints/.gitkeep s2looking/results/.gitkeep
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_dataset_indexes.py -v`
Expected: PASS (the S2Looking tests)

- [ ] **Step 8: Commit**

```bash
git add s2looking/ tests/test_dataset_indexes.py
git commit -m "feat(s2looking): add dataset folder with index builder, config, checkpoint slot

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 8: Create `valais_bmd/` dataset folder

**Files:**
- Create: `valais_bmd/__init__.py`, `valais_bmd/config.yaml`, `valais_bmd/dataset.py`, `valais_bmd/diagnostics.py`, `valais_bmd/checkpoints/.gitkeep`, `valais_bmd/results/.gitkeep`
- Test: `tests/test_dataset_indexes.py` (Valais section, appended)

**Interfaces:**
- Produces: `valais_bmd.dataset.build_index(root, split, pruned_names: set[str] | None = None) -> SplitPaths`, `valais_bmd.dataset.READERS: Readers` (cv2 reader, `make_grayscale_mask_reader(threshold=0)` — masks are `{0,1}`, per the findings table), `valais_bmd.dataset.load_pruned_names(pickle_path) -> set[str]`, `valais_bmd.diagnostics.component_recall_by_size(prediction, target, bins=((0,50),(50,200),(200,800),(800,10**9))) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dataset_indexes.py
import pickle
from valais_bmd.dataset import build_index, load_pruned_names, READERS


def _make_valais_pair(root, split, name):
    for sub in ("2017", "2023", "labels"):
        (root / split / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / split / "2017" / f"{name}.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(root / split / "2023" / f"{name}.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(root / split / "labels" / f"{name}.png"), np.zeros((4, 4), dtype=np.uint8))


def test_valais_build_index_respects_pruned_name_filter(tmp_path):
    _make_valais_pair(tmp_path, "test", "a")
    _make_valais_pair(tmp_path, "test", "b")

    full = build_index(tmp_path, "test")
    assert set(full.names) == {"a.png", "b.png"}

    pruned = build_index(tmp_path, "test", pruned_names={"a.png"})
    assert pruned.names == ["a.png"]


def test_valais_readers_use_zero_threshold_for_binary_masks(tmp_path):
    mask_path = tmp_path / "m01.png"
    cv2.imwrite(str(mask_path), np.array([[0, 1]], dtype=np.uint8))
    assert READERS.read_mask(mask_path).sum() == 1  # would be 0 with a >127 threshold


def test_load_pruned_names_reads_pickle(tmp_path):
    pickle_path = tmp_path / "names.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(["a.png", "b.png"], handle)
    assert load_pruned_names(pickle_path) == {"a.png", "b.png"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_indexes.py -v -k valais`
Expected: FAIL (`ModuleNotFoundError: No module named 'valais_bmd'`)

- [ ] **Step 3: Implement `valais_bmd/dataset.py`**

```python
# valais_bmd/dataset.py
from __future__ import annotations

import pickle
from pathlib import Path

from qgmamba_cd.data import Readers, SplitPaths, make_grayscale_mask_reader, read_rgb_cv2

# ValaisCD masks are {0,1}, not {0,255} -- a >127 threshold (correct for S2Looking/LEVIR)
# silently zeros every label here. This is the exact bug class the original notebook
# guarded against with a runtime assert; encoding it as threshold=0 removes the class entirely.
READERS = Readers(read_image=read_rgb_cv2, read_mask=make_grayscale_mask_reader(threshold=0))

_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def load_pruned_names(pickle_path: str | Path) -> set[str]:
    with Path(pickle_path).open("rb") as handle:
        names = pickle.load(handle)
    return set(names)


def build_index(root: str | Path, split: str, pruned_names: set[str] | None = None) -> SplitPaths:
    root = Path(root)
    dir_2017 = root / split / "2017"
    dir_2023 = root / split / "2023"
    dir_labels = root / split / "labels"
    if not dir_2017.exists():
        raise FileNotFoundError(f"ValaisCD split layout missing: {dir_2017}")

    a_paths, b_paths, label_paths, names = [], [], [], []
    for candidate in sorted(dir_2017.iterdir()):
        if candidate.suffix.lower() not in _EXTENSIONS:
            continue
        if pruned_names is not None and candidate.name not in pruned_names:
            continue
        image2_path = dir_2023 / candidate.name
        label_path = dir_labels / candidate.name
        if image2_path.exists() and label_path.exists():
            a_paths.append(candidate)
            b_paths.append(image2_path)
            label_paths.append(label_path)
            names.append(candidate.name)
    if not names:
        raise ValueError(f"No complete ValaisCD triplets found under {root / split}")
    return SplitPaths(a=a_paths, b=b_paths, labels=label_paths, names=names)
```

- [ ] **Step 4: Implement `valais_bmd/diagnostics.py`** (ports the working component-recall cell from the Valais notebook, generalized to accept bins as a parameter instead of hardcoding them)

```python
# valais_bmd/diagnostics.py
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_SIZE_BINS = ((0, 50), (50, 200), (200, 800), (800, 10**9))  # pixel-area bins at 0.5m/px GSD


def component_recall_by_size(
    prediction: np.ndarray, target: np.ndarray, bins: tuple[tuple[int, int], ...] = DEFAULT_SIZE_BINS
) -> dict[str, float]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(target.astype(np.uint8), connectivity=8)
    hits = {bin_range: 0 for bin_range in bins}
    totals = {bin_range: 0 for bin_range in bins}
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        for low, high in bins:
            if low <= area < high:
                totals[(low, high)] += 1
                component_mask = labels == component_id
                if (prediction.astype(bool) & component_mask).any():
                    hits[(low, high)] += 1
                break
    return {
        f"{low}-{high}": (hits[(low, high)] / totals[(low, high)] if totals[(low, high)] else float("nan"))
        for low, high in bins
    }
```

- [ ] **Step 5: Add `valais_bmd/__init__.py`**

```python
# valais_bmd/__init__.py
from .dataset import build_index, load_pruned_names, READERS
from .diagnostics import component_recall_by_size

__all__ = ["build_index", "load_pruned_names", "READERS", "component_recall_by_size"]
```

- [ ] **Step 6: Write `valais_bmd/config.yaml`**

Port from `valais-bmd-cd (4).ipynb`: `pos_weight=5.0`, `CHANGE_TARGET_FRAC=0.30`, transfer-learning config incompatible with the S2Looking checkpoint architecture (swin_tiny + `StochasticDiffusionDecoder`, not swin_small + `LegacyDiffusionDecoder`), threshold grid `[0.05, 0.65]` step `0.025` with plateau selection:

```yaml
# valais_bmd/config.yaml
paths:
  data_root: "${VALAIS_DATA_ROOT}"          # expects <root>/{train,val,test}/{2017,2023,labels}
  pruned_names_pickle: "${VALAIS_DATA_ROOT}/test_imgs_filtered_5000_2.pkl"
  transfer_checkpoint_path: "s2looking/checkpoints/qgmamba_s2looking_best_11.pt"  # cross-dataset transfer source
  model_config_path: "config/valais_cd_transfer.yaml"   # swin_tiny + StochasticDiffusionDecoder -- NOT compatible with the S2Looking checkpoint's architecture; strict=True load will fail if pointed at the wrong config
  output_dir: "valais_bmd/results"

data:
  image_size: 256
  change_target_frac: 0.30
  num_workers: 4

train:
  ema_decay: 0.999
  pos_weight: 5.0   # notebook's QG_POS_WEIGHT was a hardcoded constant, deliberately above estimate_pos_weight()'s [1,4] clip range -- requires Task 2's override hook to be applied before this key is valid (otherwise load_config raises "Unknown configuration keys")

eval:
  tile_size: 256
  tile_stride: 256
  final_stride: 256   # a ValaisCD patch already equals one model tile -- no overlap tiling gain here
  d4_tta: true

threshold_grid:
  final: [0.05, 0.675, 0.025]
  selection_strategy: plateau   # use qgmamba_cd.threshold.pick_threshold_plateau, not argmax

diagnostics:
  component_recall_bins: [[0, 50], [50, 200], [200, 800], [800, 1000000000]]
```

- [ ] **Step 7: Create empty tracked directories**

```bash
mkdir -p valais_bmd/checkpoints valais_bmd/results
touch valais_bmd/checkpoints/.gitkeep valais_bmd/results/.gitkeep
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_dataset_indexes.py -v -k valais`
Expected: PASS (4 tests)

- [ ] **Step 9: Commit**

```bash
git add valais_bmd/ tests/test_dataset_indexes.py
git commit -m "feat(valais_bmd): add dataset folder with pruned-index builder, zero-threshold mask reader, diagnostics

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 9: Create `b_flair/` dataset folder

**Files:**
- Create: `b_flair/__init__.py`, `b_flair/config.yaml`, `b_flair/dataset.py`, `b_flair/checkpoints/.gitkeep`, `b_flair/results/.gitkeep`
- Test: `tests/test_dataset_indexes.py` (b-FLAIR section, appended)

**Interfaces:**
- Produces: `b_flair.dataset.build_index(root: str | Path) -> SplitPaths` (no splits — b-FLAIR is flat/full-set-only per the findings table), `b_flair.dataset.READERS: Readers` (tifffile-based, drops IR+Elevation bands to keep the first 3, `>0` mask threshold).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dataset_indexes.py
import tifffile
from b_flair.dataset import build_index, READERS


def test_bflair_build_index_flat_layout(tmp_path):
    for sub in ("t1", "t2", "annotations"):
        (tmp_path / sub).mkdir(parents=True)
    tifffile.imwrite(tmp_path / "t1" / "0001.tif", np.zeros((4, 4, 5), dtype=np.uint8))
    tifffile.imwrite(tmp_path / "t2" / "0001.tif", np.zeros((4, 4, 5), dtype=np.uint8))
    tifffile.imwrite(tmp_path / "annotations" / "0001.tif", np.zeros((4, 4), dtype=np.uint8))

    result = build_index(tmp_path)

    assert result.names == ["0001.tif"]


def test_bflair_read_image_drops_ir_and_elevation_bands(tmp_path):
    image_path = tmp_path / "img5.tif"
    data = np.zeros((4, 4, 5), dtype=np.uint8)
    data[0, 0] = [10, 20, 30, 99, 200]  # R, G, B, IR, Elevation
    tifffile.imwrite(image_path, data)

    result = READERS.read_image(image_path)

    assert result.shape == (4, 4, 3)
    assert result[0, 0].tolist() == [10, 20, 30]


def test_bflair_read_mask_threshold_zero(tmp_path):
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(mask_path, np.array([[0, 255]], dtype=np.uint8))
    assert READERS.read_mask(mask_path).tolist() == [[0, 1]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_indexes.py -v -k bflair`
Expected: FAIL (`ModuleNotFoundError: No module named 'b_flair'`)

- [ ] **Step 3: Implement `b_flair/dataset.py`**

```python
# b_flair/dataset.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from qgmamba_cd.data import Readers, SplitPaths

_EXTENSIONS = (".tif", ".tiff")


def read_rgb_tifffile(path: Path) -> np.ndarray:
    image = tifffile.imread(str(path))
    if image.ndim != 3 or image.shape[-1] < 3:
        raise RuntimeError(f"Expected a multi-band TIFF with >= 3 bands: {path}")
    return image[..., :3]  # drop IR (band 4) and Elevation (band 5); encoder is 3-channel ImageNet Swin


def read_mask_tifffile(path: Path) -> np.ndarray:
    mask = tifffile.imread(str(path))
    if mask.ndim == 3:
        mask = mask.squeeze()
    return (mask > 0).astype(np.uint8)


READERS = Readers(read_image=read_rgb_tifffile, read_mask=read_mask_tifffile)


def build_index(root: str | Path) -> SplitPaths:
    root = Path(root)
    t1_dir, t2_dir, ann_dir = root / "t1", root / "t2", root / "annotations"
    if not t1_dir.exists():
        raise FileNotFoundError(f"b-FLAIR layout missing: {t1_dir}")

    a_paths, b_paths, label_paths, names = [], [], [], []
    for candidate in sorted(t1_dir.iterdir()):
        if candidate.suffix.lower() not in _EXTENSIONS:
            continue
        image2_path = t2_dir / candidate.name
        label_path = ann_dir / candidate.name
        if image2_path.exists() and label_path.exists():
            a_paths.append(candidate)
            b_paths.append(image2_path)
            label_paths.append(label_path)
            names.append(candidate.name)
    if not names:
        raise ValueError(f"No complete b-FLAIR triplets found under {root}")
    return SplitPaths(a=a_paths, b=b_paths, labels=label_paths, names=names)
```

- [ ] **Step 4: Add `b_flair/__init__.py`**

```python
# b_flair/__init__.py
from .dataset import build_index, READERS

__all__ = ["build_index", "READERS"]
```

- [ ] **Step 5: Write `b_flair/config.yaml`**

This notebook is zero-shot-only (evaluating an S2Looking-trained checkpoint against b-FLAIR, no training loop) — the config only needs eval-time settings:

```yaml
# b_flair/config.yaml
paths:
  data_root: "${BFLAIR_DATA_ROOT}"     # expects <root>/{t1,t2,annotations}, flat, no splits
  metadata_path: "${BFLAIR_DATA_ROOT}/metadata.json"
  checkpoint_path: "s2looking/checkpoints/qgmamba_s2looking_best_11.pt"  # zero-shot transfer source
  model_config_path: "config/levir_cd_accuracy_4x40gb.yaml"  # must match checkpoint architecture, NOT configs/bflair_cd_paper.yaml
  output_dir: "b_flair/results"

eval:
  tile_size: 256
  tile_stride: 256
  final_stride: 128
  d4_tta: false

threshold_grid:
  reference_only: [0.05, 0.925, 0.025]   # in-sample reference curve only -- operating threshold comes from the checkpoint's stored value, per the original notebook's zero-shot protocol
```

- [ ] **Step 6: Create empty tracked directories**

```bash
mkdir -p b_flair/checkpoints b_flair/results
touch b_flair/checkpoints/.gitkeep b_flair/results/.gitkeep
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_dataset_indexes.py -v -k bflair`
Expected: PASS (3 tests)

- [ ] **Step 8: Commit**

```bash
git add b_flair/ tests/test_dataset_indexes.py
git commit -m "feat(b_flair): add dataset folder with tifffile readers and flat-layout index builder

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 10: Repo hygiene — `.gitignore` and root `requirements.txt` check

**Files:**
- Create: `.gitignore`
- Modify: none (verify `requirements.txt` presence/content; the original notebooks `%pip install`-ed `albumentations`, `albucore`, `segmentation-models-pytorch`, `timm`, `tifffile` ad hoc — consolidate into one file)

**Interfaces:** none (infra-only task).

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# .gitignore
__pycache__/
*.pyc
.pytest_cache/

# Dataset checkpoints and results are large/binary and machine-specific
s2looking/checkpoints/*
!s2looking/checkpoints/.gitkeep
s2looking/results/*
!s2looking/results/.gitkeep

valais_bmd/checkpoints/*
!valais_bmd/checkpoints/.gitkeep
valais_bmd/results/*
!valais_bmd/results/.gitkeep

b_flair/checkpoints/*
!b_flair/checkpoints/.gitkeep
b_flair/results/*
!b_flair/results/.gitkeep

.ipynb_checkpoints/
```

- [ ] **Step 2: Check/create `requirements.txt`**

Run: `pip freeze | Select-String -Pattern "torch|albumentations|opencv|tifffile|timm|segmentation-models-pytorch|pandas|numpy|matplotlib|pyyaml|nbformat"` (or `pip list` and grep) to see what's already installed, then write a `requirements.txt` covering exactly what the notebooks need:

```
torch
opencv-python
tifffile
albumentations
albucore
timm
pandas
numpy
matplotlib
tqdm
pyyaml
nbformat
pytest
```

Do not pin exact versions unless the current environment shows a version conflict — the original notebooks didn't pin either.

- [ ] **Step 3: Commit**

```bash
git add .gitignore requirements.txt
git commit -m "chore: add .gitignore for checkpoints/results and consolidate requirements.txt

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 11: Generate the common `experiments.ipynb` via `scripts/build_experiments_notebook.py`

**Files:**
- Create: `scripts/build_experiments_notebook.py`
- Create (generated, not hand-edited): `experiments.ipynb`

**Interfaces:**
- Consumes: `build_model` (`qgmamba_cd.factory`, now takes `overrides` per Task 3), `build_datasets` (`qgmamba_cd.data`), `fit` (`qgmamba_cd.engine`), `sweep_thresholds`/`evaluate_threshold` (`qgmamba_cd.evaluation`), `pick_threshold_plateau` (`qgmamba_cd.threshold`), `qg_show` (`qgmamba_cd.viz`), `profile_model` (`qgmamba_cd.benchmark`), and each dataset package's `build_index`/`READERS` (`s2looking`, `valais_bmd`, `b_flair`).
- Produces: `experiments.ipynb` at repo root, and a rebuild command any contributor can re-run after editing cell source: `python scripts/build_experiments_notebook.py`.

Generating the notebook from a script (rather than hand-editing a 3MB JSON blob, which is how the original three notebooks became unmaintainable) keeps the true source of cell content in reviewable `.py`-string form and makes future edits a normal code diff.

- [ ] **Step 1: Confirm `nbformat` is importable**

Run: `python -c "import nbformat; print(nbformat.__version__)"`
Expected: prints a version string. If it fails, `pip install nbformat` (it is a transitive Jupyter dependency and is almost always already present in any environment that can open `.ipynb` files).

- [ ] **Step 2: Write `scripts/build_experiments_notebook.py`**

```python
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
```

- [ ] **Step 3: Generate the notebook**

Run: `python scripts/build_experiments_notebook.py`
Expected: `Wrote D:\Public_WACV\Public_WACV\experiments.ipynb`

- [ ] **Step 4: Validate the generated notebook is well-formed**

Run: `python -c "import nbformat; nb = nbformat.read('experiments.ipynb', as_version=4); nbformat.validate(nb); print('valid, cells:', len(nb.cells))"`
Expected: `valid, cells: 9`

- [ ] **Step 5: Smoke-run against whichever dataset has data available locally** (manual verification — not automatable without real dataset files/checkpoints present)

Set `DATASET` to a dataset whose `data_root`/`checkpoint_path` env vars are set on this machine, run the notebook top-to-bottom (`jupyter nbconvert --to notebook --execute experiments.ipynb --output experiments.ipynb` or run interactively), and confirm the printed zero-shot metrics are in the same ballpark as that dataset's original notebook's final report JSON (`*_metrics.json` in that original notebook's `OUT_DIR`). Record the comparison in the PR description; do not silently accept a metric drift of more than a few points without investigating (most likely cause: a config/checkpoint architecture mismatch, e.g. accidentally pointing Valais at the S2Looking-shaped config).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_experiments_notebook.py experiments.ipynb
git commit -m "feat: add generated common experiments.ipynb driving all three datasets

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 12: Retire the three legacy notebooks

**Files:**
- Create: `archive/` (move destination)
- Modify: none in `qgmamba_cd`/dataset folders

**Interfaces:** none.

- [ ] **Step 1: Confirm Task 11's smoke run (Step 5) matched each notebook's headline metric before archiving anything** — do not archive a legacy notebook until `experiments.ipynb` has been verified against it at least once; if only one dataset's data/checkpoint was available locally, archive only that one notebook now and leave the other two until they can be verified.

- [ ] **Step 2: Move verified notebooks to `archive/`, preserving git history**

```bash
mkdir -p archive
git mv "unet-s2looking_best (1) (5).ipynb" "archive/unet-s2looking_best (1) (5).ipynb"
# repeat per-notebook only after that notebook's dataset has been verified in Step 1
```

- [ ] **Step 3: Add a one-line pointer in `archive/README.md`**

```markdown
# Archived notebooks

Superseded by `experiments.ipynb` (see `docs/superpowers/plans/2026-09-19-notebook-modularization.md`).
Kept for reference to original hyperparameter tuning notes and metric baselines.
```

- [ ] **Step 4: Commit**

```bash
git add archive/
git commit -m "chore: archive legacy per-dataset notebooks, superseded by experiments.ipynb

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** every row of the source-notebook findings table maps to a task — directory layouts → Tasks 7-9 `build_index`; mask thresholds → Task 1 (`make_grayscale_mask_reader`) + Tasks 7-9 (`READERS`); model build duplication → Task 3; checkpoint-load duplication → Task 3 (faithful reimplementation of the notebooks' `eval_weights` + `strict=True` pattern, deliberately not `initialize_from_checkpoint`); training-loop duplication → Task 2 (reuses existing `fit`, threading the reader through and adding the `pos_weight` override the automatic estimator cannot reach); threshold selection → Task 6 (flagged for manual verification against the real cell); visualization → Task 4; benchmarking → Task 5; diagnostics → Task 8 Step 4; environment-specific absolute paths → Tasks 7-9 `config.yaml` env-var placeholders; per-dataset hyperparameter overrides actually reaching the model/training config → Task 3's `apply_overrides` + Task 11's `CELL_BUILD_MODEL`; the one common notebook itself → Task 11; cleanup of the old notebooks → Task 12.

**Placeholder scan:** no task step says "add error handling" or "similar to Task N" without code; every code step above is a complete, runnable snippet an executor can paste in directly.

**Type consistency:** `Readers`, `SplitPaths`, `DatasetBundle`, `EvalConfig`, `ExperimentConfig` are used with identical field names across every task that touches them (verified against the actual current `qgmamba_cd/{data,config,engine,evaluation,checkpoint,losses,model}.py` source read during planning, not from memory).

**Logic-preservation audit (done in response to an explicit request to verify no fundamental logic changed) — four real discrepancies were found by re-reading the actual source and fixed before this plan was finalized:**
1. `estimate_pos_weight`'s original return statement is `sqrt((total-positives)/(positives+1e-6))` clipped to `[1,4]`, not a plain ratio — the first draft of Task 2 silently substituted a different formula. Fixed: Task 2 now copies the exact formula and only swaps the mask-reading line, with a test that specifically proves the sqrt/clip behavior (asserting the clip ceiling is hit) rather than a formula-agnostic assertion.
2. `qgmamba_cd.checkpoint.initialize_from_checkpoint` (`strict=False`, caller-supplied `prefer_ema` flag) is a *different mechanism* from what all three notebooks actually do (checkpoint's own `eval_weights` field selects ema-vs-model, `strict=True` load that fails loudly on mismatch) — the first draft of Task 3 called the former where the notebooks did the latter, which would have silently downgraded every mismatch from a loud failure to a silent partial load. Fixed: Task 3 now reimplements the notebooks' exact selection+strictness policy directly, reusing only `normalize_state_dict`.
3. All three notebooks hardcode `pos_weight` (S2Looking: 3.0, Valais: 5.0 — outside the automatic estimator's `[1,4]` range) rather than computing it, but `TrainConfig` had no override field and `engine.fit()` always called the estimator unconditionally — the per-dataset `config.yaml` files would have silently used the wrong loss weighting. Fixed: added `TrainConfig.pos_weight` and threaded it through `fit()` in Task 2.
4. The per-dataset `config.yaml` files' `data`/`train`/`eval` override sections were never actually merged onto the `ExperimentConfig` object anywhere in the original Task 11 notebook cells — they would have been silently inert. Fixed: added `apply_overrides()` to Task 3 and wired it into Task 11's model-building cell.

`pick_threshold_plateau` (Task 6) could not be verified this way because it doesn't exist in `qgmamba_cd` today and the only source is a natural-language description of the Valais notebook cell, not its literal code — Task 6 now carries an explicit, mandatory step to diff the implementation against that notebook's real cell before relying on it.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-19-notebook-modularization.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
