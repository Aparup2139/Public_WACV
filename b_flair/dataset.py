from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from qgmamba_cd.data import Readers, SplitPaths

_EXTENSIONS = (".tif", ".tiff")
_HF_REPO_ID = "elliotvincent/b-FLAIR-test"
_EXPECTED_COUNT = 1730


def _triplet_count(root: Path) -> int:
    directories = [root / name for name in ("t1", "t2", "annotations")]
    if not all(directory.is_dir() for directory in directories):
        return 0
    names = [
        {path.name for path in directory.iterdir() if path.suffix.lower() in _EXTENSIONS}
        for directory in directories
    ]
    return len(set.intersection(*names))


def prepare_data_root(root: str | Path, *, download: bool = False) -> Path:
    """Download and validate the official b-FLAIR test dataset from the Hugging Face platform."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if _triplet_count(root) == _EXPECTED_COUNT:
        return root
    if not download:
        raise FileNotFoundError(f"b-FLAIR is incomplete at {root}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Automatic b-FLAIR setup requires huggingface_hub") from exc
    snapshot_download(
        repo_id=_HF_REPO_ID,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=["t1/*.tif", "t2/*.tif", "annotations/*.tif", "metadata.json", "README.md"],
    )
    count = _triplet_count(root)
    if count != _EXPECTED_COUNT:
        raise RuntimeError(f"Expected {_EXPECTED_COUNT} b-FLAIR triplets, found {count} at {root}")
    return root


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
