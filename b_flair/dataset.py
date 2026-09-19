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
