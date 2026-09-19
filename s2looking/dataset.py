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
