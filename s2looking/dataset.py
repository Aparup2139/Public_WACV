from __future__ import annotations

from pathlib import Path

from qgmamba_cd.data import DEFAULT_READERS, SplitPaths

READERS = DEFAULT_READERS  # 3-band PNG via cv2, {0,255} masks -> threshold 127, both match package defaults

_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_HF_REPO_ID = "EVER-Z/torchange_s2looking"
_EXPECTED_COUNTS = {"train": 3500, "val": 500, "test": 1000}


def _triplet_count(root: Path, split: str) -> int:
    directories = [root / split / name for name in ("Image1", "Image2", "label")]
    if not all(directory.is_dir() for directory in directories):
        return 0
    names = [
        {path.name for path in directory.iterdir() if path.suffix.lower() in _EXTENSIONS}
        for directory in directories
    ]
    return len(set.intersection(*names))


def prepare_data_root(root: str | Path, *, download: bool = False) -> Path:
    """Materialize the Hugging Face S2Looking Parquet dataset into the expected layout."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    missing = [split for split, count in _EXPECTED_COUNTS.items() if _triplet_count(root, split) < count]
    if not missing:
        return root
    if not download:
        raise FileNotFoundError(f"S2Looking is incomplete at {root}; missing: {', '.join(missing)}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Automatic S2Looking setup requires the 'datasets' package") from exc

    for split in missing:
        dataset = load_dataset(_HF_REPO_ID, split=split)
        split_root = root / split
        for directory in (split_root / "Image1", split_root / "Image2", split_root / "label"):
            directory.mkdir(parents=True, exist_ok=True)
        for row_number, row in enumerate(dataset):
            name = Path(row.get("image_name") or f"{row_number:06d}.png").stem + ".png"
            outputs = {
                "Image1": row["t1_image"],
                "Image2": row["t2_image"],
                "label": row["change_mask"],
            }
            for directory_name, image in outputs.items():
                output = split_root / directory_name / name
                if not output.exists():
                    image.save(output)
        if _triplet_count(root, split) != _EXPECTED_COUNTS[split]:
            raise RuntimeError(f"S2Looking {split} materialization did not produce the expected triplets")
    return root


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
