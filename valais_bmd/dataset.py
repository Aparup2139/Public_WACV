from __future__ import annotations

import pickle
import shutil
import tempfile
import zipfile
from pathlib import Path

from qgmamba_cd.data import Readers, SplitPaths, make_grayscale_mask_reader, read_rgb_cv2
READERS = Readers(read_image=read_rgb_cv2, read_mask=make_grayscale_mask_reader(threshold=0))

_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_EXPECTED_SPLITS = ("train", "val", "test")
_ARCHIVE_SIZES = {
    "train": 3_750_792_668,
    "val": 603_524_101,
    "test": 1_139_409_214,
}
_HF_REPO_ID = "EPFL-ECEO/ValaisCD"


def _complete_triplet_count(root: Path, split: str) -> int:
    split_root = root / split
    directories = [split_root / name for name in ("2017", "2023", "labels")]
    if not all(directory.is_dir() for directory in directories):
        return 0
    names = [
        {path.name for path in directory.iterdir() if path.suffix.lower() in _EXTENSIONS}
        for directory in directories
    ]
    return len(set.intersection(*names))


def prepare_data_root(root: str | Path, *, download: bool = False) -> Path:
    """Return a validated ValaisCD root, optionally fetching missing official splits.

    The official Hub repository stores train/val/test as ZIP files. Downloads are verified
    by their published byte sizes before extraction, and extraction happens in a temporary
    directory so an interrupted run cannot replace a usable split.
    """
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    missing = [split for split in _EXPECTED_SPLITS if _complete_triplet_count(root, split) == 0]
    if not missing:
        return root
    if not download:
        raise FileNotFoundError(
            f"ValaisCD is incomplete at {root}. Missing complete triplets for: {', '.join(missing)}. "
            "Call prepare_data_root(path, download=True) or set AUTO_DOWNLOAD_DATA=True in the notebook."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Automatic ValaisCD download requires huggingface_hub. Run: pip install huggingface_hub"
        ) from exc

    for split in missing:
        archive_name = f"{split}.zip"
        archive = Path(hf_hub_download(
            repo_id=_HF_REPO_ID,
            repo_type="dataset",
            filename=archive_name,
            local_dir=root,
            force_download=(root / archive_name).exists()
            and (root / archive_name).stat().st_size != _ARCHIVE_SIZES[split],
        ))
        if archive.stat().st_size != _ARCHIVE_SIZES[split]:
            raise RuntimeError(
                f"Incomplete {archive_name}: got {archive.stat().st_size:,} bytes; "
                f"expected {_ARCHIVE_SIZES[split]:,}. Re-run the cell to resume downloading."
            )
        with tempfile.TemporaryDirectory(prefix=f"valais-{split}-", dir=root) as temporary:
            temporary_root = Path(temporary)
            with zipfile.ZipFile(archive) as source:
                source.extractall(temporary_root)
            extracted = temporary_root / split
            if _complete_triplet_count(temporary_root, split) == 0:
                raise RuntimeError(f"{archive_name} extracted without complete image/label triplets")
            destination = root / split
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(extracted), str(destination))
    return root


def find_pruned_names_pickle(root: str | Path, split: str = "test") -> Path | None:
    candidate = Path(root) / split / "test_imgs_filtered_5000_2.pkl"
    return candidate if candidate.is_file() else None


def load_pruned_names(pickle_path: str | Path) -> set[str]:
    # Trusted, locally-produced file (the original notebook's own preprocessing output),
    # not untrusted external input -- plain pickle.load matches the notebook, no restricted
    # unpickler needed here.
    with Path(pickle_path).open("rb") as handle:
        names = pickle.load(handle)
    return set(names)


def build_index(root: str | Path, split: str, pruned_names: set[str] | None = None) -> SplitPaths:
    root = Path(root)
    dir_2017 = root / split / "2017"
    dir_2023 = root / split / "2023"
    dir_labels = root / split / "labels"
    if not all(path.is_dir() for path in (dir_2017, dir_2023, dir_labels)):
        raise FileNotFoundError(
            f"ValaisCD split '{split}' is incomplete under {root}; expected 2017/, 2023/, labels/. "
            "Run prepare_data_root(root, download=True) first."
        )

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
