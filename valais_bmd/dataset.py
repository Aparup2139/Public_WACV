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
