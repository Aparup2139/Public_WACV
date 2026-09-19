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
