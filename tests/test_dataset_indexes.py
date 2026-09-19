import pickle
from pathlib import Path
import numpy as np
import cv2
import tifffile

from s2looking.dataset import build_index, READERS
from valais_bmd.dataset import build_index as valais_build_index, load_pruned_names, READERS as VALAIS_READERS
from b_flair.dataset import build_index as bflair_build_index, READERS as BFLAIR_READERS


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


def _make_valais_pair(root, split, name):
    for sub in ("2017", "2023", "labels"):
        (root / split / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / split / "2017" / f"{name}.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(root / split / "2023" / f"{name}.png"), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(root / split / "labels" / f"{name}.png"), np.zeros((4, 4), dtype=np.uint8))


def test_valais_build_index_respects_pruned_name_filter(tmp_path):
    _make_valais_pair(tmp_path, "test", "a")
    _make_valais_pair(tmp_path, "test", "b")

    full = valais_build_index(tmp_path, "test")
    assert set(full.names) == {"a.png", "b.png"}

    pruned = valais_build_index(tmp_path, "test", pruned_names={"a.png"})
    assert pruned.names == ["a.png"]


def test_valais_readers_use_zero_threshold_for_binary_masks(tmp_path):
    mask_path = tmp_path / "m01.png"
    cv2.imwrite(str(mask_path), np.array([[0, 1]], dtype=np.uint8))
    assert VALAIS_READERS.read_mask(mask_path).sum() == 1  # would be 0 with a >127 threshold


def test_load_pruned_names_reads_pickle(tmp_path):
    pickle_path = tmp_path / "names.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(["a.png", "b.png"], handle)
    assert load_pruned_names(pickle_path) == {"a.png", "b.png"}


def test_bflair_build_index_flat_layout(tmp_path):
    for sub in ("t1", "t2", "annotations"):
        (tmp_path / sub).mkdir(parents=True)
    tifffile.imwrite(tmp_path / "t1" / "0001.tif", np.zeros((4, 4, 5), dtype=np.uint8))
    tifffile.imwrite(tmp_path / "t2" / "0001.tif", np.zeros((4, 4, 5), dtype=np.uint8))
    tifffile.imwrite(tmp_path / "annotations" / "0001.tif", np.zeros((4, 4), dtype=np.uint8))

    result = bflair_build_index(tmp_path)

    assert result.names == ["0001.tif"]


def test_bflair_read_image_drops_ir_and_elevation_bands(tmp_path):
    image_path = tmp_path / "img5.tif"
    data = np.zeros((4, 4, 5), dtype=np.uint8)
    data[0, 0] = [10, 20, 30, 99, 200]  # R, G, B, IR, Elevation
    tifffile.imwrite(image_path, data)

    result = BFLAIR_READERS.read_image(image_path)

    assert result.shape == (4, 4, 3)
    assert result[0, 0].tolist() == [10, 20, 30]


def test_bflair_read_mask_threshold_zero(tmp_path):
    mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(mask_path, np.array([[0, 255]], dtype=np.uint8))
    assert BFLAIR_READERS.read_mask(mask_path).tolist() == [[0, 1]]
