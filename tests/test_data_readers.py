import numpy as np
import cv2
from pathlib import Path

from qgmamba_cd.data import make_grayscale_mask_reader, read_rgb_cv2, DEFAULT_READERS, SplitPaths
from qgmamba_cd.losses import estimate_pos_weight

"""Unit tests for the Public_WACV/qgmamba_cd/data.py module."""
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
