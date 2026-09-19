import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import torch

from qgmamba_cd.data import IMAGENET_MEAN, IMAGENET_STD
from qgmamba_cd.viz import denorm, qg_show
import qgmamba_cd.viz as viz


def test_denorm_inverts_normalization():
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    original = torch.full((3, 2, 2), 0.6)
    normalized = (original - mean) / std

    recovered = denorm(normalized)

    assert recovered.shape == (2, 2, 3)
    assert torch.allclose(torch.tensor(recovered), original.permute(1, 2, 0), atol=1e-5)


def test_denorm_clamps_out_of_range_values():
    image = torch.full((3, 1, 1), 5.0)  # far outside the valid normalized range

    recovered = denorm(image)

    assert recovered.max() <= 1.0
    assert recovered.min() >= 0.0


class _FakeDataset:
    """Minimal stand-in for PairedFullDataset: (image_a, image_b, target, name)."""

    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        target = self.targets[index]
        image = torch.zeros(3, *target.shape)
        return image, image.clone(), target, f"sample_{index}"


class _ConstantModel(torch.nn.Module):
    def forward(self, image_a, image_b):
        batch, _, height, width = image_a.shape
        return torch.zeros(batch, 1, height, width), None


class _CapturingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image_a, image_b):
        self.calls.append((image_a.shape, image_a.dtype, image_b.shape, image_b.dtype))
        batch, _, height, width = image_a.shape
        return torch.zeros(batch, 1, height, width), None


def _row_names(figure):
    return [axis.get_title().split(":")[0] for axis in figure.axes[::4]]


def test_qg_show_prefer_changed_selects_only_changed_samples(monkeypatch):
    targets = [np.zeros((2, 2), dtype=np.uint8) for _ in range(6)]
    targets[3] = np.ones((2, 2), dtype=np.uint8)
    targets[4] = np.ones((2, 2), dtype=np.uint8)
    dataset = _FakeDataset(targets)
    monkeypatch.setattr(viz.random, "shuffle", lambda seq: None)
    monkeypatch.setattr(viz.random, "sample", lambda pool, k: list(pool)[:k])

    figure = qg_show(
        _ConstantModel(), dataset, threshold=0.5, device=torch.device("cpu"),
        n=2, tile=2, stride=2, amp=False,
    )

    assert _row_names(figure) == ["sample_3", "sample_4"]


def test_qg_show_prefer_changed_scan_is_bounded(monkeypatch):
    limit = viz._PREFER_CHANGED_SCAN_LIMIT
    targets = [np.zeros((2, 2), dtype=np.uint8) for _ in range(limit + 5)]
    targets[-1] = np.ones((2, 2), dtype=np.uint8)  # changed sample sits beyond the scan bound
    dataset = _FakeDataset(targets)
    monkeypatch.setattr(viz.random, "shuffle", lambda seq: None)
    monkeypatch.setattr(viz.random, "sample", lambda pool, k: list(pool)[:k])

    figure = qg_show(
        _ConstantModel(), dataset, threshold=0.5, device=torch.device("cpu"),
        n=1, tile=2, stride=2, amp=False,
    )

    # The one changed sample sits past the scan bound, so the bounded scan
    # never sees it and falls back to the unfiltered pool (sample_0 first).
    assert _row_names(figure) == ["sample_0"]


def test_qg_show_feeds_correctly_shaped_batches_to_model(monkeypatch):
    dataset = _FakeDataset([np.zeros((4, 4), dtype=np.uint8)])
    monkeypatch.setattr(viz.random, "shuffle", lambda seq: None)
    monkeypatch.setattr(viz.random, "sample", lambda pool, k: list(pool)[:k])
    model = _CapturingModel()

    qg_show(
        model, dataset, threshold=0.5, device=torch.device("cpu"),
        n=1, prefer_changed=False, tile=2, stride=2, amp=False,
    )

    assert model.calls, "model was never invoked"
    for a_shape, a_dtype, b_shape, b_dtype in model.calls:
        assert a_shape[1:] == (3, 2, 2)
        assert a_dtype == torch.float32
        assert b_shape == a_shape
        assert b_dtype == a_dtype
