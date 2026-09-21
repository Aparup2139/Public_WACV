from pathlib import Path

import pytest

from qgmamba_cd.artifacts import ensure_checkpoint


def test_ensure_checkpoint_returns_existing_file(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    assert ensure_checkpoint(checkpoint) == checkpoint.resolve()


def test_ensure_checkpoint_requires_a_published_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="Inference never falls back"):
        ensure_checkpoint(tmp_path / "missing.pt")
