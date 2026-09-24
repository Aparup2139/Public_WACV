from __future__ import annotations

from pathlib import Path


def ensure_checkpoint(
    local_path: str | Path,
    *,
    repo_id: str | None = None,
    filename: str | None = None,
    revision: str | None = None,
) -> Path:
    """Resolve the checkpoint locally, downloading it from a Hugging Face model repo if needed."""
    destination = Path(local_path).expanduser().resolve()
    if destination.is_file():
        return destination
    if not repo_id or not filename:
        raise FileNotFoundError(
            f"Checkpoint not found at {destination}. Publish the weights in this repository "
            "(preferably with Git LFS), or set checkpoint_hf_repo and checkpoint_hf_filename "
            "in the selected dataset config.yaml. Inference never falls back to random weights."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=filename,
        revision=revision,
        local_dir=destination.parent,
    )
    downloaded_path = Path(downloaded)
    if downloaded_path != destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path.replace(destination)
    return destination
