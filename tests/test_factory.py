import copy

import torch
import yaml

from qgmamba_cd.factory import build_model


def _write_minimal_config(path):
    path.write_text(yaml.safe_dump({
        "model": {"input_size": 64, "pretrained": False, "encoder_name": "swin_tiny_patch4_window7_224.ms_in1k"},
        "data": {"image_size": 64},
    }))


def test_build_model_without_checkpoint_forces_ddim(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    _write_minimal_config(config_path)

    model, config, checkpoint = build_model(config_path, checkpoint_path=None, device=torch.device("cpu"))

    assert config.model.use_ddim_at_inference is True
    assert model.diffusion.use_ddim
    assert checkpoint == {}


def test_build_model_loads_model_state_when_eval_weights_is_model(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    _write_minimal_config(config_path)
    model, config, _ = build_model(config_path, device=torch.device("cpu"))
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state": model.state_dict(), "threshold": 0.42, "eval_weights": "model"}, ckpt_path)

    reloaded, _, checkpoint = build_model(config_path, checkpoint_path=ckpt_path, device=torch.device("cpu"))

    assert checkpoint["threshold"] == 0.42
    for key, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key])


def test_build_model_prefers_ema_state_when_checkpoint_says_eval_weights_is_ema(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    _write_minimal_config(config_path)
    model, _, _ = build_model(config_path, device=torch.device("cpu"))
    model_state = copy.deepcopy(model.state_dict())
    ema_state = {key: value + 1.0 for key, value in model_state.items()}  # deliberately different values
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state": model_state, "ema_state": ema_state, "eval_weights": "ema"}, ckpt_path)

    reloaded, _, _ = build_model(config_path, checkpoint_path=ckpt_path, device=torch.device("cpu"))

    for key, value in ema_state.items():
        assert torch.equal(value, reloaded.state_dict()[key])  # ema_state won, not model_state


def test_build_model_strict_load_raises_on_key_mismatch(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    _write_minimal_config(config_path)
    model, _, _ = build_model(config_path, device=torch.device("cpu"))
    bad_state = {"totally_unexpected_key": torch.zeros(1)}
    ckpt_path = tmp_path / "bad_ckpt.pt"
    torch.save({"model_state": bad_state}, ckpt_path)

    try:
        build_model(config_path, checkpoint_path=ckpt_path, device=torch.device("cpu"))
        assert False, "expected a strict state_dict load to raise"
    except RuntimeError:
        pass


def test_build_model_applies_dataset_overrides(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    _write_minimal_config(config_path)

    model, config, _ = build_model(
        config_path,
        device=torch.device("cpu"),
        overrides={"data": {"change_focus_prob": 0.55}, "train": {"pos_weight": 3.0}},
    )

    assert config.data.change_focus_prob == 0.55
    assert config.train.pos_weight == 3.0


def test_apply_overrides_rejects_unknown_section(tmp_path):
    from qgmamba_cd.config import apply_overrides, ExperimentConfig

    try:
        apply_overrides(ExperimentConfig(), {"not_a_real_section": {}})
        assert False, "expected a ValueError for an unknown top-level section"
    except ValueError:
        pass
