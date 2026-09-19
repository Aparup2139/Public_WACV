import torch

from qgmamba_cd.benchmark import count_parameters, profile_model


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.register_buffer("running_stat", torch.zeros(2))

    def forward(self, a, b):
        return (self.linear(a.mean(dim=(2, 3))),)


def test_count_parameters():
    model = Tiny()
    counts = count_parameters(model)
    assert counts["total"] == counts["trainable"] == 4 * 4 + 4
    assert counts["buffers"] == 2


def test_count_parameters_distinguishes_frozen_params():
    model = Tiny()
    model.linear.weight.requires_grad_(False)
    counts = count_parameters(model)
    assert counts["total"] == 4 * 4 + 4
    assert counts["trainable"] == 4


def test_profile_model_runs_on_cpu():
    model = Tiny()
    result = profile_model(
        model, input_size=4, device=torch.device("cpu"), batch_sizes=(1, 2), repeats=2, channels=4
    )
    assert set(result["latency_ms"].keys()) == {1, 2}
    assert set(result["throughput_ips"].keys()) == {1, 2}
    assert result["params"]["total"] > 0
    assert all(v > 0 for v in result["latency_ms"].values())
    assert result["peak_memory_mb"] is None
    assert result["flops"] is None or result["flops"] > 0
