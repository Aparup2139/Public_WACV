from __future__ import annotations

import time

import torch


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    return {"total": total, "trainable": trainable, "buffers": buffers}


def _measure_flops(model: torch.nn.Module, inputs: tuple) -> float | None:
    """Best-effort FLOPs for one forward pass. None if unavailable."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    try:
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(*inputs)
        return float(counter.get_total_flops())
    except Exception:
        return None


def profile_model(
    model: torch.nn.Module,
    input_size: int,
    device: torch.device,
    batch_sizes: tuple[int, ...] = (1, 4, 8),
    repeats: int = 20,
    channels: int = 3,
) -> dict:
    model = model.to(device).eval()
    latency_ms: dict[int, float] = {}
    throughput_ips: dict[int, float] = {}
    peak_memory_mb: dict[int, float] | None = {} if device.type == "cuda" else None
    flops = None
    for batch_size in batch_sizes:
        image_a = torch.randn(batch_size, channels, input_size, input_size, device=device)
        image_b = torch.randn(batch_size, channels, input_size, input_size, device=device)
        if flops is None:
            flops = _measure_flops(model, (image_a, image_b))
        with torch.no_grad():
            for _ in range(3):  # warmup
                model(image_a, image_b)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            for _ in range(repeats):
                model(image_a, image_b)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        per_call_ms = (elapsed / repeats) * 1000
        latency_ms[batch_size] = per_call_ms
        throughput_ips[batch_size] = batch_size / (per_call_ms / 1000)
        if device.type == "cuda":
            peak_memory_mb[batch_size] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {
        "params": count_parameters(model),
        "flops": flops,
        "latency_ms": latency_ms,
        "throughput_ips": throughput_ips,
        "peak_memory_mb": peak_memory_mb,
    }
