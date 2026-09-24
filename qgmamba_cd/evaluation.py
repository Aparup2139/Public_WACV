from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_opening, label as connected_components
from tqdm import tqdm

from .config import EvalConfig
from .data import IMAGENET_MEAN, IMAGENET_STD, LEVIRFullDataset
from .distributed import DistributedContext, reduce_sum, resolve_amp_dtype
""" Evaluation utilities for the QG-Mamba-Diff change-detection package."""

def confusion_counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    prediction = prediction.astype(np.uint8)
    target = target.astype(np.uint8)
    true_positive = int(np.sum((prediction == 1) & (target == 1)))
    true_negative = int(np.sum((prediction == 0) & (target == 0)))
    false_positive = int(np.sum((prediction == 1) & (target == 0)))
    false_negative = int(np.sum((prediction == 0) & (target == 1)))
    return true_positive, true_negative, false_positive, false_negative


def metrics_from_confusion(true_positive: float, true_negative: float, false_positive: float, false_negative: float):
    precision = true_positive / (true_positive + false_positive + 1.0e-7)
    recall = true_positive / (true_positive + false_negative + 1.0e-7)
    f1 = 2.0 * precision * recall / (precision + recall + 1.0e-7)
    iou = true_positive / (true_positive + false_positive + false_negative + 1.0e-7)
    accuracy = (true_positive + true_negative) / (
        true_positive + true_negative + false_positive + false_negative + 1.0e-7
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "accuracy": float(accuracy),
    }


def tile_coordinates(height: int, width: int, tile: int, stride: int):
    if height < tile or width < tile:
        raise ValueError(f"Full image {height}x{width} is smaller than eval tile_size={tile}")
    ys = list(range(0, max(height - tile + 1, 1), stride)) or [0]
    xs = list(range(0, max(width - tile + 1, 1), stride)) or [0]
    if ys[-1] != height - tile:
        ys.append(height - tile)
    if xs[-1] != width - tile:
        xs.append(width - tile)
    return ys, xs


def _autocast(device: torch.device, enabled: bool, amp_dtype: str):
    return torch.autocast(
        device_type=device.type,
        enabled=enabled and device.type == "cuda",
        dtype=resolve_amp_dtype(amp_dtype, device),
    )


@torch.inference_mode()
def predict_probability_map(
    model: torch.nn.Module,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    device: torch.device,
    tile: int,
    stride: int,
    batch_tiles: int,
    tta: bool,
    amp: bool,
    amp_dtype: str,
    channels_last: bool,
    d4_tta: bool,
) -> np.ndarray:
    model.eval()
    height, width = image_a.shape[-2:]
    accumulator = np.zeros((height, width), dtype=np.float32)
    counter = np.zeros((height, width), dtype=np.float32)
    ys, xs = tile_coordinates(height, width, tile, stride)
    patches: list[tuple[torch.Tensor, torch.Tensor]] = []
    coordinates: list[tuple[int, int]] = []

    def flush() -> None:
        if not patches:
            return
        batch_a = torch.stack([patch[0] for patch in patches]).to(device, non_blocking=True)
        batch_b = torch.stack([patch[1] for patch in patches]).to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            batch_a = batch_a.contiguous(memory_format=torch.channels_last)
            batch_b = batch_b.contiguous(memory_format=torch.channels_last)
        with _autocast(device, amp, amp_dtype):
            logits = model(batch_a, batch_b)[0]
            if tta:
                if d4_tta:
                    predictions = [logits]
                    for rotations in range(4):
                        rotated_a = torch.rot90(batch_a, rotations, dims=(2, 3))
                        rotated_b = torch.rot90(batch_b, rotations, dims=(2, 3))
                        for reflect in (False, True):
                            if rotations == 0 and not reflect:
                                continue
                            transformed_a = torch.flip(rotated_a, [3]) if reflect else rotated_a
                            transformed_b = torch.flip(rotated_b, [3]) if reflect else rotated_b
                            prediction = model(transformed_a, transformed_b)[0]
                            if reflect:
                                prediction = torch.flip(prediction, [3])
                            predictions.append(torch.rot90(prediction, -rotations, dims=(2, 3)))
                    logits = torch.stack(predictions).mean(0)
                else:
                    horizontal = model(torch.flip(batch_a, [3]), torch.flip(batch_b, [3]))[0]
                    vertical = model(torch.flip(batch_a, [2]), torch.flip(batch_b, [2]))[0]
                    both = model(torch.flip(batch_a, [2, 3]), torch.flip(batch_b, [2, 3]))[0]
                    logits = (
                        logits
                        + torch.flip(horizontal, [3])
                        + torch.flip(vertical, [2])
                        + torch.flip(both, [2, 3])
                    ) / 4.0
        probabilities = torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
        for index, (y, x) in enumerate(coordinates):
            accumulator[y : y + tile, x : x + tile] += probabilities[index]
            counter[y : y + tile, x : x + tile] += 1.0
        patches.clear()
        coordinates.clear()

    for y in ys:
        for x in xs:
            patches.append((image_a[:, y : y + tile, x : x + tile], image_b[:, y : y + tile, x : x + tile]))
            coordinates.append((y, x))
            if len(patches) == batch_tiles:
                flush()
    flush()
    return accumulator / np.maximum(counter, 1.0e-6)


def postprocess_prediction(prediction: np.ndarray, min_area: int, morph_kernel: int) -> np.ndarray:
    kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
    prediction = binary_opening(prediction.astype(bool), structure=kernel).astype(np.uint8)
    prediction = binary_closing(prediction.astype(bool), structure=kernel).astype(np.uint8)
    labeled, count = connected_components(prediction)
    for index in range(1, count + 1):
        if np.sum(labeled == index) < min_area:
            prediction[labeled == index] = 0
    return prediction


def sweep_thresholds(
    model: torch.nn.Module,
    dataset: LEVIRFullDataset,
    thresholds: list[float],
    config: EvalConfig,
    context: DistributedContext,
    amp: bool,
    amp_dtype: str = "float16",
    channels_last: bool = False,
):
    threshold_array = np.asarray(thresholds, dtype=np.float32)
    confusion = np.zeros((len(threshold_array), 4), dtype=np.float64)
    total_images = len(dataset) if config.quick_val_images is None else min(len(dataset), config.quick_val_images)
    indices = range(context.rank, total_images, context.world_size)
    progress = tqdm(indices, desc="Validation", disable=not context.is_main, leave=False)
    for index in progress:
        image_a, image_b, target, _ = dataset[index]
        probability = predict_probability_map(
            model,
            image_a,
            image_b,
            context.device,
            config.tile_size,
            config.tile_stride,
            config.batch_tiles,
            config.tta,
            amp,
            amp_dtype,
            channels_last,
            config.d4_tta,
        )
        for threshold_index, threshold in enumerate(threshold_array):
            prediction = (probability > threshold).astype(np.uint8)
            if config.postprocess:
                prediction = postprocess_prediction(prediction, config.min_area, config.morph_kernel)
            confusion[threshold_index] += confusion_counts(prediction, target)
    confusion_tensor = torch.tensor(confusion, dtype=torch.float64, device=context.device)
    confusion = reduce_sum(confusion_tensor).cpu().numpy()
    rows = []
    for threshold, counts in zip(threshold_array, confusion):
        rows.append({"threshold": float(threshold), **metrics_from_confusion(*counts)})
    best = max(rows, key=lambda row: row["f1"])
    return float(best["threshold"]), {key: best[key] for key in ("precision", "recall", "f1", "iou", "accuracy")}, rows


def evaluate_threshold(
    model: torch.nn.Module,
    dataset: LEVIRFullDataset,
    threshold: float,
    config: EvalConfig,
    context: DistributedContext,
    amp: bool,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    final: bool = True,
):
    confusion = np.zeros(4, dtype=np.float64)
    stride = config.final_stride if final else config.tile_stride
    tta = config.final_tta if final else config.tta
    indices = range(context.rank, len(dataset), context.world_size)
    progress = tqdm(indices, desc="Evaluation", disable=not context.is_main, leave=False)
    for index in progress:
        image_a, image_b, target, _ = dataset[index]
        probability = predict_probability_map(
            model,
            image_a,
            image_b,
            context.device,
            config.tile_size,
            stride,
            config.batch_tiles,
            tta,
            amp,
            amp_dtype,
            channels_last,
            config.d4_tta,
        )
        prediction = (probability > threshold).astype(np.uint8)
        if config.postprocess:
            prediction = postprocess_prediction(prediction, config.min_area, config.morph_kernel)
        confusion += confusion_counts(prediction, target)
    confusion_tensor = torch.tensor(confusion, dtype=torch.float64, device=context.device)
    confusion = reduce_sum(confusion_tensor).cpu().numpy()
    return metrics_from_confusion(*confusion)


def save_predictions(
    model: torch.nn.Module,
    dataset: LEVIRFullDataset,
    threshold: float,
    config: EvalConfig,
    context: DistributedContext,
    amp: bool,
    output_dir: str | Path,
    amp_dtype: str = "float16",
    channels_last: bool = False,
) -> None:
    output_dir = Path(output_dir)
    for subdirectory in ("probability", "mask", "overlay"):
        (output_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    indices = range(context.rank, len(dataset), context.world_size)
    progress = tqdm(indices, desc="Prediction", disable=not context.is_main)
    for index in progress:
        image_a, image_b, _, name = dataset[index]
        probability = predict_probability_map(
            model,
            image_a,
            image_b,
            context.device,
            config.tile_size,
            config.final_stride,
            config.batch_tiles,
            config.final_tta,
            amp,
            amp_dtype,
            channels_last,
            config.d4_tta,
        )
        prediction = (probability > threshold).astype(np.uint8)
        if config.postprocess:
            prediction = postprocess_prediction(prediction, config.min_area, config.morph_kernel)
        stem = Path(name).stem
        cv2.imwrite(str(output_dir / "probability" / f"{stem}_prob.png"), (probability * 255).astype(np.uint8))
        cv2.imwrite(str(output_dir / "mask" / f"{stem}_mask.png"), prediction * 255)
        rgb = image_b.cpu().numpy().transpose(1, 2, 0)
        rgb = np.clip((rgb * std + mean) * 255, 0, 255).astype(np.uint8)
        overlay = rgb.copy()
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        selected = prediction.astype(bool)
        overlay[selected] = (0.45 * overlay[selected] + 0.55 * red[selected]).astype(np.uint8)
        cv2.imwrite(
            str(output_dir / "overlay" / f"{stem}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
