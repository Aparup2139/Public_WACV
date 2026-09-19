from __future__ import annotations

import cv2
import numpy as np

DEFAULT_SIZE_BINS = ((0, 50), (50, 200), (200, 800), (800, 10**9))  # pixel-area bins at 0.5m/px GSD


def component_recall_by_size(
    prediction: np.ndarray, target: np.ndarray, bins: tuple[tuple[int, int], ...] = DEFAULT_SIZE_BINS
) -> dict[str, float]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(target.astype(np.uint8), connectivity=8)
    hits = {bin_range: 0 for bin_range in bins}
    totals = {bin_range: 0 for bin_range in bins}
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        for low, high in bins:
            if low <= area < high:
                totals[(low, high)] += 1
                component_mask = labels == component_id
                if (prediction.astype(bool) & component_mask).any():
                    hits[(low, high)] += 1
                break
    return {
        f"{low}-{high}": (hits[(low, high)] / totals[(low, high)] if totals[(low, high)] else float("nan"))
        for low, high in bins
    }
