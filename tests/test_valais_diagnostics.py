import math

import numpy as np

from valais_bmd.diagnostics import component_recall_by_size


def test_component_recall_by_size_hits_small_misses_large_nan_for_empty_bin():
    target = np.zeros((100, 100), dtype=np.uint8)
    target[5:8, 5:8] = 1  # small component, area 9 -> bin (0, 50)
    target[50:70, 50:70] = 1  # large component, area 400 -> bin (200, 800)
    # no component falls in (50, 200) or (800, 10**9)

    prediction = np.zeros((100, 100), dtype=np.uint8)
    prediction[5:8, 5:8] = 1  # fully covers the small component
    # large component left entirely unpredicted

    recall = component_recall_by_size(prediction, target)

    assert recall["0-50"] == 1.0
    assert recall["200-800"] == 0.0
    assert math.isnan(recall["50-200"])
    assert math.isnan(recall["800-1000000000"])
