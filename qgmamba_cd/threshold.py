"""Threshold-selection strategies for change-detection evaluation.

Ported verbatim from the Valais notebook's ``pick_threshold_plateau`` (see
``valais-bmd-cd (4).ipynb``, cell A / "EVAL HELPERS"). Consumes the
list-of-dicts shape produced by ``qgmamba_cd.evaluation.sweep_thresholds``.
"""

from __future__ import annotations


def pick_threshold_plateau(rows: list[dict], tolerance: float = 0.99) -> float:
    """Median-by-index threshold of the CONTIGUOUS run of within-tolerance
    rows that contains the argmax (not every within-tolerance row anywhere
    on the grid).

    The argmax of a 50-object F1 curve is noise; the plateau centre is
    stable. Restricting to the contiguous run guards against a
    multi-lobed curve: taking the median of ALL within-tolerance rows can
    land in the valley between two lobes, an F1 below tolerance that the
    argmax rule was supposed to avoid returning. Returns one of the
    threshold values passed in, unchanged (not an interpolated average --
    for an even-length plateau this picks the upper-middle element by
    index, not the statistical mean of the two middle thresholds, so the
    result is always a threshold that was actually swept).
    """

    ordered = sorted(rows, key=lambda row: row["threshold"])
    best_f1 = max(row["f1"] for row in ordered)
    within = [row["f1"] >= tolerance * best_f1 for row in ordered]
    argmax_index = max(range(len(ordered)), key=lambda i: ordered[i]["f1"])

    lo = argmax_index
    while lo > 0 and within[lo - 1]:
        lo -= 1

    hi = argmax_index
    while hi < len(ordered) - 1 and within[hi + 1]:
        hi += 1

    plateau = ordered[lo:hi + 1]
    picked = plateau[len(plateau) // 2]

    if picked["f1"] < tolerance * best_f1:
        raise ValueError("pick_threshold_plateau selected a threshold below tolerance")

    return picked["threshold"]
