from qgmamba_cd.threshold import pick_threshold_plateau


def _row(threshold, f1):
    return {"threshold": threshold, "f1": f1}


def test_plateau_picks_middle_of_odd_length_contiguous_run():
    # F1 sweep that rises, plateaus (within 1% of best across 3 contiguous
    # thresholds), then falls -- a realistic threshold-sweep shape.
    rows = [
        _row(0.10, 0.60),
        _row(0.15, 0.85),
        _row(0.20, 0.945),
        _row(0.25, 0.95),  # best (argmax)
        _row(0.30, 0.943),
        _row(0.35, 0.80),
        _row(0.40, 0.55),
    ]
    # cutoff = 0.99 * 0.95 = 0.9405 -> thresholds 0.20, 0.25, 0.30 all qualify
    # and are contiguous around the argmax (0.25). Odd-length plateau (3):
    # middle-by-index element is 0.25.
    assert pick_threshold_plateau(rows, tolerance=0.99) == 0.25


def test_plateau_even_length_run_picks_upper_middle_by_index_not_average():
    # Regression test: an even-length plateau must return one of the ACTUAL
    # swept thresholds (index-based middle element), never an interpolated
    # average of the two middle thresholds. Downstream code looks up the
    # returned value by exact match against the swept threshold grid
    # (`next(i for i, r in enumerate(rows) if r["threshold"] == thr)`), so
    # an interpolated value (e.g. statistics.median giving 0.45 here) would
    # break that lookup.
    rows = [
        _row(0.1, 0.50),
        _row(0.2, 0.90),
        _row(0.3, 0.93),
        _row(0.4, 0.94),  # tied best
        _row(0.5, 0.94),  # tied best
        _row(0.6, 0.93),
        _row(0.7, 0.40),
    ]
    # cutoff = 0.99 * 0.94 = 0.9306 -> plateau is [0.4, 0.5] (length 2).
    # plateau[len(plateau) // 2] == plateau[1] -> 0.5, not mean(0.4, 0.5) = 0.45.
    result = pick_threshold_plateau(rows, tolerance=0.99)
    assert result == 0.5
    assert result in {row["threshold"] for row in rows}


def test_plateau_falls_back_to_argmax_for_single_point_peak():
    rows = [_row(0.1, 0.10), _row(0.2, 0.95), _row(0.3, 0.10)]
    assert pick_threshold_plateau(rows, tolerance=0.999) == 0.2
