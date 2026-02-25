from collections.abc import Sequence

import numpy as np
from scipy import stats as statspy


ALPHA = 0.05
MIN_SAMPLE_SIZE = 31


def absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> np.ndarray:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    if true_arr.shape != pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return np.abs(true_arr - pred_arr)


def evaluate_quality(
    y_true: Sequence[float],
    y_pred_new: Sequence[float],
    y_pred_old: Sequence[float],
    alpha: float = ALPHA,
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> dict[str, float | bool | int | None]:
    err_new = absolute_error(y_true, y_pred_new)
    err_old = absolute_error(y_true, y_pred_old)

    sample_size = int(err_new.size)
    mean_new = float(err_new.mean())
    mean_old = float(err_old.mean())
    mean_error_improved = mean_new < mean_old

    if sample_size < min_sample_size:
        return {
            "sample_size": sample_size,
            "minimum_sample_size": min_sample_size,
            "mean_error_new": mean_new,
            "mean_error_old": mean_old,
            "mean_error_improved": mean_error_improved,
            "ks_statistic": None,
            "ks_pvalue": None,
            "ks_improved": None,
            "quality_gate_passed": None,
        }

    ks_result = statspy.ks_2samp(err_new, err_old, alternative="greater", method="auto")
    ks_improved = ks_result.pvalue < alpha and ks_result.statistic > 0
    quality_gate_passed = mean_error_improved and ks_improved

    return {
        "sample_size": sample_size,
        "minimum_sample_size": min_sample_size,
        "alpha": alpha,
        "mean_error_new": mean_new,
        "mean_error_old": mean_old,
        "mean_error_improved": mean_error_improved,
        "ks_statistic": float(ks_result.statistic),
        "ks_pvalue": float(ks_result.pvalue),
        "ks_improved": bool(ks_improved),
        "quality_gate_passed": bool(quality_gate_passed),
    }
