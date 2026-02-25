"""quality.py

Provides statistical quality evaluation for model predictions using
mean-error comparison and the two-sample Kolmogorov–Smirnov test.

The module exposes a quality gate that passes only when **both** criteria
are satisfied:
    1. The new model has a lower mean absolute error than the old one.
    2. The KS test (one-sided, *greater*) confirms the error distribution
       of the new model is stochastically smaller at significance level ``alpha``.

Constants:
    ALPHA (float): Default significance level for the KS test (0.05).
    MIN_SAMPLE_SIZE (int): Minimum number of observations required to
        run the statistical test (31).
"""

from collections.abc import Sequence

import numpy as np
from scipy import stats as statspy


ALPHA: float = 0.05
"""Default significance level for the Kolmogorov–Smirnov test."""

MIN_SAMPLE_SIZE: int = 31
"""Minimum number of observations before a statistical test is run."""


def absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> np.ndarray:
    """Compute the element-wise absolute error between true and predicted values.

    Args:
        y_true (Sequence[float]): Ground-truth values.
        y_pred (Sequence[float]): Predicted values (same length as *y_true*).

    Returns:
        np.ndarray: 1-D array of absolute errors.

    Raises:
        ValueError: If *y_true* and *y_pred* have different shapes.
    """
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
    """Evaluate prediction quality of a new model versus an old (baseline) model.

    Computes the mean absolute error of both models and, when there are
    enough samples, applies a two-sample Kolmogorov–Smirnov test to
    assess whether the new model's error distribution is stochastically
    smaller.

    Args:
        y_true (Sequence[float]): Ground-truth values.
        y_pred_new (Sequence[float]): Predictions from the **new** model.
        y_pred_old (Sequence[float]): Predictions from the **old** (baseline) model.
        alpha (float, optional): Significance level for the KS test.
            Defaults to ``ALPHA`` (0.05).
        min_sample_size (int, optional): Minimum number of observations required
            to perform the statistical test.  Defaults to ``MIN_SAMPLE_SIZE`` (31).

    Returns:
        dict[str, float | bool | int | None]: Dictionary containing:
            - **sample_size** (*int*) – Number of observations evaluated.
            - **minimum_sample_size** (*int*) – Threshold for statistical testing.
            - **mean_error_new** (*float*) – Mean absolute error of the new model.
            - **mean_error_old** (*float*) – Mean absolute error of the old model.
            - **mean_error_improved** (*bool*) – ``True`` if ``mean_error_new < mean_error_old``.
            - **ks_statistic** (*float | None*) – KS test statistic (``None`` if sample too small).
            - **ks_pvalue** (*float | None*) – KS p-value (``None`` if sample too small).
            - **ks_improved** (*bool | None*) – ``True`` if the KS test supports the new model.
            - **quality_gate_passed** (*bool | None*) – ``True`` when **both** gates pass.
    """
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
