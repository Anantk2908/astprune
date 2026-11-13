"""Bootstrap confidence intervals over issues.

All methods run on the same issues, so two methods are compared by resampling
issues and looking at the difference each time (a paired bootstrap).
"""

from __future__ import annotations

import numpy as np

N_RESAMPLES = 10_000
SEED = 20261113


def _resample_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def mean_ci(values, n_resamples: int = N_RESAMPLES, seed: int = SEED, level: float = 0.95) -> dict:
    """Mean of per-issue values with a percentile bootstrap interval."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    means = x[_resample_indices(x.size, n_resamples, seed)].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return {"mean": float(x.mean()), "lo": float(np.quantile(means, alpha)),
            "hi": float(np.quantile(means, 1.0 - alpha)), "n": int(x.size)}


def paired_diff(a, b, n_resamples: int = N_RESAMPLES, seed: int = SEED, level: float = 0.95) -> dict:
    """Mean of ``a - b`` over the same issues, with interval and a two-sided p-value.

    The p-value is the share of resamples where the difference has the
    opposite sign (doubled), the usual bootstrap test for "no difference".
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs the same issues for both methods")
    if a.size == 0:
        return {"diff": None, "lo": None, "hi": None, "p": None, "n": 0}
    d = a - b
    diffs = d[_resample_indices(d.size, n_resamples, seed)].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    p = 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {"diff": float(d.mean()), "lo": float(np.quantile(diffs, alpha)),
            "hi": float(np.quantile(diffs, 1.0 - alpha)), "p": min(p, 1.0), "n": int(d.size)}
