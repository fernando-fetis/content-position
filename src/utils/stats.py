from __future__ import annotations

from collections.abc import Callable
from statistics import NormalDist

import numpy as np


def rank(x: np.ndarray) -> np.ndarray:
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    return ((upper - (counts - 1) / 2) - 1)[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.ravel(x), np.ravel(y)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(rank(x[ok]), rank(y[ok]))[0, 1]) if ok.sum() > 2 else float('nan')


def _bca(vals: np.ndarray, theta: float, jack: np.ndarray) -> tuple[float, float]:
    '''The 95% BCa interval from the bootstrap values of a statistic, its value on the sample and its jackknife values; a degenerate resampling gives (theta, theta), an undefined correction falls back to the percentile interval.'''
    normal = NormalDist()
    ok = vals[np.isfinite(vals)]
    if ok.size == 0 or np.ptp(ok) == 0:
        return (float(theta), float(theta)) if ok.size else (float('nan'), float('nan'))
    below = (np.sum(ok < theta) + 0.5 * np.sum(ok == theta)) / ok.size
    z0 = normal.inv_cdf(min(max(below, 1 / (ok.size + 1)), ok.size / (ok.size + 1)))
    jack = jack[np.isfinite(jack)]
    d = jack.mean() - jack if jack.size else jack
    den = 6 * np.sum(d ** 2) ** 1.5
    a = float(np.sum(d ** 3) / den) if den > 0 else 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        lo, hi = (normal.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))) if 1 - a * (z0 + z) > 0 else float('nan') for z in (normal.inv_cdf(0.025), normal.inv_cdf(0.975)))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = 0.025, 0.975
    return float(np.quantile(ok, lo)), float(np.quantile(ok, hi))


def _resampled(x: np.ndarray, stat: Callable[[np.ndarray], np.ndarray], n_boot: int, seed: int, block: int = 10_000_000) -> np.ndarray:
    '''stat on n_boot resamples of the first axis of x, drawn as index matrices (B, n) in blocks of about block indices.'''
    rng, n = np.random.default_rng(seed), x.shape[0]
    rows = max(1, block // n)
    return np.concatenate([np.asarray(stat(x[rng.integers(0, n, (min(rows, n_boot - start), n))]), float) for start in range(0, n_boot, rows)])


def bootstrap_mean(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    '''95% BCa interval of the mean of x (flattened), resampled in blocks (_resampled), with the jackknife in closed form, (sum - x_i) / (n - 1).'''
    x = np.asarray(x, float).ravel()
    n = x.size
    if n == 0:
        return float('nan'), float('nan')
    theta = float(x.mean())
    if n == 1:
        return theta, theta
    return _bca(_resampled(x, lambda r: r.mean(1), n_boot, seed), theta, (x.sum() - x) / (n - 1))
