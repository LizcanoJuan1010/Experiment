"""Confidence interval utilities for deep-analysis figures.

Centralizes Wilson / bootstrap / paired-bootstrap / ellipse CI helpers so all
visualization scripts (V1-V7 and run_analysis.py plots 01-06) share the same
statistical backbone and consistent visual style.

All functions return either a ``(lo, hi)`` tuple or a ``DataFrame`` with
``mean, ci_lo, ci_hi, n`` columns. Degenerate cases (n<2, all-NaN, std=0)
return NaN rather than raising.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.patches import Ellipse
import matplotlib.transforms as mtransforms

__all__ = [
    "wilson_ci", "bootstrap_ci", "paired_bootstrap_ci",
    "agg_with_ci", "delta_accuracy_wilson",
    "confidence_ellipse", "annotate_heatmap_with_ci", "significance_mark",
]

_RNG = np.random.default_rng(20260417)


def wilson_ci(n_correct: int, n_total: int, conf: float = 0.95) -> Tuple[float, float]:
    """Wilson score CI for a binomial proportion; robust for small n and p near 0/1."""
    if n_total is None or n_total <= 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = n_correct / n_total
    denom = 1 + z * z / n_total
    center = (p + z * z / (2 * n_total)) / denom
    half = (z * np.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total * n_total))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(values: Sequence[float], statistic: Callable = np.mean,
                 n_boot: int = 2000, conf: float = 0.95,
                 rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """Percentile bootstrap CI for any statistic on a 1-D sample."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return (np.nan, np.nan)
    rng = rng or _RNG
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    try:
        boot = np.apply_along_axis(statistic, 1, v[idx])
    except Exception:
        boot = np.array([statistic(v[i]) for i in idx])
    lo, hi = np.quantile(boot, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return (float(lo), float(hi))


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float],
                        statistic: Callable = lambda x, y: np.mean(x) - np.mean(y),
                        n_boot: int = 2000, conf: float = 0.95,
                        rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """Paired bootstrap CI for a contrast between two aligned samples of equal length."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if a.size < 2:
        return (np.nan, np.nan)
    rng = rng or _RNG
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot = np.empty(n_boot)
    for k in range(n_boot):
        boot[k] = statistic(a[idx[k]], b[idx[k]])
    lo, hi = np.quantile(boot, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return (float(lo), float(hi))


def _kv(cols, key):
    key = key if isinstance(key, tuple) else (key,)
    return dict(zip(cols, key))


def agg_with_ci(df: pd.DataFrame, group_cols: list, value_col: str,
                method: str = "bootstrap", conf: float = 0.95,
                n_boot: int = 2000) -> pd.DataFrame:
    """Group-by aggregation returning mean + (ci_lo, ci_hi, n).

    ``method`` in {"bootstrap", "wilson", "normal"}. "wilson" requires
    ``n_correct`` and ``n_items`` columns in ``df``.
    """
    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        vals = sub[value_col].dropna().to_numpy()
        n = int(vals.size)
        mean = float(np.mean(vals)) if n else np.nan
        if n < 2:
            lo = hi = np.nan
        elif method == "bootstrap":
            lo, hi = bootstrap_ci(vals, n_boot=n_boot, conf=conf)
        elif method == "wilson" and {"n_correct", "n_items"}.issubset(sub.columns):
            lo, hi = wilson_ci(int(sub["n_correct"].sum()),
                               int(sub["n_items"].sum()), conf)
        else:
            s = float(vals.std(ddof=1)) if n > 1 else 0.0
            se = s / np.sqrt(n) if s > 0 else 0.0
            z = stats.norm.ppf(1 - (1 - conf) / 2)
            lo, hi = mean - z * se, mean + z * se
        rows.append({**_kv(group_cols, key), "mean": mean,
                     "ci_lo": float(lo) if lo == lo else np.nan,
                     "ci_hi": float(hi) if hi == hi else np.nan,
                     "n": n})
    return pd.DataFrame(rows)


def delta_accuracy_wilson(df: pd.DataFrame, group_cols: Optional[list] = None,
                          conf: float = 0.95) -> pd.DataFrame | dict:
    """Delta accuracy CI via Wilson on pooled counts (Newcombe-simple for difference).

    Convention: ``delta = baseline - condition`` (positive = bigger drop),
    matching the ``delta_accuracy`` column in ``unified_cct_results*.csv``.

    With ``group_cols=None`` returns a single dict for the whole ``df``.
    Requires columns ``n_correct``, ``n_baseline_correct``, ``n_items``.
    """
    def _one(sub: pd.DataFrame) -> dict:
        nt = int(sub["n_items"].sum()) if "n_items" in sub.columns else 0
        if nt == 0:
            return {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": 0}
        nc = int(sub["n_correct"].sum())
        nb = int(sub["n_baseline_correct"].sum())
        p_abl_lo, p_abl_hi = wilson_ci(nc, nt, conf)
        p_base_lo, p_base_hi = wilson_ci(nb, nt, conf)
        delta = (nb - nc) / nt
        lo = p_base_lo - p_abl_hi
        hi = p_base_hi - p_abl_lo
        return {"mean": float(delta), "ci_lo": float(lo),
                "ci_hi": float(hi), "n": nt}

    if group_cols is None:
        return _one(df)
    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        rows.append({**_kv(group_cols, key), **_one(sub)})
    return pd.DataFrame(rows)


def confidence_ellipse(x, y, ax, conf: float = 0.95,
                       edgecolor: str = "black",
                       facecolor: str = "none",
                       alpha: float = 0.4,
                       linewidth: float = 1.4,
                       linestyle: str = "-",
                       **kwargs) -> Optional[Ellipse]:
    """Draw a chi-square 2D confidence ellipse on ``ax`` from (x, y) samples."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if x.size < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    pearson = np.clip(pearson, -1 + 1e-9, 1 - 1e-9)
    rx = np.sqrt(1 + pearson)
    ry = np.sqrt(1 - pearson)
    chi2 = stats.chi2.ppf(conf, df=2)
    scale_x = np.sqrt(cov[0, 0] * chi2)
    scale_y = np.sqrt(cov[1, 1] * chi2)
    ell = Ellipse((0, 0), width=2 * rx, height=2 * ry,
                  facecolor=facecolor, edgecolor=edgecolor,
                  alpha=alpha, linewidth=linewidth, linestyle=linestyle, **kwargs)
    tr = (mtransforms.Affine2D()
          .rotate_deg(45).scale(scale_x, scale_y)
          .translate(float(np.mean(x)), float(np.mean(y))))
    ell.set_transform(tr + ax.transData)
    ax.add_patch(ell)
    return ell


def annotate_heatmap_with_ci(ax, means: np.ndarray,
                             ci_lo: np.ndarray, ci_hi: np.ndarray,
                             fmt: str = "{:.2f}", fontsize: float = 7.0,
                             color_thresh: float = 0.35,
                             show_ci_above: float = 0.0) -> None:
    """Annotate each heatmap cell with ``mean`` and ``[lo, hi]`` underneath.

    - ``color_thresh``: ``|mean|`` above this renders white text, else black.
    - ``show_ci_above``: skip CI line for cells with ``|mean|`` below this
      (keeps faint cells visually clean).
    """
    nrows, ncols = means.shape
    for i in range(nrows):
        for j in range(ncols):
            m = means[i, j]
            if np.isnan(m):
                continue
            color = "white" if abs(m) > color_thresh else "black"
            if abs(m) < show_ci_above or np.isnan(ci_lo[i, j]) or np.isnan(ci_hi[i, j]):
                txt = fmt.format(m)
            else:
                txt = f"{fmt.format(m)}\n[{fmt.format(ci_lo[i, j])},{fmt.format(ci_hi[i, j])}]"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=fontsize, linespacing=0.95)


def significance_mark(ci_lo: float, ci_hi: float) -> str:
    """Star marker from a CI: '' if CI crosses 0, else '*', '**', or '***' by distance."""
    if ci_lo is None or ci_hi is None:
        return ""
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return ""
    if ci_lo <= 0 <= ci_hi:
        return ""
    dist = min(abs(ci_lo), abs(ci_hi))
    if dist > 0.10:
        return "***"
    if dist > 0.05:
        return "**"
    if dist > 0.01:
        return "*"
    return ""


if __name__ == "__main__":
    print("smoke test of ci_utils")
    print(f"  wilson_ci(42, 50) = {wilson_ci(42, 50)}")
    print(f"  wilson_ci(0, 10)  = {wilson_ci(0, 10)}")
    print(f"  wilson_ci(10, 10) = {wilson_ci(10, 10)}")
    rng = np.random.default_rng(0)
    sample = rng.normal(0.3, 0.1, 80)
    print(f"  bootstrap_ci(normal(0.3,0.1)) = {bootstrap_ci(sample)}")
    a = rng.normal(0.5, 0.1, 40)
    b = rng.normal(0.3, 0.1, 40)
    print(f"  paired_bootstrap_ci(a-b) = {paired_bootstrap_ci(a, b)}")
    df = pd.DataFrame({
        "g": list("AAAABBBB"),
        "v": [0.1, 0.2, 0.15, 0.22, 0.4, 0.35, 0.5, 0.45],
        "n_correct": [4, 8, 6, 9, 16, 14, 20, 18],
        "n_items":   [20, 20, 20, 20, 20, 20, 20, 20],
        "n_baseline_correct": [18, 18, 18, 18, 18, 18, 18, 18],
    })
    print(f"  agg_with_ci bootstrap:\n{agg_with_ci(df, ['g'], 'v', method='bootstrap')}")
    print(f"  agg_with_ci wilson:\n{agg_with_ci(df, ['g'], 'v', method='wilson')}")
    print(f"  delta_accuracy_wilson grouped:\n{delta_accuracy_wilson(df, ['g'])}")
    print(f"  significance_mark(-0.20,-0.05) = {significance_mark(-0.20, -0.05)!r}")
    print(f"  significance_mark(-0.02, 0.05) = {significance_mark(-0.02, 0.05)!r}")
    print("OK")
