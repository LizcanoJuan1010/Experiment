"""V3: Cross-model dose-response heatmap with Wilson / bootstrap CIs.

Per model: two heatmaps stacked.
  (a) On-target Δaccuracy (rows = layer-depth fraction, cols = α intensity).
  (b) Selectivity = on-target Δ − off-target Δ  (desired: positive).
All panels share a divergent RdBu_r colormap around zero.

Every cell is annotated with ``mean [lo, hi]`` (95% CI) plus a ``*`` marker
when the CI excludes zero, turning a descriptive heatmap into a contrast plot.
CIs: Wilson (on-target, single concept) and paired bootstrap (selectivity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from viz_style import (MODEL_META, apply_style, RESULTS_DIR, savefig,
                       depth_frac, MODEL_COLORS)
from ci_utils import (bootstrap_ci, paired_bootstrap_ci,
                      annotate_heatmap_with_ci, significance_mark,
                      delta_accuracy_wilson)

apply_style()
MODELS = ["gpt2_small", "pythia_28b", "llava_15_7b"]


def _cell_ci(sub_df: pd.DataFrame) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for a sub-DataFrame of one (layer, α) cell."""
    if len(sub_df) == 0:
        return (np.nan, np.nan, np.nan)
    if {"n_correct", "n_baseline_correct", "n_items"}.issubset(sub_df.columns) and \
            (sub_df["n_baseline_correct"] <= sub_df["n_items"]).all():
        s = delta_accuracy_wilson(sub_df)
        return (s["mean"], s["ci_lo"], s["ci_hi"])
    vals = sub_df["delta_accuracy"].dropna().to_numpy()
    if vals.size < 2:
        m = float(vals.mean()) if vals.size else np.nan
        return (m, np.nan, np.nan)
    lo, hi = bootstrap_ci(vals)
    return (float(vals.mean()), lo, hi)


def build_grid_on(df_model):
    """On-target grid: mean + Wilson CI per (layer, α)."""
    d = df_model[(df_model["layer_numeric"] >= 0) &
                 (df_model["on_target"] == True)]
    alphas = sorted(d["alpha"].unique())
    layers = sorted(d["layer_numeric"].unique())
    mean = np.full((len(layers), len(alphas)), np.nan)
    lo   = np.full_like(mean, np.nan)
    hi   = np.full_like(mean, np.nan)
    for i, L in enumerate(layers):
        for j, a in enumerate(alphas):
            sel = d[(d["layer_numeric"] == L) & (d["alpha"] == a)]
            mean[i, j], lo[i, j], hi[i, j] = _cell_ci(sel)
    return layers, alphas, mean, lo, hi


def build_grid_sel(df_model):
    """Selectivity grid: on − off mean + paired bootstrap CI per (layer, α).

    Pairing is by ``ablated_concept`` so bootstrap preserves the natural
    covariance between on-target and off-target effects for each concept.
    """
    d = df_model[df_model["layer_numeric"] >= 0]
    alphas = sorted(d["alpha"].unique())
    layers = sorted(d["layer_numeric"].unique())
    mean = np.full((len(layers), len(alphas)), np.nan)
    lo   = np.full_like(mean, np.nan)
    hi   = np.full_like(mean, np.nan)
    for i, L in enumerate(layers):
        for j, a in enumerate(alphas):
            cell = d[(d["layer_numeric"] == L) & (d["alpha"] == a)]
            on_vals = cell[cell["on_target"] == True]["delta_accuracy"].to_numpy()
            off_vals = cell[cell["on_target"] == False]["delta_accuracy"].to_numpy()
            if on_vals.size == 0 and off_vals.size == 0:
                continue
            m_on = float(np.mean(on_vals)) if on_vals.size else np.nan
            m_off = float(np.mean(off_vals)) if off_vals.size else np.nan
            mean[i, j] = (m_on if not np.isnan(m_on) else 0.0) - \
                         (m_off if not np.isnan(m_off) else 0.0)
            n = min(on_vals.size, off_vals.size)
            if n >= 2:
                lo[i, j], hi[i, j] = paired_bootstrap_ci(on_vals[:n], off_vals[:n])
    return layers, alphas, mean, lo, hi


def plot_heat(ax, grid, lo, hi, alphas, layers, model_key, title, vmin, vmax):
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(grid, cmap="RdBu_r", norm=norm, aspect="auto", origin="lower")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{L}\n({depth_frac(model_key, L):.2f})" for L in layers],
                       fontsize=8)
    ax.set_xlabel("α (intensidad)")
    ax.set_ylabel("capa (fracción de profundidad)")
    ax.set_title(title, fontsize=10)
    annotate_heatmap_with_ci(ax, grid, lo, hi, fmt="{:.2f}", fontsize=6.3,
                             color_thresh=vmax * 0.6, show_ci_above=0.02)
    # Significance asterisk (upper-right corner)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            mark = significance_mark(lo[i, j], hi[i, j])
            if mark:
                ax.text(j + 0.40, i + 0.40, mark, ha="right", va="top",
                        fontsize=8, fontweight="bold",
                        color="white" if abs(grid[i, j]) > vmax * 0.6 else "black")
    return im


def main():
    df = pd.read_csv(RESULTS_DIR / "unified_cct_results_agg.csv")
    df = df[(df["cav_method"] == "svm") & (df["ablation_mode"] == "all_tokens")]

    fig, axes = plt.subplots(2, len(MODELS), figsize=(16, 9))
    grids = {}
    all_abs = []
    for m in MODELS:
        dm = df[df["model"] == m]
        L, A, g_on,  lo_on,  hi_on  = build_grid_on(dm)
        _, _, g_sel, lo_sel, hi_sel = build_grid_sel(dm)
        grids[m] = (L, A, g_on, lo_on, hi_on, g_sel, lo_sel, hi_sel)
        all_abs += [np.nanmax(np.abs(g_on)) if np.any(~np.isnan(g_on)) else 0,
                    np.nanmax(np.abs(g_sel)) if np.any(~np.isnan(g_sel)) else 0]
    vmax = max(0.05, float(np.nanmax(all_abs)))
    vmin = -vmax

    for ci, m in enumerate(MODELS):
        L, A, g_on, lo_on, hi_on, g_sel, lo_sel, hi_sel = grids[m]
        im1 = plot_heat(axes[0, ci], g_on,  lo_on,  hi_on,  A, L, m,
                        f"{MODEL_META[m]['label']} · Δ on-target", vmin, vmax)
        im2 = plot_heat(axes[1, ci], g_sel, lo_sel, hi_sel, A, L, m,
                        f"{MODEL_META[m]['label']} · selectividad (on − off)", vmin, vmax)

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cbar.set_label("Δ precisión", fontsize=9)

    fig.suptitle("Superficies dosis-respuesta por modelo, profundidad e intensidad α\n"
                 "Arriba: degradación on-target (Wilson 95% CI) · "
                 "Abajo: selectividad on − off (bootstrap pareado 95% CI) · "
                 "* = CI excluye 0",
                 fontsize=12, fontweight="semibold", y=0.995)
    savefig(fig, "dose_response_heatmap")


if __name__ == "__main__":
    main()
