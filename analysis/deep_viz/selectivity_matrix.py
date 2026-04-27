"""V6: Selectivity confusion matrix with bootstrap CIs.

For each (model, cav_method), matrix M[i,j] = mean Δaccuracy when ablating
concept_i and measuring concept_j at the most intense α.
Strong diagonal + pale off-diagonal = selective ablation.

Contrast-plot upgrade:
- Each cell gets a bootstrap 95% CI across its layer/α replicates.
- Cells whose CI excludes 0 are framed with a bold border (visual = "robust
  effect"), replacing the undifferentiated diagonal rectangle.
- Annotations are shown only on the diagonal plus the top-K off-diagonal
  by |mean| to keep large matrices readable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from viz_style import (MODEL_META, apply_style, RESULTS_DIR, savefig,
                       concept_category)
from ci_utils import bootstrap_ci

apply_style()
MODELS = ["gpt2_small", "pythia_28b"]
METHODS = ["svm", "mean_diff"]
TOP_K_OFFDIAG = 8


def build_matrix(df):
    """Return (mean_mat, lo_mat, hi_mat, concepts).

    Averaged across layers/α at the largest α per (ablated, measured).
    Bootstrap CI uses the delta_accuracy values within each cell (rows).
    """
    d = df[df["layer_numeric"] >= 0]
    if len(d) == 0:
        return None, None, None, []
    alpha_max = d["alpha"].max()
    d = d[d["alpha"] == alpha_max]

    concepts = sorted(set(d["ablated_concept"]).intersection(d["measured_concept"]),
                      key=lambda c: (concept_category(c), c))
    if len(concepts) < 2:
        return None, None, None, []

    n = len(concepts)
    mean_mat = np.full((n, n), np.nan)
    lo_mat = np.full_like(mean_mat, np.nan)
    hi_mat = np.full_like(mean_mat, np.nan)
    for i, ca in enumerate(concepts):
        for j, cm in enumerate(concepts):
            cell = d[(d["ablated_concept"] == ca) & (d["measured_concept"] == cm)]
            vals = cell["delta_accuracy"].dropna().to_numpy()
            if vals.size == 0:
                continue
            mean_mat[i, j] = float(vals.mean())
            if vals.size >= 2:
                lo_mat[i, j], hi_mat[i, j] = bootstrap_ci(vals)
    return mean_mat, lo_mat, hi_mat, concepts


def _top_offdiag_indices(mat: np.ndarray, k: int) -> list[tuple[int, int]]:
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    flat = np.where(mask, np.abs(mat), 0.0).ravel()
    # Ignore NaN-derived zeros
    flat = np.nan_to_num(flat, nan=0.0)
    order = np.argsort(-flat)
    idx = []
    for p in order[:k]:
        i, j = divmod(int(p), n)
        if flat[p] > 0:
            idx.append((i, j))
    return idx


def main():
    df = pd.read_csv(RESULTS_DIR / "unified_cct_results_agg.csv")
    df = df[df["ablation_mode"] == "all_tokens"]

    fig, axes = plt.subplots(len(MODELS), len(METHODS), figsize=(15, 13))

    vmax = 0.5
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for ri, m in enumerate(MODELS):
        for ci, method in enumerate(METHODS):
            ax = axes[ri, ci]
            sub = df[(df["model"] == m) & (df["cav_method"] == method)]
            mat, lo, hi, concepts = build_matrix(sub)
            if mat is None:
                ax.axis("off"); continue

            im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="equal")
            ax.set_xticks(range(len(concepts)))
            ax.set_xticklabels(concepts, rotation=90, fontsize=6)
            ax.set_yticks(range(len(concepts)))
            ax.set_yticklabels(concepts, fontsize=6)
            ax.set_xlabel("concepto medido", fontsize=9)
            if ci == 0:
                ax.set_ylabel("concepto ablacionado", fontsize=9)
            ax.set_title(f"{MODEL_META[m]['label']} · {method}", fontsize=10)

            # Annotate: diagonal + top-K off-diagonal by |mean|
            cells = [(k, k) for k in range(len(concepts))] + \
                    _top_offdiag_indices(mat, TOP_K_OFFDIAG)
            for (i, j) in cells:
                v = mat[i, j]
                if np.isnan(v):
                    continue
                color = "white" if abs(v) > 0.35 else "black"
                if not np.isnan(lo[i, j]) and not np.isnan(hi[i, j]):
                    txt = f"{v:.2f}\n[{lo[i, j]:.2f},{hi[i, j]:.2f}]"
                    fs = 5.2
                else:
                    txt = f"{v:.2f}"
                    fs = 6.0
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontsize=fs, linespacing=0.95)

            # Bold border on cells whose CI excludes 0 (robust effects)
            for i in range(len(concepts)):
                for j in range(len(concepts)):
                    if np.isnan(lo[i, j]) or np.isnan(hi[i, j]):
                        # Keep a thin border on the diagonal even without CI
                        if i == j:
                            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                        fill=False, edgecolor="#888",
                                                        linewidth=0.5))
                        continue
                    if lo[i, j] > 0 or hi[i, j] < 0:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                    fill=False, edgecolor="#111",
                                                    linewidth=1.5))
                    elif i == j:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                    fill=False, edgecolor="#888",
                                                    linewidth=0.5))

            # Summary stats
            diag = np.nanmean(np.diag(mat))
            off_mask = ~np.eye(len(concepts), dtype=bool)
            off = float(np.nanmean(mat[off_mask])) if off_mask.any() else np.nan
            n_sig_diag = sum(1 for k in range(len(concepts))
                             if not np.isnan(lo[k, k]) and (lo[k, k] > 0 or hi[k, k] < 0))
            ax.text(0.02, 0.98,
                    f"diagonal μ = {diag:.3f}\nfuera-diagonal μ = {off:.3f}\n"
                    f"celdas diag. robustas: {n_sig_diag}/{len(concepts)}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="#BBB", alpha=0.9))

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02)
    cbar.set_label("Δ precisión (media)", fontsize=9)

    fig.suptitle("Matriz de selectividad: concepto ablacionado × concepto medido\n"
                 "Diagonal fuerte = selectivo · bordes gruesos = CI 95% bootstrap excluye 0",
                 fontsize=13, fontweight="semibold", y=0.995)
    savefig(fig, "selectivity_matrix")


if __name__ == "__main__":
    main()
