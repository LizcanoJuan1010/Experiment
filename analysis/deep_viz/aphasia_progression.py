"""V7: LLaVA dose-response as aphasia analogue.

Panel A — Ridge-like stacked distributions of on-target accuracy drop across
           α ∈ {3, 6, 10, 15, 20}, one curve per layer: shows progressive
           collapse with dose.
Panel B — Mean Δaccuracy vs α per layer with clinical-stage annotations
           (coherent → paraphasia → perseveration → collapse) drawn from
           the qualitative protocol described in the thesis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from viz_style import apply_style, RESULTS_DIR, savefig

apply_style()

STAGE_ANNOTATIONS = [
    (3, "coherente"),
    (6, "deriva léxica"),
    (10, "parafasia semántica"),
    (15, "perseveración"),
    (20, "colapso global"),
]


def main():
    df = pd.read_csv(RESULTS_DIR / "unified_cct_results.csv", low_memory=False)
    d = df[(df["model"] == "llava_15_7b") &
           (df["on_target"] == True) &
           (df["cav_method"] == "svm") &
           (df["ablation_mode"] == "all_tokens") &
           (df["layer_numeric"] >= 0)].copy()

    alphas = sorted(d["alpha"].unique())
    layers = sorted(d["layer_numeric"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    # ==== Panel A: ridge by alpha ====
    ax = axes[0]
    palette = plt.cm.magma(np.linspace(0.15, 0.85, len(alphas)))
    xs = np.linspace(-1.05, 0.3, 400)
    offset = 0
    step = 1.0
    for i, a in enumerate(alphas):
        vals = d[d["alpha"] == a]["delta_accuracy"].dropna().values
        if len(vals) < 3:
            continue
        try:
            kde = gaussian_kde(vals, bw_method=0.25)
            ys = kde(xs)
            ys = ys / ys.max() * 0.8
        except np.linalg.LinAlgError:
            ys = np.zeros_like(xs)
        ax.fill_between(xs, offset, offset + ys, color=palette[i], alpha=0.85,
                        edgecolor="white", linewidth=0.6)
        ax.axhline(offset, color="#CCC", lw=0.5)
        ax.text(0.28, offset + 0.1, f"α={a:g}  (n={len(vals)})",
                fontsize=9, va="bottom")
        stage = next((s for al, s in STAGE_ANNOTATIONS if al == a), "")
        if stage:
            ax.text(-1.03, offset + 0.55, stage, fontsize=9,
                    fontstyle="italic", color="#444", va="center")
        offset += step

    ax.set_xlim(-1.05, 0.35)
    ax.set_ylim(-0.1, offset + 0.5)
    ax.axvline(0, color="#444", lw=0.8, alpha=0.6)
    ax.set_xlabel("Δ precisión  (negativo = degradación)")
    ax.set_yticks([])
    ax.set_title("A · Densidad de Δ precisión por α (LLaVA, on-target, SVM)",
                 fontsize=10)

    # ==== Panel B: mean Δ with CI per layer ====
    ax = axes[1]
    layer_colors = plt.cm.cividis(np.linspace(0.15, 0.85, len(layers)))
    for li, L in enumerate(layers):
        means = []
        sems = []
        xs_a = []
        for a in alphas:
            v = d[(d["alpha"] == a) & (d["layer_numeric"] == L)]["delta_accuracy"].dropna()
            if len(v) == 0:
                continue
            means.append(v.mean())
            sems.append(v.std() / np.sqrt(max(len(v), 1)))
            xs_a.append(a)
        ax.errorbar(xs_a, means, yerr=sems, marker="o", color=layer_colors[li],
                    lw=2.0, markersize=8, markeredgecolor="white",
                    label=f"capa {L}", capsize=3)

    # stage annotations
    for a, stage in STAGE_ANNOTATIONS:
        ax.axvline(a, color="#DDD", lw=0.6, zorder=0)
        ax.text(a, 0.08, stage, rotation=35, fontsize=8,
                color="#666", ha="left", va="bottom")

    ax.axhline(0, color="#444", lw=0.8, alpha=0.6)
    ax.set_xlabel("α (intensidad)")
    ax.set_ylabel("Δ precisión (media)")
    ax.set_xticks(alphas)
    ax.set_title("B · Degradación progresiva por α con etapas cualitativas",
                 fontsize=10)
    ax.legend(loc="lower left", fontsize=9)

    fig.suptitle("Analogía de afasia dependiente de dosis (LLaVA-1.5-7B)",
                 fontsize=13, fontweight="semibold", y=1.01)
    fig.tight_layout()
    savefig(fig, "aphasia_progression")


if __name__ == "__main__":
    main()
