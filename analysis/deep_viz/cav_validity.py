"""V2: CAV validity sheet.

For top-N concepts of each model, stack four evidence panels:
  (1) SVM cross-validated accuracy vs layer (separability).
  (2) Cohen's d vs layer (effect size in projection space).
  (3) Inter-layer CAV stability — cosine matrix across layers for the same concept.
  (4) Cross-concept "isolation" — distribution of cosines vs other concepts' CAVs
      (a genuine CAV should be near-orthogonal to random peers).

Demonstrates multi-evidential convergence: a real CAV must pass ALL four.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from viz_style import (MODEL_META, MODEL_COLORS, apply_style, load_cavs,
                       MODELS_DIR, savefig, depth_frac)

apply_style()

MODELS = ["gpt2_small", "pythia_28b"]           # text models with extraction reports
TOP_N = 5


def load_report(model_key: str) -> dict:
    d = MODELS_DIR / MODEL_META[model_key]["dir"] / "cavs"
    p = d / "extraction_report_individual.json"
    if not p.exists():
        p = d / "extraction_report.json"
    return json.load(open(p))


def pick_top_concepts(report: dict, n: int = TOP_N) -> list[str]:
    """Rank concepts by mean CV accuracy across available layers."""
    scored = []
    for c, layers in report["results"].items():
        accs = [layers[lk]["svm"]["cv_accuracy_mean"] for lk in layers
                if "svm" in layers[lk]]
        if accs:
            scored.append((c, float(np.mean(accs))))
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:n]]


def panel_cv_accuracy(ax, report, concept, model_key):
    layers = sorted([int(k.split("_")[1]) for k in report["results"][concept]])
    accs = [report["results"][concept][f"layer_{L}"]["svm"]["cv_accuracy_mean"] for L in layers]
    stds = [report["results"][concept][f"layer_{L}"]["svm"].get("cv_accuracy_std", 0) for L in layers]
    xs = [depth_frac(model_key, L) for L in layers]
    ax.axhspan(0.4, 0.6, color="#EEE", label="azar ±10%")
    ax.errorbar(xs, accs, yerr=stds, marker="o", color=MODEL_COLORS[model_key],
                ecolor=MODEL_COLORS[model_key], alpha=0.9, capsize=3)
    ax.set_ylim(0.4, 1.05)
    ax.set_xlim(0, 1.02)
    ax.set_ylabel("Precisión CV (SVM)", fontsize=8)
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=7)


def panel_cohens_d(ax, report, concept, model_key):
    layers = sorted([int(k.split("_")[1]) for k in report["results"][concept]])
    ds = [report["results"][concept][f"layer_{L}"]["svm"].get("cohens_d", np.nan) for L in layers]
    xs = [depth_frac(model_key, L) for L in layers]
    ax.axhspan(0, 0.8, color="#FBECEC", label="pequeño")
    ax.axhspan(0.8, 1.5, color="#FDF3D7", label="medio")
    ax.bar(xs, ds, width=0.07, color=MODEL_COLORS[model_key], alpha=0.85,
           edgecolor="white", linewidth=0.6)
    ax.set_ylabel("d de Cohen", fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=7)


def panel_inter_layer(ax, cavs_by_layer, concept):
    all_layers = sorted(cavs_by_layer.keys())
    layers = [L for L in all_layers if concept in cavs_by_layer[L]]
    vecs = []
    for L in layers:
        v = cavs_by_layer[L][concept]
        v = v / (np.linalg.norm(v) + 1e-12)
        vecs.append(v)
    if not vecs:
        ax.axis("off"); return
    V = np.stack(vecs)
    C = V @ V.T
    im = ax.imshow(C, cmap="viridis", vmin=-0.3, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(layers))); ax.set_yticks(range(len(layers)))
    ax.set_xticklabels([f"L{L}" for L in layers], fontsize=7)
    ax.set_yticklabels([f"L{L}" for L in layers], fontsize=7)
    for i in range(len(layers)):
        for j in range(len(layers)):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if C[i, j] > 0.5 else "black")
    ax.set_ylabel("estabilidad", fontsize=8)


def panel_isolation(ax, cavs_by_layer, concept, model_key):
    """Cosine of this concept's mean-layer CAV vs every other concept."""
    layers = sorted(cavs_by_layer.keys())
    # use middle layer
    L = layers[len(layers) // 2]
    if concept not in cavs_by_layer[L]:
        ax.axis("off"); return
    v = cavs_by_layer[L][concept]
    v = v / (np.linalg.norm(v) + 1e-12)
    others = [c for c in cavs_by_layer[L] if c != concept]
    cos = []
    for c in others:
        w = cavs_by_layer[L][c]
        w = w / (np.linalg.norm(w) + 1e-12)
        cos.append(float(v @ w))
    cos = np.array(cos)
    ax.hist(cos, bins=20, color=MODEL_COLORS[model_key], alpha=0.7,
            edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="k", linewidth=0.8, alpha=0.5)
    ax.axvline(cos.mean(), color="#C44", linewidth=1.4, linestyle="--",
               label=f"μ={cos.mean():.2f}")
    ax.set_xlabel("cos(vs. otros CAVs)", fontsize=8)
    ax.set_ylabel("frecuencia", fontsize=8)
    ax.set_xlim(-0.5, 1.0)
    ax.legend(loc="upper right", fontsize=7)
    ax.tick_params(labelsize=7)


def main():
    n_rows = TOP_N
    n_cols = 4 * len(MODELS)  # 4 panels per model
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * len(MODELS) * 1.6, 2.1 * n_rows))

    for mi, model_key in enumerate(MODELS):
        report = load_report(model_key)
        cavs = load_cavs(model_key, "svm")
        top = pick_top_concepts(report, TOP_N)

        col0 = mi * 4
        # Column group header
        axes[0, col0].set_title(
            f"{MODEL_META[model_key]['label']}\nseparabilidad",
            fontsize=9, color=MODEL_COLORS[model_key])
        axes[0, col0 + 1].set_title("tamaño del efecto", fontsize=9, color=MODEL_COLORS[model_key])
        axes[0, col0 + 2].set_title("estabilidad entre capas", fontsize=9, color=MODEL_COLORS[model_key])
        axes[0, col0 + 3].set_title("aislamiento vs. otros CAVs", fontsize=9, color=MODEL_COLORS[model_key])

        for ri, concept in enumerate(top):
            panel_cv_accuracy(axes[ri, col0], report, concept, model_key)
            panel_cohens_d(axes[ri, col0 + 1], report, concept, model_key)
            panel_inter_layer(axes[ri, col0 + 2], cavs, concept)
            panel_isolation(axes[ri, col0 + 3], cavs, concept, model_key)
            axes[ri, col0].set_ylabel(f"{concept}\nPrecisión CV", fontsize=8)

    fig.suptitle("Hoja de validez de los CAVs: cuatro líneas de evidencia independientes\n"
                 "Un CAV es genuino solo si supera las CUATRO pruebas",
                 fontsize=13, fontweight="semibold", y=1.00)
    fig.tight_layout()
    savefig(fig, "cav_validity")


if __name__ == "__main__":
    main()
