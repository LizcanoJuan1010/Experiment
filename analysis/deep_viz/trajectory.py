"""V5: Cross-model trajectory of separability and effect size.

Left axis: SVM CV-accuracy (line + bootstrap band over concepts).
Right axis: Cohen's d of CAV projection (bar, same x = depth fraction).
One panel per model, all sharing a depth fraction (0-1) x-axis so that
a 12-layer model and a 32-layer model are directly comparable.

Replaces raw vector norm (which is ~1 because CAVs are stored unit-normalized)
with Cohen's d, the normalization-invariant signal-to-noise analogue.
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt

from viz_style import (MODEL_META, MODEL_COLORS, apply_style, MODELS_DIR,
                       savefig, depth_frac)

apply_style()
MODELS = ["gpt2_small", "pythia_28b", "llava_15_7b"]


def load_summary(model_key):
    d = MODELS_DIR / MODEL_META[model_key]["dir"] / "cavs"
    p = d / "extraction_report_individual.json"
    if p.exists():
        return json.load(open(p))
    # Pythia writes its report as extraction_report.json with the same schema.
    # LLaVA also has that filename but with empty `results`, so validate content.
    p2 = d / "extraction_report.json"
    if p2.exists():
        data = json.load(open(p2))
        if data.get("results") and any(
            "svm" in meta
            for layers in data["results"].values()
            for meta in layers.values()
        ):
            return data
    # LLaVA: use statistical_validation.json (different schema)
    sv = d / "statistical_validation.json"
    if sv.exists():
        raw = json.load(open(sv))["results"]
        out = {"results": {}}
        for c, layers in raw.items():
            out["results"][c] = {}
            for lk, meta in layers.items():
                svm = meta.get("svm", {})
                out["results"][c][lk] = {"svm": {
                    "cv_accuracy_mean": svm.get("separability", {}).get("cv_accuracy_mean",
                                          meta.get("mean_diff", {}).get("selectivity", {}).get("real_accuracy", np.nan)),
                    "cv_accuracy_std": 0.0,
                    "cohens_d": svm.get("cohens_d", {}).get("cohens_d",
                                 meta.get("mean_diff", {}).get("cohens_d", {}).get("cohens_d", np.nan)),
                }}
        return out
    return None


def per_layer_stats(report, model_key):
    """Aggregate (mean, std) of CV-acc and cohen's d across concepts per layer."""
    by_layer = {}
    for c, layers in report["results"].items():
        for lk, meta in layers.items():
            L = int(lk.split("_")[1])
            svm = meta.get("svm", {})
            if "cv_accuracy_mean" not in svm:
                continue
            by_layer.setdefault(L, {"cv": [], "d": []})
            by_layer[L]["cv"].append(svm.get("cv_accuracy_mean", np.nan))
            by_layer[L]["d"].append(svm.get("cohens_d", np.nan))
    xs, cv_m, cv_s, d_m, d_s = [], [], [], [], []
    for L in sorted(by_layer):
        cvs = np.array(by_layer[L]["cv"], dtype=float)
        ds = np.array(by_layer[L]["d"], dtype=float)
        cvs = cvs[~np.isnan(cvs)]; ds = ds[~np.isnan(ds)]
        if len(cvs) == 0:
            continue
        xs.append(depth_frac(model_key, L))
        cv_m.append(cvs.mean()); cv_s.append(cvs.std() if len(cvs) > 1 else 0)
        d_m.append(ds.mean() if len(ds) else np.nan)
        d_s.append(ds.std() if len(ds) > 1 else 0)
    return (np.array(xs), np.array(cv_m), np.array(cv_s),
            np.array(d_m), np.array(d_s))


def main():
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4.6), sharey=True)

    for i, m in enumerate(MODELS):
        ax = axes[i]
        report = load_summary(m)
        if not report:
            ax.set_visible(False); continue
        xs, cv_m, cv_s, d_m, d_s = per_layer_stats(report, m)
        if len(xs) == 0:
            ax.text(0.5, 0.5, "no data", ha="center"); continue

        # Cohen's d as bars on secondary axis
        ax2 = ax.twinx()
        width = 0.03
        bar = ax2.bar(xs, d_m, width=width, yerr=d_s, color=MODEL_COLORS[m],
                      alpha=0.25, edgecolor="white", linewidth=0.6,
                      label="d de Cohen (media ± DE)", ecolor="#888", capsize=2)
        ax2.set_ylabel("d de Cohen", color="#555", fontsize=9)
        ax2.tick_params(axis="y", colors="#555")
        ax2.grid(False)
        ax2.set_ylim(0, max(6.5, float(np.nanmax(d_m)) * 1.2))

        # CV accuracy line
        ax.plot(xs, cv_m, "-o", color=MODEL_COLORS[m], lw=2.0,
                markersize=8, markeredgecolor="white", label="Precisión CV (SVM)")
        ax.fill_between(xs, cv_m - cv_s, cv_m + cv_s,
                        color=MODEL_COLORS[m], alpha=0.18)

        ax.axhspan(0.4, 0.6, color="#EEE", zorder=0)
        ax.set_xlim(0, 1.02); ax.set_ylim(0.4, 1.05)
        ax.set_xlabel("fracción de profundidad de la capa")
        if i == 0:
            ax.set_ylabel("Precisión CV (SVM)")
        ax.set_title(f"{MODEL_META[m]['label']} ({MODEL_META[m]['n_layers']} capas)",
                     color=MODEL_COLORS[m], fontsize=11)

        lines = ax.get_lines() + [bar]
        ax.legend(lines, [l.get_label() if hasattr(l, "get_label") else "d de Cohen"
                          for l in lines],
                  loc="lower right", fontsize=8)

    fig.suptitle("Separabilidad y tamaño del efecto a lo largo de la profundidad\n"
                 "Línea: precisión CV (SVM) (media ± DE entre conceptos) · "
                 "Barras: d de Cohen de la proyección del CAV",
                 fontsize=13, fontweight="semibold", y=1.02)
    fig.tight_layout()
    savefig(fig, "trajectory")


if __name__ == "__main__":
    main()
