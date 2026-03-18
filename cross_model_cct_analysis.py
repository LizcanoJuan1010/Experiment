"""
Cross-Model CCT Analysis
=========================
Statistical analysis of unified CCT results across GPT2-small, Pythia-2.8B,
and LLaVA-1.5-7B. Tests whether CAV ablation effectiveness varies by model
scale, layer depth, intensity, and CAV method.

Reads:  results/unified_cct_results.csv
Writes: results/cross_model_analysis_results.json
        results/figures/cross_model/  (plots)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    from statsmodels.regression.mixed_linear_model import MixedLM
except ImportError:
    raise ImportError("statsmodels required: pip install statsmodels")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning if 'ConvergenceWarning' in dir() else FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "results", "unified_cct_results.csv")
OUT_JSON = os.path.join(SCRIPT_DIR, "results", "cross_model_analysis_results.json")
FIG_DIR = os.path.join(SCRIPT_DIR, "results", "figures", "cross_model")
os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    """Load the unified CSV."""
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"  Models: {df['model'].unique().tolist()}")
    print(f"  Overlap tiers: {df['concept_overlap_tier'].value_counts().to_dict()}")
    return df


# ===================================================================
# Analysis 1: Primary Mixed-Effects Model (all_3 concepts)
# ===================================================================
def analysis_primary(df):
    """
    Primary analysis: Mixed-effects regression on the 5 shared concepts.
    DV: delta_accuracy
    Fixed: model * alpha + cav_method + layer_type
    Random: (1 | ablated_concept)
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Primary Mixed-Effects Model")
    print("=" * 60)

    sub = df[
        (df["concept_overlap_tier"] == "all_3") &
        (df["on_target"] == True) &
        (df["ablation_mode"] == "all_tokens")
    ].copy()

    print(f"  Subset: {len(sub)} rows (all_3, on_target, all_tokens)")
    print(f"  Models: {sub['model'].value_counts().to_dict()}")

    if len(sub) < 10:
        print("  SKIP: Too few rows")
        return {"status": "skipped", "reason": "too few rows"}

    # Descriptive stats
    desc = sub.groupby("model")["delta_accuracy"].agg(["mean", "std", "count"])
    print(f"\n  Descriptive stats (delta_accuracy by model):")
    print(desc.to_string(float_format="%.4f"))

    # Mixed-effects model
    try:
        md = MixedLM.from_formula(
            "delta_accuracy ~ C(model) * alpha + C(cav_method) + C(layer_type)",
            groups="ablated_concept",
            data=sub,
        )
        result = md.fit(reml=True)
        print(f"\n  Mixed-Effects Model Summary:")
        print(result.summary().tables[1].to_string())

        coefficients = {}
        for name, val in result.params.items():
            coefficients[name] = {
                "estimate": float(val),
                "std_error": float(result.bse[name]),
                "z_value": float(result.tvalues[name]),
                "p_value": float(result.pvalues[name]),
            }

        return {
            "status": "converged",
            "n_observations": len(sub),
            "descriptive": {
                model: {
                    "mean_delta": float(desc.loc[model, "mean"]),
                    "std_delta": float(desc.loc[model, "std"]),
                    "n": int(desc.loc[model, "count"]),
                }
                for model in desc.index
            },
            "fixed_effects": coefficients,
            "aic": float(result.aic),
            "bic": float(result.bic),
            "log_likelihood": float(result.llf),
        }
    except Exception as e:
        print(f"  Mixed model failed: {e}")
        # Fallback to OLS ANOVA
        print("  Falling back to OLS ANOVA...")
        try:
            ols_model = smf.ols(
                "delta_accuracy ~ C(model) * alpha + C(cav_method) + C(layer_type)",
                data=sub,
            ).fit()
            aov = anova_lm(ols_model, typ=2)
            print(aov.to_string(float_format="%.4f"))
            return {
                "status": "ols_fallback",
                "n_observations": len(sub),
                "descriptive": {
                    model: {
                        "mean_delta": float(desc.loc[model, "mean"]),
                        "std_delta": float(desc.loc[model, "std"]),
                        "n": int(desc.loc[model, "count"]),
                    }
                    for model in desc.index
                },
                "anova_table": aov.to_dict(),
                "r_squared": float(ols_model.rsquared),
                "r_squared_adj": float(ols_model.rsquared_adj),
            }
        except Exception as e2:
            print(f"  OLS also failed: {e2}")
            return {"status": "failed", "error": str(e2)}


# ===================================================================
# Analysis 2: Dose-Response (alpha effect by model)
# ===================================================================
def analysis_dose_response(df):
    """Does the alpha slope differ across models?"""
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Dose-Response (alpha x model)")
    print("=" * 60)

    sub = df[
        (df["concept_overlap_tier"] == "all_3") &
        (df["on_target"] == True) &
        (df["ablation_mode"] == "all_tokens") &
        (df["layer_type"] != "all_layers")
    ].copy()

    print(f"  Subset: {len(sub)} rows")

    # Mean delta by model x alpha
    pivot = sub.groupby(["model", "alpha"])["delta_accuracy"].mean().unstack("alpha")
    print(f"\n  Mean delta_accuracy (model x alpha):")
    print(pivot.to_string(float_format="%.4f"))

    # OLS with interaction
    try:
        model = smf.ols("delta_accuracy ~ C(model) * alpha", data=sub).fit()
        aov = anova_lm(model, typ=2)
        print(f"\n  ANOVA (Type II):")
        print(aov.to_string(float_format="%.4f"))

        result = {
            "n_observations": len(sub),
            "mean_by_model_alpha": pivot.to_dict(),
            "anova": {
                k: {
                    "sum_sq": float(v["sum_sq"]),
                    "df": float(v["df"]),
                    "F": float(v["F"]) if not np.isnan(v["F"]) else None,
                    "PR(>F)": float(v["PR(>F)"]) if not np.isnan(v["PR(>F)"]) else None,
                }
                for k, v in aov.to_dict("index").items()
            },
            "r_squared": float(model.rsquared),
        }
    except Exception as e:
        print(f"  Failed: {e}")
        result = {"error": str(e)}

    # Plot
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 5))
        alphas = sorted(sub["alpha"].unique())
        markers = {"gpt2_small": "o", "pythia_28b": "s", "llava_15_7b": "^"}
        colors = {"gpt2_small": "#1f77b4", "pythia_28b": "#ff7f0e", "llava_15_7b": "#2ca02c"}
        labels = {"gpt2_small": "GPT2-small (124M)", "pythia_28b": "Pythia-2.8B", "llava_15_7b": "LLaVA-1.5-7B"}

        for m in sub["model"].unique():
            ms = sub[sub["model"] == m]
            means = ms.groupby("alpha")["delta_accuracy"].mean()
            sems = ms.groupby("alpha")["delta_accuracy"].sem()
            ax.errorbar(
                means.index, means.values, yerr=sems.values,
                marker=markers.get(m, "o"), color=colors.get(m, "gray"),
                label=labels.get(m, m), capsize=3, linewidth=2,
            )
        ax.set_xlabel("Alpha (ablation intensity)", fontsize=12)
        ax.set_ylabel("Delta accuracy (on-target deficit)", fontsize=12)
        ax.set_title("Dose-Response: CAV Ablation by Model", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "dose_response.png"), dpi=150)
        plt.close(fig)
        print(f"  Plot saved: dose_response.png")

    return result


# ===================================================================
# Analysis 3: Specificity (on-target vs off-target by model)
# ===================================================================
def analysis_specificity(df):
    """Is ablation selective across models?"""
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Specificity (on_target x model)")
    print("=" * 60)

    sub = df[
        (df["concept_overlap_tier"] == "all_3") &
        (df["ablation_mode"] == "all_tokens")
    ].copy()

    print(f"  Subset: {len(sub)} rows")

    pivot = sub.groupby(["model", "on_target"])["delta_accuracy"].mean().unstack("on_target")
    pivot.columns = ["off_target", "on_target"]
    pivot["ratio"] = pivot["on_target"] / pivot["off_target"].replace(0, np.nan)
    print(f"\n  Specificity by model:")
    print(pivot.to_string(float_format="%.4f"))

    try:
        model = smf.ols("delta_accuracy ~ C(model) * C(on_target)", data=sub).fit()
        aov = anova_lm(model, typ=2)
        print(f"\n  ANOVA (Type II):")
        print(aov.to_string(float_format="%.4f"))

        result = {
            "n_observations": len(sub),
            "specificity_by_model": pivot.to_dict("index"),
            "anova": {
                k: {
                    "sum_sq": float(v["sum_sq"]),
                    "df": float(v["df"]),
                    "F": float(v["F"]) if not np.isnan(v["F"]) else None,
                    "p_value": float(v["PR(>F)"]) if not np.isnan(v["PR(>F)"]) else None,
                }
                for k, v in aov.to_dict("index").items()
            },
        }
    except Exception as e:
        print(f"  Failed: {e}")
        result = {"error": str(e)}

    # Plot
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(7, 5))
        models = pivot.index.tolist()
        x = np.arange(len(models))
        w = 0.35
        ax.bar(x - w/2, pivot["on_target"], w, label="On-target", color="#d62728")
        ax.bar(x + w/2, pivot["off_target"], w, label="Off-target", color="#7f7f7f")
        ax.set_xticks(x)
        ax.set_xticklabels(["GPT2-small\n(124M)", "Pythia-2.8B", "LLaVA-1.5-7B"][:len(models)])
        ax.set_ylabel("Mean delta accuracy", fontsize=12)
        ax.set_title("Ablation Specificity by Model", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "specificity.png"), dpi=150)
        plt.close(fig)
        print(f"  Plot saved: specificity.png")

    return result


# ===================================================================
# Analysis 4: SVM vs Mean-Diff across models
# ===================================================================
def analysis_cav_method(df):
    """Does SVM superiority generalize across models?"""
    print("\n" + "=" * 60)
    print("ANALYSIS 4: CAV Method (svm vs mean_diff)")
    print("=" * 60)

    sub = df[
        (df["concept_overlap_tier"] == "all_3") &
        (df["on_target"] == True) &
        (df["ablation_mode"] == "all_tokens")
    ].copy()

    pivot = sub.groupby(["model", "cav_method"])["delta_accuracy"].mean().unstack("cav_method")
    print(f"\n  Mean delta by model x method:")
    print(pivot.to_string(float_format="%.4f"))

    try:
        model = smf.ols("delta_accuracy ~ C(model) * C(cav_method)", data=sub).fit()
        aov = anova_lm(model, typ=2)
        print(f"\n  ANOVA (Type II):")
        print(aov.to_string(float_format="%.4f"))

        result = {
            "n_observations": len(sub),
            "mean_by_model_method": pivot.to_dict(),
            "anova": {
                k: {
                    "sum_sq": float(v["sum_sq"]),
                    "df": float(v["df"]),
                    "F": float(v["F"]) if not np.isnan(v["F"]) else None,
                    "p_value": float(v["PR(>F)"]) if not np.isnan(v["PR(>F)"]) else None,
                }
                for k, v in aov.to_dict("index").items()
            },
        }
    except Exception as e:
        print(f"  Failed: {e}")
        result = {"error": str(e)}

    return result


# ===================================================================
# Analysis 5: Layer Depth effect
# ===================================================================
def analysis_layer_depth(df):
    """Does normalized layer depth have similar effects across models?"""
    print("\n" + "=" * 60)
    print("ANALYSIS 5: Layer Depth x Model")
    print("=" * 60)

    sub = df[
        (df["concept_overlap_tier"] == "all_3") &
        (df["on_target"] == True) &
        (df["ablation_mode"] == "all_tokens") &
        (df["layer_type"] != "all_layers")
    ].copy()

    pivot = sub.groupby(["model", "layer_type"])["delta_accuracy"].mean().unstack("layer_type")
    print(f"\n  Mean delta by model x layer:")
    print(pivot.to_string(float_format="%.4f"))

    try:
        model = smf.ols("delta_accuracy ~ C(model) * C(layer_type)", data=sub).fit()
        aov = anova_lm(model, typ=2)
        print(f"\n  ANOVA (Type II):")
        print(aov.to_string(float_format="%.4f"))

        result = {
            "n_observations": len(sub),
            "mean_by_model_layer": pivot.to_dict(),
            "anova": {
                k: {
                    "sum_sq": float(v["sum_sq"]),
                    "df": float(v["df"]),
                    "F": float(v["F"]) if not np.isnan(v["F"]) else None,
                    "p_value": float(v["PR(>F)"]) if not np.isnan(v["PR(>F)"]) else None,
                }
                for k, v in aov.to_dict("index").items()
            },
        }
    except Exception as e:
        print(f"  Failed: {e}")
        result = {"error": str(e)}

    # Heatmap
    if HAS_MPL and not pivot.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(pivot.values, cmap="Reds", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)
        ax.set_title("Mean On-Target Delta by Model x Layer", fontsize=13)
        fig.colorbar(im, ax=ax, label="delta_accuracy")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "layer_depth_heatmap.png"), dpi=150)
        plt.close(fig)
        print(f"  Plot saved: layer_depth_heatmap.png")

    return result


# ===================================================================
# Analysis 6: LLaVA ablation mode (image_only vs all_tokens)
# ===================================================================
def analysis_llava_ablation_mode(df):
    """LLaVA-specific: image_tokens_only vs all_tokens."""
    print("\n" + "=" * 60)
    print("ANALYSIS 6: LLaVA Ablation Mode")
    print("=" * 60)

    sub = df[
        (df["model"] == "llava_15_7b") &
        (df["on_target"] == True)
    ].copy()

    if len(sub) < 5:
        print("  SKIP: Not enough LLaVA data")
        return {"status": "skipped"}

    pivot = sub.groupby(["ablation_mode", "alpha"])["delta_accuracy"].mean().unstack("alpha")
    print(f"\n  Mean delta by ablation_mode x alpha:")
    print(pivot.to_string(float_format="%.4f"))

    mode_means = sub.groupby("ablation_mode")["delta_accuracy"].agg(["mean", "std", "count"])
    print(f"\n  Summary:")
    print(mode_means.to_string(float_format="%.4f"))

    # t-test between modes
    all_tok = sub[sub["ablation_mode"] == "all_tokens"]["delta_accuracy"]
    img_only = sub[sub["ablation_mode"] == "image_tokens_only"]["delta_accuracy"]

    if len(all_tok) > 0 and len(img_only) > 0:
        t_stat, p_val = stats.ttest_ind(all_tok, img_only, equal_var=False)
        cohens_d = (all_tok.mean() - img_only.mean()) / np.sqrt(
            (all_tok.std()**2 + img_only.std()**2) / 2
        )
        print(f"\n  Welch t-test: t={t_stat:.3f}, p={p_val:.6f}")
        print(f"  Cohen's d: {cohens_d:.3f}")

        result = {
            "n_all_tokens": len(all_tok),
            "n_image_only": len(img_only),
            "mean_all_tokens": float(all_tok.mean()),
            "mean_image_only": float(img_only.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "mean_by_mode_alpha": pivot.to_dict(),
        }
    else:
        result = {"status": "insufficient_data"}

    return result


# ===================================================================
# Summary table
# ===================================================================
def summary_table(df):
    """Generate a quick descriptive summary."""
    print("\n" + "=" * 60)
    print("DESCRIPTIVE SUMMARY")
    print("=" * 60)

    sub = df[
        (df["concept_overlap_tier"] == "all_3") &
        (df["on_target"] == True) &
        (df["ablation_mode"] == "all_tokens")
    ]

    print("\n  Mean on-target delta by model x alpha (shared concepts only):")
    pivot = sub.groupby(["model", "alpha"])["delta_accuracy"].mean().unstack("alpha")
    print(pivot.to_string(float_format="%.4f"))

    print("\n  Overall by model:")
    overall = sub.groupby("model")["delta_accuracy"].agg(["mean", "std", "median", "count"])
    print(overall.to_string(float_format="%.4f"))

    return {
        "mean_by_model_alpha": {
            str(k): {str(a): float(v) for a, v in row.items()}
            for k, row in pivot.to_dict("index").items()
        },
        "overall_by_model": overall.to_dict("index"),
    }


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("  Cross-Model CCT Analysis")
    print("=" * 60)

    df = load_data()

    results = {}
    results["summary"] = summary_table(df)
    results["primary_mixed_effects"] = analysis_primary(df)
    results["dose_response"] = analysis_dose_response(df)
    results["specificity"] = analysis_specificity(df)
    results["cav_method"] = analysis_cav_method(df)
    results["layer_depth"] = analysis_layer_depth(df)
    results["llava_ablation_mode"] = analysis_llava_ablation_mode(df)

    # Save
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Results saved: {OUT_JSON}")
    print(f"  Figures saved: {FIG_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
