"""
Cross-Model CCT Experiment Analysis
====================================
Robust statistical analysis of CAV ablation across GPT2-small, Pythia-2.8B,
and LLaVA-1.5-7B. Designed for thesis-quality reporting.

Reads:
    ../results/unified_cct_results.csv       (item-level)
    ../results/unified_cct_results_agg.csv   (condition-level)

Outputs:
    results/experiment_report.json           (full statistical results)
    results/latex_tables.tex                 (publication-ready tables)
    figures/*.png                            (all plots)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from collections import OrderedDict

try:
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from statsmodels.regression.mixed_linear_model import MixedLM
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests
    HAS_SM = True
except ImportError:
    print("WARNING: statsmodels not found. pip install statsmodels")
    HAS_SM = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
ITEM_CSV = os.path.join(ROOT_DIR, "results", "unified_cct_results.csv")
AGG_CSV = os.path.join(ROOT_DIR, "results", "unified_cct_results_agg.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "results")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Model display names
MODEL_LABELS = {
    "gpt2_small": "GPT-2 Small (124M)",
    "pythia_28b": "Pythia-2.8B",
    "llava_15_7b": "LLaVA-1.5-7B",
}
MODEL_ORDER = ["gpt2_small", "pythia_28b", "llava_15_7b"]
MODEL_COLORS = {"gpt2_small": "#1f77b4", "pythia_28b": "#ff7f0e", "llava_15_7b": "#2ca02c"}
MODEL_MARKERS = {"gpt2_small": "o", "pythia_28b": "s", "llava_15_7b": "^"}

ALPHA_VALUES = [3.0, 6.0, 10.0, 15.0, 20.0]
ALL_3_CONCEPTS = {"camel", "bee", "penguin", "violin", "candle"}


def load_data():
    """Load both CSVs."""
    print("Loading data...")
    df_item = pd.read_csv(ITEM_CSV, low_memory=False)
    df_agg = pd.read_csv(AGG_CSV)
    # Ensure is_correct is numeric (mixed bool/str from CSV)
    df_item["is_correct"] = pd.to_numeric(
        df_item["is_correct"].map({"True": 1, "False": 0, True: 1, False: 0}),
        errors="coerce",
    ).fillna(0).astype(int)
    print(f"  Item-level:      {len(df_item):,} rows x {len(df_item.columns)} cols")
    print(f"  Condition-level: {len(df_agg):,} rows x {len(df_agg.columns)} cols")
    return df_item, df_agg


# ===================================================================
# 1. DESCRIPTIVE STATISTICS
# ===================================================================
def descriptive_stats(df_agg):
    """Comprehensive descriptive statistics."""
    print(f"\n{'='*70}")
    print("1. DESCRIPTIVE STATISTICS")
    print(f"{'='*70}")

    report = {}

    # 1a. Overall dataset composition
    print("\n  1a. Dataset composition")
    for m in MODEL_ORDER:
        sub = df_agg[df_agg["model"] == m]
        print(f"    {MODEL_LABELS[m]}: {len(sub)} conditions")
    report["n_conditions"] = df_agg["model"].value_counts().to_dict()

    # 1b. Baseline accuracy (before ablation)
    # Use condition_accuracy from non-ablated conditions or baseline_accuracy
    print("\n  1b. Concepts evaluated per model")
    for m in MODEL_ORDER:
        sub = df_agg[df_agg["model"] == m]
        concepts = sorted(sub["ablated_concept"].unique())
        print(f"    {MODEL_LABELS[m]}: {len(concepts)} concepts")

    # 1c. Core descriptive: delta_accuracy by model (on-target, all_tokens, all_3)
    core = df_agg[
        (df_agg["concept_overlap_tier"] == "all_3") &
        (df_agg["on_target"] == True) &
        (df_agg["ablation_mode"] == "all_tokens")
    ]

    print("\n  1c. On-target delta_accuracy (shared 5 concepts, all_tokens)")
    desc = core.groupby("model")["delta_accuracy"].describe()
    print(desc.to_string(float_format=lambda x: f"{x:.4f}"))
    report["on_target_descriptive"] = desc.to_dict()

    # 1d. Mean delta by model x alpha
    pivot_ma = core.groupby(["model", "alpha"])["delta_accuracy"].mean().unstack("alpha")
    print("\n  1d. Mean delta by Model x Alpha")
    print(pivot_ma.to_string(float_format=lambda x: f"{x:.4f}"))
    report["mean_delta_model_alpha"] = pivot_ma.to_dict()

    # 1e. Mean delta by model x cav_method
    pivot_mm = core.groupby(["model", "cav_method"])["delta_accuracy"].mean().unstack("cav_method")
    print("\n  1e. Mean delta by Model x CAV Method")
    print(pivot_mm.to_string(float_format=lambda x: f"{x:.4f}"))
    report["mean_delta_model_method"] = pivot_mm.to_dict()

    # 1f. Mean delta by model x layer_type
    core_single = core[core["layer_type"] != "all_layers"]
    pivot_ml = core_single.groupby(["model", "layer_type"])["delta_accuracy"].mean().unstack("layer_type")
    print("\n  1f. Mean delta by Model x Layer")
    print(pivot_ml.to_string(float_format=lambda x: f"{x:.4f}"))
    report["mean_delta_model_layer"] = pivot_ml.to_dict()

    return report


# ===================================================================
# 2. ASSUMPTION CHECKS
# ===================================================================
def assumption_checks(df_agg):
    """Shapiro-Wilk normality and Levene homogeneity tests."""
    print(f"\n{'='*70}")
    print("2. ASSUMPTION CHECKS")
    print(f"{'='*70}")

    core = df_agg[
        (df_agg["concept_overlap_tier"] == "all_3") &
        (df_agg["on_target"] == True) &
        (df_agg["ablation_mode"] == "all_tokens")
    ].copy()

    report = {}

    # 2a. Shapiro-Wilk per model
    print("\n  2a. Shapiro-Wilk normality (delta_accuracy per model)")
    shapiro_results = {}
    for m in MODEL_ORDER:
        vals = core[core["model"] == m]["delta_accuracy"].dropna()
        if len(vals) >= 8:
            w, p = stats.shapiro(vals)
            status = "NORMAL" if p > 0.05 else "NON-NORMAL"
            print(f"    {MODEL_LABELS[m]}: W={w:.4f}, p={p:.6f} [{status}]")
            shapiro_results[m] = {"W": float(w), "p": float(p), "normal": p > 0.05}
    report["shapiro_wilk"] = shapiro_results

    # 2b. Levene's test
    print("\n  2b. Levene's test (homogeneity of variance across models)")
    groups = [core[core["model"] == m]["delta_accuracy"].dropna().values for m in MODEL_ORDER]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        lev_stat, lev_p = stats.levene(*groups)
        status = "HOMOGENEOUS" if lev_p > 0.05 else "HETEROGENEOUS"
        print(f"    Levene F={lev_stat:.4f}, p={lev_p:.6f} [{status}]")
        report["levene"] = {"F": float(lev_stat), "p": float(lev_p), "homogeneous": lev_p > 0.05}

    # 2c. Recommendation
    any_nonnormal = any(not v["normal"] for v in shapiro_results.values())
    heterogeneous = report.get("levene", {}).get("homogeneous", True) is False

    if any_nonnormal or heterogeneous:
        rec = "NON-PARAMETRIC recommended (Kruskal-Wallis). Reporting both parametric and non-parametric."
    else:
        rec = "PARAMETRIC OK (ANOVA assumptions met)."
    print(f"\n  Recommendation: {rec}")
    report["recommendation"] = rec

    return report


# ===================================================================
# 3. MAIN ANOVA: Model x Alpha x Method
# ===================================================================
def main_anova(df_agg):
    """Three-way ANOVA: Model x Alpha x CAV Method on delta_accuracy."""
    print(f"\n{'='*70}")
    print("3. MAIN ANOVA: Model x Alpha x CAV Method")
    print(f"{'='*70}")

    core = df_agg[
        (df_agg["concept_overlap_tier"] == "all_3") &
        (df_agg["on_target"] == True) &
        (df_agg["ablation_mode"] == "all_tokens") &
        (df_agg["layer_type"] != "all_layers")
    ].copy()

    core["alpha_cat"] = core["alpha"].astype(str)
    print(f"  N = {len(core)} observations")

    report = {}

    if not HAS_SM:
        print("  SKIP: statsmodels not available")
        return report

    # Parametric ANOVA
    try:
        model = ols(
            "delta_accuracy ~ C(model) * C(alpha_cat) + C(cav_method) + C(layer_type)",
            data=core
        ).fit()
        aov = anova_lm(model, typ=2)

        # Compute eta-squared
        ss_total = aov["sum_sq"].sum()
        aov["eta_sq"] = aov["sum_sq"] / ss_total
        aov["sig"] = aov["PR(>F)"].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            if not np.isnan(p) else ""
        )

        print("\n  Type II ANOVA:")
        print(aov[["sum_sq", "df", "F", "PR(>F)", "eta_sq", "sig"]].to_string(float_format=lambda x: f"{x:.4f}"))

        report["parametric"] = {
            "type": "Type II ANOVA",
            "r_squared": float(model.rsquared),
            "r_squared_adj": float(model.rsquared_adj),
            "table": {}
        }
        for factor in aov.index:
            if factor == "Residual":
                continue
            report["parametric"]["table"][factor] = {
                "SS": float(aov.loc[factor, "sum_sq"]),
                "df": int(aov.loc[factor, "df"]),
                "F": float(aov.loc[factor, "F"]) if not np.isnan(aov.loc[factor, "F"]) else None,
                "p": float(aov.loc[factor, "PR(>F)"]) if not np.isnan(aov.loc[factor, "PR(>F)"]) else None,
                "eta_sq": float(aov.loc[factor, "eta_sq"]),
                "sig": aov.loc[factor, "sig"],
            }
    except Exception as e:
        print(f"  ANOVA failed: {e}")
        report["parametric"] = {"error": str(e)}

    # Non-parametric: Kruskal-Wallis by model
    print("\n  Kruskal-Wallis (non-parametric, by model):")
    groups = [core[core["model"] == m]["delta_accuracy"].values for m in MODEL_ORDER]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        h_stat, h_p = stats.kruskal(*groups)
        print(f"    H={h_stat:.4f}, p={h_p:.6f}")
        report["kruskal_wallis_model"] = {"H": float(h_stat), "p": float(h_p)}

    # Post-hoc: pairwise Mann-Whitney U
    print("\n  Post-hoc pairwise Mann-Whitney U (model pairs):")
    posthoc = {}
    model_names = [m for m in MODEL_ORDER if m in core["model"].unique()]
    p_values = []
    pairs = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            g1 = core[core["model"] == model_names[i]]["delta_accuracy"].values
            g2 = core[core["model"] == model_names[j]]["delta_accuracy"].values
            u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            cohens_d = (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)
            pair = f"{model_names[i]} vs {model_names[j]}"
            pairs.append(pair)
            p_values.append(u_p)
            posthoc[pair] = {
                "U": float(u_stat), "p_raw": float(u_p), "cohens_d": float(cohens_d),
            }

    # Holm-Bonferroni correction
    if p_values:
        reject, p_corrected, _, _ = multipletests(p_values, method="holm")
        for idx, pair in enumerate(pairs):
            posthoc[pair]["p_corrected"] = float(p_corrected[idx])
            posthoc[pair]["significant"] = bool(reject[idx])
            sig = "*" if reject[idx] else "ns"
            print(f"    {pair}: U={posthoc[pair]['U']:.0f}, "
                  f"p_raw={posthoc[pair]['p_raw']:.6f}, "
                  f"p_corrected={p_corrected[idx]:.6f}, "
                  f"d={posthoc[pair]['cohens_d']:+.3f} [{sig}]")
    report["posthoc_mannwhitney"] = posthoc

    return report


# ===================================================================
# 4. MIXED-EFFECTS MODEL
# ===================================================================
def mixed_effects(df_agg):
    """Mixed-effects regression with concept as random intercept."""
    print(f"\n{'='*70}")
    print("4. MIXED-EFFECTS LINEAR MODEL")
    print(f"{'='*70}")

    core = df_agg[
        (df_agg["concept_overlap_tier"] == "all_3") &
        (df_agg["on_target"] == True) &
        (df_agg["ablation_mode"] == "all_tokens")
    ].copy()

    print(f"  N = {len(core)} observations, {core['ablated_concept'].nunique()} concept groups")

    report = {}
    if not HAS_SM:
        return report

    try:
        md = MixedLM.from_formula(
            "delta_accuracy ~ C(model) * alpha + C(cav_method) + C(layer_type)",
            groups="ablated_concept",
            data=core,
        )
        result = md.fit(reml=True)

        print("\n  Fixed Effects:")
        summary_df = pd.DataFrame({
            "Coef": result.params,
            "SE": result.bse,
            "z": result.tvalues,
            "p": result.pvalues,
        })
        print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"\n  AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")
        print(f"  Random intercept variance: {result.cov_re.iloc[0, 0]:.6f}")

        report = {
            "converged": True,
            "n_obs": len(core),
            "n_groups": int(core["ablated_concept"].nunique()),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "log_likelihood": float(result.llf),
            "random_intercept_var": float(result.cov_re.iloc[0, 0]),
            "fixed_effects": {
                name: {
                    "estimate": float(result.params[name]),
                    "std_error": float(result.bse[name]),
                    "z": float(result.tvalues[name]),
                    "p": float(result.pvalues[name]),
                    "sig": "***" if result.pvalues[name] < 0.001 else
                           "**" if result.pvalues[name] < 0.01 else
                           "*" if result.pvalues[name] < 0.05 else "ns"
                }
                for name in result.params.index
            },
        }
    except Exception as e:
        print(f"  Failed: {e}")
        report = {"converged": False, "error": str(e)}

    return report


# ===================================================================
# 5. SPECIFICITY ANALYSIS
# ===================================================================
def specificity_analysis(df_agg):
    """On-target vs off-target: is ablation selective?"""
    print(f"\n{'='*70}")
    print("5. SPECIFICITY ANALYSIS")
    print(f"{'='*70}")

    core = df_agg[
        (df_agg["concept_overlap_tier"] == "all_3") &
        (df_agg["ablation_mode"] == "all_tokens")
    ].copy()

    report = {}

    # Per-model specificity ratios
    print("\n  Specificity ratios:")
    for m in MODEL_ORDER:
        sub = core[core["model"] == m]
        on = sub[sub["on_target"] == True]["delta_accuracy"].mean()
        off = sub[sub["on_target"] == False]["delta_accuracy"].mean()
        ratio = on / off if off > 0 else float("inf")
        print(f"    {MODEL_LABELS[m]}: on={on:.4f}, off={off:.4f}, ratio={ratio:.2f}")
        report[m] = {"on_target_mean": float(on), "off_target_mean": float(off), "ratio": float(ratio)}

    # ANOVA: on_target x model interaction
    if HAS_SM:
        try:
            model = ols("delta_accuracy ~ C(model) * C(on_target)", data=core).fit()
            aov = anova_lm(model, typ=2)
            ss_total = aov["sum_sq"].sum()
            aov["eta_sq"] = aov["sum_sq"] / ss_total
            print("\n  ANOVA: model x on_target")
            print(aov[["sum_sq", "df", "F", "PR(>F)", "eta_sq"]].to_string(float_format=lambda x: f"{x:.4f}"))
            report["anova_interaction"] = {
                "F": float(aov.loc["C(model):C(on_target)", "F"]),
                "p": float(aov.loc["C(model):C(on_target)", "PR(>F)"]),
                "eta_sq": float(aov.loc["C(model):C(on_target)", "eta_sq"]),
            }
        except Exception as e:
            print(f"  ANOVA failed: {e}")

    return report


# ===================================================================
# 6. LLAVA MULTIMODAL ANALYSIS
# ===================================================================
def llava_multimodal(df_agg):
    """LLaVA-specific: all_tokens vs image_tokens_only."""
    print(f"\n{'='*70}")
    print("6. LLaVA MULTIMODAL ANALYSIS (all_tokens vs image_tokens_only)")
    print(f"{'='*70}")

    sub = df_agg[
        (df_agg["model"] == "llava_15_7b") &
        (df_agg["on_target"] == True)
    ].copy()

    if len(sub) < 5:
        print("  SKIP: not enough data")
        return {"skipped": True}

    report = {}

    # Descriptive
    desc = sub.groupby("ablation_mode")["delta_accuracy"].describe()
    print(desc.to_string(float_format=lambda x: f"{x:.4f}"))

    # By alpha
    pivot = sub.groupby(["ablation_mode", "alpha"])["delta_accuracy"].mean().unstack("alpha")
    print(f"\n  Mean delta by mode x alpha:")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    # Welch t-test
    all_tok = sub[sub["ablation_mode"] == "all_tokens"]["delta_accuracy"]
    img_only = sub[sub["ablation_mode"] == "image_tokens_only"]["delta_accuracy"]
    t, p = stats.ttest_ind(all_tok, img_only, equal_var=False)
    d = (all_tok.mean() - img_only.mean()) / np.sqrt((all_tok.std()**2 + img_only.std()**2) / 2)

    print(f"\n  Welch t-test: t={t:.4f}, p={p:.2e}, Cohen's d={d:.3f}")

    # Effect at each alpha
    print("\n  Per-alpha Welch t-tests:")
    per_alpha = {}
    for a in ALPHA_VALUES:
        g1 = sub[(sub["ablation_mode"] == "all_tokens") & (sub["alpha"] == a)]["delta_accuracy"]
        g2 = sub[(sub["ablation_mode"] == "image_tokens_only") & (sub["alpha"] == a)]["delta_accuracy"]
        if len(g1) > 1 and len(g2) > 1:
            t_a, p_a = stats.ttest_ind(g1, g2, equal_var=False)
            d_a = (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)
            sig = "***" if p_a < 0.001 else "**" if p_a < 0.01 else "*" if p_a < 0.05 else "ns"
            print(f"    alpha={a}: t={t_a:.3f}, p={p_a:.4f}, d={d_a:.3f} [{sig}]")
            per_alpha[str(a)] = {"t": float(t_a), "p": float(p_a), "d": float(d_a)}

    report = {
        "mean_all_tokens": float(all_tok.mean()),
        "mean_image_only": float(img_only.mean()),
        "welch_t": float(t),
        "p_value": float(p),
        "cohens_d": float(d),
        "per_alpha": per_alpha,
    }
    return report


# ===================================================================
# 7. ITEM-LEVEL LOGIT ANALYSIS
# ===================================================================
def item_level_analysis(df_item):
    """Analyze raw logits/scores at item level."""
    print(f"\n{'='*70}")
    print("7. ITEM-LEVEL LOGIT ANALYSIS")
    print(f"{'='*70}")

    # Only rows with actual scores
    scored = df_item[df_item["score_correct"].notna() & (df_item["score_correct"] != "")].copy()
    scored["score_correct"] = pd.to_numeric(scored["score_correct"], errors="coerce")
    scored["score_margin"] = pd.to_numeric(scored["score_margin"], errors="coerce")

    print(f"  {len(scored):,} items with logit scores")

    report = {}

    # Mean score margin by model (on-target, all_tokens)
    core = scored[
        (scored["concept_overlap_tier"] == "all_3") &
        (scored["on_target"] == True) &
        (scored["ablation_mode"] == "all_tokens")
    ]

    print("\n  Score margin (correct - chosen) by model x alpha:")
    print("  (More negative = model more confidently wrong after ablation)")
    pivot = core.groupby(["model", "alpha"])["score_margin"].mean().unstack("alpha")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
    report["score_margin_model_alpha"] = pivot.to_dict()

    # Proportion correct at item level
    core_correct = core.groupby(["model", "alpha"])["is_correct"].mean().unstack("alpha")
    print("\n  Item-level accuracy by model x alpha:")
    print(core_correct.to_string(float_format=lambda x: f"{x:.4f}"))
    report["item_accuracy_model_alpha"] = core_correct.to_dict()

    return report


# ===================================================================
# 8. PLOTS
# ===================================================================
def generate_plots(df_agg, df_item):
    """Generate all publication-quality plots."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping plots")
        return

    print(f"\n{'='*70}")
    print("8. GENERATING PLOTS")
    print(f"{'='*70}")

    core = df_agg[
        (df_agg["concept_overlap_tier"] == "all_3") &
        (df_agg["on_target"] == True) &
        (df_agg["ablation_mode"] == "all_tokens")
    ].copy()

    # --- Plot 1: Dose-Response Curves ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in MODEL_ORDER:
        ms = core[core["model"] == m]
        if ms.empty:
            continue
        means = ms.groupby("alpha")["delta_accuracy"].mean()
        sems = ms.groupby("alpha")["delta_accuracy"].sem()
        ax.errorbar(
            means.index, means.values, yerr=sems.values,
            marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
            label=MODEL_LABELS[m], capsize=4, linewidth=2.5, markersize=8,
        )
    ax.set_xlabel("Ablation Intensity (alpha)", fontsize=13)
    ax.set_ylabel("On-Target Delta Accuracy", fontsize=13)
    ax.set_title("Dose-Response: CAV Ablation Effect by Model Scale", fontsize=14)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_dose_response.png"), dpi=200)
    plt.close(fig)
    print("  Saved: 01_dose_response.png")

    # --- Plot 2: Specificity (on vs off target) ---
    spec_data = df_agg[
        (df_agg["concept_overlap_tier"] == "all_3") &
        (df_agg["ablation_mode"] == "all_tokens")
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODEL_ORDER))
    w = 0.35
    on_means = [spec_data[(spec_data["model"] == m) & (spec_data["on_target"] == True)]["delta_accuracy"].mean()
                for m in MODEL_ORDER]
    off_means = [spec_data[(spec_data["model"] == m) & (spec_data["on_target"] == False)]["delta_accuracy"].mean()
                 for m in MODEL_ORDER]
    ax.bar(x - w/2, on_means, w, label="On-target (ablated concept)", color="#d62728", alpha=0.85)
    ax.bar(x + w/2, off_means, w, label="Off-target (other concepts)", color="#7f7f7f", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=11)
    ax.set_ylabel("Mean Delta Accuracy", fontsize=13)
    ax.set_title("Ablation Specificity: On-Target vs Off-Target Effect", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_specificity.png"), dpi=200)
    plt.close(fig)
    print("  Saved: 02_specificity.png")

    # --- Plot 3: CAV Method Comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ["mean_diff", "svm"]
    method_colors = {"mean_diff": "#9467bd", "svm": "#e377c2"}
    for i, method in enumerate(methods):
        ms = core[core["cav_method"] == method]
        means = [ms[ms["model"] == m]["delta_accuracy"].mean() for m in MODEL_ORDER]
        ax.bar(x + (i - 0.5) * w, means, w, label=method.upper(), color=method_colors[method], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=11)
    ax.set_ylabel("Mean On-Target Delta", fontsize=13)
    ax.set_title("CAV Method: Mean-Diff vs SVM by Model", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_cav_method.png"), dpi=200)
    plt.close(fig)
    print("  Saved: 03_cav_method.png")

    # --- Plot 4: Layer Depth Heatmap ---
    core_single = core[core["layer_type"] != "all_layers"]
    pivot = core_single.groupby(["model", "layer_type"])["delta_accuracy"].mean().unstack("layer_type")
    layer_order = ["single_mid", "single_late", "single_deep"]
    pivot = pivot.reindex(columns=[c for c in layer_order if c in pivot.columns])
    pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])

    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(["Mid (~50%)", "Late (~75%)", "Deep (~83%)"], fontsize=11)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in pivot.index], fontsize=11)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=12, fontweight="bold")
        ax.set_title("On-Target Delta by Model x Layer Depth", fontsize=14)
        fig.colorbar(im, ax=ax, label="Delta Accuracy", shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "04_layer_heatmap.png"), dpi=200)
        plt.close(fig)
        print("  Saved: 04_layer_heatmap.png")

    # --- Plot 5: LLaVA ablation mode ---
    llava = df_agg[
        (df_agg["model"] == "llava_15_7b") &
        (df_agg["on_target"] == True)
    ]
    if not llava.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode, color, label in [
            ("all_tokens", "#2ca02c", "All tokens"),
            ("image_tokens_only", "#17becf", "Image tokens only"),
        ]:
            ms = llava[llava["ablation_mode"] == mode]
            if ms.empty:
                continue
            means = ms.groupby("alpha")["delta_accuracy"].mean()
            sems = ms.groupby("alpha")["delta_accuracy"].sem()
            ax.errorbar(
                means.index, means.values, yerr=sems.values,
                marker="^", color=color, label=label, capsize=4, linewidth=2.5, markersize=8,
            )
        ax.set_xlabel("Alpha", fontsize=13)
        ax.set_ylabel("On-Target Delta Accuracy", fontsize=13)
        ax.set_title("LLaVA-1.5-7B: All Tokens vs Image Tokens Only", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "05_llava_ablation_mode.png"), dpi=200)
        plt.close(fig)
        print("  Saved: 05_llava_ablation_mode.png")

    # --- Plot 6: Per-concept delta across models ---
    concepts_all3 = sorted(ALL_3_CONCEPTS)
    core_concept = core[core["ablated_concept"].isin(concepts_all3)]
    pivot_c = core_concept.groupby(["model", "ablated_concept"])["delta_accuracy"].mean().unstack("ablated_concept")
    pivot_c = pivot_c.reindex([m for m in MODEL_ORDER if m in pivot_c.index])

    if not pivot_c.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        n_concepts = len(pivot_c.columns)
        n_models = len(pivot_c.index)
        bar_w = 0.8 / n_models
        for i, m in enumerate(pivot_c.index):
            positions = np.arange(n_concepts) + (i - n_models/2 + 0.5) * bar_w
            vals = pivot_c.loc[m].values
            ax.bar(positions, vals, bar_w, label=MODEL_LABELS[m], color=MODEL_COLORS[m], alpha=0.85)
        ax.set_xticks(np.arange(n_concepts))
        ax.set_xticklabels([c.title() for c in pivot_c.columns], fontsize=11)
        ax.set_ylabel("Mean On-Target Delta", fontsize=13)
        ax.set_title("Per-Concept Ablation Effect by Model", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "06_per_concept.png"), dpi=200)
        plt.close(fig)
        print("  Saved: 06_per_concept.png")



# ===================================================================
# 9. LATEX TABLES
# ===================================================================
def generate_latex(df_agg):
    """Generate LaTeX tables for thesis."""
    print(f"\n{'='*70}")
    print("9. GENERATING LATEX TABLES")
    print(f"{'='*70}")

    core = df_agg[
        (df_agg["concept_overlap_tier"] == "all_3") &
        (df_agg["on_target"] == True) &
        (df_agg["ablation_mode"] == "all_tokens")
    ]

    tex_path = os.path.join(OUT_DIR, "latex_tables.tex")
    lines = []

    # Table 1: Model x Alpha
    lines.append("% Table 1: Mean On-Target Delta Accuracy by Model x Alpha")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Mean on-target $\\Delta$ accuracy by model and ablation intensity ($\\alpha$).}")
    lines.append("\\label{tab:model_alpha}")
    lines.append("\\begin{tabular}{l" + "r" * 5 + "r}")
    lines.append("\\toprule")
    lines.append("Model & $\\alpha=3$ & $\\alpha=6$ & $\\alpha=10$ & $\\alpha=15$ & $\\alpha=20$ & Mean \\\\")
    lines.append("\\midrule")

    for m in MODEL_ORDER:
        ms = core[core["model"] == m]
        if ms.empty:
            continue
        vals = [ms[ms["alpha"] == a]["delta_accuracy"].mean() for a in ALPHA_VALUES]
        overall = ms["delta_accuracy"].mean()
        label = MODEL_LABELS[m]
        row = f"{label} & " + " & ".join(f"{v:.3f}" for v in vals) + f" & {overall:.3f} \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 2: Specificity
    spec_data = df_agg[
        (df_agg["concept_overlap_tier"] == "all_3") &
        (df_agg["ablation_mode"] == "all_tokens")
    ]
    lines.append("% Table 2: Ablation Specificity")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation specificity: on-target vs.~off-target mean $\\Delta$ accuracy.}")
    lines.append("\\label{tab:specificity}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Model & On-target & Off-target & Ratio \\\\")
    lines.append("\\midrule")

    for m in MODEL_ORDER:
        ms = spec_data[spec_data["model"] == m]
        if ms.empty:
            continue
        on = ms[ms["on_target"] == True]["delta_accuracy"].mean()
        off = ms[ms["on_target"] == False]["delta_accuracy"].mean()
        ratio = on / off if off > 0 else float("inf")
        lines.append(f"{MODEL_LABELS[m]} & {on:.3f} & {off:.3f} & {ratio:.2f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Saved: {tex_path}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 70)
    print("  CROSS-MODEL CCT EXPERIMENT ANALYSIS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    df_item, df_agg = load_data()

    report = OrderedDict()
    report["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "item_csv": ITEM_CSV,
        "agg_csv": AGG_CSV,
        "n_items": len(df_item),
        "n_conditions": len(df_agg),
    }

    report["1_descriptive"] = descriptive_stats(df_agg)
    report["2_assumptions"] = assumption_checks(df_agg)
    report["3_main_anova"] = main_anova(df_agg)
    report["4_mixed_effects"] = mixed_effects(df_agg)
    report["5_specificity"] = specificity_analysis(df_agg)
    report["6_llava_multimodal"] = llava_multimodal(df_agg)
    report["7_item_level"] = item_level_analysis(df_item)

    generate_plots(df_agg, df_item)
    generate_latex(df_agg)

    # Save report
    report_path = os.path.join(OUT_DIR, "experiment_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Report: {report_path}")
    print(f"  Figures: {FIG_DIR}/")
    print(f"  LaTeX: {os.path.join(OUT_DIR, 'latex_tables.tex')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
