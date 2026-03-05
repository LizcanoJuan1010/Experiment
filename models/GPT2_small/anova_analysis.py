"""
ANOVA Analysis for GPT-2 Semantic Aphasia Experiment.

Reads results_factorial.csv and results_multilayer/results_factorial.csv
and produces:
  - Two-way ANOVA tables (aligned with thesis hypotheses H1A, H1B, H1C)
  - Assumption checks (Shapiro-Wilk, Levene)
  - Post-hoc Tukey HSD comparisons
  - Effect sizes (eta-squared, partial eta-squared, omega-squared)
  - Benjamini-Hochberg FDR correction
  - Interaction plots
  - LaTeX tables for thesis

Usage:
    python anova_analysis.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

try:
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.multitest import multipletests
except ImportError:
    raise ImportError(
        "statsmodels is required. Install with: pip install statsmodels"
    )

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ===================================================================
# Paths
# ===================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
MULTILAYER_DIR = os.path.join(SCRIPT_DIR, "results_multilayer")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

FACTORIAL_CSV = os.path.join(RESULTS_DIR, "results_factorial.csv")
MULTILAYER_CSV = os.path.join(MULTILAYER_DIR, "results_factorial.csv")


# ===================================================================
# 1. Load & Prepare
# ===================================================================
def load_and_prepare(csv_path, exclude_baseline=True):
    """Load CSV and prepare for ANOVA."""
    df = pd.read_csv(csv_path)

    if exclude_baseline and "alpha" in df.columns:
        df = df[df["alpha"] > 0].copy()

    # Exclude amplification rows if present
    if "direction" in df.columns:
        df = df[df["direction"] == "ablation"].copy()

    # Ensure delta columns are positive (baseline - ablated)
    if "delta_r1" in df.columns:
        # delta_r1 should be positive when ablation works (baseline > ablated)
        # In the CSV it's already computed as baseline - ablated
        pass

    return df


# ===================================================================
# 2. Assumption Checks
# ===================================================================
def check_assumptions(data, dv, factor):
    """
    Check ANOVA assumptions per group.

    Returns dict with:
      - shapiro: per-group Shapiro-Wilk test (normality)
      - levene: Levene test (homogeneity of variances)
      - recommendation: 'parametric' or 'nonparametric'
    """
    groups = data.groupby(factor)[dv].apply(list).to_dict()
    result = {"shapiro": {}, "levene": {}, "recommendation": "parametric"}

    # Shapiro-Wilk per group
    normality_fails = 0
    for name, values in groups.items():
        if len(values) < 3:
            result["shapiro"][str(name)] = {
                "statistic": None, "p_value": None,
                "normal": None, "note": f"n={len(values)} too small"
            }
            continue
        try:
            stat, p = stats.shapiro(values)
            is_normal = p > 0.05
            if not is_normal:
                normality_fails += 1
            result["shapiro"][str(name)] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p), 4),
                "normal": is_normal,
            }
        except Exception as e:
            result["shapiro"][str(name)] = {
                "statistic": None, "p_value": None,
                "normal": None, "note": str(e)
            }

    # Levene test (need at least 2 groups with >= 2 observations)
    valid_groups = [v for v in groups.values() if len(v) >= 2]
    if len(valid_groups) >= 2:
        try:
            stat, p = stats.levene(*valid_groups)
            result["levene"] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p), 4),
                "homogeneous": p > 0.05,
            }
        except Exception as e:
            result["levene"] = {"error": str(e)}
    else:
        result["levene"] = {"note": "Not enough groups with n>=2"}

    # Recommendation
    if normality_fails > len(groups) / 2:
        result["recommendation"] = "nonparametric"

    return result


# ===================================================================
# 3. Two-Way ANOVA
# ===================================================================
def run_twoway_anova(data, dv, factor1, factor2, concept_filter=None):
    """
    Run two-way ANOVA: dv ~ C(factor1) * C(factor2).

    If concept_filter is given, filter data to that concept first.
    Returns ANOVA table as dict + metadata.
    """
    df = data.copy()
    if concept_filter:
        df = df[df["concept"] == concept_filter]

    if len(df) < 4:
        return {"error": f"Not enough data (n={len(df)}) for concept={concept_filter}"}

    # Check if factors have multiple levels
    f1_levels = df[factor1].nunique()
    f2_levels = df[factor2].nunique()
    if f1_levels < 2 or f2_levels < 2:
        return {"error": f"Need >= 2 levels per factor. {factor1}={f1_levels}, {factor2}={f2_levels}"}

    formula = f"{dv} ~ C({factor1}) * C({factor2})"

    try:
        model = ols(formula, data=df).fit()
        table = anova_lm(model, typ=2)

        # Convert to serializable dict
        result = {
            "concept": concept_filter,
            "formula": formula,
            "n_observations": len(df),
            "factors": {factor1: sorted(df[factor1].unique().tolist()),
                        factor2: sorted(df[factor2].unique().tolist())},
            "anova_table": {},
            "r_squared": round(float(model.rsquared), 4),
            "r_squared_adj": round(float(model.rsquared_adj), 4),
        }

        ss_total = table["sum_sq"].sum()
        for source in table.index:
            row = table.loc[source]
            eta_sq = float(row["sum_sq"]) / ss_total if ss_total > 0 else 0
            result["anova_table"][source] = {
                "SS": round(float(row["sum_sq"]), 6),
                "df": int(row["df"]) if not np.isnan(row["df"]) else None,
                "F": round(float(row["F"]), 4) if not np.isnan(row["F"]) else None,
                "p": round(float(row["PR(>F)"]), 6) if not np.isnan(row["PR(>F)"]) else None,
                "eta_squared": round(eta_sq, 4),
                "significant": bool(row["PR(>F)"] < 0.05) if not np.isnan(row["PR(>F)"]) else False,
            }

        return result

    except Exception as e:
        return {"error": str(e), "concept": concept_filter}


def run_oneway_anova(data, dv, factor, concept_filter=None):
    """
    Run one-way ANOVA: dv ~ C(factor).
    Also runs Kruskal-Wallis as nonparametric alternative.
    """
    df = data.copy()
    if concept_filter:
        df = df[df["concept"] == concept_filter]

    groups = [grp[dv].values for _, grp in df.groupby(factor)]
    valid_groups = [g for g in groups if len(g) >= 1]

    if len(valid_groups) < 2:
        return {"error": "Need >= 2 groups"}

    result = {"concept": concept_filter, "factor": factor, "dv": dv}

    # Parametric
    try:
        f_stat, p_val = stats.f_oneway(*valid_groups)
        result["f_oneway"] = {
            "F": round(float(f_stat), 4) if not np.isnan(f_stat) else None,
            "p": round(float(p_val), 6) if not np.isnan(p_val) else None,
            "significant": bool(p_val < 0.05) if not np.isnan(p_val) else False,
        }
    except Exception as e:
        result["f_oneway"] = {"error": str(e)}

    # Non-parametric alternative
    try:
        h_stat, p_val = stats.kruskal(*valid_groups)
        result["kruskal_wallis"] = {
            "H": round(float(h_stat), 4),
            "p": round(float(p_val), 6),
            "significant": bool(p_val < 0.05),
        }
    except Exception as e:
        result["kruskal_wallis"] = {"error": str(e)}

    return result


# ===================================================================
# 4. Post-hoc Tests
# ===================================================================
def posthoc_tukey(data, dv, factor, concept_filter=None):
    """Tukey HSD post-hoc pairwise comparisons."""
    df = data.copy()
    if concept_filter:
        df = df[df["concept"] == concept_filter]

    if df[factor].nunique() < 2:
        return {"error": "Need >= 2 levels"}

    try:
        tukey = pairwise_tukeyhsd(df[dv], df[factor], alpha=0.05)
        rows = []
        for i in range(len(tukey.summary().data) - 1):
            row = tukey.summary().data[i + 1]
            rows.append({
                "group1": str(row[0]),
                "group2": str(row[1]),
                "meandiff": round(float(row[2]), 6),
                "p_adj": round(float(row[3]), 6),
                "lower": round(float(row[4]), 6),
                "upper": round(float(row[5]), 6),
                "reject": bool(row[6]),
            })
        return {"concept": concept_filter, "factor": factor, "comparisons": rows}
    except Exception as e:
        return {"error": str(e)}


# ===================================================================
# 5. Effect Sizes
# ===================================================================
def compute_effect_sizes(anova_result):
    """Compute eta-squared, partial eta-squared, omega-squared from ANOVA table."""
    if "error" in anova_result:
        return anova_result

    table = anova_result.get("anova_table", {})
    ss_residual = table.get("Residual", {}).get("SS", 0)
    n = anova_result.get("n_observations", 0)

    effects = {}
    for source, vals in table.items():
        if source == "Residual":
            continue
        ss = vals.get("SS", 0)
        df = vals.get("df", 0)
        f = vals.get("F", 0)

        # Eta-squared (already computed)
        eta_sq = vals.get("eta_squared", 0)

        # Partial eta-squared = SS_effect / (SS_effect + SS_residual)
        partial_eta_sq = ss / (ss + ss_residual) if (ss + ss_residual) > 0 else 0

        # Omega-squared = (SS_effect - df * MS_residual) / (SS_total + MS_residual)
        ms_residual = ss_residual / table.get("Residual", {}).get("df", 1) if table.get("Residual", {}).get("df", 0) > 0 else 0
        ss_total = sum(v.get("SS", 0) for v in table.values())
        omega_sq = (ss - df * ms_residual) / (ss_total + ms_residual) if (ss_total + ms_residual) > 0 else 0

        effects[source] = {
            "eta_squared": round(eta_sq, 4),
            "partial_eta_squared": round(partial_eta_sq, 4),
            "omega_squared": round(max(omega_sq, 0), 4),
            "interpretation": (
                "large" if partial_eta_sq >= 0.14 else
                "medium" if partial_eta_sq >= 0.06 else
                "small" if partial_eta_sq >= 0.01 else
                "negligible"
            ),
        }

    return effects


# ===================================================================
# 6. Multiple Comparisons Correction
# ===================================================================
def correct_pvalues(p_values, method="fdr_bh"):
    """Apply Benjamini-Hochberg FDR or Bonferroni correction."""
    valid = [(i, p) for i, p in enumerate(p_values) if p is not None and not np.isnan(p)]
    if not valid:
        return p_values

    indices, pvals = zip(*valid)
    reject, corrected, _, _ = multipletests(pvals, method=method)

    result = list(p_values)
    for idx, corr_p, rej in zip(indices, corrected, reject):
        result[idx] = {
            "original": round(float(p_values[idx]), 6),
            "corrected": round(float(corr_p), 6),
            "reject_h0": bool(rej),
        }
    return result


# ===================================================================
# 7. Interaction Plots
# ===================================================================
def interaction_plot(data, dv, factor1, factor2, concept, save_path=None):
    """Generate interaction plot: mean DV by factor1 levels, separate lines for factor2."""
    if not HAS_MPL:
        return

    df = data[data["concept"] == concept].copy() if concept else data.copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    for level2 in sorted(df[factor2].unique()):
        subset = df[df[factor2] == level2]
        means = subset.groupby(factor1)[dv].mean()
        sems = subset.groupby(factor1)[dv].sem()
        x = range(len(means))
        ax.errorbar(x, means.values, yerr=sems.values,
                     marker="o", capsize=4, label=f"{factor2}={level2}")

    ax.set_xticks(range(len(df[factor1].unique())))
    ax.set_xticklabels(sorted(df[factor1].unique()), rotation=0)
    ax.set_xlabel(factor1)
    ax.set_ylabel(dv)
    title = f"Interaction: {factor1} x {factor2}"
    if concept:
        title += f" ({concept})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ===================================================================
# 8. LaTeX Table
# ===================================================================
def generate_latex_table(anova_results, caption="", label=""):
    """Generate LaTeX table from ANOVA results."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{lrrrrrc}")
    lines.append(r"\hline")
    lines.append(r"Source & SS & df & MS & F & p & Sig. \\")
    lines.append(r"\hline")

    for result in anova_results:
        concept = result.get("concept", "all")
        table = result.get("anova_table", {})
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{concept}}}}} \\")

        for source, vals in table.items():
            ss = vals.get("SS", 0)
            df = vals.get("df", "")
            f_val = vals.get("F", "")
            p_val = vals.get("p", "")
            sig = ""
            if isinstance(p_val, (int, float)) and p_val < 0.001:
                sig = "***"
            elif isinstance(p_val, (int, float)) and p_val < 0.01:
                sig = "**"
            elif isinstance(p_val, (int, float)) and p_val < 0.05:
                sig = "*"

            ms = ss / df if isinstance(df, (int, float)) and df > 0 else ""

            f_str = f"{f_val:.3f}" if isinstance(f_val, float) else str(f_val)
            p_str = f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val)
            ms_str = f"{ms:.6f}" if isinstance(ms, float) else str(ms)
            ss_str = f"{ss:.6f}" if isinstance(ss, float) else str(ss)

            clean_source = source.replace("C(", "").replace(")", "").replace(":", r" $\times$ ")
            lines.append(
                rf"  {clean_source} & {ss_str} & {df} & {ms_str} & {f_str} & {p_str} & {sig} \\"
            )
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ===================================================================
# 9. Main Orchestrator
# ===================================================================
def main():
    print("=" * 60)
    print("ANOVA Analysis — Semantic Aphasia Experiment")
    print("=" * 60)

    all_results = {}

    # -----------------------------------------------------------------
    # A. Single-Layer Factorial ANOVA
    # -----------------------------------------------------------------
    if os.path.exists(FACTORIAL_CSV):
        print(f"\n--- Loading factorial data: {FACTORIAL_CSV}")
        df_fact = load_and_prepare(FACTORIAL_CSV)
        print(f"  Rows after filtering: {len(df_fact)}")

        # Filter to SVM method (statistically validated, unlike mean_diff)
        df_svm = df_fact[df_fact["cav_method"] == "svm"].copy()
        print(f"  SVM-only rows: {len(df_svm)}")

        # ANOVA 1: technique x layer on delta_r1 (H1A + H1B)
        # Use alpha=3.5 (strongest effect without distortion)
        print("\n[ANOVA 1] technique x layer -> delta_r1 (SVM, alpha=3.5)")
        df_a1 = df_svm[df_svm["alpha"] == 3.5].copy()
        anova1_results = []
        for concept in ["time", "place"]:  # Exclude tools (invalid baseline)
            print(f"  Concept: {concept}")
            assumptions = check_assumptions(df_a1[df_a1["concept"] == concept],
                                            "delta_r1", "technique")
            print(f"    Assumptions: {assumptions['recommendation']}")

            result = run_twoway_anova(df_a1, "delta_r1", "technique", "layer",
                                      concept_filter=concept)
            if "error" not in result:
                effects = compute_effect_sizes(result)
                result["effect_sizes"] = effects
                result["assumptions"] = assumptions
                for src, vals in result["anova_table"].items():
                    sig = "***" if (vals.get("p") or 1) < 0.001 else \
                          "**" if (vals.get("p") or 1) < 0.01 else \
                          "*" if (vals.get("p") or 1) < 0.05 else "ns"
                    print(f"    {src}: F={vals.get('F')}, p={vals.get('p')}, "
                          f"eta2={vals.get('eta_squared')} [{sig}]")
            else:
                print(f"    Error: {result['error']}")
            anova1_results.append(result)

        all_results["anova1_technique_x_layer"] = anova1_results

        # Interaction plots for ANOVA 1
        if HAS_MPL:
            for concept in ["time", "place"]:
                interaction_plot(
                    df_a1, "delta_r1", "technique", "layer", concept,
                    save_path=os.path.join(FIGURES_DIR,
                                           f"interaction_technique_layer_{concept}.png")
                )

        # ANOVA 2: technique x alpha on delta_r1 (H1A x H1C)
        # Use SVM, layer 6
        print("\n[ANOVA 2] technique x alpha -> delta_r1 (SVM, L6)")
        df_a2 = df_svm[df_svm["layer"] == 6].copy()
        # Convert alpha to categorical for ANOVA
        df_a2["alpha_cat"] = df_a2["alpha"].astype(str)
        anova2_results = []
        for concept in ["time", "place"]:
            print(f"  Concept: {concept}")
            result = run_twoway_anova(df_a2, "delta_r1", "technique", "alpha_cat",
                                      concept_filter=concept)
            if "error" not in result:
                effects = compute_effect_sizes(result)
                result["effect_sizes"] = effects
                for src, vals in result["anova_table"].items():
                    sig = "***" if (vals.get("p") or 1) < 0.001 else \
                          "**" if (vals.get("p") or 1) < 0.01 else \
                          "*" if (vals.get("p") or 1) < 0.05 else "ns"
                    print(f"    {src}: F={vals.get('F')}, p={vals.get('p')} [{sig}]")
            else:
                print(f"    Error: {result['error']}")
            anova2_results.append(result)

        all_results["anova2_technique_x_alpha"] = anova2_results

        if HAS_MPL:
            for concept in ["time", "place"]:
                interaction_plot(
                    df_a2, "delta_r1", "alpha_cat", "technique", concept,
                    save_path=os.path.join(FIGURES_DIR,
                                           f"interaction_alpha_technique_{concept}.png")
                )

        # ANOVA 3: One-way on alpha (Dose-Response H1C)
        print("\n[ANOVA 3] One-way alpha -> delta_r1 (SVM, L6, subtraction)")
        df_a3 = df_svm[(df_svm["layer"] == 6) &
                        (df_svm["technique"] == "subtraction")].copy()
        df_a3["alpha_cat"] = df_a3["alpha"].astype(str)
        anova3_results = []
        for concept in ["time", "place"]:
            print(f"  Concept: {concept}")
            result = run_oneway_anova(df_a3, "delta_r1", "alpha_cat",
                                      concept_filter=concept)
            print(f"    F-oneway: {result.get('f_oneway', {})}")
            print(f"    Kruskal: {result.get('kruskal_wallis', {})}")
            anova3_results.append(result)

            # Spearman correlation for dose-response
            subset = df_a3[df_a3["concept"] == concept]
            if len(subset) >= 3:
                rho, p = stats.spearmanr(subset["alpha"], subset["delta_r1"])
                print(f"    Spearman rho={rho:.4f}, p={p:.4f}")
                result["spearman"] = {"rho": round(float(rho), 4),
                                      "p": round(float(p), 6)}

        all_results["anova3_dose_response"] = anova3_results

    else:
        print(f"\n  WARNING: {FACTORIAL_CSV} not found. Skipping factorial ANOVA.")

    # -----------------------------------------------------------------
    # B. Multilayer ANOVA
    # -----------------------------------------------------------------
    if os.path.exists(MULTILAYER_CSV):
        print(f"\n--- Loading multilayer data: {MULTILAYER_CSV}")
        df_ml = load_and_prepare(MULTILAYER_CSV)
        print(f"  Rows after filtering: {len(df_ml)}")

        # Filter to centered only
        if "centering" in df_ml.columns:
            df_ml = df_ml[df_ml["centering"] == "centered"].copy()
            print(f"  Centered-only rows: {len(df_ml)}")

        # ANOVA 4: mode x alpha on delta_r1 (Extension of H1B)
        print("\n[ANOVA 4] mode x alpha -> delta_r1 (multilayer, centered)")
        df_ml["alpha_cat"] = df_ml["alpha"].astype(str)
        anova4_results = []
        for concept in ["time", "place"]:
            print(f"  Concept: {concept}")
            assumptions = check_assumptions(
                df_ml[df_ml["concept"] == concept], "delta_r1", "mode"
            )
            print(f"    Assumptions: {assumptions['recommendation']}")

            result = run_twoway_anova(df_ml, "delta_r1", "mode", "alpha_cat",
                                      concept_filter=concept)
            if "error" not in result:
                effects = compute_effect_sizes(result)
                result["effect_sizes"] = effects
                result["assumptions"] = assumptions
                for src, vals in result["anova_table"].items():
                    sig = "***" if (vals.get("p") or 1) < 0.001 else \
                          "**" if (vals.get("p") or 1) < 0.01 else \
                          "*" if (vals.get("p") or 1) < 0.05 else "ns"
                    print(f"    {src}: F={vals.get('F')}, p={vals.get('p')}, "
                          f"eta2={vals.get('eta_squared')} [{sig}]")
            else:
                print(f"    Error: {result['error']}")
            anova4_results.append(result)

            # Post-hoc Tukey for mode
            if df_ml[df_ml["concept"] == concept]["mode"].nunique() >= 3:
                tukey = posthoc_tukey(df_ml, "delta_r1", "mode",
                                      concept_filter=concept)
                result["posthoc_mode"] = tukey
                if "comparisons" in tukey:
                    print(f"    Post-hoc (mode):")
                    for comp in tukey["comparisons"]:
                        sig_str = "SIG" if comp["reject"] else "ns"
                        print(f"      {comp['group1']} vs {comp['group2']}: "
                              f"diff={comp['meandiff']:.4f}, p={comp['p_adj']:.4f} [{sig_str}]")

        all_results["anova4_mode_x_alpha"] = anova4_results

        if HAS_MPL:
            for concept in ["time", "place"]:
                interaction_plot(
                    df_ml, "delta_r1", "mode", "alpha_cat", concept,
                    save_path=os.path.join(FIGURES_DIR,
                                           f"interaction_mode_alpha_{concept}.png")
                )

    else:
        print(f"\n  WARNING: {MULTILAYER_CSV} not found. Skipping multilayer ANOVA.")

    # -----------------------------------------------------------------
    # C. Multiple Comparisons Correction
    # -----------------------------------------------------------------
    print("\n--- Multiple Comparisons Correction (FDR Benjamini-Hochberg)")
    all_pvalues = []
    all_labels = []
    for key, results_list in all_results.items():
        for res in results_list:
            if isinstance(res, dict) and "anova_table" in res:
                for src, vals in res["anova_table"].items():
                    if src != "Residual" and vals.get("p") is not None:
                        all_pvalues.append(vals["p"])
                        all_labels.append(f"{key}/{res.get('concept','?')}/{src}")

    if all_pvalues:
        corrected = correct_pvalues(all_pvalues)
        print(f"  Total tests: {len(all_pvalues)}")
        fdr_results = []
        for label, orig_p, corr in zip(all_labels, all_pvalues, corrected):
            if isinstance(corr, dict):
                fdr_results.append({
                    "test": label,
                    "p_original": corr["original"],
                    "p_corrected": corr["corrected"],
                    "reject_h0": corr["reject_h0"],
                })
                sig = "SIG" if corr["reject_h0"] else "ns"
                print(f"  {label}: p={corr['original']:.4f} -> p_fdr={corr['corrected']:.4f} [{sig}]")
        all_results["fdr_correction"] = fdr_results

    # -----------------------------------------------------------------
    # D. Generate LaTeX Table
    # -----------------------------------------------------------------
    print("\n--- Generating LaTeX tables")

    # Table for factorial ANOVA
    if "anova1_technique_x_layer" in all_results:
        latex = generate_latex_table(
            all_results["anova1_technique_x_layer"],
            caption="Two-way ANOVA: Technique $\\times$ Layer on $\\Delta R1$ (SVM, $\\alpha=3.5$)",
            label="tab:anova_factorial"
        )
        tex_path = os.path.join(RESULTS_DIR, "anova_factorial_table.tex")
        with open(tex_path, "w") as f:
            f.write(latex)
        print(f"  Saved: {tex_path}")

    # Table for multilayer ANOVA
    if "anova4_mode_x_alpha" in all_results:
        latex = generate_latex_table(
            all_results["anova4_mode_x_alpha"],
            caption="Two-way ANOVA: Mode $\\times$ Alpha on $\\Delta R1$ (Multilayer, Centered)",
            label="tab:anova_multilayer"
        )
        tex_path = os.path.join(RESULTS_DIR, "anova_multilayer_table.tex")
        with open(tex_path, "w") as f:
            f.write(latex)
        print(f"  Saved: {tex_path}")

    # -----------------------------------------------------------------
    # E. Save Full Results
    # -----------------------------------------------------------------
    output_path = os.path.join(RESULTS_DIR, "anova_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved: {output_path}")

    # -----------------------------------------------------------------
    # F. Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    sig_count = 0
    total_count = 0
    for key, results_list in all_results.items():
        if key == "fdr_correction":
            continue
        for res in results_list:
            if isinstance(res, dict) and "anova_table" in res:
                concept = res.get("concept", "?")
                for src, vals in res["anova_table"].items():
                    if src != "Residual" and vals.get("p") is not None:
                        total_count += 1
                        if vals["significant"]:
                            sig_count += 1
                            print(f"  SIGNIFICANT: {key}/{concept}/{src} "
                                  f"(F={vals['F']}, p={vals['p']:.4f})")

    print(f"\n  Significant effects: {sig_count}/{total_count}")
    print("  Done.")


if __name__ == "__main__":
    main()
