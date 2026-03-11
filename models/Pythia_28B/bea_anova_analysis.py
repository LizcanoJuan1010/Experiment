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
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures", "bea_tests")
os.makedirs(FIGURES_DIR, exist_ok=True)

JSON_FILES = {
    "synonym": os.path.join(RESULTS_DIR, "bea_synonym_test_results.json"),
    "naming": os.path.join(RESULTS_DIR, "bea_naming_test_results.json"),
    "oddoneout": os.path.join(RESULTS_DIR, "bea_oddoneout_test_results.json")
}

# Pythia 2.8B: canonical middle layer is L16 (proportional to GPT-2 L6)
CANONICAL_LAYER = 16


# ===================================================================
# 1. Load & Prepare
# ===================================================================
def load_json_to_df(json_path, test_type):
    """Load JSON result list into a DataFrame."""
    if not os.path.exists(json_path):
        return pd.DataFrame()

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        return pd.DataFrame()

    # Flatten the list of dicts to a DataFrame
    # Ignore per_item for statistical modeling to save RAM
    flat_data = []
    for r in results:
        row = {k: v for k, v in r.items() if k != 'per_item'}
        # Clean layer name (e.g. from 'L16' to 16)
        if 'layer' in row and isinstance(row['layer'], str):
            val = row['layer'].replace('L', '')
            if val == 'all_layers':
                row['layer'] = -1
            else:
                try:
                    row['layer'] = int(val)
                except ValueError:
                    row['layer'] = val
        row['test_type'] = test_type
        flat_data.append(row)

    return pd.DataFrame(flat_data)


# ===================================================================
# 2. Assumption Checks
# ===================================================================
def check_assumptions(data, dv, factor):
    """Check ANOVA assumptions per group."""
    groups = data.groupby(factor)[dv].apply(list).to_dict()
    result = {"shapiro": {}, "levene": {}, "recommendation": "parametric"}

    normality_fails = 0
    for name, values in groups.items():
        if len(values) < 3:
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
        except Exception:
            pass

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

    # If >50% fails, recommend nonparametric
    if normality_fails > len(groups) / 2:
        result["recommendation"] = "nonparametric"

    return result


# ===================================================================
# 3. Two-Way ANOVA
# ===================================================================
def run_twoway_anova(data, dv, factor1, factor2, filter_desc=None):
    """Run two-way ANOVA: dv ~ C(factor1) * C(factor2)."""
    df = data.copy()

    f1_levels = df[factor1].nunique()
    f2_levels = df[factor2].nunique()
    if f1_levels < 2 or f2_levels < 2:
        return {"error": f"Need >= 2 levels per factor. {factor1}={f1_levels}, {factor2}={f2_levels}"}

    formula = f"{dv} ~ C({factor1}) * C({factor2})"

    try:
        model = ols(formula, data=df).fit()
        table = anova_lm(model, typ=2)

        result = {
            "filter_desc": filter_desc,
            "formula": formula,
            "n_observations": len(df),
            "anova_table": {},
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
        return {"error": str(e), "filter_desc": filter_desc}


def run_oneway_anova(data, dv, factor, filter_desc=None):
    """Run one-way ANOVA: dv ~ C(factor). Includes Kruskal-Wallis."""
    df = data.copy()

    groups = [grp[dv].values for _, grp in df.groupby(factor)]
    valid_groups = [g for g in groups if len(g) >= 1]

    if len(valid_groups) < 2:
        return {"error": "Need >= 2 groups"}

    result = {"filter_desc": filter_desc, "factor": factor, "dv": dv}

    # Parametric
    try:
        f_stat, p_val = stats.f_oneway(*valid_groups)
        result["f_oneway"] = {
            "F": round(float(f_stat), 4) if not np.isnan(f_stat) else None,
            "p": round(float(p_val), 6) if not np.isnan(p_val) else None,
            "significant": bool(p_val < 0.05) if not np.isnan(p_val) else False,
        }
    except Exception:
        pass

    # Non-parametric alternative
    try:
        h_stat, p_val = stats.kruskal(*valid_groups)
        result["kruskal_wallis"] = {
            "H": round(float(h_stat), 4),
            "p": round(float(p_val), 6),
            "significant": bool(p_val < 0.05),
        }
    except Exception:
        pass

    return result


# ===================================================================
# 4. Interaction Plots
# ===================================================================
def interaction_plot(data, dv, factor1, factor2, title, save_path=None):
    """Generate interaction plot: mean DV by factor1 levels, separate lines for factor2."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for level2 in sorted(data[factor2].unique()):
        subset = data[data[factor2] == level2]
        means = subset.groupby(factor1)[dv].mean()
        sems = subset.groupby(factor1)[dv].sem()
        x = range(len(means))
        ax.errorbar(x, means.values, yerr=sems.values,
                     marker="o", capsize=4, label=f"{factor2}={level2}")

    ax.set_xticks(range(len(data[factor1].unique())))
    ax.set_xticklabels(sorted(data[factor1].unique()), rotation=0)
    ax.set_xlabel(factor1)
    ax.set_ylabel(dv)
    ax.set_title(title)
    ax.legend(title=factor2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ===================================================================
# 5. LaTeX Table
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
        filter_desc = result.get("filter_desc", "All")
        table = result.get("anova_table", {})

        # Format test name properly
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{filter_desc}}}}} \\")

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
# Main Orchestrator
# ===================================================================
def main():
    print("=" * 60)
    print("ANOVA Analysis — BEA Test Suite (Pythia 2.8B)")
    print("=" * 60)

    # 1. Load Data
    dfs = {}
    for test_key, path in JSON_FILES.items():
        if os.path.exists(path):
            print(f"Loading {test_key} data from {os.path.basename(path)}")
            dfs[test_key] = load_json_to_df(path, test_key)
        else:
            print(f"Warning: {path} not found.")

    if not dfs:
        print("No data loaded. Exiting.")
        return

    all_results = {"technique_layer": [], "technique_alpha": []}

    # Helper function to generate stats for a specific test DataFrame
    def process_test(df, dv, test_name):
        print(f"\n--- Processing {test_name.upper()} (DV: {dv}) ---")

        # Only analyze on-target (concept ablation)
        df_target = df[df["on_target"] == True].copy()
        print(f"  Rows on-target: {len(df_target)}")

        # A. ANOVA 1: method x layer (at alpha=3.0)
        print("  [ANOVA 1] Method x Layer (Alpha=3.0)")
        df_a1 = df_target[df_target["alpha"] == 3.0].copy()

        if len(df_a1) > 0:
            res_a1 = run_twoway_anova(df_a1, dv, "cav_method", "layer", filter_desc=f"{test_name.capitalize()}: Method x Layer")
            if "error" not in res_a1:
                all_results["technique_layer"].append(res_a1)
                print(f"    Interaction p-val: {res_a1['anova_table'].get('C(cav_method):C(layer)', {}).get('p', 'N/A')}")
            else:
                print(f"    Error: {res_a1['error']}")

            interaction_plot(
                df_a1, dv, "layer", "cav_method",
                f"{test_name.capitalize()} (alpha=3.0): Mean {dv} by Layer",
                save_path=os.path.join(FIGURES_DIR, f"{test_name}_interaction_method_layer.png")
            )

        # B. ANOVA 2: method x alpha (at canonical middle layer L16)
        print(f"  [ANOVA 2] Method x Alpha (Layer={CANONICAL_LAYER})")
        df_a2 = df_target[df_target["layer"] == CANONICAL_LAYER].copy()
        df_a2["alpha_cat"] = df_a2["alpha"].astype(str)

        if len(df_a2) > 0:
            res_a2 = run_twoway_anova(df_a2, dv, "cav_method", "alpha_cat", filter_desc=f"{test_name.capitalize()}: Method x Alpha")
            if "error" not in res_a2:
                all_results["technique_alpha"].append(res_a2)
                print(f"    Interaction p-val: {res_a2['anova_table'].get('C(cav_method):C(alpha_cat)', {}).get('p', 'N/A')}")
            else:
                print(f"    Error: {res_a2['error']}")

            interaction_plot(
                df_a2, dv, "alpha_cat", "cav_method",
                f"{test_name.capitalize()} (layer={CANONICAL_LAYER}): Mean {dv} by Alpha",
                save_path=os.path.join(FIGURES_DIR, f"{test_name}_interaction_method_alpha.png")
            )

        # C. Dose-Response (Alpha scaling)
        print(f"  [ANOVA 3] Dose-Response Alpha scaling (SVM, Layer {CANONICAL_LAYER})")
        df_a3 = df_target[(df_target["layer"] == CANONICAL_LAYER) & (df_target["cav_method"] == "svm")].copy()
        df_a3["alpha_cat"] = df_a3["alpha"].astype(str)
        if len(df_a3) > 0:
            res_a3 = run_oneway_anova(df_a3, dv, "alpha_cat", filter_desc=f"{test_name.capitalize()} dose-response")
            f_stat = res_a3.get('f_oneway', {}).get('F')
            p_val = res_a3.get('f_oneway', {}).get('p')
            print(f"    F={f_stat}, p={p_val}")

    # Process each test
    if "synonym" in dfs and not dfs["synonym"].empty:
        process_test(dfs["synonym"], "delta_auc", "synonym")

    if "naming" in dfs and not dfs["naming"].empty:
        process_test(dfs["naming"], "delta_accuracy", "naming")

    if "oddoneout" in dfs and not dfs["oddoneout"].empty:
        process_test(dfs["oddoneout"], "delta_accuracy", "oddoneout")


    print("\n--- Generating LaTeX tables")

    # Method x Layer Table
    if all_results["technique_layer"]:
        latex = generate_latex_table(
            all_results["technique_layer"],
            caption="ANOVA: Method $\\times$ Layer on BEA suite at $\\alpha=3.0$ (Pythia 2.8B)",
            label="tab:bea_anova_layer_pythia"
        )
        tex_path = os.path.join(RESULTS_DIR, "bea_anova_table_layer.tex")
        with open(tex_path, "w") as f:
            f.write(latex)
        print(f"  Saved Method x Layer: {tex_path}")

    # Method x Alpha Table
    if all_results["technique_alpha"]:
        latex = generate_latex_table(
            all_results["technique_alpha"],
            caption=f"ANOVA: Method $\\times$ Alpha on BEA suite at Layer {CANONICAL_LAYER} (Pythia 2.8B)",
            label="tab:bea_anova_alpha_pythia"
        )
        tex_path = os.path.join(RESULTS_DIR, "bea_anova_table_alpha.tex")
        with open(tex_path, "w") as f:
            f.write(latex)
        print(f"  Saved Method x Alpha: {tex_path}")

    # Save complete JSON
    output_path = os.path.join(RESULTS_DIR, "bea_anova_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nData exported successfully to: {output_path}")

if __name__ == "__main__":
    main()
