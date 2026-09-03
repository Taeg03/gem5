"""
characterize_972_landscape.py

Comprehensive landscape characterization of the 972-point architectural space:
1. Feasibility analysis under the 1500 KB budget (overall and by core_type).
2. ANOVA variance decomposition (sensitivity of each parameter by workload).
3. Identification and sizing of near-optimal regions (>=98% of constrained optimum).
4. Core-Memory interactions (MinorCPU vs DerivO3CPU across cache tiers).
5. Impact of conditional rob_size on the agent's search and reasoning problem.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

BUDGET_KB = 1500.0

df_master = pd.read_csv("oracle_972_master.csv")

print("=" * 90)
print("             972-POINT ARCHITECTURAL ORACLE LANDSCAPE CHARACTERIZATION")
print("=" * 90)

# ------------------------------------------------------------------------------
# 1. Feasibility Analysis under 1500 KB Budget
# ------------------------------------------------------------------------------
print("\n--- 1. Feasibility Analysis under 1500 KB Budget ---")
total_points = len(df_master)
total_feasible = len(df_master[df_master["is_feasible"]])
pct_feasible = (total_feasible / total_points) * 100

df_o3 = df_master[df_master["core_type"] == "DerivO3CPU"]
df_minor = df_master[df_master["core_type"] == "MinorCPU"]

o3_feasible = len(df_o3[df_o3["is_feasible"]])
o3_pct = (o3_feasible / len(df_o3)) * 100

minor_feasible = len(df_minor[df_minor["is_feasible"]])
minor_pct = (minor_feasible / len(df_minor)) * 100

print(f"Overall Space:    {total_feasible:3d} / {total_points} feasible ({pct_feasible:.1f}%)")
print(f"DerivO3CPU:       {o3_feasible:3d} / {len(df_o3)} feasible ({o3_pct:.1f}%)")
print(f"MinorCPU:         {minor_feasible:3d} / {len(df_minor)} feasible ({minor_pct:.1f}%)")

print("\nCost Summary by Core Type:")
print(df_master.groupby("core_type")["cost"].describe().round(1))

# ------------------------------------------------------------------------------
# 2. Constrained Optima & Near-Optimal Regions (>=98% of Optimum)
# ------------------------------------------------------------------------------
print("\n--- 2. Constrained Optima & Near-Optimal Regions (>=98% of Optimum) ---")
workloads = ["compute", "latency", "concurrency"]

for wl in workloads:
    col = f"ipc_{wl}"
    feasible_df = df_master[df_master["is_feasible"]]
    best_row = feasible_df.sort_values(by=col, ascending=False).iloc[0]
    opt_ipc = best_row[col]
    threshold_98 = 0.98 * opt_ipc

    near_opt = feasible_df[feasible_df[col] >= threshold_98]
    n_near = len(near_opt)
    pct_near = (n_near / len(feasible_df)) * 100

    o3_near = len(near_opt[near_opt["core_type"] == "DerivO3CPU"])
    minor_near = len(near_opt[near_opt["core_type"] == "MinorCPU"])

    print(f"\nWorkload: {wl.upper()}")
    print(f"  Constrained Optimum IPC: {opt_ipc:.4f}")
    print(f"  Optimal Architecture:    {best_row['core_type']} | IW:{best_row['issue_width']}w | ROB:{best_row['rob_size']} | MSHR:{best_row['l1d_mshrs']} | L1:{best_row['l1d_size']}/{best_row['l1d_assoc']}w | L2:{best_row['l2_size']} (Cost: {best_row['cost']:.1f} KB)")
    print(f"  >=98% Region Threshold:  {threshold_98:.4f}")
    print(f"  >=98% Region Count:      {n_near:2d} / {len(feasible_df)} feasible points ({pct_near:.1f}%)")
    print(f"  Composition:             {o3_near} DerivO3CPU, {minor_near} MinorCPU")

# ------------------------------------------------------------------------------
# 3. Parameter Sensitivity Analysis (ANOVA Variance Decomposition)
# ------------------------------------------------------------------------------
print("\n--- 3. Parameter Sensitivity Analysis (ANOVA Variance Decomposition) ---")

for wl in workloads:
    col = f"ipc_{wl}"
    # Global ANOVA across all 972 points (using categorical encoding)
    formula = f"{col} ~ C(core_type) + C(issue_width) + C(l1d_mshrs) + C(l1d_size) + C(l1d_assoc) + C(l2_size)"
    model = ols(formula, data=df_master).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    total_ss = aov_table["sum_sq"].sum()
    aov_table["pct_variance"] = (aov_table["sum_sq"] / total_ss) * 100

    print(f"\nANOVA Variance Decomposition for '{wl.upper()}':")
    for param, row in aov_table.iterrows():
        if param != "Residual":
            clean_name = param.replace("C(", "").replace(")", "")
            print(f"  {clean_name:<16}: {row['pct_variance']:6.2f}% variance (p = {row['PR(>F)']:.4e})")
    resid = aov_table.loc["Residual", "pct_variance"]
    print(f"  {'Residual':<16}: {resid:6.2f}% variance")

# ------------------------------------------------------------------------------
# 4. DerivO3CPU Sub-Space ANOVA (with rob_size active)
# ------------------------------------------------------------------------------
print("\n--- 4. DerivO3CPU Sub-Space ANOVA (Evaluating Active rob_size Sensitivity) ---")
for wl in workloads:
    col = f"ipc_{wl}"
    formula_o3 = f"{col} ~ C(issue_width) + C(rob_size) + C(l1d_mshrs) + C(l1d_size) + C(l1d_assoc) + C(l2_size)"
    model_o3 = ols(formula_o3, data=df_o3).fit()
    aov_o3 = sm.stats.anova_lm(model_o3, typ=2)
    total_ss_o3 = aov_o3["sum_sq"].sum()
    aov_o3["pct_variance"] = (aov_o3["sum_sq"] / total_ss_o3) * 100

    print(f"\nDerivO3CPU Subspace '{wl.upper()}':")
    for param, row in aov_o3.iterrows():
        if param != "Residual":
            clean_name = param.replace("C(", "").replace(")", "")
            print(f"  {clean_name:<16}: {row['pct_variance']:6.2f}% variance (p = {row['PR(>F)']:.4e})")

# ------------------------------------------------------------------------------
# 5. Core-Memory Interactions: Does MinorCPU unlock optimal trade-offs?
# ------------------------------------------------------------------------------
print("\n--- 5. Core-Memory Interactions: Peak IPC Comparison per Core Type ---")
for wl in workloads:
    col = f"ipc_{wl}"
    peak_o3 = df_o3[df_o3["is_feasible"]][col].max()
    peak_minor = df_minor[df_minor["is_feasible"]][col].max()
    print(f"Workload {wl.upper():<12}: Peak Feasible O3 IPC = {peak_o3:.4f} | Peak Feasible Minor IPC = {peak_minor:.4f} | Ratio (Minor/O3) = {peak_minor/peak_o3:.2f}")

print("\n" + "=" * 90)
