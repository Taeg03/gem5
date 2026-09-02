import pandas as pd
import numpy as np

# Neutral workload mappings
oracles = {
    "Workload_Alpha": "oracle_l1.csv",
    "Workload_Beta": "oracle_l2.csv",
    "Workload_Gamma": "oracle_assoc.csv"
}

params = ["l1d_size", "l1d_assoc", "l2_size", "l2_assoc"]

print("==========================================================================================")
print("             4-PARAMETER ORACLE CHARACTERIZATION & PRE-LLM VERIFICATION                   ")
print("==========================================================================================")

dataframes = {}
for name, path in oracles.items():
    df = pd.read_csv(path)
    dataframes[name] = df

# 1. Summary Statistics & Global Optima
print("\n--- 1. Summary Statistics & Global Optima ---")
optima = {}
for name, df in dataframes.items():
    min_ipc = df["ipc"].min()
    mean_ipc = df["ipc"].mean()
    max_ipc = df["ipc"].max()
    std_ipc = df["ipc"].std()
    
    top_row = df.sort_values(by="ipc", ascending=False).iloc[0]
    opt_cfg = (top_row["l1d_size"], int(top_row["l1d_assoc"]), top_row["l2_size"], int(top_row["l2_assoc"]))
    optima[name] = (opt_cfg, max_ipc)
    
    print(f"\nWorkload: {name}")
    print(f"  IPC: Min={min_ipc:.4f}, Mean={mean_ipc:.4f}, Max={max_ipc:.4f}, Std={std_ipc:.4f}")
    print(f"  Global Optimum: L1:{opt_cfg[0]}/{opt_cfg[1]}-way, L2:{opt_cfg[2]}/{opt_cfg[3]}-way -> IPC: {max_ipc:.4f}")
    print(f"  Top 3 Configurations:")
    for _, r in df.sort_values(by="ipc", ascending=False).head(3).iterrows():
        print(f"    (L1:{r['l1d_size']}/{int(r['l1d_assoc'])}w, L2:{r['l2_size']}/{int(r['l2_assoc'])}w) -> IPC: {r['ipc']:.4f}")

# 2. Distance Between Workload Optima
print("\n--- 2. Parameter Distance Between Workload Optima ---")
def param_distance(cfg1, cfg2):
    mappings = {
        "l1d_size": {"16kB":0, "32kB":1, "64kB":2, "128kB":3},
        "l1d_assoc": {2:0, 4:1, 8:2, 16:3},
        "l2_size": {"512kB":0, "1MB":1, "2MB":2, "4MB":3},
        "l2_assoc": {2:0, 4:1, 8:2, 16:3}
    }
    dist = 0
    for idx, p in enumerate(params):
        v1 = cfg1[idx]
        v2 = cfg2[idx]
        dist += abs(mappings[p][v1] - mappings[p][v2])
    return dist

names = list(oracles.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        n1, n2 = names[i], names[j]
        d = param_distance(optima[n1][0], optima[n2][0])
        print(f"  Distance({n1} <-> {n2}): {d} / 12 (Normalized: {d/12*100:.1f}%)")

# 3. Parameter Sensitivity & Variance Contribution (ANOVA Main Effects)
print("\n--- 3. Parameter Sensitivity / Variance Contribution (%) ---")
sensitivity_matrix = {}
for name, df in dataframes.items():
    total_var = df["ipc"].var()
    var_contribs = {}
    for p in params:
        var_p = df.groupby(p)["ipc"].mean().var()
        var_contribs[p] = (var_p / total_var) * 100
    sensitivity_matrix[name] = var_contribs

df_sens = pd.DataFrame(sensitivity_matrix)
print(df_sens.round(2))

# Subsystem Variance Share (L1 Subsystem vs L2 Subsystem)
print("\n--- Subsystem Variance Share (L1 Subsystem vs L2 Subsystem) ---")
for name, contribs in sensitivity_matrix.items():
    l1_share = contribs["l1d_size"] + contribs["l1d_assoc"]
    l2_share = contribs["l2_size"] + contribs["l2_assoc"]
    print(f"  {name:<15}: L1 Subsystem = {l1_share:6.2f}% | L2 Subsystem = {l2_share:6.2f}%")

# 4. Near-Optimal Solution Density & Plateau Breadth
print("\n--- 4. Landscape Multiplicity & Plateau Density ---")
for name, df in dataframes.items():
    max_ipc = df["ipc"].max()
    n_95 = (df["ipc"] >= 0.95 * max_ipc).sum()
    n_98 = (df["ipc"] >= 0.98 * max_ipc).sum()
    n_99 = (df["ipc"] >= 0.99 * max_ipc).sum()
    print(f"  {name:<15}: >=95% Max: {n_95:3d} ({n_95/256*100:4.1f}%) | >=98% Max: {n_98:3d} ({n_98/256*100:4.1f}%) | >=99% Max: {n_99:3d} ({n_99/256*100:4.1f}%)")

# 5. Search Difficulty Diagnostic: "Maximize Everything" Performance
print("\n--- 5. Search Difficulty Diagnostic ('Maximize Everything' Heuristic) ---")
max_all_cfg = ("128kB", 16, "4MB", 16)
for name, df in dataframes.items():
    max_ipc = df["ipc"].max()
    row_max_all = df[(df["l1d_size"] == "128kB") & (df["l1d_assoc"] == 16) & 
                     (df["l2_size"] == "4MB") & (df["l2_assoc"] == 16)].iloc[0]
    ipc_max_all = row_max_all["ipc"]
    pct_max_all = (ipc_max_all / max_ipc) * 100
    print(f"  {name:<15}: Maximize-All (128k/16w, 4M/16w) -> IPC: {ipc_max_all:.4f} ({pct_max_all:.2f}% of Max {max_ipc:.4f})")

# 6. Random Search Benchmark (Monte Carlo 10,000 Trials)
print("\n--- 6. Random Search Difficulty Benchmark (10,000 Monte Carlo Trials) ---")
np.random.seed(42)
N_TRIALS = 10000

for name, df in dataframes.items():
    ipcs = df["ipc"].values
    max_ipc = ipcs.max()
    th_98 = 0.98 * max_ipc
    th_99 = 0.99 * max_ipc
    
    steps_to_98 = []
    steps_to_99 = []
    
    for _ in range(N_TRIALS):
        sample_indices = np.random.choice(len(ipcs), size=15, replace=False)
        sample_ipcs = ipcs[sample_indices]
        
        s98 = np.argmax(sample_ipcs >= th_98) + 1 if np.any(sample_ipcs >= th_98) else 16
        s99 = np.argmax(sample_ipcs >= th_99) + 1 if np.any(sample_ipcs >= th_99) else 16
        
        steps_to_98.append(s98)
        steps_to_99.append(s99)
        
    p_hit_10_98 = np.mean(np.array(steps_to_98) <= 10) * 100
    median_steps_98 = np.median([s for s in steps_to_98 if s <= 15])
    print(f"  {name:<15}: P(Hit >=98% within 10 random steps) = {p_hit_10_98:5.1f}% | Median random steps = {median_steps_98:.1f}")

print("\n==========================================================================================")
