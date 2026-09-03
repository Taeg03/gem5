"""
generate_oracle_optional_l2.py

Builds the 972-point Optional-L2 Architecture Oracle:
- 729 points with L2 cache (from oracle_729_{compute, latency, concurrency}.csv)
- 243 points without L2 cache (simulated across compute, latency, concurrency)
"""

import argparse
import concurrent.futures
import itertools
import os
import re
import shutil
import subprocess
import time
from typing import Any
import pandas as pd

issue_widths = [2, 4, 8]
rob_sizes = [32, 64, 128]
l1d_mshrs = [2, 4, 8]
l1d_sizes = ["16kB", "32kB", "64kB"]
l1d_assocs = [2, 4, 8]

nol2_configs = list(itertools.product(
    issue_widths, rob_sizes, l1d_mshrs, l1d_sizes, l1d_assocs
))

def parse_size_kb(size_str: str) -> float:
    size_str = str(size_str).strip()
    if size_str.endswith("kB") or size_str.endswith("KB"):
        return float(size_str[:-2])
    elif size_str.endswith("MB"):
        return float(size_str[:-2]) * 1024.0
    return float(size_str)

def compute_cost_optl2(iw: int, rob: int, mshr: int, l1s: str, l1a: int, has_l2: bool, l2s: str) -> float:
    core_cost = (iw * 50.0) + (rob * 1.5) + (mshr * 5.0)
    l1_kb = parse_size_kb(l1s)
    l1_cost = l1_kb * (1.0 + 0.1 * (l1a / 8.0))
    l2_cost = parse_size_kb(l2s) if has_l2 else 0.0
    return round(core_cost + l1_cost + l2_cost, 1)

def run_single_nol2(idx, iw, rob, mshr, l1s, l1a, workload_mode, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "build/X86/gem5.opt",
        "-d", out_dir,
        "run_dse_optional_l2.py",
        str(iw), str(rob), str(mshr), l1s, str(l1a),
        "False", "inactive", workload_mode
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ipc = None
    stats_file = os.path.join(out_dir, "stats.txt")
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            for line in f:
                if "system.cpu.ipc" in line:
                    match = re.search(r"system\.cpu\.ipc\s+([0-9.]+)", line)
                    if match:
                        ipc = float(match.group(1))
                        break
    return {
        "idx": idx,
        "issue_width": iw,
        "rob_size": rob,
        "l1d_mshrs": mshr,
        "l1d_size": l1s,
        "l1d_assoc": l1a,
        "has_l2": False,
        "l2_size": "inactive",
        "ipc": ipc
    }

def simulate_nol2_workload(workload_mode: str, max_workers: int = 12) -> pd.DataFrame:
    total_runs = len(nol2_configs)
    print(f"\n=======================================================")
    print(f"Simulating 243 No-L2 configurations for '{workload_mode}' (Workers: {max_workers})")
    print(f"=======================================================")
    temp_base = f"m5out_optl2_{workload_mode}"
    os.makedirs(temp_base, exist_ok=True)
    start_time = time.perf_counter()
    results = [None] * total_runs

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                run_single_nol2,
                idx, iw, rob, mshr, l1s, l1a,
                workload_mode,
                os.path.join(temp_base, f"w_{idx}")
            ): idx
            for idx, (iw, rob, mshr, l1s, l1a) in enumerate(nol2_configs)
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            res = future.result()
            results[res["idx"]] = res
            completed += 1
            if completed % 50 == 0 or completed == total_runs:
                elapsed = time.perf_counter() - start_time
                print(f"  [{completed:03d}/{total_runs}] Done (Elapsed: {elapsed:.1f}s, Last IPC: {res['ipc']})")

    elapsed_total = time.perf_counter() - start_time
    print(f"Completed No-L2 {workload_mode} in {elapsed_total:.1f}s")
    shutil.rmtree(temp_base, ignore_errors=True)
    return pd.DataFrame(results)

def build_optl2_oracles(max_workers: int = 12):
    workloads = ["compute", "latency", "concurrency"]
    master_dfs = {}

    for wl in workloads:
        # 1. Load 729 L2 points
        src_csv = f"oracle_729_{wl}.csv"
        df_l2 = pd.read_csv(src_csv)
        df_l2["has_l2"] = True
        df_l2["l2_size"] = df_l2["l2_size"].astype(str)

        # 2. Simulate 243 No-L2 points
        df_nol2 = simulate_nol2_workload(wl, max_workers=max_workers)

        # 3. Merge (729 with L2 + 243 without L2 = 972)
        cols = ["issue_width", "rob_size", "l1d_mshrs", "l1d_size", "l1d_assoc", "has_l2", "l2_size", "ipc"]
        df_combined = pd.concat([df_l2[cols], df_nol2[cols]], ignore_index=True)
        df_combined["config_id"] = range(len(df_combined))

        # 4. Compute Cost & Feasibility
        df_combined["cost"] = df_combined.apply(
            lambda r: compute_cost_optl2(r["issue_width"], r["rob_size"], r["l1d_mshrs"], r["l1d_size"], r["l1d_assoc"], r["has_l2"], r["l2_size"]),
            axis=1
        )
        df_combined["is_feasible"] = df_combined["cost"] <= 1500.0

        out_csv = f"oracle_optl2_{wl}.csv"
        df_combined.to_csv(out_csv, index=False)
        print(f"Saved {len(df_combined)} rows to {out_csv}")
        master_dfs[wl] = df_combined

    # 5. Build Master Oracle
    df_master = master_dfs["compute"].copy()
    df_master = df_master.rename(columns={"ipc": "ipc_compute"})
    df_master["ipc_latency"] = master_dfs["latency"]["ipc"]
    df_master["ipc_concurrency"] = master_dfs["concurrency"]["ipc"]

    df_master.to_csv("oracle_optl2_master.csv", index=False)
    print(f"\nSaved master oracle with 972 points to oracle_optl2_master.csv!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    build_optl2_oracles(max_workers=args.workers)
