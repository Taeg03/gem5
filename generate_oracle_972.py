"""
generate_oracle_972.py

Generates the 972-point architectural oracle:
- Reuses the existing 729 DerivO3CPU points from oracle_729_{compute, latency, concurrency}.csv
- Simulates the 243 X86MinorCPU points across all 3 workloads in parallel
- Evaluates the principled project cost proxy
- Exports oracle_972_compute.csv, oracle_972_latency.csv, oracle_972_concurrency.csv, and oracle_972_master.csv
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

# 243 MinorCPU configurations (rob_size is structurally inactive)
issue_widths = [2, 4, 8]
l1d_mshrs = [2, 4, 8]
l1d_sizes = ["16kB", "32kB", "64kB"]
l1d_assocs = [2, 4, 8]
l2_sizes = ["512kB", "1MB", "2MB"]

minor_configs = list(itertools.product(
    issue_widths, l1d_mshrs, l1d_sizes, l1d_assocs, l2_sizes
))

def parse_size_kb(size_str: str) -> float:
    size_str = str(size_str).strip()
    if size_str.endswith("kB") or size_str.endswith("KB"):
        return float(size_str[:-2])
    elif size_str.endswith("MB"):
        return float(size_str[:-2]) * 1024.0
    return float(size_str)

def compute_cost_972(core_type: str, iw: int, rob_size: Any, mshr: int, l1s: str, l1a: int, l2s: str) -> float:
    # Cost Model as a project cost proxy
    if core_type == "DerivO3CPU":
        rob_val = int(rob_size)
        core_cost = (iw * 50.0) + (rob_val * 1.5) + (mshr * 5.0)
    else:  # MinorCPU (rob_size is inactive -> 0)
        core_cost = (iw * 50.0) + (mshr * 5.0)

    l1_kb = parse_size_kb(l1s)
    l2_kb = parse_size_kb(l2s)
    mem_cost = l1_kb * (1.0 + 0.1 * (l1a / 8.0)) + l2_kb
    return round(core_cost + mem_cost, 1)

def run_single_minor(idx, iw, mshr, l1s, l1a, l2s, workload_mode, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "build/X86/gem5.opt",
        "-d", out_dir,
        "run_dse_972.py",
        "MinorCPU",
        str(iw),
        "inactive",
        str(mshr),
        l1s,
        str(l1a),
        l2s,
        workload_mode
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
        "minor_idx": idx,
        "core_type": "MinorCPU",
        "issue_width": iw,
        "rob_size": "inactive",
        "l1d_mshrs": mshr,
        "l1d_size": l1s,
        "l1d_assoc": l1a,
        "l2_size": l2s,
        "ipc": ipc
    }

def simulate_minor_workload(workload_mode: str, max_workers: int = 12) -> pd.DataFrame:
    total_runs = len(minor_configs)
    print(f"\n=======================================================")
    print(f"Simulating 243 MinorCPU points for '{workload_mode}' (Workers: {max_workers})")
    print(f"=======================================================")

    temp_base = f"m5out_minor_{workload_mode}"
    os.makedirs(temp_base, exist_ok=True)
    start_time = time.perf_counter()
    results = [None] * total_runs

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                run_single_minor,
                idx, iw, mshr, l1s, l1a, l2s,
                workload_mode,
                os.path.join(temp_base, f"w_{idx}")
            ): idx
            for idx, (iw, mshr, l1s, l1a, l2s) in enumerate(minor_configs)
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            res = future.result()
            idx = res["minor_idx"]
            results[idx] = res
            completed += 1
            if completed % 50 == 0 or completed == total_runs:
                elapsed = time.perf_counter() - start_time
                print(f"  [{completed:03d}/{total_runs}] Done (Elapsed: {elapsed:.1f}s, Last IPC: {res['ipc']})")

    elapsed_total = time.perf_counter() - start_time
    print(f"Completed MinorCPU {workload_mode} in {elapsed_total:.1f}s")
    shutil.rmtree(temp_base, ignore_errors=True)
    return pd.DataFrame(results)

def build_972_oracles(max_workers: int = 12):
    workloads = ["compute", "latency", "concurrency"]
    master_dfs = {}

    for wl in workloads:
        # 1. Load existing 729 O3 oracle
        src_csv = f"oracle_729_{wl}.csv"
        df_o3 = pd.read_csv(src_csv)
        df_o3["core_type"] = "DerivO3CPU"
        df_o3["rob_size"] = df_o3["rob_size"].astype(str)

        # 2. Simulate 243 MinorCPU points
        df_minor = simulate_minor_workload(wl, max_workers=max_workers)

        # 3. Concatenate (729 O3 + 243 Minor = 972)
        cols = ["core_type", "issue_width", "rob_size", "l1d_mshrs", "l1d_size", "l1d_assoc", "l2_size", "ipc"]
        df_combined = pd.concat([df_o3[cols], df_minor[cols]], ignore_index=True)
        df_combined["config_id"] = range(len(df_combined))

        # 4. Add Cost & Feasibility
        df_combined["cost"] = df_combined.apply(
            lambda r: compute_cost_972(r["core_type"], r["issue_width"], r["rob_size"], r["l1d_mshrs"], r["l1d_size"], r["l1d_assoc"], r["l2_size"]),
            axis=1
        )
        df_combined["is_feasible"] = df_combined["cost"] <= 1500.0

        out_csv = f"oracle_972_{wl}.csv"
        df_combined.to_csv(out_csv, index=False)
        print(f"Saved {len(df_combined)} rows to {out_csv}")
        master_dfs[wl] = df_combined

    # 5. Build Master Oracle (all IPCs in one table)
    df_master = master_dfs["compute"].copy()
    df_master = df_master.rename(columns={"ipc": "ipc_compute"})
    df_master["ipc_latency"] = master_dfs["latency"]["ipc"]
    df_master["ipc_concurrency"] = master_dfs["concurrency"]["ipc"]

    df_master.to_csv("oracle_972_master.csv", index=False)
    print(f"\nSaved master oracle with 972 points to oracle_972_master.csv!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    build_972_oracles(max_workers=args.workers)
