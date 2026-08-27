import argparse
import concurrent.futures
import itertools
import os
import re
import shutil
import subprocess
import time
import pandas as pd

# 1. Define the 256-point design space
l1d_sizes = ["16kB", "32kB", "64kB", "128kB"]
l1d_assocs = [2, 4, 8, 16]
l2_sizes = ["512kB", "1MB", "2MB", "4MB"]
l2_assocs = [2, 4, 8, 16]

configurations = list(itertools.product(l1d_sizes, l1d_assocs, l2_sizes, l2_assocs))

def run_single_config(idx, l1s, l1a, l2s, l2a, workload_mode, out_dir):
    """Run one gem5 simulation in an isolated output directory."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "build/X86/gem5.opt",
        "-d", out_dir,
        "run_dse.py",
        l1s, str(l1a), l2s, str(l2a), workload_mode
    ]

    process = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

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
        "config_id": idx,
        "l1d_size": l1s,
        "l1d_assoc": l1a,
        "l2_size": l2s,
        "l2_assoc": l2a,
        "ipc": ipc,
    }

def generate_oracle_for_workload(workload_mode: str, output_csv: str, max_workers: int = 8):
    total_runs = len(configurations)
    print(f"\n=======================================================")
    print(f"Generating Oracle for Workload: '{workload_mode}' -> {output_csv}")
    print(f"Total configurations: {total_runs} (Parallel Workers: {max_workers})")
    print(f"=======================================================")

    temp_base = f"m5out_par_{workload_mode}"
    os.makedirs(temp_base, exist_ok=True)

    start_time = time.perf_counter()
    results = [None] * total_runs

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                run_single_config,
                idx, l1s, l1a, l2s, l2a,
                workload_mode,
                os.path.join(temp_base, f"worker_{idx}")
            ): idx
            for idx, (l1s, l1a, l2s, l2a) in enumerate(configurations)
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            res = future.result()
            idx = res["config_id"]
            results[idx] = res
            completed += 1
            if completed % 16 == 0 or completed == total_runs:
                elapsed = time.perf_counter() - start_time
                print(f"  [{completed:03d}/{total_runs}] Done (Elapsed: {elapsed:.1f}s, Recent IPC: {res['ipc']})")

    elapsed_total = time.perf_counter() - start_time
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Completed {workload_mode} Oracle in {elapsed_total:.1f}s. Saved to {output_csv}")

    # Cleanup worker temp directories
    shutil.rmtree(temp_base, ignore_errors=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 256-point DSE Oracles for gem5 workloads")
    parser.add_argument("--workload", type=str, default="suite", choices=["l1", "l2", "assoc", "all", "suite"],
                        help="Workload mode: 'l1', 'l2', 'assoc', 'all' (composite), or 'suite' (generates l1, l2, assoc)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel gem5 simulation workers")
    args = parser.parse_args()

    if args.workload == "suite":
        for mode, filename in [("l1", "oracle_l1.csv"), ("l2", "oracle_l2.csv"), ("assoc", "oracle_assoc.csv")]:
            generate_oracle_for_workload(mode, filename, max_workers=args.workers)
    else:
        out_name = f"oracle_{args.workload}.csv" if args.workload != "all" else "oracle_results.csv"
        generate_oracle_for_workload(args.workload, out_name, max_workers=args.workers)
