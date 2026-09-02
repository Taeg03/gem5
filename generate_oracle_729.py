import argparse
import concurrent.futures
import itertools
import os
import re
import shutil
import subprocess
import time
import pandas as pd

# 6 Parameter Space
issue_widths = [2, 4, 8]
rob_sizes = [32, 64, 128]
l1d_mshrs = [2, 4, 8]
l1d_sizes = ["16kB", "32kB", "64kB"]
l1d_assocs = [2, 4, 8]
l2_sizes = ["512kB", "1MB", "2MB"]

configurations = list(itertools.product(
    issue_widths, rob_sizes, l1d_mshrs, l1d_sizes, l1d_assocs, l2_sizes
))

def run_single_config(idx, iw, rob, mshr, l1s, l1a, l2s, workload_mode, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "build/X86/gem5.opt",
        "-d", out_dir,
        "run_dse_729.py",
        str(iw), str(rob), str(mshr), l1s, str(l1a), l2s, workload_mode
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
        "issue_width": iw,
        "rob_size": rob,
        "l1d_mshrs": mshr,
        "l1d_size": l1s,
        "l1d_assoc": l1a,
        "l2_size": l2s,
        "ipc": ipc,
    }

def generate_oracle(workload_mode: str, output_csv: str, max_workers: int = 10):
    total_runs = len(configurations)
    print(f"\n=======================================================")
    print(f"Generating 729-Point Oracle for '{workload_mode}' -> {output_csv}")
    print(f"Total configurations: {total_runs} (Parallel Workers: {max_workers})")
    print(f"=======================================================")

    temp_base = f"m5out_par729_{workload_mode}"
    os.makedirs(temp_base, exist_ok=True)

    start_time = time.perf_counter()
    results = [None] * total_runs

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                run_single_config,
                idx, iw, rob, mshr, l1s, l1a, l2s,
                workload_mode,
                os.path.join(temp_base, f"worker_{idx}")
            ): idx
            for idx, (iw, rob, mshr, l1s, l1a, l2s) in enumerate(configurations)
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            res = future.result()
            idx = res["config_id"]
            results[idx] = res
            completed += 1
            if completed % 50 == 0 or completed == total_runs:
                elapsed = time.perf_counter() - start_time
                print(f"  [{completed:03d}/{total_runs}] Done (Elapsed: {elapsed:.1f}s, Recent IPC: {res['ipc']})")

    elapsed_total = time.perf_counter() - start_time
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Completed {workload_mode} Oracle in {elapsed_total:.1f}s. Saved to {output_csv}")

    shutil.rmtree(temp_base, ignore_errors=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="suite", choices=["compute", "latency", "concurrency", "suite"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    if args.workload == "suite":
        for mode, filename in [("compute", "oracle_729_compute.csv"),
                               ("latency", "oracle_729_latency.csv"),
                               ("concurrency", "oracle_729_concurrency.csv")]:
            generate_oracle(mode, filename, max_workers=args.workers)
    else:
        generate_oracle(args.workload, f"oracle_729_{args.workload}.csv", max_workers=args.workers)
