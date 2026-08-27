import itertools
import os
import re
import subprocess
import time

import pandas as pd

# 1. Define the rigorous design space
l1d_sizes = ["16kB", "32kB", "64kB", "128kB"]
l1d_assocs = [2, 4, 8, 16]
l2_sizes = ["512kB", "1MB", "2MB", "4MB"]
l2_assocs = [2, 4, 8, 16]

# Generate all 256 combinations
configurations = list(
    itertools.product(l1d_sizes, l1d_assocs, l2_sizes, l2_assocs)
)

results = []
total_runs = len(configurations)

print(f"Starting Oracle generation for {total_runs} configurations...")
overall_start_time = time.perf_counter()

for idx, (l1s, l1a, l2s, l2a) in enumerate(configurations):
    print(
        f"[{idx+1}/{total_runs}] Running L1: {l1s}/{l1a}-way | L2: {l2s}/{l2a}-way...",
        end="",
        flush=True,
    )

    # 2. Construct the gem5 command
    cmd = ["build/X86/gem5.opt", "run_dse.py", l1s, str(l1a), l2s, str(l2a)]

    run_start_time = time.perf_counter()

    # Run gem5 (suppressing standard output to keep the console clean)
    process = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    run_elapsed_time = time.perf_counter() - run_start_time

    # 3. Extract IPC from stats.txt
    ipc = None
    stats_file = "m5out/stats.txt"
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            for line in f:
                if "system.cpu.ipc" in line:
                    match = re.search(r"system\.cpu\.ipc\s+([0-9.]+)", line)
                    if match:
                        ipc = float(match.group(1))
                        break

    if ipc is not None:
        print(f" IPC: {ipc}")
    else:
        print(" FAILED")

    # 4. Store the data
    results.append(
        {
            "config_id": idx,
            "l1d_size": l1s,
            "l1d_assoc": l1a,
            "l2_size": l2s,
            "l2_assoc": l2a,
            "ipc": ipc,
        }
    )

overall_elapsed_time = time.perf_counter() - overall_start_time
mins, secs = divmod(overall_elapsed_time, 60)
hours, mins = divmod(mins, 60)

# 5. Save the Oracle to CSV
df = pd.DataFrame(results)
df.to_csv("oracle_results.csv", index=False)
print(
    f"\nOracle generation complete! Total elapsed time: {int(hours):02d}h {int(mins):02d}m {secs:05.2f}s"
)
print("Saved to oracle_results.csv")
