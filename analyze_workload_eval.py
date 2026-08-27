import json
import os
import pandas as pd

def analyze_results(results_file="workload_eval_results.json"):
    if not os.path.exists(results_file):
        print(f"File {results_file} not found.")
        return

    with open(results_file) as f:
        data = json.load(f)

    oracle_files = {
        "L1_Heavy": "oracle_l1.csv",
        "L2_Heavy": "oracle_l2.csv",
        "Assoc_Heavy": "oracle_assoc.csv",
        "Composite_Workload": "oracle_results.csv"
    }

    print(f"\n=========================================================================================")
    print(f"             WORKLOAD-DEPENDENT REPRESENTATION EVALUATION ANALYSIS                       ")
    print(f"=========================================================================================")

    summary_rows = []

    for wl_name, baselines in data.items():
        oracle_csv = oracle_files.get(wl_name, f"oracle_{wl_name.lower().split('_')[0]}.csv")
        df_oracle = pd.read_csv(oracle_csv) if os.path.exists(oracle_csv) else None
        global_max_ipc = df_oracle["ipc"].max() if df_oracle is not None else 1.0
        init_ipc = df_oracle[(df_oracle["l1d_size"] == "16kB") & (df_oracle["l1d_assoc"] == 2) & 
                             (df_oracle["l2_size"] == "512kB") & (df_oracle["l2_assoc"] == 2)]["ipc"].iloc[0]

        print(f"\n>>> Workload: {wl_name} (Global Max IPC: {global_max_ipc:.4f}, Init IPC: {init_ipc:.4f})")
        print(f"-----------------------------------------------------------------------------------------")

        for b_name, steps in baselines.items():
            best_ipc = 0.0
            step_to_98pct = None
            threshold_98 = 0.98 * global_max_ipc

            l1_mutations = 0
            l2_mutations = 0
            assoc_mutations = 0

            prev_cfg = ("16kB", 2, "512kB", 2)

            for s in steps:
                cfg = tuple(s["config"])
                ipc = s["ipc"]
                if ipc and ipc > best_ipc:
                    best_ipc = ipc
                if ipc and ipc >= threshold_98 and step_to_98pct is None:
                    step_to_98pct = s["step"]

                # Track what was changed
                if cfg[0] != prev_cfg[0]: l1_mutations += 1
                if cfg[1] != prev_cfg[1]: assoc_mutations += 1
                if cfg[2] != prev_cfg[2]: l2_mutations += 1
                if cfg[3] != prev_cfg[3]: assoc_mutations += 1
                prev_cfg = cfg

            pct_max = (best_ipc / global_max_ipc) * 100
            step_str = str(step_to_98pct) if step_to_98pct is not None else ">10"

            print(f"  {b_name:<12} | Best IPC: {best_ipc:.4f} ({pct_max:.1f}% of max) | Steps to >=98%: {step_str:<3} | Mutations: L1_size={l1_mutations}, L2_size={l2_mutations}, Assoc={assoc_mutations}")

            summary_rows.append({
                "Workload": wl_name,
                "Baseline": b_name,
                "Best_IPC": round(best_ipc, 4),
                "Pct_Global_Max": round(pct_max, 1),
                "Steps_to_98pct": step_str,
                "L1_Mutations": l1_mutations,
                "L2_Mutations": l2_mutations,
                "Assoc_Mutations": assoc_mutations
            })

    print(f"\n=========================================================================================\n")
    df_summary = pd.DataFrame(summary_rows)
    return df_summary

if __name__ == "__main__":
    analyze_results()
