import argparse
import json
import os
import re
import time
from typing import Literal

# Load .env
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v.strip("\"'")

import pandas as pd
import numpy as np
from google import genai
from google.genai import types
from pydantic import (
    BaseModel,
    Field,
)

# ---------------------------------------------------------
# 1. Cost Model & Oracle Lookup
# ---------------------------------------------------------
CURRENT_ORACLE_DF = None

def parse_size_kb(s: str) -> int:
    if "MB" in s:
        return int(s.replace("MB", "")) * 1024
    return int(s.replace("kB", ""))

def compute_cost_729(iw: int, rob: int, mshr: int, l1s: str, l1a: int, l2s: str) -> float:
    """Explicit transparent hardware cost model combining Core and Memory subsystems."""
    l1_kb = parse_size_kb(l1s)
    l2_kb = parse_size_kb(l2s)
    core_cost = (iw * 50) + (rob * 1.5) + (mshr * 5)
    mem_cost = l1_kb * (1.0 + 0.1 * (l1a / 8.0)) + l2_kb
    return round(core_cost + mem_cost, 1)

def load_oracle(oracle_path: str):
    global CURRENT_ORACLE_DF
    CURRENT_ORACLE_DF = pd.read_csv(oracle_path)
    return CURRENT_ORACLE_DF

def oracle_lookup_729(iw: int, rob: int, mshr: int, l1s: str, l1a: int, l2s: str):
    match = CURRENT_ORACLE_DF[
        (CURRENT_ORACLE_DF["issue_width"] == int(iw))
        & (CURRENT_ORACLE_DF["rob_size"] == int(rob))
        & (CURRENT_ORACLE_DF["l1d_mshrs"] == int(mshr))
        & (CURRENT_ORACLE_DF["l1d_size"] == str(l1s))
        & (CURRENT_ORACLE_DF["l1d_assoc"] == int(l1a))
        & (CURRENT_ORACLE_DF["l2_size"] == str(l2s))
    ]
    if match.empty:
        return None
    return float(match.iloc[0]["ipc"])


# ---------------------------------------------------------
# 2. Strict Pydantic Schemas for 6-Parameter Baselines
# ---------------------------------------------------------

class BaselineASchema(BaseModel):
    system_cpu_issueWidth: Literal["2", "4", "8"] = Field(..., alias="system.cpu.issueWidth")
    system_cpu_numROBEntries: Literal["32", "64", "128"] = Field(..., alias="system.cpu.numROBEntries")
    system_cpu_dcache_mshrs: Literal["2", "4", "8"] = Field(..., alias="system.cpu.dcache.mshrs")
    system_cpu_dcache_size: Literal["16kB", "32kB", "64kB"] = Field(..., alias="system.cpu.dcache.size")
    system_cpu_dcache_assoc: Literal["2", "4", "8"] = Field(..., alias="system.cpu.dcache.assoc")
    system_l2cache_size: Literal["512kB", "1MB", "2MB"] = Field(..., alias="system.l2cache.size")
    reasoning: str


class BaselineBSchema(BaseModel):
    issue_width: Literal["2", "4", "8"]
    rob_size: Literal["32", "64", "128"]
    l1d_mshrs: Literal["2", "4", "8"]
    l1d_capacity_kb: Literal["16", "32", "64"]
    l1d_associativity: Literal["2", "4", "8"]
    l2_capacity_kb: Literal["512", "1024", "2048"]
    reasoning: str


class CoreSubsystem(BaseModel):
    issue_width: Literal["2", "4", "8"]
    rob_size: Literal["32", "64", "128"]
    l1d_mshrs: Literal["2", "4", "8"]


class MemorySubsystem(BaseModel):
    l1d_capacity_kb: Literal["16", "32", "64"]
    l1d_associativity: Literal["2", "4", "8"]
    l2_capacity_kb: Literal["512", "1024", "2048"]


class BaselineCSchema(BaseModel):
    core_subsystem: CoreSubsystem
    memory_subsystem: MemorySubsystem
    reasoning: str


# ---------------------------------------------------------
# 3. Parsers
# ---------------------------------------------------------

def parse_baseline_a(data: dict):
    iw = int(data.get("system.cpu.issueWidth", data.get("system_cpu_issueWidth")))
    rob = int(data.get("system.cpu.numROBEntries", data.get("system_cpu_numROBEntries")))
    mshr = int(data.get("system.cpu.dcache.mshrs", data.get("system_cpu_dcache_mshrs")))
    l1s = str(data.get("system.cpu.dcache.size", data.get("system_cpu_dcache_size")))
    l1a = int(data.get("system.cpu.dcache.assoc", data.get("system_cpu_dcache_assoc")))
    l2s = str(data.get("system.l2cache.size", data.get("system_l2cache_size")))
    return (iw, rob, mshr, l1s, l1a, l2s)

def parse_baseline_b(data: dict):
    size_map = {"16": "16kB", "32": "32kB", "64": "64kB", "512": "512kB", "1024": "1MB", "2048": "2MB",
                16: "16kB", 32: "32kB", 64: "64kB", 512: "512kB", 1024: "1MB", 2048: "2MB"}
    iw = int(data["issue_width"])
    rob = int(data["rob_size"])
    mshr = int(data["l1d_mshrs"])
    l1s = size_map[str(data["l1d_capacity_kb"])]
    l1a = int(data["l1d_associativity"])
    l2s = size_map[str(data["l2_capacity_kb"])]
    return (iw, rob, mshr, l1s, l1a, l2s)

def parse_baseline_c(data: dict):
    size_map = {"16": "16kB", "32": "32kB", "64": "64kB", "512": "512kB", "1024": "1MB", "2048": "2MB",
                16: "16kB", 32: "32kB", 64: "64kB", 512: "512kB", 1024: "1MB", 2048: "2MB"}
    c = data["core_subsystem"]
    m = data["memory_subsystem"]
    iw = int(c["issue_width"])
    rob = int(c["rob_size"])
    mshr = int(c["l1d_mshrs"])
    l1s = size_map[str(m["l1d_capacity_kb"])]
    l1a = int(m["l1d_associativity"])
    l2s = size_map[str(m["l2_capacity_kb"])]
    return (iw, rob, mshr, l1s, l1a, l2s)


BASELINES = {
    "Baseline_A": (BaselineASchema, parse_baseline_a, "Raw simulator parameter flags"),
    "Baseline_B": (BaselineBSchema, parse_baseline_b, "Flat microarchitectural parameters"),
    "Baseline_C": (BaselineCSchema, parse_baseline_c, "Hierarchical Core and Memory subsystems"),
}


def run_trial(client, baseline_name: str, workload_name: str, oracle_path: str, budget_kb: float, trial_seed: int, max_steps: int = 10):
    load_oracle(oracle_path)
    initial_cfg = (2, 32, 2, "16kB", 2, "512kB")
    initial_ipc = oracle_lookup_729(*initial_cfg)
    initial_cost = compute_cost_729(*initial_cfg)

    # Compute ground truth constrained optimum
    CURRENT_ORACLE_DF["cost"] = CURRENT_ORACLE_DF.apply(
        lambda r: compute_cost_729(r["issue_width"], r["rob_size"], r["l1d_mshrs"], r["l1d_size"], r["l1d_assoc"], r["l2_size"]), axis=1
    )
    feasible_df = CURRENT_ORACLE_DF[CURRENT_ORACLE_DF["cost"] <= budget_kb]
    best_feasible_row = feasible_df.sort_values(by="ipc", ascending=False).iloc[0] if not feasible_df.empty else None
    opt_ipc = best_feasible_row["ipc"] if best_feasible_row is not None else 0.0
    opt_cfg = (int(best_feasible_row["issue_width"]), int(best_feasible_row["rob_size"]), int(best_feasible_row["l1d_mshrs"]),
               best_feasible_row["l1d_size"], int(best_feasible_row["l1d_assoc"]), best_feasible_row["l2_size"]) if best_feasible_row is not None else ()

    schema_cls, parser_fn, desc = BASELINES[baseline_name]

    # Neutral prompt context
    prompt_history = [
        f"You are an expert computer architect optimizing a heterogeneous processor design under a strict hardware area/cost budget.",
        f"Target Workload Identifier: {workload_name}.",
        f"HARDWARE BUDGET CONSTRAINT: Total Cost <= {budget_kb} KB equivalent.",
        f"Cost formula: Cost ≈ Core_Cost(issue_width*50 + rob_size*1.5 + mshrs*5) + Memory_Cost(L1_Cap*(1 + 0.1*L1_Assoc/8) + L2_Cap).",
        f"Goal: Maximize IPC within {max_steps} steps. Any configuration exceeding the budget of {budget_kb} KB is INFEASIBLE and cannot be selected as the final design.",
        f"Representation format: {desc}.",
        f"Available design choices:",
        f"- Core Parameters: issue_width: [2, 4, 8], rob_size: [32, 64, 128], l1d_mshrs: [2, 4, 8]",
        f"- Memory Parameters: l1d_size: [16kB, 32kB, 64kB], l1d_assoc: [2, 4, 8], l2_size: [512kB, 1MB, 2MB]",
        f"Initial baseline config: (IW:2w, ROB:32, MSHR:2, L1:16kB/2-way, L2:512kB) (Cost: {initial_cost} KB). Initial IPC: {initial_ipc:.4f}.",
        f"Propose 1 configuration per step."
    ]

    history_log = []
    prev_cfg = initial_cfg

    for step in range(1, max_steps + 1):
        full_prompt = "\n".join(prompt_history) + f"\n\nStep {step}/{max_steps}: Propose the next architecture respecting the <= {budget_kb} KB budget."

        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        max_retries = 5
        response = None
        last_error = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema_cls,
                        temperature=0.2,
                        seed=trial_seed + step
                    ),
                )
                break
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    match = re.search(r"retry in ([0-9.]+)s", err_str)
                    wait_time = float(match.group(1)) + 2.0 if match else max(35.0, 20.0 * (attempt + 1))
                    time.sleep(wait_time)
                else:
                    raise e

        if response is None:
            raise RuntimeError(f"Failed to obtain response after {max_retries} attempts: {last_error}")

        pacing_delay = float(os.environ.get("GEMINI_PACING_DELAY", "1.5"))
        if pacing_delay > 0:
            time.sleep(pacing_delay)

        proposal = json.loads(response.text)
        curr_cfg = parser_fn(proposal)
        iw, rob, mshr, l1s, l1a, l2s = curr_cfg
        ipc = oracle_lookup_729(iw, rob, mshr, l1s, l1a, l2s)
        cost = compute_cost_729(iw, rob, mshr, l1s, l1a, l2s)
        is_feasible = (cost <= budget_kb)
        feasibility_status = "FEASIBLE" if is_feasible else f"INFEASIBLE (Exceeds budget by +{cost - budget_kb:.1f} KB)"

        # Empirical variance-based sensitivity threshold (<5% is inactive)
        # Alpha: issue_width active (148%); all others inactive (<5%)
        # Beta: l2_size active (148%); all others inactive (<5%)
        # Gamma: l2_size (126%) and l1d_mshrs (14.3%) active; all others inactive (<5%)
        iw_mut = (curr_cfg[0] != prev_cfg[0])
        rob_mut = (curr_cfg[1] != prev_cfg[1])
        mshr_mut = (curr_cfg[2] != prev_cfg[2])
        l1s_mut = (curr_cfg[3] != prev_cfg[3])
        l1a_mut = (curr_cfg[4] != prev_cfg[4])
        l2s_mut = (curr_cfg[5] != prev_cfg[5])

        if workload_name == "Workload_Alpha":
            act_muts = (1 if iw_mut else 0)
            inact_muts = (1 if rob_mut else 0) + (1 if mshr_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0) + (1 if l2s_mut else 0)
        elif workload_name == "Workload_Beta":
            act_muts = (1 if l2s_mut else 0)
            inact_muts = (1 if iw_mut else 0) + (1 if rob_mut else 0) + (1 if mshr_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0)
        else: # Workload_Gamma
            act_muts = (1 if l2s_mut else 0) + (1 if mshr_mut else 0)
            inact_muts = (1 if iw_mut else 0) + (1 if rob_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0)

        history_log.append({
            "step": step,
            "config": curr_cfg,
            "ipc": ipc,
            "cost": cost,
            "feasible": is_feasible,
            "active_mutations_count": act_muts,
            "inactive_mutations_count": inact_muts,
            "reasoning": proposal.get("reasoning", "")
        })
        prev_cfg = curr_cfg

        prompt_history.append(
            f"Step {step} Evaluated: Config=(IW:{iw}w, ROB:{rob}, MSHR:{mshr}, L1:{l1s}/{l1a}w, L2:{l2s}) -> IPC: {ipc:.4f}, Cost: {cost} KB [{feasibility_status}]"
        )
        print(
            f"  [{workload_name} | Seed:{trial_seed} | {baseline_name}] Step {step:02d}: (IW:{iw}w, ROB:{rob}, MSHR:{mshr}, L1:{l1s}/{l1a}w, L2:{l2s}) -> IPC: {ipc:.4f}, Cost: {cost} KB [{feasibility_status}] (Opt: {opt_ipc:.4f})"
        )

    return history_log, opt_ipc, opt_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 729-Point Heterogeneous Representation Trials")
    parser.add_argument("--budget", type=int, default=1500, help="Budget constraint in KB")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2026], help="Random seeds")
    parser.add_argument("--steps", type=int, default=10, help="Steps per trial")
    parser.add_argument("--out", type=str, default="heterogeneous_trials_results.json", help="Output JSON file")
    args = parser.parse_args()

    client = genai.Client()

    workloads = [
        ("Workload_Alpha", "oracle_729_compute.csv"),
        ("Workload_Beta", "oracle_729_latency.csv"),
        ("Workload_Gamma", "oracle_729_concurrency.csv"),
    ]

    all_results = {}
    print(f"Beginning 729-Point Heterogeneous Trials (Budget: {args.budget} KB, Seeds: {args.seeds})...")

    for wl_name, oracle_file in workloads:
        all_results[wl_name] = {}
        for b_name in BASELINES.keys():
            all_results[wl_name][b_name] = []
            for seed in args.seeds:
                print(f"\n--- Running {b_name} on {wl_name} (Seed: {seed}, Budget: {args.budget} KB) ---")
                history, opt_ipc, opt_cfg = run_trial(
                    client, b_name, wl_name, oracle_file, budget_kb=args.budget, trial_seed=seed, max_steps=args.steps
                )
                all_results[wl_name][b_name].append({
                    "seed": seed,
                    "history": history,
                    "constrained_optimal_ipc": opt_ipc,
                    "constrained_optimal_cfg": opt_cfg
                })

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n==================================================================")
    print(f"All heterogeneous trials completed! Saved results to {args.out}")
    print(f"==================================================================")
