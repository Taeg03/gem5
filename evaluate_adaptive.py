import argparse
import json
import os
import re
import time
from typing import Literal, Optional, Tuple, Dict, Any

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
from scipy import stats
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

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

def oracle_lookup_729(iw: int, rob: int, mshr: int, l1s: str, l1a: int, l2s: str) -> Optional[float]:
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
# 2. Schemas for Baseline_B (Flat) and Baseline_D (Adaptive)
# ---------------------------------------------------------

class BaselineBSchema(BaseModel):
    issue_width: Literal["2", "4", "8"]
    rob_size: Literal["32", "64", "128"]
    l1d_mshrs: Literal["2", "4", "8"]
    l1d_capacity_kb: Literal["16", "32", "64"]
    l1d_associativity: Literal["2", "4", "8"]
    l2_capacity_kb: Literal["512", "1024", "2048"]
    reasoning: str


class CoreSubsystemParams(BaseModel):
    issue_width: Literal["2", "4", "8"]
    rob_size: Literal["32", "64", "128"]
    l1d_mshrs: Literal["2", "4", "8"]


class MemorySubsystemParams(BaseModel):
    l1d_capacity_kb: Literal["16", "32", "64"]
    l1d_associativity: Literal["2", "4", "8"]
    l2_capacity_kb: Literal["512", "1024", "2048"]


class BaselineDSchema(BaseModel):
    target_subsystem: Literal["core_execution", "memory_hierarchy", "both"]
    core_parameters: Optional[CoreSubsystemParams] = None
    memory_parameters: Optional[MemorySubsystemParams] = None
    reasoning: str


SIZE_MAP = {
    "16": "16kB", "32": "32kB", "64": "64kB", "512": "512kB", "1024": "1MB", "2048": "2MB",
    16: "16kB", 32: "32kB", 64: "64kB", 512: "512kB", 1024: "1MB", 2048: "2MB",
}


# ---------------------------------------------------------
# 3. Trial Runners
# ---------------------------------------------------------

def run_baseline_b_trial(
    client: genai.Client,
    workload_name: str,
    oracle_path: str,
    budget_kb: float,
    trial_seed: int,
    max_steps: int = 10,
    model_name: str = "gemini-3.5-flash-lite"
) -> Tuple[Dict[str, Any], float, Tuple]:
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

    prompt_history = [
        f"You are an expert computer architect optimizing a processor design under a strict hardware budget.",
        f"Target Workload Identifier: {workload_name}.",
        f"HARDWARE BUDGET CONSTRAINT: Total Cost <= {budget_kb} KB equivalent.",
        f"Cost formula: Cost ≈ Core_Cost(issue_width*50 + rob_size*1.5 + mshrs*5) + Memory_Cost(L1_Cap*(1 + 0.1*L1_Assoc/8) + L2_Cap).",
        f"Goal: Maximize IPC within {max_steps} steps. Any configuration exceeding the budget of {budget_kb} KB is INFEASIBLE and cannot be selected as the final design.",
        f"Representation format: Flat microarchitectural parameters across all subsystems.",
        f"Available design choices:",
        f"- Core Parameters: issue_width: [2, 4, 8], rob_size: [32, 64, 128], l1d_mshrs: [2, 4, 8]",
        f"- Memory Parameters: l1d_size: [16kB, 32kB, 64kB], l1d_assoc: [2, 4, 8], l2_size: [512kB, 1MB, 2MB]",
        f"Initial baseline config: (IW:2w, ROB:32, MSHR:2, L1:16kB/2-way, L2:512kB) (Cost: {initial_cost} KB). Initial IPC: {initial_ipc:.4f}.",
        f"Propose 1 configuration per step specifying all parameters."
    ]

    history_log = []
    prev_cfg = initial_cfg
    prompt_tokens_total = 0
    completion_tokens_total = 0

    for step in range(1, max_steps + 1):
        full_prompt = "\n".join(prompt_history) + f"\n\nStep {step}/{max_steps}: Propose the next architecture respecting the <= {budget_kb} KB budget."

        max_retries = 5
        response = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=BaselineBSchema,
                        temperature=0.2,
                        seed=trial_seed + step
                    ),
                )
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    match = re.search(r"retry in ([0-9.]+)s", err_str)
                    wait_time = float(match.group(1)) + 2.0 if match else max(35.0, 20.0 * (attempt + 1))
                    time.sleep(wait_time)
                else:
                    raise e

        if response is None:
            raise RuntimeError("API failed after retries.")

        if response.usage_metadata:
            prompt_tokens_total += response.usage_metadata.prompt_token_count or 0
            completion_tokens_total += response.usage_metadata.candidates_token_count or 0

        pacing_delay = float(os.environ.get("GEMINI_PACING_DELAY", "1.5"))
        if pacing_delay > 0:
            time.sleep(pacing_delay)

        proposal = json.loads(response.text)
        iw = int(proposal["issue_width"])
        rob = int(proposal["rob_size"])
        mshr = int(proposal["l1d_mshrs"])
        l1s = SIZE_MAP[str(proposal["l1d_capacity_kb"])]
        l1a = int(proposal["l1d_associativity"])
        l2s = SIZE_MAP[str(proposal["l2_capacity_kb"])]
        curr_cfg = (iw, rob, mshr, l1s, l1a, l2s)

        ipc = oracle_lookup_729(iw, rob, mshr, l1s, l1a, l2s)
        cost = compute_cost_729(iw, rob, mshr, l1s, l1a, l2s)
        is_feasible = (cost <= budget_kb)
        feasibility_status = "FEASIBLE" if is_feasible else f"INFEASIBLE (Exceeds budget by +{cost - budget_kb:.1f} KB)"

        # Calculate mutations
        iw_mut = (curr_cfg[0] != prev_cfg[0])
        rob_mut = (curr_cfg[1] != prev_cfg[1])
        mshr_mut = (curr_cfg[2] != prev_cfg[2])
        l1s_mut = (curr_cfg[3] != prev_cfg[3])
        l1a_mut = (curr_cfg[4] != prev_cfg[4])
        l2s_mut = (curr_cfg[5] != prev_cfg[5])

        if workload_name == "Workload_Alpha": # issue_width active (148%)
            act_muts = (1 if iw_mut else 0)
            inact_muts = (1 if rob_mut else 0) + (1 if mshr_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0) + (1 if l2s_mut else 0)
        elif workload_name == "Workload_Beta": # l2_size active (148%)
            act_muts = (1 if l2s_mut else 0)
            inact_muts = (1 if iw_mut else 0) + (1 if rob_mut else 0) + (1 if mshr_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0)
        else: # Workload_Gamma (l2_size + mshrs active)
            act_muts = (1 if l2s_mut else 0) + (1 if mshr_mut else 0)
            inact_muts = (1 if iw_mut else 0) + (1 if rob_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0)

        history_log.append({
            "step": step,
            "target_subsystem": "all",
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
            f"  [{workload_name} | Seed:{trial_seed} | Baseline_B] Step {step:02d}: (IW:{iw}w, ROB:{rob}, MSHR:{mshr}, L1:{l1s}/{l1a}w, L2:{l2s}) -> IPC: {ipc:.4f}, Cost: {cost} KB [{feasibility_status}] (Opt: {opt_ipc:.4f})"
        )

    trial_summary = {
        "workload": workload_name,
        "baseline": "Baseline_B",
        "seed": trial_seed,
        "history": history_log,
        "prompt_tokens": prompt_tokens_total,
        "completion_tokens": completion_tokens_total
    }
    return trial_summary, opt_ipc, opt_cfg


def run_baseline_d_trial(
    client: genai.Client,
    workload_name: str,
    oracle_path: str,
    budget_kb: float,
    trial_seed: int,
    max_steps: int = 10,
    model_name: str = "gemini-3.5-flash-lite"
) -> Tuple[Dict[str, Any], float, Tuple]:
    load_oracle(oracle_path)
    
    # State Machine: Tracks active state of both subsystems across steps
    current_state = {
        "core_execution": {"issue_width": 2, "rob_size": 32, "l1d_mshrs": 2},
        "memory_hierarchy": {"l1d_size": "16kB", "l1d_assoc": 2, "l2_size": "512kB"}
    }
    
    initial_cfg = (
        current_state["core_execution"]["issue_width"],
        current_state["core_execution"]["rob_size"],
        current_state["core_execution"]["l1d_mshrs"],
        current_state["memory_hierarchy"]["l1d_size"],
        current_state["memory_hierarchy"]["l1d_assoc"],
        current_state["memory_hierarchy"]["l2_size"]
    )
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

    prompt_history = [
        f"You are an expert computer architect with ADAPTIVE SUBSYSTEM MUTATION capabilities optimizing a processor design under a strict hardware budget.",
        f"Target Workload Identifier: {workload_name}.",
        f"HARDWARE BUDGET CONSTRAINT: Total Cost <= {budget_kb} KB equivalent.",
        f"Cost formula: Cost ≈ Core_Cost(issue_width*50 + rob_size*1.5 + mshrs*5) + Memory_Cost(L1_Cap*(1 + 0.1*L1_Assoc/8) + L2_Cap).",
        f"Goal: Maximize IPC within {max_steps} steps. Any configuration exceeding the budget of {budget_kb} KB is INFEASIBLE and cannot be selected as the final design.",
        f"ADAPTIVE ABSTRACTION MECHANISM:",
        f"You can choose WHICH subsystem to mutate at each step to avoid wasting exploration budget on inactive subsystems:",
        f"- Target 'core_execution': Mutate issue_width [2, 4, 8], rob_size [32, 64, 128], and l1d_mshrs [2, 4, 8]. The memory hierarchy remains FROZEN at its current state.",
        f"- Target 'memory_hierarchy': Mutate l1d_capacity_kb [16, 32, 64], l1d_associativity [2, 4, 8], and l2_capacity_kb [512, 1024, 2048]. Core execution remains FROZEN at its current state.",
        f"- Target 'both': Mutate all parameters simultaneously.",
        f"Initial baseline config: Core=(IW:2w, ROB:32, MSHR:2), Memory=(L1:16kB/2-way, L2:512kB) (Cost: {initial_cost} KB). Initial IPC: {initial_ipc:.4f}."
    ]

    history_log = []
    prev_cfg = initial_cfg
    prompt_tokens_total = 0
    completion_tokens_total = 0

    for step in range(1, max_steps + 1):
        # Provide current frozen state explicitly
        c_state = current_state["core_execution"]
        m_state = current_state["memory_hierarchy"]
        curr_cost = compute_cost_729(c_state["issue_width"], c_state["rob_size"], c_state["l1d_mshrs"],
                                     m_state["l1d_size"], m_state["l1d_assoc"], m_state["l2_size"])

        state_str = f"Current System State: Core=(IW:{c_state['issue_width']}w, ROB:{c_state['rob_size']}, MSHR:{c_state['l1d_mshrs']}), Memory=(L1:{m_state['l1d_size']}/{m_state['l1d_assoc']}w, L2:{m_state['l2_size']}) [Current Cost: {curr_cost:.1f} KB]."

        full_prompt = "\n".join(prompt_history) + f"\n\n{state_str}\nStep {step}/{max_steps}: Select target_subsystem ('core_execution', 'memory_hierarchy', or 'both') and provide mutated parameters."

        max_retries = 5
        response = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=BaselineDSchema,
                        temperature=0.2,
                        seed=trial_seed + step
                    ),
                )
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    match = re.search(r"retry in ([0-9.]+)s", err_str)
                    wait_time = float(match.group(1)) + 2.0 if match else max(35.0, 20.0 * (attempt + 1))
                    time.sleep(wait_time)
                else:
                    raise e

        if response is None:
            raise RuntimeError("API failed after retries.")

        if response.usage_metadata:
            prompt_tokens_total += response.usage_metadata.prompt_token_count or 0
            completion_tokens_total += response.usage_metadata.candidates_token_count or 0

        pacing_delay = float(os.environ.get("GEMINI_PACING_DELAY", "1.5"))
        if pacing_delay > 0:
            time.sleep(pacing_delay)

        proposal = json.loads(response.text)
        target = proposal.get("target_subsystem", "both")

        # Selective state update
        if target in ["core_execution", "both"] and proposal.get("core_parameters"):
            cp = proposal["core_parameters"]
            current_state["core_execution"]["issue_width"] = int(cp["issue_width"])
            current_state["core_execution"]["rob_size"] = int(cp["rob_size"])
            current_state["core_execution"]["l1d_mshrs"] = int(cp["l1d_mshrs"])

        if target in ["memory_hierarchy", "both"] and proposal.get("memory_parameters"):
            mp = proposal["memory_parameters"]
            current_state["memory_hierarchy"]["l1d_size"] = SIZE_MAP[str(mp["l1d_capacity_kb"])]
            current_state["memory_hierarchy"]["l1d_assoc"] = int(mp["l1d_associativity"])
            current_state["memory_hierarchy"]["l2_size"] = SIZE_MAP[str(mp["l2_capacity_kb"])]

        # Reconstructed 6D configuration
        c_state = current_state["core_execution"]
        m_state = current_state["memory_hierarchy"]
        curr_cfg = (
            c_state["issue_width"], c_state["rob_size"], c_state["l1d_mshrs"],
            m_state["l1d_size"], m_state["l1d_assoc"], m_state["l2_size"]
        )

        ipc = oracle_lookup_729(*curr_cfg)
        cost = compute_cost_729(*curr_cfg)
        is_feasible = (cost <= budget_kb)
        feasibility_status = "FEASIBLE" if is_feasible else f"INFEASIBLE (Exceeds budget by +{cost - budget_kb:.1f} KB)"

        # Mutation tracking
        iw_mut = (curr_cfg[0] != prev_cfg[0])
        rob_mut = (curr_cfg[1] != prev_cfg[1])
        mshr_mut = (curr_cfg[2] != prev_cfg[2])
        l1s_mut = (curr_cfg[3] != prev_cfg[3])
        l1a_mut = (curr_cfg[4] != prev_cfg[4])
        l2s_mut = (curr_cfg[5] != prev_cfg[5])

        if workload_name == "Workload_Alpha": # issue_width active (148%)
            act_muts = (1 if iw_mut else 0)
            inact_muts = (1 if rob_mut else 0) + (1 if mshr_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0) + (1 if l2s_mut else 0)
        elif workload_name == "Workload_Beta": # l2_size active (148%)
            act_muts = (1 if l2s_mut else 0)
            inact_muts = (1 if iw_mut else 0) + (1 if rob_mut else 0) + (1 if mshr_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0)
        else: # Workload_Gamma (l2_size + mshrs active)
            act_muts = (1 if l2s_mut else 0) + (1 if mshr_mut else 0)
            inact_muts = (1 if iw_mut else 0) + (1 if rob_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0)

        history_log.append({
            "step": step,
            "target_subsystem": target,
            "config": curr_cfg,
            "ipc": ipc,
            "cost": cost,
            "feasible": is_feasible,
            "active_mutations_count": act_muts,
            "inactive_mutations_count": inact_muts,
            "reasoning": proposal.get("reasoning", "")
        })
        prev_cfg = curr_cfg

        iw, rob, mshr, l1s, l1a, l2s = curr_cfg
        prompt_history.append(
            f"Step {step} Evaluated [Action: {target}]: Config=(IW:{iw}w, ROB:{rob}, MSHR:{mshr}, L1:{l1s}/{l1a}w, L2:{l2s}) -> IPC: {ipc:.4f}, Cost: {cost} KB [{feasibility_status}]"
        )
        print(
            f"  [{workload_name} | Seed:{trial_seed} | Baseline_D | Target: {target:<16}] Step {step:02d}: (IW:{iw}w, ROB:{rob}, MSHR:{mshr}, L1:{l1s}/{l1a}w, L2:{l2s}) -> IPC: {ipc:.4f}, Cost: {cost} KB [{feasibility_status}] (Opt: {opt_ipc:.4f})"
        )

    trial_summary = {
        "workload": workload_name,
        "baseline": "Baseline_D",
        "seed": trial_seed,
        "history": history_log,
        "prompt_tokens": prompt_tokens_total,
        "completion_tokens": completion_tokens_total
    }
    return trial_summary, opt_ipc, opt_cfg


# ---------------------------------------------------------
# 4. Statistical Analysis & Reporting Function
# ---------------------------------------------------------

def analyze_and_report_results(results_data: Dict[str, Any], output_md_path: Optional[str] = None):
    rows = []
    for wl_name, baselines in results_data.items():
        for b_name, trials in baselines.items():
            for t in trials:
                history = t["history"]
                opt_ipc = t["constrained_optimal_ipc"]
                feasible_steps = [s for s in history if s["feasible"]]
                infeas_steps = [s for s in history if not s["feasible"]]
                b_ipc = max([s["ipc"] for s in feasible_steps]) if feasible_steps else 0.0

                s98 = 11
                for s in history:
                    if s["feasible"] and s["ipc"] >= 0.98 * opt_ipc:
                        s98 = s["step"]
                        break

                act_muts = sum([s["active_mutations_count"] for s in history])
                inact_muts = sum([s["inactive_mutations_count"] for s in history])
                tot_muts = act_muts + inact_muts
                rel_ratio = (act_muts / tot_muts * 100) if tot_muts > 0 else 0.0

                rows.append({
                    "Workload": wl_name,
                    "Baseline": b_name,
                    "Seed": t["seed"],
                    "Best_IPC": b_ipc,
                    "Steps_to_Opt": s98,
                    "Rel_Mutation_Ratio_Pct": rel_ratio,
                    "Active_Mutations": act_muts,
                    "Inactive_Mutations": inact_muts,
                    "Infeasible_Proposals": len(infeas_steps),
                    "Prompt_Tokens": t.get("prompt_tokens", 0),
                    "Completion_Tokens": t.get("completion_tokens", 0)
                })

    df = pd.DataFrame(rows)
    
    print("\n==================================================================================================")
    print("                    PHASE 3: ADAPTIVE ABSTRACTION (BASELINE_B vs BASELINE_D)                     ")
    print("==================================================================================================")
    
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 1000)
    
    table = df.groupby(["Workload", "Baseline"])[
        ["Best_IPC", "Steps_to_Opt", "Rel_Mutation_Ratio_Pct", "Active_Mutations", "Inactive_Mutations", "Infeasible_Proposals", "Completion_Tokens"]
    ].agg(["mean", "std"]).round(2)
    print("\n--- Detailed Breakdown per Workload & Baseline (Mean +/- Std across 3 Seeds) ---")
    print(table)
    
    print("\n--- Overall Baseline Aggregates Across All 18 Trials ---")
    overall = df.groupby("Baseline")[
        ["Best_IPC", "Steps_to_Opt", "Rel_Mutation_Ratio_Pct", "Active_Mutations", "Inactive_Mutations", "Infeasible_Proposals", "Prompt_Tokens", "Completion_Tokens"]
    ].agg(["mean", "std"]).round(2)
    print(overall)
    
    print("\n--- Automated Paired Hypothesis Testing (Baseline_B vs. Baseline_D, N=9 matched pairs) ---")
    df_bd = df[df["Baseline"].isin(["Baseline_B", "Baseline_D"])]
    metrics_to_test = [
        ("Steps_to_Opt", "Steps to >=98% Optimum"),
        ("Rel_Mutation_Ratio_Pct", "Relevant Mutation Ratio (%)"),
        ("Inactive_Mutations", "Inactive Parameter Mutations"),
        ("Infeasible_Proposals", "Infeasible Proposals / Budget Violations"),
        ("Completion_Tokens", "Completion Token Consumption"),
        ("Best_IPC", "Best Feasible IPC")
    ]
    
    stat_summary = []
    for col, display_name in metrics_to_test:
        vals_b = df_bd[df_bd["Baseline"] == "Baseline_B"][col].values
        vals_d = df_bd[df_bd["Baseline"] == "Baseline_D"][col].values
        
        diff = np.mean(vals_d) - np.mean(vals_b)
        t_stat, p_val = stats.ttest_rel(vals_b, vals_d)
        w_stat, w_pval = stats.wilcoxon(vals_b, vals_d) if not np.all(vals_b == vals_d) else (0, 1.0)
        
        print(f"{display_name:<36}: B={np.mean(vals_b):.2f} +/- {np.std(vals_b):.2f} | D={np.mean(vals_d):.2f} +/- {np.std(vals_d):.2f} | Diff={diff:+6.2f} | t={t_stat:6.3f}, p={p_val:.4f} (Wilcoxon p={w_pval:.4f})")
        stat_summary.append({
            "metric": display_name,
            "mean_b": np.mean(vals_b),
            "std_b": np.std(vals_b),
            "mean_d": np.mean(vals_d),
            "std_d": np.std(vals_d),
            "diff": diff,
            "t_stat": t_stat,
            "p_val": p_val,
            "w_pval": w_pval
        })

    return df, stat_summary


# ---------------------------------------------------------
# 5. Main Execution Script
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Adaptive Abstraction (Baseline_B vs Baseline_D)")
    parser.add_argument("--budget", type=int, default=1500, help="Hardware budget constraint in KB")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2026], help="Random seeds")
    parser.add_argument("--steps", type=int, default=10, help="Steps per trial")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite", help="GenAI model to use")
    parser.add_argument("--dry-run", action="store_true", help="Run a single test trial on Workload_Alpha Seed 42")
    parser.add_argument("--out", type=str, default="adaptive_eval_results.json", help="Output JSON file")
    args = parser.parse_args()

    client = genai.Client()

    workloads = [
        ("Workload_Alpha", "oracle_729_compute.csv"),
        ("Workload_Beta", "oracle_729_latency.csv"),
        ("Workload_Gamma", "oracle_729_concurrency.csv"),
    ]

    if args.dry_run:
        print("==================================================================")
        print("DRY RUN: Testing Baseline_D State Persistence on Workload_Alpha (Seed 42)")
        print("==================================================================")
        trial_summary, opt_ipc, opt_cfg = run_baseline_d_trial(
            client, "Workload_Alpha", "oracle_729_compute.csv", budget_kb=args.budget, trial_seed=42, max_steps=4, model_name=args.model
        )
        print("\nDry Run History Log:")
        for s in trial_summary["history"]:
            print(f"  Step {s['step']}: Target={s['target_subsystem']:<16} | Config={s['config']} | Cost={s['cost']:.1f} KB | IPC={s['ipc']:.4f} | Reason: {s['reasoning'][:80]}...")
        print("\nDry run completed successfully!")
        exit(0)

    all_results = {}
    print(f"Beginning Phase 3 Adaptive Abstraction Study (Budget: {args.budget} KB, Seeds: {args.seeds}, Model: {args.model})...")

    for wl_name, oracle_file in workloads:
        all_results[wl_name] = {"Baseline_B": [], "Baseline_D": []}
        
        # 1. Baseline_B Trials
        for seed in args.seeds:
            print(f"\n--- Running Baseline_B on {wl_name} (Seed: {seed}) ---")
            trial_summary, opt_ipc, opt_cfg = run_baseline_b_trial(
                client, wl_name, oracle_file, budget_kb=args.budget, trial_seed=seed, max_steps=args.steps, model_name=args.model
            )
            trial_summary["constrained_optimal_ipc"] = opt_ipc
            trial_summary["constrained_optimal_cfg"] = opt_cfg
            all_results[wl_name]["Baseline_B"].append(trial_summary)

        # 2. Baseline_D Trials
        for seed in args.seeds:
            print(f"\n--- Running Baseline_D on {wl_name} (Seed: {seed}) ---")
            trial_summary, opt_ipc, opt_cfg = run_baseline_d_trial(
                client, wl_name, oracle_file, budget_kb=args.budget, trial_seed=seed, max_steps=args.steps, model_name=args.model
            )
            trial_summary["constrained_optimal_ipc"] = opt_ipc
            trial_summary["constrained_optimal_cfg"] = opt_cfg
            all_results[wl_name]["Baseline_D"].append(trial_summary)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n==================================================================")
    print(f"All 18 trials completed! Saved results to {args.out}")
    print(f"==================================================================")

    # Perform automated statistical analysis
    analyze_and_report_results(all_results)
