"""
evaluate_optional_l2.py

Benchmark Harness for 972-Point Optional-L2 Design Space:
Comparing:
1. Baseline_Flat_Cartesian: Static flat schema (l2_capacity_kb is always emitted)
2. ADSG_Conditional: Semantic conditional schema (l2_capacity_kb is pruned when has_l2=False)

Execution Matrix:
- 3 Workloads: Workload_Alpha (Compute), Workload_Beta (Latency), Workload_Gamma (Concurrency)
- 3 Random Seeds: 42, 100, 2026
- 2 Baselines = 18 Total Trials (10 steps per trial)
"""

import argparse
import json
import os
import re
import time
from typing import Literal, Optional, Tuple, Dict, Any, List, Union

if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v.strip("\"'")

import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

BUDGET_KB_DEFAULT = 1500.0

SIZE_MAP = {
    "16": "16kB", "32": "32kB", "64": "64kB", "512": "512kB", "1024": "1MB", "2048": "2MB",
    16: "16kB", 32: "32kB", 64: "64kB", 512: "512kB", 1024: "1MB", 2048: "2MB",
    "16kB": "16kB", "32kB": "32kB", "64kB": "64kB", "512kB": "512kB", "1MB": "1MB", "2MB": "2MB"
}

# Load Master Oracle
ORACLE_DF = pd.read_csv("oracle_optl2_master.csv")
ORACLE_DICT = {}
for _, r in ORACLE_DF.iterrows():
    k = (
        int(r["issue_width"]),
        int(r["rob_size"]),
        int(r["l1d_mshrs"]),
        str(r["l1d_size"]),
        int(r["l1d_assoc"]),
        bool(r["has_l2"]),
        str(r["l2_size"])
    )
    ORACLE_DICT[k] = {
        "cost": float(r["cost"]),
        "is_feasible": bool(r["is_feasible"]),
        "ipc_compute": float(r["ipc_compute"]),
        "ipc_latency": float(r["ipc_latency"]),
        "ipc_concurrency": float(r["ipc_concurrency"])
    }


# ==============================================================================
# 1. Action Schemas
# ==============================================================================

# Flat Cartesian: l2_capacity_kb is ALWAYS mandatory, even if direct_l1_no_l2
class FlatCartesianSchema(BaseModel):
    architecture_type: Literal["two_level_with_l2", "direct_l1_no_l2"]
    issue_width: Literal["2", "4", "8"]
    rob_size: Literal["32", "64", "128"]
    l1d_mshrs: Literal["2", "4", "8"]
    l1d_capacity_kb: Literal["16", "32", "64"]
    l1d_associativity: Literal["2", "4", "8"]
    l2_capacity_kb: Literal["512", "1024", "2048"]
    reasoning: str


# ADSG Conditional: Dynamic Schema where l2_capacity_kb only exists under with_l2
class ArchitectureWithL2(BaseModel):
    architecture_type: Literal["two_level_with_l2"]
    issue_width: Literal["2", "4", "8"]
    rob_size: Literal["32", "64", "128"]
    l1d_mshrs: Literal["2", "4", "8"]
    l1d_capacity_kb: Literal["16", "32", "64"]
    l1d_associativity: Literal["2", "4", "8"]
    l2_capacity_kb: Literal["512", "1024", "2048"]
    reasoning: str


class ArchitectureNoL2(BaseModel):
    architecture_type: Literal["direct_l1_no_l2"]
    issue_width: Literal["2", "4", "8"]
    rob_size: Literal["32", "64", "128"]
    l1d_mshrs: Literal["2", "4", "8"]
    l1d_capacity_kb: Literal["16", "32", "64"]
    l1d_associativity: Literal["2", "4", "8"]
    reasoning: str


ADSGProposalSchema = Union[ArchitectureWithL2, ArchitectureNoL2]


# ==============================================================================
# 2. Helpers
# ==============================================================================

def get_optimum(workload_key: str, budget_kb: float = 1500.0) -> Tuple[float, Tuple]:
    feasible = ORACLE_DF[ORACLE_DF["cost"] <= budget_kb]
    best_row = feasible.sort_values(by=f"ipc_{workload_key}", ascending=False).iloc[0]
    opt_cfg = (
        int(best_row["issue_width"]),
        int(best_row["rob_size"]),
        int(best_row["l1d_mshrs"]),
        str(best_row["l1d_size"]),
        int(best_row["l1d_assoc"]),
        bool(best_row["has_l2"]),
        str(best_row["l2_size"])
    )
    return float(best_row[f"ipc_{workload_key}"]), opt_cfg


def calculate_mutations(curr: Tuple, prev: Tuple, workload_key: str, flat_l2_mut: bool = False) -> Tuple[int, int]:
    # curr / prev: (iw, rob, mshr, l1s, l1a, has_l2, l2s)
    iw_mut = (curr[0] != prev[0])
    rob_mut = (curr[1] != prev[1])
    mshr_mut = (curr[2] != prev[2])
    l1s_mut = (curr[3] != prev[3])
    l1a_mut = (curr[4] != prev[4])
    has_l2_mut = (curr[5] != prev[5])
    # For l2s, if has_l2 is True, check change. If has_l2 is False and flat emitted a change, it's inactive!
    l2s_mut = (curr[6] != prev[6]) if curr[5] else False

    if workload_key == "compute":
        # Active: issue_width, has_l2
        act = (1 if iw_mut else 0) + (1 if has_l2_mut else 0)
        inact = (1 if rob_mut else 0) + (1 if mshr_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0) + (1 if l2s_mut else 0) + (1 if flat_l2_mut else 0)
    elif workload_key == "latency":
        # Active: has_l2, l2_size (when has_l2)
        act = (1 if has_l2_mut else 0) + (1 if l2s_mut else 0)
        inact = (1 if iw_mut else 0) + (1 if rob_mut else 0) + (1 if mshr_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0) + (1 if flat_l2_mut else 0)
    else:  # concurrency
        # Active: has_l2, l1d_mshrs
        act = (1 if has_l2_mut else 0) + (1 if mshr_mut else 0)
        inact = (1 if iw_mut else 0) + (1 if rob_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0) + (1 if l2s_mut else 0) + (1 if flat_l2_mut else 0)

    return act, inact


def call_llm_with_retry(client: genai.Client, model: str, prompt: str, schema: Any, seed: int, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.2,
                    seed=seed
                ),
            )
            return response
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                match = re.search(r"retry in ([0-9.]+)s", err_str)
                wait_time = float(match.group(1)) + 2.0 if match else max(35.0, 20.0 * (attempt + 1))
                print(f"    Rate limited (429), waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("API failed after max retries.")


# ==============================================================================
# 3. Runners
# ==============================================================================

def run_flat_cartesian(
    client: genai.Client,
    workload_key: str,
    workload_display: str,
    budget_kb: float,
    seed: int,
    max_steps: int = 10,
    model: str = "gemini-3.5-flash-lite"
) -> Dict[str, Any]:
    opt_ipc, opt_cfg = get_optimum(workload_key, budget_kb)

    # Initial baseline config: with L2
    initial_cfg = (2, 32, 2, "16kB", 2, True, "512kB")
    init_data = ORACLE_DICT[initial_cfg]
    initial_ipc = init_data[f"ipc_{workload_key}"]
    initial_cost = init_data["cost"]
    init_slack = round(budget_kb - initial_cost, 1)

    prompt_history = [
        f"You are an expert computer architect optimizing a processor design under a strict hardware budget.",
        f"Target Workload: {workload_display}.",
        f"HARDWARE BUDGET CONSTRAINT: Total Cost <= {budget_kb:.1f} KB equivalent.",
        f"Cost Formula: Total Cost = Core_Cost + Memory_Cost",
        f"  - Core_Cost = (issue_width * 50) + (rob_size * 1.5) + (l1d_mshrs * 5)",
        f"  - Memory_Cost = (l1d_size_kb * (1 + 0.1 * l1d_assoc/8)) + (l2_size_kb if architecture_type == 'two_level_with_l2' else 0)",
        f"Memory Hierarchy Options:",
        f"  - 'two_level_with_l2': Dedicated L2 cache. l2_capacity_kb adds directly to cost.",
        f"  - 'direct_l1_no_l2': Direct L1-to-DRAM memory bus. Saves L2 cost entirely.",
        f"Available design choices:",
        f"  - architecture_type: ['two_level_with_l2', 'direct_l1_no_l2']",
        f"  - Core: issue_width: [2, 4, 8], rob_size: [32, 64, 128], l1d_mshrs: [2, 4, 8]",
        f"  - L1 Cache: l1d_capacity_kb: [16, 32, 64], l1d_associativity: [2, 4, 8]",
        f"  - L2 Cache: l2_capacity_kb: [512, 1024, 2048] (Note: in this flat schema, you must declare l2_capacity_kb even if direct_l1_no_l2).",
        f"Initial config: (two_level_with_l2, IW:2w, ROB:32, MSHR:2, L1:16kB/2w, L2:512kB). Initial IPC: {initial_ipc:.4f}.",
        f"Initial Cost Breakdown: Total Cost = {initial_cost:.1f} KB / {budget_kb:.1f} KB (Remaining Slack Delta = +{init_slack:.1f} KB)."
    ]

    history = []
    prev_cfg = initial_cfg
    prev_flat_l2 = "512kB"
    prompt_tokens = 0
    comp_tokens = 0

    for step in range(1, max_steps + 1):
        full_prompt = "\n".join(prompt_history) + f"\n\nStep {step}/{max_steps}: Propose the next architecture respecting the <= {budget_kb:.1f} KB budget."
        response = call_llm_with_retry(client, model, full_prompt, FlatCartesianSchema, seed + step)

        if response.usage_metadata:
            prompt_tokens += response.usage_metadata.prompt_token_count or 0
            comp_tokens += response.usage_metadata.candidates_token_count or 0

        time.sleep(float(os.environ.get("GEMINI_PACING_DELAY", "1.5")))
        proposal = json.loads(response.text)

        arch_type = proposal["architecture_type"]
        has_l2 = (arch_type == "two_level_with_l2")
        flat_l2_val = SIZE_MAP[str(proposal["l2_capacity_kb"])]
        l2s = flat_l2_val if has_l2 else "inactive"

        curr_cfg = (
            int(proposal["issue_width"]),
            int(proposal["rob_size"]),
            int(proposal["l1d_mshrs"]),
            SIZE_MAP[str(proposal["l1d_capacity_kb"])],
            int(proposal["l1d_associativity"]),
            has_l2,
            l2s
        )

        # Did the flat schema emit an inactive mutation to l2_capacity_kb while has_l2 is False?
        flat_l2_mut = (not has_l2) and (flat_l2_val != prev_flat_l2)
        prev_flat_l2 = flat_l2_val

        data = ORACLE_DICT[curr_cfg]
        ipc = data[f"ipc_{workload_key}"]
        cost = data["cost"]
        is_feas = data["is_feasible"]
        slack = round(budget_kb - cost, 1)
        feas_status = "FEASIBLE" if is_feas else f"INFEASIBLE (Exceeds budget by +{-slack:.1f} KB)"

        act_muts, inact_muts = calculate_mutations(curr_cfg, prev_cfg, workload_key, flat_l2_mut=flat_l2_mut)

        history.append({
            "step": step,
            "config": curr_cfg,
            "ipc": ipc,
            "cost": cost,
            "feasible": is_feas,
            "active_mutations": act_muts,
            "inactive_mutations": inact_muts,
            "reasoning": proposal.get("reasoning", "")
        })
        prev_cfg = curr_cfg

        iw, rob, mshr, l1s, l1a, _, l2_str = curr_cfg
        core_c = (iw * 50) + (rob * 1.5) + (mshr * 5)
        mem_c = round(cost - core_c, 1)
        prompt_history.append(
            f"Step {step} Evaluated: Arch={arch_type}, Config=(IW:{iw}w, ROB:{rob}, MSHR:{mshr}, L1:{l1s}/{l1a}w, L2:{l2_str}) -> IPC: {ipc:.4f}\n"
            f"  Cost Breakdown: Core_Cost = {core_c:.1f} KB | Mem_Cost = {mem_c:.1f} KB | Total Cost = {cost:.1f} KB / {budget_kb:.1f} KB (Slack = {slack:+.1f} KB) [{feas_status}]"
        )
        print(f"  [{workload_display} | Seed:{seed} | Flat] Step {step:02d}: Arch={arch_type:<17} Config={curr_cfg[:5]} L2={curr_cfg[6]:<8} -> IPC: {ipc:.4f}, Cost: {cost:.1f} KB [{feas_status}] (Opt: {opt_ipc:.4f})")

    return {
        "workload": workload_display,
        "baseline": "Baseline_Flat_Cartesian",
        "seed": seed,
        "opt_ipc": opt_ipc,
        "opt_cfg": opt_cfg,
        "history": history,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": comp_tokens
    }


def run_adsg_conditional(
    client: genai.Client,
    workload_key: str,
    workload_display: str,
    budget_kb: float,
    seed: int,
    max_steps: int = 10,
    model: str = "gemini-3.5-flash-lite"
) -> Dict[str, Any]:
    opt_ipc, opt_cfg = get_optimum(workload_key, budget_kb)

    initial_cfg = (2, 32, 2, "16kB", 2, True, "512kB")
    init_data = ORACLE_DICT[initial_cfg]
    initial_ipc = init_data[f"ipc_{workload_key}"]
    initial_cost = init_data["cost"]
    init_slack = round(budget_kb - initial_cost, 1)

    prompt_history = [
        f"You are an expert computer architect utilizing the Architecture Design Space Graph (ADSG) framework under a strict hardware budget.",
        f"Target Workload: {workload_display}.",
        f"HARDWARE BUDGET CONSTRAINT: Total Cost <= {budget_kb:.1f} KB equivalent.",
        f"ADSG Structural Conditionality:",
        f"  - Memory hierarchy allows conditional tiering: 'two_level_with_l2' vs. 'direct_l1_no_l2'.",
        f"  - When 'two_level_with_l2' is selected, the L2 decision subtree is active and you must size l2_capacity_kb.",
        f"  - When 'direct_l1_no_l2' is selected, the L2 decision subtree is structurally pruned (l2_capacity_kb does not exist in your schema).",
        f"Cost Formula: Total Cost = Core_Cost + Memory_Cost",
        f"  - Core_Cost = (issue_width * 50) + (rob_size * 1.5) + (l1d_mshrs * 5)",
        f"  - Memory_Cost = (l1d_size_kb * (1 + 0.1 * l1d_assoc/8)) + (l2_size_kb if architecture_type == 'two_level_with_l2' else 0)",
        f"Available design choices:",
        f"  - Core: issue_width: [2, 4, 8], rob_size: [32, 64, 128], l1d_mshrs: [2, 4, 8]",
        f"  - L1 Cache: l1d_capacity_kb: [16, 32, 64], l1d_associativity: [2, 4, 8]",
        f"  - L2 Cache (only under two_level_with_l2): l2_capacity_kb: [512, 1024, 2048]",
        f"Initial config: (two_level_with_l2, IW:2w, ROB:32, MSHR:2, L1:16kB/2w, L2:512kB). Initial IPC: {initial_ipc:.4f}.",
        f"Initial Cost Breakdown: Total Cost = {initial_cost:.1f} KB / {budget_kb:.1f} KB (Remaining Slack Delta = +{init_slack:.1f} KB)."
    ]

    history = []
    prev_cfg = initial_cfg
    prompt_tokens = 0
    comp_tokens = 0

    for step in range(1, max_steps + 1):
        full_prompt = "\n".join(prompt_history) + f"\n\nStep {step}/{max_steps}: Propose the next architecture respecting the <= {budget_kb:.1f} KB budget."
        response = call_llm_with_retry(client, model, full_prompt, ADSGProposalSchema, seed + step)

        if response.usage_metadata:
            prompt_tokens += response.usage_metadata.prompt_token_count or 0
            comp_tokens += response.usage_metadata.candidates_token_count or 0

        time.sleep(float(os.environ.get("GEMINI_PACING_DELAY", "1.5")))
        proposal = json.loads(response.text)

        arch_type = proposal["architecture_type"]
        has_l2 = (arch_type == "two_level_with_l2")
        l2s = SIZE_MAP[str(proposal["l2_capacity_kb"])] if has_l2 else "inactive"

        curr_cfg = (
            int(proposal["issue_width"]),
            int(proposal["rob_size"]),
            int(proposal["l1d_mshrs"]),
            SIZE_MAP[str(proposal["l1d_capacity_kb"])],
            int(proposal["l1d_associativity"]),
            has_l2,
            l2s
        )

        data = ORACLE_DICT[curr_cfg]
        ipc = data[f"ipc_{workload_key}"]
        cost = data["cost"]
        is_feas = data["is_feasible"]
        slack = round(budget_kb - cost, 1)
        feas_status = "FEASIBLE" if is_feas else f"INFEASIBLE (Exceeds budget by +{-slack:.1f} KB)"

        act_muts, inact_muts = calculate_mutations(curr_cfg, prev_cfg, workload_key, flat_l2_mut=False)

        history.append({
            "step": step,
            "config": curr_cfg,
            "ipc": ipc,
            "cost": cost,
            "feasible": is_feas,
            "active_mutations": act_muts,
            "inactive_mutations": inact_muts,
            "reasoning": proposal.get("reasoning", "")
        })
        prev_cfg = curr_cfg

        iw, rob, mshr, l1s, l1a, _, l2_str = curr_cfg
        core_c = (iw * 50) + (rob * 1.5) + (mshr * 5)
        mem_c = round(cost - core_c, 1)
        prompt_history.append(
            f"Step {step} Evaluated: Arch={arch_type}, Config=(IW:{iw}w, ROB:{rob}, MSHR:{mshr}, L1:{l1s}/{l1a}w, L2:{l2_str}) -> IPC: {ipc:.4f}\n"
            f"  Cost Breakdown: Core_Cost = {core_c:.1f} KB | Mem_Cost = {mem_c:.1f} KB | Total Cost = {cost:.1f} KB / {budget_kb:.1f} KB (Slack = {slack:+.1f} KB) [{feas_status}]"
        )
        print(f"  [{workload_display} | Seed:{seed} | ADSG] Step {step:02d}: Arch={arch_type:<17} Config={curr_cfg[:5]} L2={curr_cfg[6]:<8} -> IPC: {ipc:.4f}, Cost: {cost:.1f} KB [{feas_status}] (Opt: {opt_ipc:.4f})")

    return {
        "workload": workload_display,
        "baseline": "ADSG_Conditional",
        "seed": seed,
        "opt_ipc": opt_ipc,
        "opt_cfg": opt_cfg,
        "history": history,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": comp_tokens
    }


# ==============================================================================
# 4. Statistical Analysis
# ==============================================================================

def analyze_and_report_results(results_data: Dict[str, Any]):
    rows = []
    for wl_name, baselines in results_data.items():
        for b_name, trials in baselines.items():
            for t in trials:
                history = t["history"]
                opt_ipc = t["opt_ipc"]
                feasible_steps = [s for s in history if s["feasible"]]
                infeas_steps = [s for s in history if not s["feasible"]]
                b_ipc = max([s["ipc"] for s in feasible_steps]) if feasible_steps else 0.0

                s98 = 11
                for s in history:
                    if s["feasible"] and s["ipc"] >= 0.98 * opt_ipc:
                        s98 = s["step"]
                        break

                act_muts = sum([s["active_mutations"] for s in history])
                inact_muts = sum([s["inactive_mutations"] for s in history])
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

    print("\n" + "=" * 100)
    print("           EMPIRICAL RESULTS: OPTIONAL-L2 CONDITIONAL REPRESENTATION BENCHMARK")
    print("=" * 100)

    pd.set_option("display.max_columns", 14)
    pd.set_option("display.width", 1000)

    for wl in df["Workload"].unique():
        print(f"\n==================== {wl.upper()} RESULTS ====================")
        wl_df = df[df["Workload"] == wl]
        table = wl_df.groupby("Baseline")[
            ["Best_IPC", "Steps_to_Opt", "Rel_Mutation_Ratio_Pct", "Active_Mutations", "Inactive_Mutations", "Infeasible_Proposals", "Completion_Tokens"]
        ].agg(["mean", "std"]).round(2)
        print(table)

        print(f"\n--- Paired Comparisons for {wl} (N=3 seeds) ---")
        for col in ["Steps_to_Opt", "Rel_Mutation_Ratio_Pct", "Inactive_Mutations", "Infeasible_Proposals", "Completion_Tokens", "Best_IPC"]:
            v_flat = wl_df[wl_df["Baseline"] == "Baseline_Flat_Cartesian"][col].values
            v_adsg = wl_df[wl_df["Baseline"] == "ADSG_Conditional"][col].values
            diff = np.mean(v_adsg) - np.mean(v_flat)
            t_stat, p_val = stats.ttest_rel(v_flat, v_adsg)
            print(f"  {col:<24}: Flat={np.mean(v_flat):.2f} | ADSG={np.mean(v_adsg):.2f} | Diff={diff:+6.2f} | t={t_stat:6.3f}, p={p_val:.4f}")

    print("\n==================== OVERALL AGGREGATES ACROSS ALL 18 TRIALS ====================")
    overall = df.groupby("Baseline")[
        ["Best_IPC", "Steps_to_Opt", "Rel_Mutation_Ratio_Pct", "Active_Mutations", "Inactive_Mutations", "Infeasible_Proposals", "Prompt_Tokens", "Completion_Tokens"]
    ].agg(["mean", "std"]).round(2)
    print(overall)

    print("\n--- Overall Paired Comparisons (N=9 matched pairs) ---")
    for col in ["Steps_to_Opt", "Rel_Mutation_Ratio_Pct", "Inactive_Mutations", "Infeasible_Proposals", "Completion_Tokens", "Best_IPC"]:
        v_flat = df[df["Baseline"] == "Baseline_Flat_Cartesian"][col].values
        v_adsg = df[df["Baseline"] == "ADSG_Conditional"][col].values
        diff = np.mean(v_adsg) - np.mean(v_flat)
        t_stat, p_val = stats.ttest_rel(v_flat, v_adsg)
        w_stat, w_pval = stats.wilcoxon(v_flat, v_adsg) if not np.all(v_flat == v_adsg) else (0, 1.0)
        print(f"  {col:<24}: Flat={np.mean(v_flat):.2f} | ADSG={np.mean(v_adsg):.2f} | Diff={diff:+6.2f} | t={t_stat:6.3f}, p={p_val:.4f} (Wilcoxon p={w_pval:.4f})")

    return df


# ==============================================================================
# 5. Main Execution
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=BUDGET_KB_DEFAULT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2026])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=str, default="optl2_eval_results.json")
    args = parser.parse_args()

    client = genai.Client()

    workload_map = [
        ("compute", "Workload_Alpha (Compute)"),
        ("latency", "Workload_Beta (Latency)"),
        ("concurrency", "Workload_Gamma (Concurrency)")
    ]

    if args.dry_run:
        print("Running 1-step dry run on Compute Seed 42...")
        run_flat_cartesian(client, "compute", "Workload_Alpha", args.budget, seed=42, max_steps=1, model=args.model)
        run_adsg_conditional(client, "compute", "Workload_Alpha", args.budget, seed=42, max_steps=1, model=args.model)
        print("Dry run passed!")
        exit(0)

    all_results = {disp: {"Baseline_Flat_Cartesian": [], "ADSG_Conditional": []} for _, disp in workload_map}

    for k, disp in workload_map:
        for seed in args.seeds:
            print(f"\n>>> Running Baseline_Flat_Cartesian on {disp} (Seed: {seed}) <<<")
            res_flat = run_flat_cartesian(client, k, disp, args.budget, seed, max_steps=args.steps, model=args.model)
            all_results[disp]["Baseline_Flat_Cartesian"].append(res_flat)

        for seed in args.seeds:
            print(f"\n>>> Running ADSG_Conditional on {disp} (Seed: {seed}) <<<")
            res_adsg = run_adsg_conditional(client, k, disp, args.budget, seed, max_steps=args.steps, model=args.model)
            all_results[disp]["ADSG_Conditional"].append(res_adsg)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    analyze_and_report_results(all_results)
