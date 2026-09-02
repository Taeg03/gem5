"""
evaluate_adsg_agent.py

3-Arm Benchmark Harness for ADSG Agent-Facing Semantic Abstraction Study:
1. Baseline B (Flat Microarch): Flat 6-parameter schema + aggregate cost.
2. Baseline B + Cost Breakdown (Ablation Control): Flat 6-parameter schema + Core_Cost, Mem_Cost, Slack Delta.
3. ADSG Candidate Interface: Scheduled focal view + compensatory adjustment channel + Core_Cost, Mem_Cost, Slack Delta.

Execution Matrix:
- 3 Workloads: Workload_Alpha, Workload_Beta, Workload_Gamma
- 3 Random Seeds: 42, 100, 2026
- 3 Baselines = 27 Total Trials (10 steps per trial)
"""

import argparse
import json
import os
import re
import time
from typing import Literal, Optional, Tuple, Dict, Any, List

# Load .env
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

import adsg_cpu_model as acm
import adsg_translator as at

BUDGET_KB_DEFAULT = 1500.0

SIZE_MAP = {
    "16": "16kB", "32": "32kB", "64": "64kB", "512": "512kB", "1024": "1MB", "2048": "2MB",
    16: "16kB", 32: "32kB", 64: "64kB", 512: "512kB", 1024: "1MB", 2048: "2MB",
    "16kB": "16kB", "32kB": "32kB", "64kB": "64kB", "512kB": "512kB", "1MB": "1MB", "2MB": "2MB"
}


# ==============================================================================
# 1. Pydantic Action Schemas
# ==============================================================================

# Flat Schema (used by Baseline B and Baseline B + Cost Breakdown)
class BaselineBSchema(BaseModel):
    issue_width: Literal["2", "4", "8"]
    rob_size: Literal["32", "64", "128"]
    l1d_mshrs: Literal["2", "4", "8"]
    l1d_capacity_kb: Literal["16", "32", "64"]
    l1d_associativity: Literal["2", "4", "8"]
    l2_capacity_kb: Literal["512", "1024", "2048"]
    reasoning: str


# Component Schemas for ADSG Candidate
class CoreSubsystemUpdate(BaseModel):
    issue_width: Literal["2", "4", "8"]
    rob_size: Literal["32", "64", "128"]
    l1d_mshrs: Literal["2", "4", "8"]


class MemorySubsystemUpdate(BaseModel):
    l1d_capacity_kb: Literal["16", "32", "64"]
    l1d_associativity: Literal["2", "4", "8"]
    l2_capacity_kb: Literal["512", "1024", "2048"]


class CoreFocalActionSchema(BaseModel):
    focal_core_mutations: CoreSubsystemUpdate
    compensatory_memory_adjustments: Optional[MemorySubsystemUpdate] = None
    reasoning: str


class MemoryFocalActionSchema(BaseModel):
    focal_memory_mutations: MemorySubsystemUpdate
    compensatory_core_adjustments: Optional[CoreSubsystemUpdate] = None
    reasoning: str


# ==============================================================================
# 2. Helper Functions
# ==============================================================================

def get_constrained_optimum(workload_name: str, budget_kb: float) -> Tuple[float, Tuple]:
    df = at.get_oracle_df(workload_name)
    df["cost"] = df.apply(
        lambda r: at.compute_cost_729(r["issue_width"], r["rob_size"], r["l1d_mshrs"], r["l1d_size"], r["l1d_assoc"], r["l2_size"]),
        axis=1
    )
    feasible = df[df["cost"] <= budget_kb]
    if feasible.empty:
        return 0.0, ()
    best_row = feasible.sort_values(by="ipc", ascending=False).iloc[0]
    opt_cfg = (
        int(best_row["issue_width"]),
        int(best_row["rob_size"]),
        int(best_row["l1d_mshrs"]),
        str(best_row["l1d_size"]),
        int(best_row["l1d_assoc"]),
        str(best_row["l2_size"])
    )
    return float(best_row["ipc"]), opt_cfg


def calculate_mutations(curr_cfg: Tuple, prev_cfg: Tuple, workload_name: str) -> Tuple[int, int]:
    iw_mut = (curr_cfg[0] != prev_cfg[0])
    rob_mut = (curr_cfg[1] != prev_cfg[1])
    mshr_mut = (curr_cfg[2] != prev_cfg[2])
    l1s_mut = (curr_cfg[3] != prev_cfg[3])
    l1a_mut = (curr_cfg[4] != prev_cfg[4])
    l2s_mut = (curr_cfg[5] != prev_cfg[5])

    if workload_name == "Workload_Alpha":  # issue_width active (148%)
        act_muts = 1 if iw_mut else 0
        inact_muts = (1 if rob_mut else 0) + (1 if mshr_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0) + (1 if l2s_mut else 0)
    elif workload_name == "Workload_Beta":  # l2_size active (148%)
        act_muts = 1 if l2s_mut else 0
        inact_muts = (1 if iw_mut else 0) + (1 if rob_mut else 0) + (1 if mshr_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0)
    else:  # Workload_Gamma (l2_size + mshrs active)
        act_muts = (1 if l2s_mut else 0) + (1 if mshr_mut else 0)
        inact_muts = (1 if iw_mut else 0) + (1 if rob_mut else 0) + (1 if l1s_mut else 0) + (1 if l1a_mut else 0)

    return act_muts, inact_muts


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
# 3. Trial Runners for the 3 Arms
# ==============================================================================

def run_baseline_b(
    client: genai.Client,
    workload_name: str,
    budget_kb: float,
    seed: int,
    max_steps: int = 10,
    model: str = "gemini-3.5-flash-lite"
) -> Dict[str, Any]:
    opt_ipc, opt_cfg = get_constrained_optimum(workload_name, budget_kb)
    initial_cfg = (2, 32, 2, "16kB", 2, "512kB")
    initial_ipc = at.lookup_oracle_ipc(*initial_cfg, workload_name)
    initial_cost = at.compute_cost_729(*initial_cfg)

    prompt_history = [
        f"You are an expert computer architect optimizing a processor design under a strict hardware budget.",
        f"Target Workload Identifier: {workload_name}.",
        f"HARDWARE BUDGET CONSTRAINT: Total Cost <= {budget_kb:.1f} KB equivalent.",
        f"Cost formula: Cost ≈ Core_Cost(issue_width*50 + rob_size*1.5 + mshrs*5) + Memory_Cost(L1_Cap*(1 + 0.1*L1_Assoc/8) + L2_Cap).",
        f"Goal: Maximize IPC within {max_steps} steps. Any configuration exceeding {budget_kb:.1f} KB is INFEASIBLE.",
        f"Representation format: Flat microarchitectural parameters across all subsystems.",
        f"Available design choices:",
        f"- Core Parameters: issue_width: [2, 4, 8], rob_size: [32, 64, 128], l1d_mshrs: [2, 4, 8]",
        f"- Memory Parameters: l1d_size: [16kB, 32kB, 64kB], l1d_assoc: [2, 4, 8], l2_size: [512kB, 1MB, 2MB]",
        f"Initial baseline config: (IW:2w, ROB:32, MSHR:2, L1:16kB/2w, L2:512kB) (Cost: {initial_cost:.1f} KB). Initial IPC: {initial_ipc:.4f}."
    ]

    history = []
    prev_cfg = initial_cfg
    prompt_tokens_total = 0
    completion_tokens_total = 0

    for step in range(1, max_steps + 1):
        full_prompt = "\n".join(prompt_history) + f"\n\nStep {step}/{max_steps}: Propose the next architecture respecting the <= {budget_kb:.1f} KB budget."
        response = call_llm_with_retry(client, model, full_prompt, BaselineBSchema, seed + step)

        if response.usage_metadata:
            prompt_tokens_total += response.usage_metadata.prompt_token_count or 0
            completion_tokens_total += response.usage_metadata.candidates_token_count or 0

        time.sleep(float(os.environ.get("GEMINI_PACING_DELAY", "1.5")))
        proposal = json.loads(response.text)

        curr_cfg = (
            int(proposal["issue_width"]),
            int(proposal["rob_size"]),
            int(proposal["l1d_mshrs"]),
            SIZE_MAP[str(proposal["l1d_capacity_kb"])],
            int(proposal["l1d_associativity"]),
            SIZE_MAP[str(proposal["l2_capacity_kb"])]
        )

        cost = at.compute_cost_729(*curr_cfg)
        ipc = at.lookup_oracle_ipc(*curr_cfg, workload_name)
        is_feasible = (cost <= budget_kb)
        feas_status = "FEASIBLE" if is_feasible else f"INFEASIBLE (Exceeds budget by +{cost - budget_kb:.1f} KB)"

        act_muts, inact_muts = calculate_mutations(curr_cfg, prev_cfg, workload_name)
        history.append({
            "step": step,
            "config": curr_cfg,
            "ipc": ipc,
            "cost": cost,
            "feasible": is_feasible,
            "active_mutations": act_muts,
            "inactive_mutations": inact_muts,
            "reasoning": proposal.get("reasoning", "")
        })
        prev_cfg = curr_cfg

        prompt_history.append(
            f"Step {step} Evaluated: Config=(IW:{curr_cfg[0]}w, ROB:{curr_cfg[1]}, MSHR:{curr_cfg[2]}, L1:{curr_cfg[3]}/{curr_cfg[4]}w, L2:{curr_cfg[5]}) -> IPC: {ipc:.4f}, Cost: {cost:.1f} KB [{feas_status}]"
        )
        print(f"  [{workload_name} | Seed:{seed} | Baseline_B] Step {step:02d}: Config={curr_cfg} -> IPC: {ipc:.4f}, Cost: {cost:.1f} KB [{feas_status}] (Opt: {opt_ipc:.4f})")

    return {
        "workload": workload_name,
        "baseline": "Baseline_B",
        "seed": seed,
        "opt_ipc": opt_ipc,
        "opt_cfg": opt_cfg,
        "history": history,
        "prompt_tokens": prompt_tokens_total,
        "completion_tokens": completion_tokens_total
    }


def run_baseline_b_cost_breakdown(
    client: genai.Client,
    workload_name: str,
    budget_kb: float,
    seed: int,
    max_steps: int = 10,
    model: str = "gemini-3.5-flash-lite"
) -> Dict[str, Any]:
    opt_ipc, opt_cfg = get_constrained_optimum(workload_name, budget_kb)
    initial_cfg = (2, 32, 2, "16kB", 2, "512kB")
    initial_ipc = at.lookup_oracle_ipc(*initial_cfg, workload_name)

    # Initial cost breakdown
    init_core_cost = (2 * 50) + (32 * 1.5) + (2 * 5)
    init_mem_cost = 16 * (1.0 + 0.1 * (2 / 8.0)) + 512
    initial_cost = round(init_core_cost + init_mem_cost, 1)
    init_slack = round(budget_kb - initial_cost, 1)

    prompt_history = [
        f"You are an expert computer architect optimizing a processor design under a strict hardware budget.",
        f"Target Workload Identifier: {workload_name}.",
        f"HARDWARE BUDGET CONSTRAINT: Total Cost <= {budget_kb:.1f} KB equivalent.",
        f"Detailed Cost Formula: Total Cost = Core_Cost + Memory_Cost",
        f"  - Core_Cost = (issue_width * 50) + (rob_size * 1.5) + (l1d_mshrs * 5)",
        f"  - Memory_Cost = (l1d_size_kb * (1 + 0.1 * l1d_assoc/8)) + l2_size_kb",
        f"Goal: Maximize IPC within {max_steps} steps. Any configuration exceeding {budget_kb:.1f} KB is INFEASIBLE.",
        f"Representation format: Flat microarchitectural parameters across all subsystems.",
        f"Available design choices:",
        f"- Core Parameters: issue_width: [2, 4, 8], rob_size: [32, 64, 128], l1d_mshrs: [2, 4, 8]",
        f"- Memory Parameters: l1d_size: [16kB, 32kB, 64kB], l1d_assoc: [2, 4, 8], l2_size: [512kB, 1MB, 2MB]",
        f"Initial baseline config: (IW:2w, ROB:32, MSHR:2, L1:16kB/2w, L2:512kB). Initial IPC: {initial_ipc:.4f}.",
        f"Initial Cost Breakdown: Core_Cost = {init_core_cost:.1f} KB | Mem_Cost = {init_mem_cost:.1f} KB | Total Cost = {initial_cost:.1f} KB (Remaining Slack Delta = +{init_slack:.1f} KB)."
    ]

    history = []
    prev_cfg = initial_cfg
    prompt_tokens_total = 0
    completion_tokens_total = 0

    for step in range(1, max_steps + 1):
        full_prompt = "\n".join(prompt_history) + f"\n\nStep {step}/{max_steps}: Propose the next architecture respecting the <= {budget_kb:.1f} KB budget."
        response = call_llm_with_retry(client, model, full_prompt, BaselineBSchema, seed + step)

        if response.usage_metadata:
            prompt_tokens_total += response.usage_metadata.prompt_token_count or 0
            completion_tokens_total += response.usage_metadata.candidates_token_count or 0

        time.sleep(float(os.environ.get("GEMINI_PACING_DELAY", "1.5")))
        proposal = json.loads(response.text)

        curr_cfg = (
            int(proposal["issue_width"]),
            int(proposal["rob_size"]),
            int(proposal["l1d_mshrs"]),
            SIZE_MAP[str(proposal["l1d_capacity_kb"])],
            int(proposal["l1d_associativity"]),
            SIZE_MAP[str(proposal["l2_capacity_kb"])]
        )

        iw, rob, mshr, l1s, l1a, l2s = curr_cfg
        core_cost = (iw * 50) + (rob * 1.5) + (mshr * 5)
        l1_kb = at.parse_size_kb(l1s)
        l2_kb = at.parse_size_kb(l2s)
        mem_cost = l1_kb * (1.0 + 0.1 * (l1a / 8.0)) + l2_kb
        total_cost = round(core_cost + mem_cost, 1)
        slack = round(budget_kb - total_cost, 1)

        ipc = at.lookup_oracle_ipc(*curr_cfg, workload_name)
        is_feasible = (total_cost <= budget_kb)
        feas_status = "FEASIBLE" if is_feasible else f"INFEASIBLE (Exceeds budget by +{-slack:.1f} KB)"

        act_muts, inact_muts = calculate_mutations(curr_cfg, prev_cfg, workload_name)
        history.append({
            "step": step,
            "config": curr_cfg,
            "ipc": ipc,
            "cost": total_cost,
            "feasible": is_feasible,
            "active_mutations": act_muts,
            "inactive_mutations": inact_muts,
            "reasoning": proposal.get("reasoning", "")
        })
        prev_cfg = curr_cfg

        prompt_history.append(
            f"Step {step} Evaluated: Config=(IW:{iw}w, ROB:{rob}, MSHR:{mshr}, L1:{l1s}/{l1a}w, L2:{l2s}) -> IPC: {ipc:.4f}\n"
            f"  Cost Breakdown: Core_Cost = {core_cost:.1f} KB | Mem_Cost = {mem_cost:.1f} KB | Total Cost = {total_cost:.1f} KB / {budget_kb:.1f} KB (Remaining Slack Delta = {slack:+.1f} KB) [{feas_status}]"
        )
        print(f"  [{workload_name} | Seed:{seed} | Baseline_B_CostBreakdown] Step {step:02d}: Config={curr_cfg} -> IPC: {ipc:.4f}, Cost: {total_cost:.1f} KB [{feas_status}] (Opt: {opt_ipc:.4f})")

    return {
        "workload": workload_name,
        "baseline": "Baseline_B_CostBreakdown",
        "seed": seed,
        "opt_ipc": opt_ipc,
        "opt_cfg": opt_cfg,
        "history": history,
        "prompt_tokens": prompt_tokens_total,
        "completion_tokens": completion_tokens_total
    }


def run_adsg_candidate(
    client: genai.Client,
    workload_name: str,
    budget_kb: float,
    seed: int,
    max_steps: int = 10,
    model: str = "gemini-3.5-flash-lite"
) -> Dict[str, Any]:
    # Initialize ADSG Model & Processor
    dsg = acm.build_cpu_adsg()
    gp = acm.get_cpu_graph_processor(dsg)

    opt_ipc, opt_cfg = get_constrained_optimum(workload_name, budget_kb)

    # Initial system state
    current_state = {
        "core_execution": {"issue_width": 2, "rob_size": 32, "l1d_mshrs": 2},
        "memory_hierarchy": {"l1d_size": "16kB", "l1d_assoc": 2, "l2_size": "512kB"}
    }

    initial_cfg = (2, 32, 2, "16kB", 2, "512kB")
    initial_ipc = at.lookup_oracle_ipc(*initial_cfg, workload_name)

    init_core_cost = (2 * 50) + (32 * 1.5) + (2 * 5)
    init_mem_cost = 16 * (1.0 + 0.1 * (2 / 8.0)) + 512
    initial_cost = round(init_core_cost + init_mem_cost, 1)
    init_slack = round(budget_kb - initial_cost, 1)

    prompt_history = [
        f"You are an expert computer architect utilizing the Architecture Design Space Graph (ADSG) framework under a strict hardware budget.",
        f"Target Workload Identifier: {workload_name}.",
        f"HARDWARE BUDGET CONSTRAINT: Total Cost <= {budget_kb:.1f} KB equivalent.",
        f"ADSG System Hierarchy & Cost Decomposition:",
        f"  - System Root: [CPU] derives [Core] and [Memory].",
        f"  - Metric Constraint: PR[cost] <= {budget_kb:.1f} KB.",
        f"  - Total Cost = Core_Cost + Memory_Cost",
        f"      Core_Cost = (issue_width * 50) + (rob_size * 1.5) + (l1d_mshrs * 5)",
        f"      Memory_Cost = (l1d_size_kb * (1 + 0.1 * l1d_assoc/8)) + l2_size_kb",
        f"ADSG-Grounded Representation Mechanism:",
        f"To eliminate search dispersion over irrelevant parameters, the environment designates a SCHEDULED FOCAL SUBSYSTEM for each turn.",
        f"- Mandatory: Provide primary architectural mutations for the focal subsystem.",
        f"- Optional Compensatory Channel: If your focal mutations increase cost, you can simultaneously adjust the background subsystem to offset cost and maintain feasibility in the same turn.",
        f"Initial baseline config: (IW:2w, ROB:32, MSHR:2, L1:16kB/2w, L2:512kB). Initial IPC: {initial_ipc:.4f}.",
        f"Initial Cost Breakdown: Core_Cost = {init_core_cost:.1f} KB | Mem_Cost = {init_mem_cost:.1f} KB | Total Cost = {initial_cost:.1f} KB (Remaining Slack Delta = +{init_slack:.1f} KB)."
    ]

    history = []
    prev_cfg = initial_cfg
    prompt_tokens_total = 0
    completion_tokens_total = 0

    for step in range(1, max_steps + 1):
        # Scheduled focal subsystem: Alternates Core (odd) and Memory (even)
        is_core_focal = (step % 2 == 1)
        focal_name = "Core Execution Engine" if is_core_focal else "Memory Hierarchy"
        action_schema = CoreFocalActionSchema if is_core_focal else MemoryFocalActionSchema

        c_state = current_state["core_execution"]
        m_state = current_state["memory_hierarchy"]
        core_cost = (c_state["issue_width"] * 50) + (c_state["rob_size"] * 1.5) + (c_state["l1d_mshrs"] * 5)
        l1_kb = at.parse_size_kb(m_state["l1d_size"])
        l2_kb = at.parse_size_kb(m_state["l2_size"])
        mem_cost = l1_kb * (1.0 + 0.1 * (m_state["l1d_assoc"] / 8.0)) + l2_kb
        curr_total_cost = round(core_cost + mem_cost, 1)
        curr_slack = round(budget_kb - curr_total_cost, 1)

        state_status = (
            f"Current Confirmed System State:\n"
            f"  Core State:   IW={c_state['issue_width']}w, ROB={c_state['rob_size']}, MSHR={c_state['l1d_mshrs']} (Core_Cost: {core_cost:.1f} KB)\n"
            f"  Memory State: L1={m_state['l1d_size']}/{m_state['l1d_assoc']}w, L2={m_state['l2_size']} (Mem_Cost: {mem_cost:.1f} KB)\n"
            f"  Cost Status:  Total Cost = {curr_total_cost:.1f} KB / {budget_kb:.1f} KB (Remaining Slack Delta = {curr_slack:+.1f} KB)"
        )

        focus_instruction = (
            f"Turn {step}/{max_steps} Schedule: FOCUS ON [{focal_name.upper()}].\n"
            f"Formulate primary mutations for [{focal_name}]. "
            f"If your upgrade requires budget rebalancing, use the compensatory adjustment channel to adjust the background subsystem in the same turn."
        )

        full_prompt = "\n".join(prompt_history) + f"\n\n{state_status}\n{focus_instruction}"
        response = call_llm_with_retry(client, model, full_prompt, action_schema, seed + step)

        if response.usage_metadata:
            prompt_tokens_total += response.usage_metadata.prompt_token_count or 0
            completion_tokens_total += response.usage_metadata.candidates_token_count or 0

        time.sleep(float(os.environ.get("GEMINI_PACING_DELAY", "1.5")))
        proposal = json.loads(response.text)

        # Atomic Reconstruction Pipeline
        candidate_iw = c_state["issue_width"]
        candidate_rob = c_state["rob_size"]
        candidate_mshr = c_state["l1d_mshrs"]
        candidate_l1s = m_state["l1d_size"]
        candidate_l1a = m_state["l1d_assoc"]
        candidate_l2s = m_state["l2_size"]

        if is_core_focal:
            # Mandatory focal core mutations
            focal_core = proposal.get("focal_core_mutations", {})
            candidate_iw = int(focal_core["issue_width"])
            candidate_rob = int(focal_core["rob_size"])
            candidate_mshr = int(focal_core["l1d_mshrs"])

            # Optional compensatory memory adjustments
            comp_mem = proposal.get("compensatory_memory_adjustments")
            if comp_mem:
                candidate_l1s = SIZE_MAP[str(comp_mem["l1d_capacity_kb"])]
                candidate_l1a = int(comp_mem["l1d_associativity"])
                candidate_l2s = SIZE_MAP[str(comp_mem["l2_capacity_kb"])]
        else:
            # Mandatory focal memory mutations
            focal_mem = proposal.get("focal_memory_mutations", {})
            candidate_l1s = SIZE_MAP[str(focal_mem["l1d_capacity_kb"])]
            candidate_l1a = int(focal_mem["l1d_associativity"])
            candidate_l2s = SIZE_MAP[str(focal_mem["l2_capacity_kb"])]

            # Optional compensatory core adjustments
            comp_core = proposal.get("compensatory_core_adjustments")
            if comp_core:
                candidate_iw = int(comp_core["issue_width"])
                candidate_rob = int(comp_core["rob_size"])
                candidate_mshr = int(comp_core["l1d_mshrs"])

        candidate_vector = (candidate_iw, candidate_rob, candidate_mshr, candidate_l1s, candidate_l1a, candidate_l2s)

        # Resolve atomically via ADSG model
        g_inst, res_idx, res_val = acm.resolve_adsg_vector(gp, candidate_vector)
        gem5_cfg = at.adsg_to_gem5_config(g_inst, budget_kb=budget_kb)

        # Update confirmed state
        current_state["core_execution"]["issue_width"] = gem5_cfg.issue_width
        current_state["core_execution"]["rob_size"] = gem5_cfg.rob_size
        current_state["core_execution"]["l1d_mshrs"] = gem5_cfg.l1d_mshrs
        current_state["memory_hierarchy"]["l1d_size"] = gem5_cfg.l1d_size
        current_state["memory_hierarchy"]["l1d_assoc"] = gem5_cfg.l1d_assoc
        current_state["memory_hierarchy"]["l2_size"] = gem5_cfg.l2_size

        ipc = at.lookup_oracle_ipc(*gem5_cfg.as_param_tuple(), workload_name)
        cost = gem5_cfg.cost
        is_feasible = gem5_cfg.is_feasible
        slack = round(budget_kb - cost, 1)
        feas_status = "FEASIBLE" if is_feasible else f"INFEASIBLE (Exceeds budget by +{-slack:.1f} KB)"

        curr_cfg = gem5_cfg.as_param_tuple()
        act_muts, inact_muts = calculate_mutations(curr_cfg, prev_cfg, workload_name)

        history.append({
            "step": step,
            "focal_subsystem": focal_name,
            "config": curr_cfg,
            "ipc": ipc,
            "cost": cost,
            "feasible": is_feasible,
            "active_mutations": act_muts,
            "inactive_mutations": inact_muts,
            "reasoning": proposal.get("reasoning", "")
        })
        prev_cfg = curr_cfg

        iw, rob, mshr, l1s, l1a, l2s = curr_cfg
        new_core_cost = (iw * 50) + (rob * 1.5) + (mshr * 5)
        new_l1_kb = at.parse_size_kb(l1s)
        new_l2_kb = at.parse_size_kb(l2s)
        new_mem_cost = new_l1_kb * (1.0 + 0.1 * (l1a / 8.0)) + new_l2_kb

        prompt_history.append(
            f"Step {step} Evaluated [Focal: {focal_name}]: Config=(IW:{iw}w, ROB:{rob}, MSHR:{mshr}, L1:{l1s}/{l1a}w, L2:{l2s}) -> IPC: {ipc:.4f}\n"
            f"  Cost Breakdown: Core_Cost = {new_core_cost:.1f} KB | Mem_Cost = {new_mem_cost:.1f} KB | Total Cost = {cost:.1f} KB / {budget_kb:.1f} KB (Remaining Slack Delta = {slack:+.1f} KB) [{feas_status}]"
        )
        print(f"  [{workload_name} | Seed:{seed} | ADSG_Candidate | Focal: {focal_name:<16}] Step {step:02d}: Config={curr_cfg} -> IPC: {ipc:.4f}, Cost: {cost:.1f} KB [{feas_status}] (Opt: {opt_ipc:.4f})")

    return {
        "workload": workload_name,
        "baseline": "ADSG_Candidate",
        "seed": seed,
        "opt_ipc": opt_ipc,
        "opt_cfg": opt_cfg,
        "history": history,
        "prompt_tokens": prompt_tokens_total,
        "completion_tokens": completion_tokens_total
    }


# ==============================================================================
# 4. Statistical Analysis & Reporting
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
    print("           EMPIRICAL BENCHMARK RESULTS: 3-ARM ADSG AGENT ABLATION STUDY")
    print("=" * 100)

    pd.set_option("display.max_columns", 14)
    pd.set_option("display.width", 1000)

    table = df.groupby(["Workload", "Baseline"])[
        ["Best_IPC", "Steps_to_Opt", "Rel_Mutation_Ratio_Pct", "Inactive_Mutations", "Infeasible_Proposals", "Completion_Tokens"]
    ].agg(["mean", "std"]).round(2)
    print("\n--- Detailed Breakdown per Workload & Baseline (Mean +/- Std across 3 Seeds) ---")
    print(table)

    print("\n--- Overall Baseline Aggregates Across All 27 Trials ---")
    overall = df.groupby("Baseline")[
        ["Best_IPC", "Steps_to_Opt", "Rel_Mutation_Ratio_Pct", "Active_Mutations", "Inactive_Mutations", "Infeasible_Proposals", "Prompt_Tokens", "Completion_Tokens"]
    ].agg(["mean", "std"]).round(2)
    print(overall)

    metrics_to_test = [
        ("Steps_to_Opt", "Steps to >=98% Optimum"),
        ("Rel_Mutation_Ratio_Pct", "Relevant Mutation Ratio (%)"),
        ("Inactive_Mutations", "Inactive Parameter Mutations"),
        ("Infeasible_Proposals", "Infeasible Proposals / Violations"),
        ("Completion_Tokens", "Completion Token Consumption"),
        ("Best_IPC", "Best Feasible IPC")
    ]

    # Paired comparisons:
    comparisons = [
        ("Baseline_B", "Baseline_B_CostBreakdown", "Ablation Test 1: Arithmetic Cost Feedback Effect (B vs. B+Cost)"),
        ("Baseline_B_CostBreakdown", "ADSG_Candidate", "Ablation Test 2: ADSG Representational Focus Effect (B+Cost vs. ADSG)"),
        ("Baseline_B", "ADSG_Candidate", "Net Effect: Standard Control vs. ADSG Candidate (B vs. ADSG)")
    ]

    for b1, b2, title in comparisons:
        print(f"\n--- {title} (N=9 matched pairs) ---")
        df_pair = df[df["Baseline"].isin([b1, b2])]
        for col, display_name in metrics_to_test:
            vals_1 = df_pair[df_pair["Baseline"] == b1][col].values
            vals_2 = df_pair[df_pair["Baseline"] == b2][col].values
            diff = np.mean(vals_2) - np.mean(vals_1)

            t_stat, p_val = stats.ttest_rel(vals_1, vals_2)
            w_stat, w_pval = stats.wilcoxon(vals_1, vals_2) if not np.all(vals_1 == vals_2) else (0, 1.0)
            print(f"  {display_name:<34}: {b1}={np.mean(vals_1):.2f} | {b2}={np.mean(vals_2):.2f} | Diff={diff:+6.2f} | t={t_stat:6.3f}, p={p_val:.4f} (Wilcoxon p={w_pval:.4f})")

    return df


# ==============================================================================
# 5. Main Execution Script
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ADSG Agent 3-Arm Benchmark Suite")
    parser.add_argument("--budget", type=float, default=BUDGET_KB_DEFAULT, help="Hardware budget constraint in KB")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2026], help="Random seeds")
    parser.add_argument("--steps", type=int, default=10, help="Steps per trial")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite", help="GenAI model to use")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 dry run trial for each baseline on Workload_Alpha Seed 42")
    parser.add_argument("--out", type=str, default="adsg_eval_3arm_results.json", help="Output JSON file")
    args = parser.parse_args()

    client = genai.Client()

    workloads = ["Workload_Alpha", "Workload_Beta", "Workload_Gamma"]

    if args.dry_run:
        print("==================================================================")
        print("DRY RUN: Testing All 3 Arms on Workload_Alpha (Seed 42, 2 steps)")
        print("==================================================================")
        res_b = run_baseline_b(client, "Workload_Alpha", args.budget, seed=42, max_steps=2, model=args.model)
        res_b_cost = run_baseline_b_cost_breakdown(client, "Workload_Alpha", args.budget, seed=42, max_steps=2, model=args.model)
        res_adsg = run_adsg_candidate(client, "Workload_Alpha", args.budget, seed=42, max_steps=2, model=args.model)
        print("\nAll 3 baselines executed dry run successfully!")
        exit(0)

    all_results = {wl: {"Baseline_B": [], "Baseline_B_CostBreakdown": [], "ADSG_Candidate": []} for wl in workloads}

    print(f"Beginning 3-Arm ADSG Benchmark Suite (Budget: {args.budget} KB, Seeds: {args.seeds}, Model: {args.model})...")

    for wl in workloads:
        for seed in args.seeds:
            print(f"\n>>> Running Baseline_B on {wl} (Seed: {seed}) <<<")
            res = run_baseline_b(client, wl, args.budget, seed, max_steps=args.steps, model=args.model)
            all_results[wl]["Baseline_B"].append(res)

        for seed in args.seeds:
            print(f"\n>>> Running Baseline_B_CostBreakdown on {wl} (Seed: {seed}) <<<")
            res = run_baseline_b_cost_breakdown(client, wl, args.budget, seed, max_steps=args.steps, model=args.model)
            all_results[wl]["Baseline_B_CostBreakdown"].append(res)

        for seed in args.seeds:
            print(f"\n>>> Running ADSG_Candidate on {wl} (Seed: {seed}) <<<")
            res = run_adsg_candidate(client, wl, args.budget, seed, max_steps=args.steps, model=args.model)
            all_results[wl]["ADSG_Candidate"].append(res)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n==================================================================")
    print(f"All 27 trials completed! Saved results to {args.out}")
    print(f"==================================================================")

    # Perform statistical analysis
    analyze_and_report_results(all_results)
