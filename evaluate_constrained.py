import argparse
import json
import os
import re
import time
from typing import Literal

# Load .env if present
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v.strip("\"'")

import pandas as pd
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

def compute_cost(l1_size: str, l1_assoc: int, l2_size: str, l2_assoc: int) -> float:
    """Explicit, transparent cost model: Capacity (KB) + 10% max assoc scaling."""
    l1_kb = parse_size_kb(l1_size)
    l2_kb = parse_size_kb(l2_size)
    cost = l1_kb * (1.0 + 0.1 * (l1_assoc / 16.0)) + l2_kb * (1.0 + 0.1 * (l2_assoc / 16.0))
    return round(cost, 1)

def load_oracle(oracle_path: str):
    global CURRENT_ORACLE_DF
    CURRENT_ORACLE_DF = pd.read_csv(oracle_path)
    return CURRENT_ORACLE_DF

def oracle_lookup(l1_size: str, l1_assoc: int, l2_size: str, l2_assoc: int):
    match = CURRENT_ORACLE_DF[
        (CURRENT_ORACLE_DF["l1d_size"] == str(l1_size))
        & (CURRENT_ORACLE_DF["l1d_assoc"] == int(l1_assoc))
        & (CURRENT_ORACLE_DF["l2_size"] == str(l2_size))
        & (CURRENT_ORACLE_DF["l2_assoc"] == int(l2_assoc))
    ]
    if match.empty:
        return None
    return float(match.iloc[0]["ipc"])


# ---------------------------------------------------------
# 2. Strict Pydantic Schemas for Structured Outputs
# ---------------------------------------------------------

class BaselineASchema(BaseModel):
    system_cpu_dcache_size: Literal["16kB", "32kB", "64kB", "128kB"] = Field(
        ..., alias="system.cpu.dcache.size"
    )
    system_cpu_dcache_assoc: Literal["2", "4", "8", "16"] = Field(
        ..., alias="system.cpu.dcache.assoc"
    )
    system_l2cache_size: Literal["512kB", "1MB", "2MB", "4MB"] = Field(
        ..., alias="system.l2cache.size"
    )
    system_l2cache_assoc: Literal["2", "4", "8", "16"] = Field(
        ..., alias="system.l2cache.assoc"
    )
    reasoning: str


class BaselineBSchema(BaseModel):
    l1d_capacity_kb: Literal["16", "32", "64", "128"]
    l1d_associativity: Literal["2", "4", "8", "16"]
    l2_capacity_kb: Literal["512", "1024", "2048", "4096"]
    l2_associativity: Literal["2", "4", "8", "16"]
    reasoning: str


class L1CacheConfig(BaseModel):
    capacity_kb: Literal["16", "32", "64", "128"]
    associativity_ways: Literal["2", "4", "8", "16"]


class L2CacheConfig(BaseModel):
    capacity_kb: Literal["512", "1024", "2048", "4096"]
    associativity_ways: Literal["2", "4", "8", "16"]


class MemoryHierarchy(BaseModel):
    l1_data_cache: L1CacheConfig
    l2_unified_cache: L2CacheConfig


class BaselineCSchema(BaseModel):
    memory_hierarchy: MemoryHierarchy
    reasoning: str


# ---------------------------------------------------------
# 3. Adapters
# ---------------------------------------------------------

def parse_baseline_a(data: dict):
    l1_size = data.get("system.cpu.dcache.size", data.get("system_cpu_dcache_size"))
    l1_assoc = data.get("system.cpu.dcache.assoc", data.get("system_cpu_dcache_assoc"))
    l2_size = data.get("system.l2cache.size", data.get("system_l2cache_size"))
    l2_assoc = data.get("system.l2cache.assoc", data.get("system_l2cache_assoc"))
    return (str(l1_size), int(l1_assoc), str(l2_size), int(l2_assoc))

def parse_baseline_b(data: dict):
    size_map = {
        "16": "16kB", "32": "32kB", "64": "64kB", "128": "128kB",
        "512": "512kB", "1024": "1MB", "2048": "2MB", "4096": "4MB",
        16: "16kB", 32: "32kB", 64: "64kB", 128: "128kB",
        512: "512kB", 1024: "1MB", 2048: "2MB", 4096: "4MB",
    }
    return (
        size_map[str(data["l1d_capacity_kb"])],
        int(data["l1d_associativity"]),
        size_map[str(data["l2_capacity_kb"])],
        int(data["l2_associativity"]),
    )

def parse_baseline_c(data: dict):
    size_map = {
        "16": "16kB", "32": "32kB", "64": "64kB", "128": "128kB",
        "512": "512kB", "1024": "1MB", "2048": "2MB", "4096": "4MB",
        16: "16kB", 32: "32kB", 64: "64kB", 128: "128kB",
        512: "512kB", 1024: "1MB", 2048: "2MB", 4096: "4MB",
    }
    l1 = data["memory_hierarchy"]["l1_data_cache"]
    l2 = data["memory_hierarchy"]["l2_unified_cache"]
    return (
        size_map[str(l1["capacity_kb"])],
        int(l1["associativity_ways"]),
        size_map[str(l2["capacity_kb"])],
        int(l2["associativity_ways"]),
    )


BASELINES = {
    "Baseline_A": (BaselineASchema, parse_baseline_a, "Raw simulator parameter flags"),
    "Baseline_B": (BaselineBSchema, parse_baseline_b, "Microarchitectural parameters"),
    "Baseline_C": (BaselineCSchema, parse_baseline_c, "Hierarchical structural topology"),
}


def run_constrained_trial(client, baseline_name: str, workload_name: str, oracle_path: str, budget_kb: float, max_steps: int = 10):
    load_oracle(oracle_path)
    initial_ipc = oracle_lookup("16kB", 2, "512kB", 2)
    initial_cost = compute_cost("16kB", 2, "512kB", 2)
    
    # Compute ground truth constrained optimum
    CURRENT_ORACLE_DF["cost"] = CURRENT_ORACLE_DF.apply(
        lambda r: compute_cost(r["l1d_size"], r["l1d_assoc"], r["l2_size"], r["l2_assoc"]), axis=1
    )
    feasible_df = CURRENT_ORACLE_DF[CURRENT_ORACLE_DF["cost"] <= budget_kb]
    best_feasible_row = feasible_df.sort_values(by="ipc", ascending=False).iloc[0] if not feasible_df.empty else None
    opt_ipc = best_feasible_row["ipc"] if best_feasible_row is not None else 0.0
    opt_cfg = (best_feasible_row["l1d_size"], int(best_feasible_row["l1d_assoc"]),
               best_feasible_row["l2_size"], int(best_feasible_row["l2_assoc"])) if best_feasible_row is not None else ()

    schema_cls, parser_fn, desc = BASELINES[baseline_name]

    prompt_history = [
        f"You are an expert computer architect optimizing cache hierarchy parameters under a strict hardware area/cost budget.",
        f"Target workload: {workload_name}.",
        f"HARDWARE BUDGET CONSTRAINT: Total Cost <= {budget_kb} KB.",
        f"Cost formula: Cost ≈ L1_Capacity*(1 + 0.1*L1_Assoc/16) + L2_Capacity*(1 + 0.1*L2_Assoc/16).",
        f"Goal: Maximize IPC within 10 steps. Any configuration exceeding the budget of {budget_kb} KB is INFEASIBLE and cannot be selected as the final design.",
        f"Representation format: {desc}.",
        f"Available design choices:",
        f"- L1D Size: 16kB, 32kB, 64kB, 128kB | L1D Assoc: 2, 4, 8, 16",
        f"- L2 Size: 512kB, 1MB, 2MB, 4MB | L2 Assoc: 2, 4, 8, 16",
        f"Initial baseline config: L1D=16kB/2-way, L2=512kB/2-way (Cost: {initial_cost} KB). Initial IPC: {initial_ipc:.4f}.",
        f"Propose 1 configuration per step."
    ]

    history_log = []

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

        pacing_delay = float(os.environ.get("GEMINI_PACING_DELAY", "2.0"))
        if pacing_delay > 0:
            time.sleep(pacing_delay)

        proposal = json.loads(response.text)
        l1s, l1a, l2s, l2a = parser_fn(proposal)
        ipc = oracle_lookup(l1s, l1a, l2s, l2a)
        cost = compute_cost(l1s, l1a, l2s, l2a)
        is_feasible = (cost <= budget_kb)
        feasibility_status = "FEASIBLE" if is_feasible else f"INFEASIBLE (Exceeds budget by +{cost - budget_kb:.1f} KB)"

        history_log.append({
            "step": step,
            "config": (l1s, l1a, l2s, l2a),
            "ipc": ipc,
            "cost": cost,
            "feasible": is_feasible,
            "reasoning": proposal.get("reasoning", "")
        })

        # Feedback loop providing true IPC + explicit feasibility signal
        prompt_history.append(
            f"Step {step} Evaluated: Config=(L1:{l1s}/{l1a}-way, L2:{l2s}/{l2a}-way) -> IPC: {ipc:.4f}, Cost: {cost} KB [{feasibility_status}]"
        )
        print(
            f"  [{workload_name} | B={budget_kb} | {baseline_name}] Step {step:02d}: (L1:{l1s}/{l1a}w, L2:{l2s}/{l2a}w) -> IPC: {ipc:.4f}, Cost: {cost} KB [{feasibility_status}] (Opt: {opt_ipc:.4f})"
        )

    return history_log, opt_ipc, opt_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DSE Representations under Resource Constraints")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1100, 2200], help="List of budget constraints in KB")
    parser.add_argument("--steps", type=int, default=10, help="Steps per baseline trial")
    parser.add_argument("--out", type=str, default="constrained_eval_results.json", help="Output JSON results")
    args = parser.parse_args()

    client = genai.Client()

    workloads = [
        ("L1_Heavy", "oracle_l1.csv"),
        ("L2_Heavy", "oracle_l2.csv"),
        ("Assoc_Heavy", "oracle_assoc.csv"),
    ]

    all_results = {}
    print(f"Beginning Constrained Representation Sweep across {len(workloads)} workloads and {len(args.budgets)} budgets...")

    for B in args.budgets:
        all_results[f"Budget_{B}KB"] = {}
        for wl_name, oracle_file in workloads:
            print(f"\n==================================================================")
            print(f"Workload: {wl_name} | Budget Constraint: {B} KB (Oracle: {oracle_file})")
            print(f"==================================================================")
            all_results[f"Budget_{B}KB"][wl_name] = {}
            for b_name in BASELINES.keys():
                print(f"\n--- Running {b_name} on {wl_name} (Budget: {B} KB) ---")
                history, opt_ipc, opt_cfg = run_constrained_trial(
                    client, b_name, wl_name, oracle_file, budget_kb=B, max_steps=args.steps
                )
                all_results[f"Budget_{B}KB"][wl_name][b_name] = {
                    "history": history,
                    "constrained_optimal_ipc": opt_ipc,
                    "constrained_optimal_cfg": opt_cfg
                }

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n==================================================================")
    print(f"Constrained sweeps completed! Saved to {args.out}")
    print(f"==================================================================")
