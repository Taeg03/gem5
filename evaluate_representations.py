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
# 1. Oracle Lookup
# ---------------------------------------------------------
CURRENT_ORACLE_DF = None

def load_oracle(oracle_path: str):
    global CURRENT_ORACLE_DF
    CURRENT_ORACLE_DF = pd.read_csv(oracle_path)
    return CURRENT_ORACLE_DF

def oracle_lookup(l1_size: str, l1_assoc: int, l2_size: str, l2_assoc: int):
    """Deterministic CSV lookup for simulated metrics."""
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

# Baseline A: Implementation Level (Raw Simulator Flags)
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


# Baseline B: Microarchitectural Level (Textbook Variables)
class BaselineBSchema(BaseModel):
    l1d_capacity_kb: Literal["16", "32", "64", "128"]
    l1d_associativity: Literal["2", "4", "8", "16"]
    l2_capacity_kb: Literal["512", "1024", "2048", "4096"]
    l2_associativity: Literal["2", "4", "8", "16"]
    reasoning: str


# Baseline C: Semantic / Hierarchical Level
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
# 3. Adapters: Standardize Agent Output to Oracle Format
# ---------------------------------------------------------

def parse_baseline_a(data: dict):
    l1_size = data.get(
        "system.cpu.dcache.size", data.get("system_cpu_dcache_size")
    )
    l1_assoc = data.get(
        "system.cpu.dcache.assoc", data.get("system_cpu_dcache_assoc")
    )
    l2_size = data.get("system.l2cache.size", data.get("system_l2cache_size"))
    l2_assoc = data.get(
        "system.l2cache.assoc", data.get("system_l2cache_assoc")
    )
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


# ---------------------------------------------------------
# 4. Evaluation Loop
# ---------------------------------------------------------

BASELINES = {
    "Baseline_A": (
        BaselineASchema,
        parse_baseline_a,
        "Raw simulator parameter flags",
    ),
    "Baseline_B": (
        BaselineBSchema,
        parse_baseline_b,
        "Microarchitectural parameters",
    ),
    "Baseline_C": (
        BaselineCSchema,
        parse_baseline_c,
        "Hierarchical structural topology",
    ),
}


def run_trial(client, baseline_name: str, workload_name: str, oracle_path: str, max_steps: int = 10):
    load_oracle(oracle_path)
    initial_ipc = oracle_lookup("16kB", 2, "512kB", 2)
    best_oracle_ipc = CURRENT_ORACLE_DF["ipc"].max()
    schema_cls, parser_fn, desc = BASELINES[baseline_name]

    prompt_history = [
        f"You are an expert computer architect optimizing cache hierarchy parameters to maximize IPC on a target workload ({workload_name}).",
        f"Representation format: {desc}.",
        f"Available design choices:",
        f"- L1D Size: 16kB, 32kB, 64kB, 128kB | L1D Assoc: 2, 4, 8, 16",
        f"- L2 Size: 512kB, 1MB, 2MB, 4MB | L2 Assoc: 2, 4, 8, 16",
        f"Initial baseline config: L1D=16kB/2-way, L2=512kB/2-way. Initial IPC: {initial_ipc:.4f}.",
        f"Propose 1 configuration per step to find the optimal IPC within {max_steps} steps."
    ]

    history_log = []

    for step in range(1, max_steps + 1):
        full_prompt = (
            "\n".join(prompt_history)
            + f"\n\nStep {step}/{max_steps}: Propose the next architecture."
        )

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
                    print(
                        f"    [Rate Limited 429] Server requested backoff. Waiting {wait_time:.1f}s before retrying (attempt {attempt+1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                else:
                    raise e

        if response is None:
            raise RuntimeError(
                f"Failed to obtain response after {max_retries} attempts. Last error: {last_error}"
            )

        pacing_delay = float(os.environ.get("GEMINI_PACING_DELAY", "2.0"))
        if pacing_delay > 0:
            time.sleep(pacing_delay)

        proposal = json.loads(response.text)
        l1s, l1a, l2s, l2a = parser_fn(proposal)
        ipc = oracle_lookup(l1s, l1a, l2s, l2a)

        history_log.append(
            {
                "step": step,
                "config": (l1s, l1a, l2s, l2a),
                "ipc": ipc,
                "reasoning": proposal.get("reasoning", ""),
            }
        )

        # Feedback loop
        prompt_history.append(
            f"Step {step} Evaluated: Config=(L1:{l1s}/{l1a}-way, L2:{l2s}/{l2a}-way) -> Resulting IPC: {ipc:.4f}"
        )
        print(
            f"  [{workload_name} | {baseline_name}] Step {step:02d}: (L1:{l1s}/{l1a}-way, L2:{l2s}/{l2a}-way) -> IPC: {ipc:.4f} (Global Max: {best_oracle_ipc:.4f})"
        )

    return history_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DSE Representations across Workload Landscapes")
    parser.add_argument("--workload", type=str, default="suite", choices=["l1", "l2", "assoc", "composite", "suite"],
                        help="Target workload: 'l1', 'l2', 'assoc', 'composite', or 'suite' (all three)")
    parser.add_argument("--steps", type=int, default=10, help="Number of exploration steps per baseline")
    parser.add_argument("--out", type=str, default="workload_eval_results.json", help="Output JSON file for results")
    args = parser.parse_args()

    client = genai.Client()

    workloads = []
    if args.workload == "suite":
        workloads = [
            ("L1_Heavy", "oracle_l1.csv"),
            ("L2_Heavy", "oracle_l2.csv"),
            ("Assoc_Heavy", "oracle_assoc.csv"),
        ]
    elif args.workload == "composite":
        workloads = [("Composite_Workload", "oracle_results.csv")]
    else:
        workloads = [(f"{args.workload.upper()}_Workload", f"oracle_{args.workload}.csv")]

    all_results = {}
    print(f"Beginning Multi-Workload Representation Sweep across {len(workloads)} workload(s)...")

    for wl_name, oracle_file in workloads:
        print(f"\n==================================================================")
        print(f"Evaluating Workload: {wl_name} (Oracle: {oracle_file})")
        print(f"==================================================================")
        all_results[wl_name] = {}
        for b_name in BASELINES.keys():
            print(f"\n--- Running {b_name} on {wl_name} ---")
            all_results[wl_name][b_name] = run_trial(
                client, b_name, wl_name, oracle_file, max_steps=args.steps
            )

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n==================================================================")
    print(f"All sweeps completed! Results saved to {args.out}")
    print(f"==================================================================")
