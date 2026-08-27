import json
import os
import re
import time
from typing import Literal

import pandas as pd
from google import genai
from google.genai import types
from pydantic import (
    BaseModel,
    Field,
)

# ---------------------------------------------------------
# 1. Load Oracle Dataset
# ---------------------------------------------------------
ORACLE_DF = pd.read_csv("oracle_results.csv")


def oracle_lookup(l1_size: str, l1_assoc: int, l2_size: str, l2_assoc: int):
    """Deterministic CSV lookup for simulated metrics."""
    match = ORACLE_DF[
        (ORACLE_DF["l1d_size"] == str(l1_size))
        & (ORACLE_DF["l1d_assoc"] == int(l1_assoc))
        & (ORACLE_DF["l2_size"] == str(l2_size))
        & (ORACLE_DF["l2_assoc"] == int(l2_assoc))
    ]
    if match.empty:
        return None
    return match.iloc[0]["ipc"]


# ---------------------------------------------------------
# 2. Strict Pydantic Schemas for Structured Outputs
# ---------------------------------------------------------
# Note: Google GenAI Schema enums require string values (Literal["..."]).
# Numeric parameters are typed as string literals here and converted
# to integers in the adapter/parser layer.


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
        "16": "16kB",
        "32": "32kB",
        "64": "64kB",
        "128": "128kB",
        "512": "512kB",
        "1024": "1MB",
        "2048": "2MB",
        "4096": "4MB",
        16: "16kB",
        32: "32kB",
        64: "64kB",
        128: "128kB",
        512: "512kB",
        1024: "1MB",
        2048: "2MB",
        4096: "4MB",
    }
    return (
        size_map[str(data["l1d_capacity_kb"])],
        int(data["l1d_associativity"]),
        size_map[str(data["l2_capacity_kb"])],
        int(data["l2_associativity"]),
    )


def parse_baseline_c(data: dict):
    size_map = {
        "16": "16kB",
        "32": "32kB",
        "64": "64kB",
        "128": "128kB",
        "512": "512kB",
        "1024": "1MB",
        "2048": "2MB",
        "4096": "4MB",
        16: "16kB",
        32: "32kB",
        64: "64kB",
        128: "128kB",
        512: "512kB",
        1024: "1MB",
        2048: "2MB",
        4096: "4MB",
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


def run_trial(client, baseline_name: str, max_steps: int = 10):
    schema_cls, parser_fn, desc = BASELINES[baseline_name]

    prompt_history = [
        f"You are an expert computer architect optimizing cache hierarchy parameters to maximize IPC.",
        f"Representation format: {desc}.",
        f"Available design choices:",
        f"- L1D Size: 16kB, 32kB, 64kB, 128kB | L1D Assoc: 2, 4, 8, 16",
        f"- L2 Size: 512kB, 1MB, 2MB, 4MB | L2 Assoc: 2, 4, 8, 16",
        f"Initial baseline config: L1D=16kB/2-way, L2=512kB/2-way. Initial IPC: 0.5959.",
        f"Propose 1 configuration per step to find the optimal IPC within {max_steps} steps.",
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
                    # Extract exact retry-after duration from Google response if available
                    match = re.search(r"retry in ([0-9.]+)s", err_str)
                    if match:
                        wait_time = float(match.group(1)) + 2.0
                    else:
                        wait_time = max(35.0, 20.0 * (attempt + 1))
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

        # Pacing sleep to stay under free tier 5 RPM limit (12.5s pacing = ~4.8 RPM)
        pacing_delay = float(os.environ.get("GEMINI_PACING_DELAY", "12.5"))
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
            f"  [{baseline_name}] Step {step:02d}: (L1:{l1s}/{l1a}-way, L2:{l2s}/{l2a}-way) -> IPC: {ipc:.4f}"
        )

    return history_log


if __name__ == "__main__":
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in (
        "1",
        "true",
        "yes",
    )
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
        "VERTEX_PROJECT"
    )
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    if use_vertex or project:
        print(
            f"Connecting to Gemini via Vertex AI (Project: {project}, Location: {location})..."
        )
        client = genai.Client(
            vertexai=True, project=project, location=location
        )
    else:
        client = genai.Client()
    print("Beginning Agentic Representation Evaluation Sweep...")

    experiment_results = {}
    for name in BASELINES.keys():
        print(f"\n--- Running {name} ---")
        experiment_results[name] = run_trial(client, name, max_steps=10)

    with open("experiment_eval_results.json", "w") as f:
        json.dump(experiment_results, f, indent=2)

    print(
        "\nEvaluation complete! Results saved to experiment_eval_results.json"
    )
