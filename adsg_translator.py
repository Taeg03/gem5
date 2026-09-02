"""
adsg_translator.py

Deterministic Translator from Resolved ADSG Architecture to gem5 Configuration,
Cost Evaluation, and Oracle Lookup.

Guarantees:
- Strict preservation of all 6 architectural parameter values.
- Zero silent substitution of defaults or value alteration.
- Identical gem5 SimObject attribute construction as run_dse_729.py.
- Exact cost function and budget matching the reference oracle pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Union
import pandas as pd

# Global cache for oracle tables
_ORACLE_CACHE: Dict[str, pd.DataFrame] = {}

ORACLE_FILES = {
    "compute": "oracle_729_compute.csv",
    "latency": "oracle_729_latency.csv",
    "concurrency": "oracle_729_concurrency.csv",
    # Workload aliases
    "Workload_Alpha": "oracle_729_compute.csv",
    "Workload_Beta": "oracle_729_latency.csv",
    "Workload_Gamma": "oracle_729_concurrency.csv",
}

BUDGET_KB_DEFAULT = 1500.0


def parse_size_kb(s: str) -> int:
    """Parses size strings like '16kB', '512kB', '1MB', '2MB' into integer KB."""
    if "MB" in s:
        return int(s.replace("MB", "")) * 1024
    return int(s.replace("kB", ""))


def compute_cost_729(iw: int, rob: int, mshr: int, l1s: str, l1a: int, l2s: str) -> float:
    """
    Transparent hardware cost function combining Core and Memory subsystems:
    Cost = Core_Cost(iw*50 + rob*1.5 + mshr*5) + Memory_Cost(l1_kb*(1 + 0.1*l1a/8) + l2_kb)
    """
    l1_kb = parse_size_kb(l1s)
    l2_kb = parse_size_kb(l2s)
    core_cost = (int(iw) * 50) + (int(rob) * 1.5) + (int(mshr) * 5)
    mem_cost = l1_kb * (1.0 + 0.1 * (int(l1a) / 8.0)) + l2_kb
    return round(core_cost + mem_cost, 1)


@dataclass(frozen=True)
class Gem5Configuration:
    """
    Complete, verified representation of a gem5 CPU simulation configuration.
    """
    issue_width: int
    rob_size: int
    l1d_mshrs: int
    l1d_size: str
    l1d_assoc: int
    l2_size: str

    cost: float
    is_feasible: bool

    # Complete SimObject attributes dictionary exactly matching run_dse_729.py
    simobject_attrs: Dict[str, Any]

    # CLI args list matching run_dse_729.py invocation
    cli_args: Tuple[str, ...]

    def as_param_tuple(self) -> Tuple[int, int, int, str, int, str]:
        return (
            self.issue_width,
            self.rob_size,
            self.l1d_mshrs,
            self.l1d_size,
            self.l1d_assoc,
            self.l2_size,
        )


def build_gem5_config_from_params(
    iw: int, rob: int, mshr: int, l1s: str, l1a: int, l2s: str, budget_kb: float = BUDGET_KB_DEFAULT
) -> Gem5Configuration:
    """Constructs a Gem5Configuration directly from 6 architectural parameters."""
    cost = compute_cost_729(iw, rob, mshr, l1s, l1a, l2s)
    feasible = (cost <= budget_kb)

    simobject_attrs = {
        # Core pipeline widths
        "system.cpu.fetchWidth": int(iw),
        "system.cpu.decodeWidth": int(iw),
        "system.cpu.renameWidth": int(iw),
        "system.cpu.dispatchWidth": int(iw),
        "system.cpu.issueWidth": int(iw),
        "system.cpu.wbWidth": int(iw),
        "system.cpu.commitWidth": int(iw),
        # Speculative window and register file
        "system.cpu.numROBEntries": int(rob),
        "system.cpu.LQEntries": max(16, int(rob) // 4),
        "system.cpu.SQEntries": max(16, int(rob) // 4),
        "system.cpu.numPhysIntRegs": max(64, int(rob) + 32),
        "system.cpu.numPhysFloatRegs": max(64, int(rob) + 32),
        # Fixed L1I cache
        "system.cpu.icache.size": "32kB",
        "system.cpu.icache.assoc": 4,
        "system.cpu.icache.mshrs": 4,
        # Parameterized L1D cache
        "system.cpu.dcache.size": str(l1s),
        "system.cpu.dcache.assoc": int(l1a),
        "system.cpu.dcache.mshrs": int(mshr),
        # Parameterized L2 unified cache
        "system.l2cache.size": str(l2s),
        "system.l2cache.assoc": 8,
        "system.l2cache.mshrs": 20,
    }

    cli_args = (str(iw), str(rob), str(mshr), str(l1s), str(l1a), str(l2s))

    return Gem5Configuration(
        issue_width=int(iw),
        rob_size=int(rob),
        l1d_mshrs=int(mshr),
        l1d_size=str(l1s),
        l1d_assoc=int(l1a),
        l2_size=str(l2s),
        cost=cost,
        is_feasible=feasible,
        simobject_attrs=simobject_attrs,
        cli_args=cli_args,
    )


def extract_params_from_resolved_adsg(g_inst: Any) -> Tuple[int, int, int, str, int, str]:
    """
    Extracts the 6 architectural parameters from a resolved ADSG instance (DSGType).
    """
    extracted = {}
    for node in g_inst.graph.nodes:
        if hasattr(node, "obj_ref") and isinstance(node.obj_ref, dict) and "param" in node.obj_ref:
            extracted[node.obj_ref["param"]] = node.obj_ref["val"]

    required_keys = ["issue_width", "rob_size", "l1d_mshrs", "l1d_size", "l1d_assoc", "l2_size"]
    for k in required_keys:
        if k not in extracted:
            raise KeyError(f"Parameter '{k}' missing in resolved ADSG instance!")

    return (
        int(extracted["issue_width"]),
        int(extracted["rob_size"]),
        int(extracted["l1d_mshrs"]),
        str(extracted["l1d_size"]),
        int(extracted["l1d_assoc"]),
        str(extracted["l2_size"]),
    )


def adsg_to_gem5_config(g_inst: Any, budget_kb: float = BUDGET_KB_DEFAULT) -> Gem5Configuration:
    """
    Translates a resolved ADSG architecture instance into a Gem5Configuration.
    """
    params = extract_params_from_resolved_adsg(g_inst)
    return build_gem5_config_from_params(*params, budget_kb=budget_kb)


def get_oracle_df(workload_mode: str) -> pd.DataFrame:
    """Loads and caches oracle CSVs."""
    file_path = ORACLE_FILES.get(workload_mode, workload_mode)
    if file_path not in _ORACLE_CACHE:
        df = pd.read_csv(file_path)
        # Ensure proper types
        df["issue_width"] = df["issue_width"].astype(int)
        df["rob_size"] = df["rob_size"].astype(int)
        df["l1d_mshrs"] = df["l1d_mshrs"].astype(int)
        df["l1d_size"] = df["l1d_size"].astype(str)
        df["l1d_assoc"] = df["l1d_assoc"].astype(int)
        df["l2_size"] = df["l2_size"].astype(str)
        df["ipc"] = df["ipc"].astype(float)
        _ORACLE_CACHE[file_path] = df
    return _ORACLE_CACHE[file_path]


def lookup_oracle_ipc(
    iw: int, rob: int, mshr: int, l1s: str, l1a: int, l2s: str, workload_mode: str = "compute"
) -> float:
    """Queries ground-truth oracle for exact IPC."""
    df = get_oracle_df(workload_mode)
    match = df[
        (df["issue_width"] == int(iw))
        & (df["rob_size"] == int(rob))
        & (df["l1d_mshrs"] == int(mshr))
        & (df["l1d_size"] == str(l1s))
        & (df["l1d_assoc"] == int(l1a))
        & (df["l2_size"] == str(l2s))
    ]
    if match.empty:
        raise ValueError(
            f"Configuration (IW:{iw}, ROB:{rob}, MSHR:{mshr}, L1:{l1s}/{l1a}w, L2:{l2s}) not found in oracle '{workload_mode}'!"
        )
    return float(match.iloc[0]["ipc"])
