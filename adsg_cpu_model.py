"""
adsg_cpu_model.py

Minimal Faithful Architecture Design Space Graph (ADSG) Model
for the 6-Parameter Heterogeneous CPU Design Space (729 points).

Semantic Boundary & Rationale:
-------------------------------
This model represents the minimal, non-speculative structural and choice
decomposition of the 6-parameter CPU architecture space used in gem5:
1. Root Node ('CPU'):
   - Top-level system container.
2. Subsystems ('Core', 'Memory'):
   - Derived directly from 'CPU' via EdgeType.DERIVES.
   - Represents the physical compositional hierarchy (a CPU consists of an
     execution core and a memory hierarchy).
3. Selection Choices (SelectionChoiceNode):
   - Attached to 'Core':
     * 'issue_width' ∈ {2, 4, 8} (pipeline bandwidth)
     * 'rob_size'    ∈ {32, 64, 128} (instruction window depth)
     * 'l1d_mshrs'   ∈ {2, 4, 8} (L1D non-blocking miss registers)
   - Attached to 'Memory':
     * 'l1d_size'    ∈ {16kB, 32kB, 64kB} (L1 Data cache capacity)
     * 'l1d_assoc'   ∈ {2, 4, 8} (L1 Data cache set associativity)
     * 'l2_size'     ∈ {512kB, 1MB, 2MB} (L2 unified cache capacity)
4. Metrics (MetricNode):
   - 'cost': Hardware area/cost constraint (<= 1500 KB equivalent)
   - 'ipc': Performance objective (to maximize)

No artificial incompatibility rules, spurious bypass edges, or unneeded
connection choices are added. The model adheres strictly to the 3^6 = 729
space without relying on projection or imputation for legal vectors.
"""

from typing import List, Dict, Tuple, Any, Sequence, Union
from adsg_core.graph import (
    BasicDSG,
    NamedNode,
    SelectionChoiceNode,
    MetricNode,
    MetricType,
    EdgeType,
)
from adsg_core.optimization import GraphProcessor

# Canonical parameter definitions and discrete option values
PARAM_NAMES: List[str] = [
    "issue_width",
    "rob_size",
    "l1d_mshrs",
    "l1d_size",
    "l1d_assoc",
    "l2_size",
]

PARAM_OPTIONS: Dict[str, List[Any]] = {
    "issue_width": [2, 4, 8],
    "rob_size": [32, 64, 128],
    "l1d_mshrs": [2, 4, 8],
    "l1d_size": ["16kB", "32kB", "64kB"],
    "l1d_assoc": [2, 4, 8],
    "l2_size": ["512kB", "1MB", "2MB"],
}

# Value-to-index mapping for fast vector encoding
VAL_TO_IDX: Dict[str, Dict[Any, int]] = {
    p: {val: idx for idx, val in enumerate(opts)}
    for p, opts in PARAM_OPTIONS.items()
}


class CPUArchitectureDSG(BasicDSG):
    """
    Custom BasicDSG subclass ensuring canonical sorting order
    for the 6 architectural choices:
    issue_width -> rob_size -> l1d_mshrs -> l1d_size -> l1d_assoc -> l2_size.
    """

    def _choice_sort_key(self, choice_node) -> Tuple[int, str]:
        dec_id = getattr(choice_node, "decision_id", "")
        idx = PARAM_NAMES.index(dec_id) if dec_id in PARAM_NAMES else 99
        return idx, dec_id


def build_cpu_adsg() -> CPUArchitectureDSG:
    """
    Constructs the minimal faithful 6-parameter CPU ADSG.
    Total declared designs = 729, valid designs = 729.
    """
    dsg = CPUArchitectureDSG()

    # 1. Structural Subsystem Nodes
    cpu = NamedNode("CPU")
    core = NamedNode("Core")
    mem = NamedNode("Memory")

    # Derivation edges: A CPU requires both a Core and a Memory hierarchy
    dsg.add_edge(cpu, core, edge_type=EdgeType.DERIVES)
    dsg.add_edge(cpu, mem, edge_type=EdgeType.DERIVES)

    # 2. Core Execution Engine Selection Choices
    for param in ["issue_width", "rob_size", "l1d_mshrs"]:
        options = [
            NamedNode(
                f"{param}_{val}",
                obj_ref={"param": param, "val": val, "idx": idx}
            )
            for idx, val in enumerate(PARAM_OPTIONS[param])
        ]
        dsg.add_selection_choice(param, core, options, is_ordinal=True)

    # 3. Memory Hierarchy Selection Choices
    for param in ["l1d_size", "l1d_assoc", "l2_size"]:
        options = [
            NamedNode(
                f"{param}_{val}",
                obj_ref={"param": param, "val": val, "idx": idx}
            )
            for idx, val in enumerate(PARAM_OPTIONS[param])
        ]
        dsg.add_selection_choice(param, mem, options, is_ordinal=True)

    # 4. Problem Metrics
    cost_metric = MetricNode("cost", direction=-1, ref=1500.0, type_=MetricType.CONSTRAINT)
    ipc_metric = MetricNode("ipc", direction=1, type_=MetricType.OBJECTIVE)
    dsg.add_edge(cpu, cost_metric, edge_type=EdgeType.DERIVES)
    dsg.add_edge(cpu, ipc_metric, edge_type=EdgeType.DERIVES)

    # Set root starting node
    dsg = dsg.set_start_nodes({cpu})
    return dsg


def get_cpu_graph_processor(dsg: CPUArchitectureDSG = None) -> GraphProcessor:
    """Returns a GraphProcessor initialized with the CPU ADSG."""
    if dsg is None:
        dsg = build_cpu_adsg()
    return GraphProcessor(dsg)


def encode_values_to_indices(values: Sequence[Any]) -> List[int]:
    """Converts a 6-parameter value sequence to a discrete index vector x in {0, 1, 2}^6."""
    if len(values) != 6:
        raise ValueError(f"Expected 6 parameter values, got {len(values)}: {values}")
    indices = []
    for p, val in zip(PARAM_NAMES, values):
        if val not in VAL_TO_IDX[p]:
            raise ValueError(f"Value '{val}' is not valid for parameter '{p}' (allowed: {PARAM_OPTIONS[p]})")
        indices.append(VAL_TO_IDX[p][val])
    return indices


def decode_indices_to_values(indices: Sequence[int]) -> Tuple[int, int, int, str, int, str]:
    """Converts discrete index vector x in {0, 1, 2}^6 to canonical 6-parameter value tuple."""
    if len(indices) != 6:
        raise ValueError(f"Expected 6 indices, got {len(indices)}: {indices}")
    return (
        PARAM_OPTIONS["issue_width"][indices[0]],
        PARAM_OPTIONS["rob_size"][indices[1]],
        PARAM_OPTIONS["l1d_mshrs"][indices[2]],
        PARAM_OPTIONS["l1d_size"][indices[3]],
        PARAM_OPTIONS["l1d_assoc"][indices[4]],
        PARAM_OPTIONS["l2_size"][indices[5]],
    )


def resolve_adsg_vector(
    gp: GraphProcessor,
    vector: Sequence[Union[int, Any]]
) -> Tuple[Any, List[int], Tuple[int, int, int, str, int, str]]:
    """
    Resolves an input vector (either discrete indices or values) into a final ADSG instance.
    Returns:
      (resolved_graph_instance, resolved_indices, resolved_values)
    """
    # Determine if input is discrete indices or values
    if all(isinstance(v, (int, float)) and int(v) in (0, 1, 2) for v in vector) and len(vector) == 6:
        idx_vector = [int(v) for v in vector]
    else:
        idx_vector = encode_values_to_indices(vector)

    # Resolve graph through ADSG GraphProcessor
    g_inst, imputed_x, is_active = gp.get_graph(idx_vector, create=True)

    # Extract verified parameters from resolved graph nodes
    extracted_params = {}
    for node in g_inst.graph.nodes:
        if hasattr(node, "obj_ref") and isinstance(node.obj_ref, dict) and "param" in node.obj_ref:
            p = node.obj_ref["param"]
            extracted_params[p] = {
                "val": node.obj_ref["val"],
                "idx": node.obj_ref["idx"],
            }

    if len(extracted_params) != 6:
        raise RuntimeError(
            f"Failed to extract all 6 parameters from resolved ADSG! Found {len(extracted_params)}: {extracted_params}"
        )

    resolved_values = (
        extracted_params["issue_width"]["val"],
        extracted_params["rob_size"]["val"],
        extracted_params["l1d_mshrs"]["val"],
        extracted_params["l1d_size"]["val"],
        extracted_params["l1d_assoc"]["val"],
        extracted_params["l2_size"]["val"],
    )

    resolved_indices = [extracted_params[p]["idx"] for p in PARAM_NAMES]

    return g_inst, resolved_indices, resolved_values
