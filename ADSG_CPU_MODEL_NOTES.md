# ADSG Representation & Equivalence Verification Notes

## 1. Overview & Scientific Boundary

This document records the design decisions, semantic mappings, and empirical equivalence verification for the Architecture Design Space Graph (ADSG) representation of the 6-parameter heterogeneous CPU design space ($3^6 = 729$ configurations).

As specified in the research directive:
- **ADSG is an underlying formal representation layer, not a search algorithm.**
- **ADSG structural validity $\neq$ gem5 simulation validity.**
- **No speculative or artificial constructs were introduced:** Every node, edge, and choice directly mirrors the physical and microarchitectural reality of the existing gem5 CPU model.
- **Strict Equivalence:** The ADSG pipeline must reproduce the reference pipeline with zero tolerance for value alterations, silent projections, or imputation of legal vectors.

---

## 2. Parameter Mapping to ADSG Primitives

The 6 parameters are partitioned into the Core Execution Engine and Memory Hierarchy subsystems. In ADSG, architectural decisions are modeled as `SelectionChoiceNode`s attached to their respective subsystem nodes via `EdgeType.DERIVES`.

| Architectural Parameter | Subsystem Parent | ADSG Node Class | Choice Options (`NamedNode`) | gem5 SimObject Attribute Target |
| :--- | :--- | :--- | :--- | :--- |
| **`issue_width`** | `Core` | `SelectionChoiceNode(is_ordinal=True)` | `IW_2`, `IW_4`, `IW_8` | `system.cpu.{fetch,decode,rename,dispatch,issue,wb,commit}Width` |
| **`rob_size`** | `Core` | `SelectionChoiceNode(is_ordinal=True)` | `ROB_32`, `ROB_64`, `ROB_128` | `system.cpu.numROBEntries`, `LQEntries`, `SQEntries`, `numPhys{Int,Float}Regs` |
| **`l1d_mshrs`** | `Core` | `SelectionChoiceNode(is_ordinal=True)` | `MSHR_2`, `MSHR_4`, `MSHR_8` | `system.cpu.dcache.mshrs` |
| **`l1d_size`** | `Memory` | `SelectionChoiceNode(is_ordinal=True)` | `L1_16kB`, `L1_32kB`, `L1_64kB` | `system.cpu.dcache.size` |
| **`l1d_assoc`** | `Memory` | `SelectionChoiceNode(is_ordinal=True)` | `L1A_2`, `L1A_4`, `L1A_8` | `system.cpu.dcache.assoc` |
| **`l2_size`** | `Memory` | `SelectionChoiceNode(is_ordinal=True)` | `L2_512kB`, `L2_1MB`, `L2_2MB` | `system.l2cache.size` (assoc=8 fixed) |

In addition, two `MetricNode`s are attached to the root `CPU` node:
- `MetricNode("cost", direction=-1, ref=1500.0, type_=MetricType.CONSTRAINT)`: Defines the hardware area constraint ($\le 1500\text{ KB}$).
- `MetricNode("ipc", direction=1, type_=MetricType.OBJECTIVE)`: Defines the objective metric to maximize.

---

## 3. ADSG Nodes & Edges: Semantic Rationale

### A. Nodes
1. **`NamedNode("CPU")`**: Root starting node (`derivation_start_nodes`). Represents the complete processor system instance.
2. **`NamedNode("Core")`**: Represents the Out-of-Order superscalar execution engine subsystem.
3. **`NamedNode("Memory")`**: Represents the non-blocking cache hierarchy subsystem.
4. **`SelectionChoiceNode(...)`**: Represents mutually exclusive architectural decisions. The `is_ordinal=True` flag denotes that options have a natural monotone ordering ($2 < 4 < 8$).
5. **Option Nodes (`NamedNode(...)`)**: Concrete option choices carrying `obj_ref` metadata specifying exact parameter names, string/integer values, and discrete indices.

### B. Edges
1. **`EdgeType.DERIVES` (CPU $\to$ Core, CPU $\to$ Memory)**:
   - *Semantics:* Compositional requirement. Any valid CPU architecture instance requires both an execution core and a memory hierarchy.
2. **`EdgeType.DERIVES` (Core $\to$ SelectionChoiceNodes, Memory $\to$ SelectionChoiceNodes)**:
   - *Semantics:* Activation dependency. Instantiating the Core activates decisions for `issue_width`, `rob_size`, and `l1d_mshrs`. Instantiating Memory activates decisions for `l1d_size`, `l1d_assoc`, and `l2_size`.
3. **`EdgeType.DERIVES` (SelectionChoiceNodes $\to$ Option Nodes)**:
   - *Semantics:* Alternative branches. When a decision is resolved, the chosen branch is retained and confirmed; unselected alternative branches are pruned.

---

## 4. ADSG Capabilities NOT Exercised by the Current 729-Point Space

To ensure scientific control and prevent artificial complexity, several advanced ADSG capabilities were intentionally omitted:

1. **`EdgeType.INCOMPATIBILITY` (Cross-Subsystem Pruning)**:
   - *Why Omitted:* In gem5's `DerivO3CPU` and classic cache model, all 729 configurations are syntactically and architecturally simulatable (e.g. an 8-wide core with a 32-entry ROB or a 16kB 2-way L1 is physical in gem5, even if bottlenecked). Adding synthetic incompatibility edges would violate ground-truth correspondence.
2. **`ConnectionChoiceNode` / `ConnectorNode` (Dynamic Topologies)**:
   - *Why Omitted:* The interconnect topology in `run_dse_729.py` is fixed (Core $\to$ L1D $\to$ L2XBar $\to$ L2 $\to$ MemBus $\to$ DRAM). The 729-point space evaluates sizing and pipeline bandwidth rather than dynamic bus routing or crossbar permutations.
3. **Conditional Subsystem Existence ($\delta = \text{False}$)**:
   - *Why Omitted:* L2 cache is always present in this study (ranging from 512kB to 2MB). There is no "No-L2" option node that would conditionally deactivate L2 sizing parameters.
4. **`ChoiceConstraint` (e.g. `LINKED`, `PERMUTATION`)**:
   - *Why Omitted:* The 6 parameters vary independently across their discrete value grids.

---

## 5. Verification Results

All 729 vectors were exhaustively verified using [`verify_adsg_equivalence.py`](verify_adsg_equivalence.py):
- **Design-Vector Equivalence [ADSG(x) = x]:** 729 / 729 (100.0%)
- **gem5 Configuration Equivalence:** 729 / 729 (100.0%)
- **Feasibility Equivalence ($\le 1500.0\text{ KB}$):** 729 / 729 (100.0%)
- **IPC Equivalence (`compute`):** 729 / 729 (100.0%)
- **IPC Equivalence (`latency`):** 729 / 729 (100.0%)
- **IPC Equivalence (`concurrency`):** 729 / 729 (100.0%)
- **Unresolved / Unmapped States:** 0 / 729 (0.0%)

**Conclusion:** `ADSG REPRESENTATION VALIDATED: PASS`
