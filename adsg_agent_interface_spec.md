# ADSG Agent Interface Specification: Schema, Action Grammar, & Focal Mechanics

## 1. Objective & Scope

This specification formalizes the exact schemas, parsing pipelines, and execution mechanisms for the **ADSG-Grounded Candidate Agent Interface**.

---

## 2. Action Grammar Definition: Unified Atomic 6D Proposal

### Problem Statement
In Baseline D, actions were restricted to mutating only one subsystem while freezing the other. This created a serial dependency: an agent that scaled the core and violated the budget had to wait until the *next turn* to mutate memory to regain feasibility, doubling exploration latency and inactive mutations.

### Solution: Dual-Channel Schema with Atomic Evaluation
The candidate action schema provides two distinct channels within a **single turn**:
1. **Focal Channel (`focal_subsystem_mutations`)**: Mandatory mutations concentrated on the designated focal subsystem.
2. **Compensatory Channel (`compensatory_adjustments`)**: Optional adjustments to the background subsystem to offset cost or fine-tune balance.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    AGENT PROPOSAL (Single JSON Object)                     │
│                                                                            │
│  1. focal_subsystem_mutations: { ... }  [MANDATORY FOCUS OF REASONING]     │
│  2. compensatory_adjustments:   { ... }  [OPTIONAL BUDGET REBALANCING]      │
│  3. reasoning:                 "..."                                      │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│               ATOMIC 6D RECONSTRUCTION & ADSG RESOLUTION                   │
│                                                                            │
│  Step A: Retrieve confirmed system state x_prev in Z^6                     │
│  Step B: Overlay focal mutations -> x_intermediate                         │
│  Step C: Overlay compensatory adjustments (if provided) -> x_candidate     │
│  Step D: If any parameter is unmentioned, retain from x_prev               │
│  Step E: Resolve x_candidate atomically via adsg_cpu_model.py              │
│  Step F: Evaluate single 6D Gem5Configuration in gem5 / oracle             │
└────────────────────────────────────────────────────────────────────────────┘
```

> ### [!IMPORTANT]
> **Atomic Execution Guarantee**
> The candidate proposal is **NOT** evaluated as two sequential operations. It is parsed, reconstructed, and evaluated **atomically as a single 6D configuration**.
> There are no intermediate partial states, no intermediate gem5 runs, and no half-step budget penalties.

### Concrete Pydantic Schemas

#### Core Subsystem Sizing Schema
```python
class CoreSubsystemUpdate(BaseModel):
    issue_width: Optional[Literal["2", "4", "8"]] = Field(
        default=None, description="Pipeline issue, decode, and commit width."
    )
    rob_size: Optional[Literal["32", "64", "128"]] = Field(
        default=None, description="Reorder buffer entries and register file scaling."
    )
    l1d_mshrs: Optional[Literal["2", "4", "8"]] = Field(
        default=None, description="L1 Data cache Miss Status Holding Registers."
    )
```

#### Memory Subsystem Sizing Schema
```python
class MemorySubsystemUpdate(BaseModel):
    l1d_capacity_kb: Optional[Literal["16", "32", "64"]] = Field(
        default=None, description="L1 Data cache capacity."
    )
    l1d_associativity: Optional[Literal["2", "4", "8"]] = Field(
        default=None, description="L1 Data cache associativity."
    )
    l2_capacity_kb: Optional[Literal["512", "1024", "2048"]] = Field(
        default=None, description="L2 unified cache capacity."
    )
```

#### Candidate Interface Action Schemas (Per Focal Subsystem Mode)

##### Mode 1: Core-Focal Action Schema (`CoreFocalActionSchema`)
When the environment or scheduler sets the focus to **Core Execution Engine**:
```python
class CoreFocalActionSchema(BaseModel):
    focal_core_mutations: CoreSubsystemUpdate = Field(
        ..., description="Primary architectural mutations targeting the Core Execution Engine."
    )
    compensatory_memory_adjustments: Optional[MemorySubsystemUpdate] = Field(
        default=None, description="Optional adjustments to Memory Hierarchy to offset area or balance bandwidth."
    )
    reasoning: str = Field(
        ..., description="Architectural rationale explaining expected IPC benefit and budget impact."
    )
```

##### Mode 2: Memory-Focal Action Schema (`MemoryFocalActionSchema`)
When the environment or scheduler sets the focus to **Memory Hierarchy**:
```python
class MemoryFocalActionSchema(BaseModel):
    focal_memory_mutations: MemorySubsystemUpdate = Field(
        ..., description="Primary architectural mutations targeting the Memory Hierarchy."
    )
    compensatory_core_adjustments: Optional[CoreSubsystemUpdate] = Field(
        default=None, description="Optional adjustments to Core Execution to offset area or balance issue width."
    )
    reasoning: str = Field(
        ..., description="Architectural rationale explaining expected IPC benefit and budget impact."
    )
```

---

## 3. Focal Selection Mechanism: Eliminating Cognitive Meta-Reasoning Overhead

### The Baseline D Meta-Reasoning Trap
In Baseline D, the LLM was prompted:
`"Select target_subsystem ('core_execution', 'memory_hierarchy', or 'both') and provide parameters."`
This introduced severe failure modes:
1. **Misattribution Latency**: On `Workload_Beta` (pure memory latency bound), the agent spent 3 consecutive steps selecting `target_subsystem: core_execution`. Because core scaling produced zero IPC gain, the agent was baffled and repeatedly mutated ROB sizes before switching.
2. **Attention Dilution**: The agent spent reasoning tokens debating *which* subsystem to pick rather than optimizing parameters.

### Focal Selection Design Alternatives

| Mechanism | Description | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Option 1: Agent-Chosen Focus** | Agent selects `target_subsystem` dynamically in the JSON prompt. | Maximum agent autonomy. | Reintroduces the cognitive overhead and misattribution traps of Baseline D. |
| **Option 2: External Round-Robin / Alternating Scheduler** | Environment alternates focal subsystem every step ($t_1 = \text{Core}, t_2 = \text{Mem}, t_3 = \text{Core} \dots$). | Eliminates meta-reasoning overhead; forces balanced exploration. | May force exploration of an inactive subsystem on alternating turns. |
| **Option 3: Diagnostic Variance-Directed Scheduler** | Environment detects whether previous turn resulted in performance plateau and shifts focus, or provides initial bottleneck guidance. | Focuses on active parameters rapidly. | Requires diagnostic oracle knowledge outside standard black-box DSE. |
| **Option 4: Dual-Channel Dynamic Focus (Unified Schema)** | Schema allows primary mutations on one subsystem and compensatory adjustments on the other, but the agent specifies the primary intent. | Seamless expressivity. | Must be carefully parsed to avoid ambiguity. |

### Specification Choice for Equivalence & Experimental Cleanliness
For this study, we specify **Option 2 (Scheduled Focal Subsystem)** with **Atomic Compensatory Rebalancing** as the primary scientific candidate, compared alongside **Option 1 (Agent-Directed Dynamic Focus with Compensation)**:
- In the primary candidate, the focal subsystem is defined by the environment turn schedule. The prompt states:
  `"Current Focus: Core Execution Engine. Formulate primary mutations for the Core. If this exceeds the 1500 KB budget, use compensatory_memory_adjustments to downscale cache capacity in the same step."`
- This completely removes the cognitive selection burden while preserving $100\%$ of the agent's ability to maintain feasibility and execute global trade-offs.

---

## 4. Reconstructing a Full ADSG Configuration

The deterministic reconstruction algorithm is implemented as:
```python
def reconstruct_and_resolve(
    current_state: Tuple[int, int, int, str, int, str],
    proposal: Dict[str, Any],
    gp: GraphProcessor,
    budget_kb: float = 1500.0
) -> Tuple[Gem5Configuration, bool]:
    # 1. Unpack current state
    iw, rob, mshr, l1s, l1a, l2s = current_state

    # 2. Extract focal and compensatory updates
    focal = proposal.get("focal_core_mutations") or proposal.get("focal_memory_mutations") or {}
    comp = proposal.get("compensatory_memory_adjustments") or proposal.get("compensatory_core_adjustments") or {}

    merged_updates = {**focal, **comp}

    # 3. Apply updates
    if "issue_width" in merged_updates and merged_updates["issue_width"] is not None:
        iw = int(merged_updates["issue_width"])
    if "rob_size" in merged_updates and merged_updates["rob_size"] is not None:
        rob = int(merged_updates["rob_size"])
    if "l1d_mshrs" in merged_updates and merged_updates["l1d_mshrs"] is not None:
        mshr = int(merged_updates["l1d_mshrs"])
    if "l1d_capacity_kb" in merged_updates and merged_updates["l1d_capacity_kb"] is not None:
        l1s = f"{merged_updates['l1d_capacity_kb']}kB"
    if "l1d_associativity" in merged_updates and merged_updates["l1d_associativity"] is not None:
        l1a = int(merged_updates["l1d_associativity"])
    if "l2_capacity_kb" in merged_updates and merged_updates["l2_capacity_kb"] is not None:
        l2s = "1MB" if merged_updates["l2_capacity_kb"] == "1024" else f"{merged_updates['l2_capacity_kb']}kB"

    candidate_vector = (iw, rob, mshr, l1s, l1a, l2s)

    # 4. Resolve via ADSG
    g_inst, res_idx, res_val = acm.resolve_adsg_vector(gp, candidate_vector)

    # 5. Build verified gem5 configuration
    gem5_cfg = at.adsg_to_gem5_config(g_inst, budget_kb=budget_kb)
    return gem5_cfg, gem5_cfg.is_feasible
```
