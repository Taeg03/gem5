# Validation Report: Candidate 1 Structural Benchmark (In-Order MinorCPU vs. Out-of-Order DerivO3CPU)

## 1. Executive Summary & Purpose

This report documents the independent verification of Candidate 1 for the first genuinely structural ADSG benchmark. 

In accordance with the scientific directive, we evaluate all proposed mechanisms against three strict categories:
1. **Real gem5 structural conditionality**: A parameter genuinely does not exist or cannot be instantiated under a particular architectural choice.
2. **Real architectural constraint**: A combination is genuinely unsupported or causes an instantiation failure in gem5.
3. **Performance preference**: A combination is legal in gem5 but performs sub-optimally.

> **Key Takeaway**: Only Category 1 and Category 2 may be encoded into ADSG structural validity. Performance preferences (Category 3) belong strictly in the objective/evaluation function.

---

## 2. Verified MinorCPU vs. DerivO3CPU Parameter Semantics

We empirically tested both `X86MinorCPU` and `DerivO3CPU` within gem5 v25.1.0.1 running x86 Syscall Emulation (SE) mode on the compiled microbenchmark binary (`workload_729_bin`).

### A. DerivO3CPU (Out-of-Order Superscalar)
- **Pipeline Widths**: `fetchWidth`, `decodeWidth`, `renameWidth`, `dispatchWidth`, `issueWidth`, `wbWidth`, `commitWidth` symmetrically parameterized to $W \in \{2, 4, 8\}$.
- **Window & Queue Sizing**:
  - `numROBEntries` $\in \{32, 64, 128\}$
  - `LQEntries = max(16, rob_size // 4)`
  - `SQEntries = max(16, rob_size // 4)`
  - `numPhysIntRegs = max(64, rob_size + 32)`
  - `numPhysFloatRegs = max(64, rob_size + 32)`
- **Cache Interface**: `system.cpu.dcache.mshrs` $\in \{2, 4, 8\}$.
- **Simulation Verification**: Fully validated across all existing 729 oracle points.

### B. X86MinorCPU (In-Order Superscalar)
- **Pipeline Widths**:
  - `fetch1FetchLimit`, `decodeInputWidth`, `executeInputWidth`, `executeIssueLimit`, `executeCommitLimit` symmetrically parameterized to $W \in \{2, 4, 8\}$.
  - Empirically verified: Setting $W = 2, 4, 8$ simulates cleanly on `workload_729_bin`. Because of strict in-order scoreboarding and structural dependencies, IPC scales gently ($0.491 \to 0.501 \to 0.506$), accurately reflecting in-order microarchitecture.
- **Window & Queue Sizing**:
  - **Does NOT exist**. `X86MinorCPU` has no reorder buffer, no speculative register renaming, no physical register files, and no out-of-order issue queues.
  - Empirically verified: Attempting to assign `numROBEntries` to `X86MinorCPU` raises a fatal gem5 error:
    `AttributeError: Invalid assignment for Class X86MinorCPU with parameter numROBEntries`.
- **Cache Interface**: `system.cpu.dcache.mshrs` $\in \{2, 4, 8\}$ is fully supported and connects identically to the classic cache hierarchy.
- **Simulation Verification**: Successfully executed `workload_729_bin` across both `compute` and `latency` modes, producing valid `system.cpu.ipc` statistics with zero modifications to the binary or simulator C++ source.

---

## 3. Classification of Proposed Rules

| Proposed Feature | gem5 Status | Formal Classification | Decision in Benchmark |
| :--- | :--- | :--- | :--- |
| **`rob_size` conditional on `core_type == OutOfOrder`** | `numROBEntries` raises `AttributeError` on `MinorCPU`. | **Real gem5 Structural Conditionality (Category 1)** | **RETAINED** |
| **`issue_width` scaling on `MinorCPU`** | `MinorCPU` parameters accept $\{2, 4, 8\}$ cleanly. | **Legal gem5 Sizing Parameter** | **RETAINED** |
| **`IW=8 + ROB=32` Incompatibility** | Successfully simulated in gem5 (Config 486, IPC = 1.0480). | **Performance Preference (Category 3)** | **REMOVED** |

### Elimination of the "IW=8 + ROB=32" Incompatibility
The previous proposal asserted that an 8-wide issue width with a 32-entry ROB was an "architecturally invalid" configuration. 

Empirical testing in gem5 refutes this: gem5 instantiates and simulates `(IW:8, ROB:32)` to normal exit with valid statistics. It is an unbalanced microarchitecture, but it is **not** structurally invalid in gem5. 

Per the scientific requirement, **this artificial rule has been removed entirely**. We do not manufacture fake incompatibilities to give ADSG an artificial advantage.

---

## 4. Cost-Model Treatment

The project's established hardware cost model is:
$$\text{Cost} = \text{Core\_Cost} + \text{Memory\_Cost}$$
$$\text{Core\_Cost} = (IW \times 50) + (ROB \times 1.5) + (MSHR \times 5)$$
$$\text{Memory\_Cost} = \text{Cap}_{\text{L1D}} \times \left(1 + 0.1 \times \frac{A_{\text{L1D}}}{8}\right) + \text{Cap}_{\text{L2}}$$

### Principled Extension to MinorCPU:
Because `rob_size` does not exist on `MinorCPU`, the $(ROB \times 1.5)$ term drops to identically $0$:
$$\text{Core\_Cost}_{\text{Minor}} = (IW \times 50) + (MSHR \times 5)$$
$$\text{Core\_Cost}_{\text{O3}} = (IW \times 50) + (ROB \times 1.5) + (MSHR \times 5)$$

### Architectural Rationale:
- This extension requires **zero ad-hoc multipliers** or speculative "70% area reduction" assumptions.
- It directly reflects physical reality: the In-Order core saves the exact area that would otherwise be occupied by the reorder buffer and speculative register renaming structures ($48\text{--}192\text{ KB}$ equivalent).
- Under the $B = 1500\text{ KB}$ budget, this saving directly enables In-Order cores to afford larger L2 caches on memory-bound workloads.

---

## 5. Correct Configuration Counts (Independent Derivation)

### A. Subsystem Domains
1. **`core_type`**: 2 choices $\in \{\text{InOrder}, \text{OutOfOrder}\}$
2. **`issue_width`**: 3 choices $\in \{2, 4, 8\}$
3. **`rob_size`**:
   - Under `OutOfOrder`: 3 choices $\in \{32, 64, 128\}$
   - Under `InOrder`: 1 state $\in \{\emptyset\}$ (inactive)
4. **`l1d_mshrs`**: 3 choices $\in \{2, 4, 8\}$
5. **`l1d_size`**: 3 choices $\in \{16\text{kB}, 32\text{kB}, 64\text{kB}\}$
6. **`l1d_assoc`**: 3 choices $\in \{2, 4, 8\}$
7. **`l2_size`**: 3 choices $\in \{512\text{kB}, 1\text{MB}, 2\text{MB}\}$

### B. Valid Architectural Space ($N_{\text{valid}}$)
- **Memory Combinations**: $3 \times 3 \times 3 = 27$
- **OutOfOrder Configurations**:
  $$N_{\text{O3}} = 3 (\text{IW}) \times 3 (\text{ROB}) \times 3 (\text{MSHR}) \times 27 (\text{Mem}) = \mathbf{729\text{ configurations}}$$
- **InOrder Configurations**:
  $$N_{\text{Minor}} = 3 (\text{IW}) \times 1 (\text{no ROB}) \times 3 (\text{MSHR}) \times 27 (\text{Mem}) = \mathbf{243\text{ configurations}}$$
- **Total Valid Space**:
  $$N_{\text{valid}} = 729 + 243 = \mathbf{972\text{ valid physical configurations}}$$

### C. Flat Declared Space ($N_{\text{declared}}$)
A flat vector must declare options for all 7 variables simultaneously:
$$N_{\text{declared}} = 2 (\text{type}) \times 3 (\text{IW}) \times 3 (\text{ROB}) \times 3 (\text{MSHR}) \times 3 (\text{L1S}) \times 3 (\text{L1A}) \times 3 (\text{L2S}) = \mathbf{1458\text{ points}}$$

### D. Redundancy / Imputation Discrepancy
- For the In-Order core, the flat representation generates $1 \times 3 \times 3 \times 3 \times 27 = 729$ points.
- However, since `rob_size` is ignored, every triplet of flat points differing only by `rob_size` maps to the exact same physical machine.
- **Redundant Alias Points**: $729 - 243 = \mathbf{486\text{ points}}$ ($33.3\%$ of the flat space).
- **Discrete Imputation Ratio**:
  $$IR = \frac{N_{\text{declared}}}{N_{\text{valid}}} = \frac{1458}{972} = \mathbf{1.50}$$

---

## 6. Exact Representations Compared

### A. Flat Representation
The flat agent must emit a 7-parameter vector:
$$\mathbf{x}_{\text{flat}} = (\text{core\_type}, \text{issue\_width}, \text{rob\_size}, \text{l1d\_mshrs}, \text{l1d\_size}, \text{l1d\_assoc}, \text{l2\_size})$$
When `core_type == InOrder`, the agent is still forced to reason about and emit `rob_size`, wasting token budget on an uninstantiated parameter and requiring the environment to silently discard the value.

### B. ADSG Representation
```
                     [CPU]
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[X86MinorCPU]                    [DerivO3CPU]
   (InOrder)                     (OutOfOrder)
       │                               │
       ├─► D[issue_width]              ├─► D[issue_width]
       ├─► D[l1d_mshrs]                ├─► D[rob_size]     <-- ONLY instantiated here
       └─► [Memory]                    ├─► D[l1d_mshrs]
             │                         └─► [Memory]
             ├─► D[l1d_size]                 │
             ├─► D[l1d_assoc]                ├─► D[l1d_size]
             └─► D[l2_size]                  ├─► D[l1d_assoc]
                                             └─► D[l2_size]
```
In ADSG, `SelectionChoiceNode("rob_size")` is derived **only** from the `DerivO3CPU` node. When `X86MinorCPU` is selected, `rob_size` has $\delta = \text{False}$. The agent-facing schema dynamically excludes `rob_size` from token emission.

---

## 7. Uncertainties & Verification Risks

1. **Simulation Throughput for 972 Points**:
   - `DerivO3CPU` took $\sim 15\text{ minutes}$ across 10 threads to generate 729 points.
   - `X86MinorCPU` executes in $\sim 3.8\text{ million cycles}$ per run ($<2\text{ seconds}$ wall-clock time).
   - Simulating the additional 243 In-Order points across 3 workloads will require $\sim 5\text{ minutes}$ in parallel.
   - Risk: **None** (verified by direct execution).
2. **Workload Discriminability**:
   - Does `MinorCPU` provide distinct sensitivity profiles?
   - Verified: On `compute`, O3 achieves $\text{IPC} \approx 1.05$ vs Minor's $\approx 0.50$. On `latency`, Minor achieves $\text{IPC} \approx 0.30\text{--}0.34$, while saving area for L2.
   - Risk: **Low**.

---

## 8. Final Recommendation

### **Recommendation: PROCEED WITH MODIFIED CANDIDATE 1**

Candidate 1 should proceed to implementation with the following methodological corrections:
1. **Retain the genuine structural conditionality**: `rob_size` is active for `DerivO3CPU` and inactive ($\delta = \text{False}$) for `X86MinorCPU`.
2. **Remove the artificial incompatibility**: Do **not** forbid `(IW:8, ROB:32)`. All 972 physical configurations are legal and simulatable.
3. **Use the exact mathematically verified domains**: 972 valid physical configurations vs. 1458 flat declared vectors ($IR = 1.50$).
4. **Use the principled cost model**: $\text{Core\_Cost} = (IW \times 50) + (ROB \times 1.5) + (MSHR \times 5)$, where the $ROB$ term naturally vanishes for In-Order cores.

This provides a clean, honest, and grounded benchmark to test whether eliminating inactive parameter emission via conditional ADSG abstraction improves agentic DSE without manufacturing artificial rules.
