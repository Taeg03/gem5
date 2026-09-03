# Empirical Characterization Report: 972-Point Structural Architectural Space

## 1. Executive Summary

This report documents the exhaustive simulation and statistical characterization of the **972-point CPU architectural design space**. 

This space extends the static 729-point Cartesian space by introducing a high-level microarchitectural paradigm choice: **In-Order (`X86MinorCPU`) vs. Out-of-Order (`DerivO3CPU`)**, creating **genuine gem5 structural conditionality** where `rob_size` exists exclusively for `DerivO3CPU` ($\delta = \text{False}$ for `MinorCPU`).

All 972 configurations were simulated to completion in gem5 across three neutral workloads:
- `oracle_972_compute.csv`
- `oracle_972_latency.csv`
- `oracle_972_concurrency.csv`
- `oracle_972_master.csv`

---

## 2. Representation of Inactive `rob_size` in the Flat Control

### The Methodological Question
> *Is the flat control allowed to omit `rob_size` when `core_type = MinorCPU`, or does it have to represent the full declared Cartesian schema?*

### Theoretical & Empirical Analysis of the Options:

| Control Formulation | Schema Definition | Handling of `rob_size` under `MinorCPU` | Cognitive Load & Attention | Scientific Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Option A: Static Declared Cartesian Flat Schema** *(Strict Baseline)* | Fixed 7-variable JSON schema. Every field is mandatory. | Forced emission of dummy `rob_size` (e.g. `rob_size: "32"`). Simulator harness silently discards the value. | Forces LLM to reason about and emit tokens for a non-existent parameter. Changes produce zero $\Delta$ in IPC or cost. | Models the standard baseline in current DSE literature (static schemas unaware of downstream conditionality). |
| **Option B: Syntactic Flat-Conditional Schema** *(Ablation Baseline)* | Single flat schema with `rob_size: Optional[Literal["32","64","128"]] = None`. | Prompt instructs: *"Omit/set null when core_type is MinorCPU"*. | Relies on LLM in-context compliance. Schema still declares the field globally. | Tests whether simple syntactic nullability is sufficient without formal structural isolation. |
| **Option C: Semantic ADSG-Grounded Conditional Schema** *(Candidate)* | Dynamic Schema Union via `Discriminator(field="core_type")` or ADSG-guided prompt. | When `core_type == MinorCPU`, `rob_size` does **not exist in the schema**. It is structurally impossible to emit. | Prunes `rob_size` from token emission and attention budget entirely. | Tests whether formal structural conditionality eliminates irrelevant reasoning. |

### Recommended Scientific Protocol:
To rigorously isolate the mechanism, the upcoming agent evaluation should compare:
1. **Baseline B (Static Flat Cartesian)**: Forced emission of all 7 variables (Option A).
2. **Baseline B_Optional (Syntactic Flat-Conditional)**: Flat schema with `Optional[rob_size]` (Option B).
3. **ADSG_Candidate (Structural Semantic Conditional)**: ADSG-grounded schema that prunes `rob_size` when `MinorCPU` is selected (Option C).

---

## 3. Feasibility Analysis under 1500 KB Budget

- **Hardware Budget Constraint**: $B = 1500.0\text{ KB}$ project cost proxy.
  $$\text{Core\_Cost}_{\text{O3}} = (IW \times 50) + (ROB \times 1.5) + (MSHR \times 5)$$
  $$\text{Core\_Cost}_{\text{Minor}} = (IW \times 50) + (MSHR \times 5)$$
  $$\text{Memory\_Cost} = \text{Cap}_{\text{L1D}} \times \left(1 + 0.1 \times \frac{A_{\text{L1D}}}{8}\right) + \text{Cap}_{\text{L2}}$$

### Empirical Feasibility Breakdown:
- **Overall Space**: $556 / 972\text{ configurations}$ are feasible (**$57.2\%$**).
- **DerivO3CPU**: $402 / 729\text{ configurations}$ are feasible (**$55.1\%$**).
- **MinorCPU**: $154 / 243\text{ configurations}$ are feasible (**$63.4\%$**).
- **Cost Distribution**:
  - `DerivO3CPU`: Mean = $1602.8\text{ KB}$, Median = $1397.2\text{ KB}$, Range = $[686.4, 2750.4]\text{ KB}$.
  - `MinorCPU`: Mean = $1490.8\text{ KB}$, Median = $1280.8\text{ KB}$, Range = $[638.4, 2558.4]\text{ KB}$.
  - MinorCPU saves an average of **$112.0\text{ KB}$** in core cost, increasing the feasible budget fraction by $+8.3\%$.

---

## 4. Parameter Sensitivity Analysis (ANOVA Variance Decomposition)

### Global Space (All 972 Points)

| Parameter | Compute Workload (% Var) | Latency Workload (% Var) | Concurrency Workload (% Var) |
| :--- | :--- | :--- | :--- |
| **`core_type`** | **$89.46\%$** ($p < 10^{-300}$) | **$80.30\%$** ($p < 10^{-300}$) | **$93.65\%$** ($p < 10^{-300}$) |
| **`issue_width`** | **$8.17\%$** ($p < 10^{-300}$) | $0.09\%$ ($p = 4.2 \times 10^{-9}$) | $0.08\%$ ($p = 1.9 \times 10^{-28}$) |
| **`l2_size`** | $0.00\%$ ($p = 1.0$) | **$17.47\%$** ($p < 10^{-300}$) | **$5.27\%$** ($p < 10^{-300}$) |
| **`l1d_mshrs`** | $0.00\%$ ($p = 0.995$) | $0.00\%$ ($p = 0.996$) | **$0.41\%$** ($p < 10^{-111}$) |
| **`l1d_size`** | $0.00\%$ ($p = 0.998$) | $0.00\%$ ($p = 0.526$) | $0.00\%$ ($p = 0.025$) |
| **`l1d_assoc`** | $0.00\%$ ($p = 1.0$) | $0.00\%$ ($p = 0.998$) | $0.00\%$ ($p = 0.888$) |
| **Residual** | $2.37\%$ | $2.14\%$ | $0.58\%$ |

### DerivO3CPU Subspace Alone (729 Points, `rob_size` Active)

| Parameter | Compute Workload (% Var) | Latency Workload (% Var) | Concurrency Workload (% Var) |
| :--- | :--- | :--- | :--- |
| **`issue_width`** | **$98.88\%$** ($p < 10^{-300}$) | $0.44\%$ ($p < 10^{-300}$) | $1.44\%$ ($p = 2.1 \times 10^{-46}$) |
| **`l2_size`** | $0.00\%$ ($p = 1.0$) | **$99.37\%$** ($p < 10^{-300}$) | **$84.28\%$** ($p < 10^{-300}$) |
| **`l1d_mshrs`** | $0.00\%$ ($p = 0.902$) | $0.00\%$ ($p = 0.302$) | **$9.53\%$** ($p < 10^{-183}$) |
| **`rob_size`** | $0.15\%$ ($p = 2.8 \times 10^{-23}$) | $0.14\%$ ($p = 1.6 \times 10^{-256}$) | $0.43\%$ ($p = 6.1 \times 10^{-16}$) |
| **`l1d_size`** | $0.00\%$ ($p = 0.967$) | $0.02\%$ ($p = 3.3 \times 10^{-62}$) | $0.07\%$ ($p = 0.002$) |
| **`l1d_assoc`** | $0.00\%$ ($p = 0.997$) | $0.00\%$ ($p = 0.597$) | $0.00\%$ ($p = 0.872$) |

> ### Key ANOVA Insights:
> 1. Across all 972 points, **`core_type` is by far the single most dominant parameter** ($80.3\%\text{--}93.7\%$ of total variance). The high-level decision to use an out-of-order engine vs. an in-order pipeline completely dominates IPC.
> 2. Within `DerivO3CPU`, `rob_size` has a statistically significant but small direct effect ($0.14\%\text{--}0.43\%$), primarily serving to support larger issue widths and latency tolerance.

---

## 5. Constrained Optima & Near-Optimal Landscape ($\ge 98\%$ of Optimum)

| Workload | Constrained Optimum Architecture | Optimum IPC | $\ge 98\%$ Optimum Region Count | Composition of $\ge 98\%$ Region |
| :--- | :--- | :--- | :--- | :--- |
| **Compute** | `DerivO3CPU` (IW:8w, ROB:128, MSHR:8, L1:64kB/4w, L2:512kB) | **$1.0733$** | $156 / 556\text{ feasible points}$ ($28.1\%$) | **$156\text{ O3}, 0\text{ Minor}$** |
| **Latency** | `DerivO3CPU` (IW:4w, ROB:128, MSHR:2, L1:64kB/2w, L2:1MB) | **$0.5422$** | $141 / 556\text{ feasible points}$ ($25.4\%$) | **$141\text{ O3}, 0\text{ Minor}$** |
| **Concurrency** | `DerivO3CPU` (IW:4w, ROB:64, MSHR:4, L1:64kB/2w, L2:1MB) | **$0.5998$** | $96 / 556\text{ feasible points}$ ($17.3\%$) | **$96\text{ O3}, 0\text{ Minor}$** |

### Peak Feasible IPC Comparison by Core Type:
- **Compute**: $\text{Peak O3} = 1.0733$ vs. $\text{Peak Minor} = 0.5060$ (Minor achieves $47.1\%$ of O3 IPC).
- **Latency**: $\text{Peak O3} = 0.5422$ vs. $\text{Peak Minor} = 0.3265$ (Minor achieves $60.2\%$ of O3 IPC).
- **Concurrency**: $\text{Peak O3} = 0.5998$ vs. $\text{Peak Minor} = 0.3231$ (Minor achieves $53.9\%$ of O3 IPC).

---

## 6. Critical Architectural Finding: Core-Memory Trade-off & Pareto Frontier

### Did `MinorCPU` Win on Memory Latency?
In our initial hypothesis, we conjectured that on `Workload_Beta` (memory latency bound), `MinorCPU`'s area savings might allow it to afford a larger cache and outperform `DerivO3CPU`.

**The Empirical Data Refutes This**:
- In this design space, a $2\text{MB}$ L2 cache costs $2048.0\text{ KB}$ by itself.
- Therefore, under $B = 1500.0\text{ KB}$, a $2\text{MB}$ L2 cache is **$100\%$ infeasible for BOTH `MinorCPU` and `DerivO3CPU`**.
- Both `MinorCPU` and `DerivO3CPU` are constrained to at most a $1\text{MB}$ L2 cache ($1024\text{ KB}$).
- With a $1\text{MB}$ L2 cache, `DerivO3CPU` is fully feasible (Cost: $1491.6\text{ KB} \le 1500\text{ KB}$) and achieves **$\text{IPC} = 0.5422$**.
- With a $1\text{MB}$ L2 cache, `MinorCPU` costs $1190.8\text{ KB}$, but only achieves **$\text{IPC} = 0.3265$**.
- **Conclusion**: In terms of pure IPC, `DerivO3CPU` strictly dominates `MinorCPU` across all 3 workloads.

### How Does This Impact the Agent's Search Problem?
1. **`core_type` is a high-level pruning decision**:
   - The agent must first discover that `core_type = DerivO3CPU` is required for competitive IPC.
   - If an agent chooses `MinorCPU`, it gets trapped in an inferior region ($\text{IPC} \le 0.506$).
2. **The Conditional `rob_size` Trap**:
   - If the agent chooses `core_type = MinorCPU` (or starts there), in a flat representation it will waste mutations tweaking `rob_size` ($32 \to 64 \to 128$), observing zero IPC delta and zero cost delta!
   - In an ADSG semantic representation, `rob_size` is immediately removed from the action space, preventing this wasted exploration.
