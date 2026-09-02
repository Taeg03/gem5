# Empirical Experiment Plan: Evaluating ADSG-Grounded Semantic Abstraction

## 1. Objective & Hypothesis

The goal of this experiment is to empirically evaluate whether an **ADSG-grounded focal representation** improves LLM architectural design space exploration (DSE) efficiency over flat representations, while rigorously isolating representational focus from arithmetic cost feedback.

### Research Hypothesis
> *An agent-facing representation that structures reasoning around an ADSG-derived focal subsystem while retaining a compensatory rebalancing channel reduces exploration of inactive parameters without inducing the constraint violations of hard subsystem isolation.*

---

## 2. Experimental Matrix: Mandatory Ablation Control

To prevent false attribution (confounding representational focus with arithmetic cost transparency), the experimental matrix compares **three tightly controlled configurations**:

| Configuration ID | Name | Action Representation | Cost / Metric Feedback | Semantic Representation Role |
| :--- | :--- | :--- | :--- | :--- |
| **Config 1** | **`Baseline_B`** (Standard Control) | Flat 6-parameter vector emitted every turn. | Aggregate Cost only: `Cost = X KB / 1500 KB` `[FEASIBLE / INFEASIBLE]`. | Purely flat microarchitectural variable list. |
| **Config 2** | **`Baseline_B_CostBreakdown`** (Ablation Control) | Flat 6-parameter vector emitted every turn. | Detailed Subsystem Cost Breakdown: `Core_Cost = A KB`, `Mem_Cost = B KB`, `Total = X KB / 1500 KB (Slack Delta = Z KB)`. | Flat microarchitecture with explicit arithmetic transparency. |
| **Config 3** | **`ADSG_Candidate`** (Proposed Interface) | Dual-Channel: Focal Subsystem Mutations + Optional Compensatory Rebalancing. | Detailed Subsystem Cost Breakdown: `Core_Cost = A KB`, `Mem_Cost = B KB`, `Total = X KB / 1500 KB (Slack Delta = Z KB)`. | ADSG-grounded focal reasoning + compensatory rebalancing + arithmetic transparency. |

---

## 3. Scientific Decision Rules & Causal Attribution Logic

By comparing all three configurations across matched pairs (same workload, same seed, same step budget):

```
                                  EMPIRICAL OUTCOME MATRIX
                                             │
      ┌──────────────────────────────────────┴──────────────────────────────────────┐
      ▼                                                                             ▼
Config 3 > Config 1, but Config 3 == Config 2                Config 3 > Config 2 AND Config 3 > Config 1
─────────────────────────────────────────────                ─────────────────────────────────────────────
• Attribution: Arithmetic Transparency                       • Attribution: Representational Focus
• Conclusion: The performance gain was NOT                   • Conclusion: The ADSG-grounded focal view
  caused by ADSG representation or focal focus;                genuinely reduces attention dispersion and
  it was driven entirely by knowing the numeric                inactive mutations beyond simple arithmetic
  subsystem cost split and slack.                              transparency.
```

### Formal Decision Criteria:
1. **Arithmetic Transparency Effect**: Measured by comparing `Baseline_B_CostBreakdown` vs. `Baseline_B`. If providing `Core_Cost` and `Mem_Cost` alone reduces infeasible proposals and convergence steps, arithmetic opacity was the root bottleneck in Baseline B.
2. **Representational Focus Effect**: Measured by comparing `ADSG_Candidate` vs. `Baseline_B_CostBreakdown`. Because both configurations receive the exact same arithmetic cost feedback, any reduction in inactive parameter mutations ($<5\%$ ANOVA variance) or steps-to-optimum is causally attributable to the **focal action abstraction**.

---

## 4. Controlled Evaluation Methodology

### A. Design Space & Simulation Ground Truth
- Space: 6-parameter Core + Memory space ($3^6 = 729$ points).
- Hardware Budget: $B = 1500.0\text{ KB}$ equivalent area.
- Oracles: Complete exhaustive ground-truth gem5 datasets:
  - `Workload_Alpha` (Compute / ILP): [`oracle_729_compute.csv`](oracle_729_compute.csv)
  - `Workload_Beta` (Memory Latency): [`oracle_729_latency.csv`](oracle_729_latency.csv)
  - `Workload_Gamma` (Memory Concurrency): [`oracle_729_concurrency.csv`](oracle_729_concurrency.csv)

### B. Statistical Power & Trial Budget
- Workloads: 3 neutral workloads (`Workload_Alpha`, `Workload_Beta`, `Workload_Gamma`).
- Seeds: 3 independent seeds (`42`, `100`, `2026`).
- Steps per Trial: 10 exploration steps.
- Total Trials: $3\text{ configs} \times 3\text{ workloads} \times 3\text{ seeds} = \mathbf{27\text{ total trials}}$.

### C. Metrics Tracked per Trial:
1. **Best Feasible IPC**: Maximum IPC achieved among valid configurations respecting the $\le 1500\text{ KB}$ budget.
2. **Steps to $\ge 98\%$ Constrained Optimum**: Search speed to reach near-optimal region.
3. **Relevant Mutation Ratio (%)**: Percentage of parameter mutations applied to active parameters (parameters with $\ge 5\%$ ANOVA variance contribution for the target workload).
4. **Inactive Parameter Mutations**: Absolute count of mutations wasted on parameters with $<5\%$ variance.
5. **Infeasible Proposals / Budget Violations**: Number of proposed configurations exceeding $1500\text{ KB}$.
6. **Token Consumption**: Total prompt tokens and completion tokens recorded via API usage metadata.

### D. Hypothesis Testing
- Paired two-tailed Student's $t$-tests and non-parametric Wilcoxon signed-rank tests across the 9 matched pairs per baseline comparison.
- Significance threshold: $\alpha = 0.05$.
