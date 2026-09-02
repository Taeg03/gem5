# ADSG Agent Interface Design: Semantic Abstraction & Theoretical Grounding

## 1. Executive Summary & Research Directive

This document defines the conceptual architecture for the agent-facing representation layer built on top of the Architecture Design Space Graph (ADSG) backend for the 6-parameter heterogeneous CPU design space ($3^6 = 729$ configurations).

### Research Question
> *How should the semantics of a formal architecture design space be exposed to an LLM agent to enable coordinated exploration while reducing irrelevant reasoning overhead?*

### Foundational Empirical Context
- **Baseline B (Flat Microarchitectural)**: Presents all 6 parameters simultaneously. Permits single-step coordinated cross-subsystem trade-offs, but causes search dispersion ($75\%\text{--}85\%$ mutations wasted on inactive parameters).
- **Baseline C (Static JSON Hierarchy)**: Syntactically groups variables into `core_subsystem` and `memory_hierarchy`. Empirically indistinguishable from Baseline B ($p = 1.0000$); syntactic nesting alone fails to filter LLM attention.
- **Baseline D (Hard Subsystem Isolation)**: Forces the agent to pick a single subsystem and freezes the other. Severely degrades discovery latency ($4.56$ vs $1.11$ steps, $p = 0.0367$), skyrockets budget violations ($3.44$ vs $0.22$, $p < 0.0001$), and causes multi-turn serialized "ping-pong" state oscillation due to the **Coupled-Budget Blindspot**.

The candidate agent interface must achieve **attention focus without structural isolation**.

---

## 2. Theoretical Clarification: The "Zero-Semantics" Trap

It is scientifically essential to state explicitly what the ADSG model represents in this experiment and what it does **not**:

> ### [!IMPORTANT]
> **The "Zero-Semantics" Boundary**
> Because this 729-point design space deliberately omits incompatibility edges (`EdgeType.INCOMPATIBILITY`) and conditional existence nodes ($\delta = \text{False}$), the ADSG backend in this study operates **solely as a formal structural partitioner and metric container**.
>
> We do **NOT** claim that the LLM agent is traversing complex graph logic, satisfying topological path constraints, or navigating non-trivial graph grammars.
>
> The underlying ADSG model is a 2-level star-like derivation tree:
> $$\text{CPU} \xrightarrow{\text{DERIVES}} \{\text{Core}, \text{Memory}\} \xrightarrow{\text{DERIVES}} \{\text{ChoiceNodes}\} \xrightarrow{\text{DERIVES}} \{\text{OptionNodes}\}$$
> The true utility of ADSG in this milestone is providing a mathematically grounded component-to-decision ownership mapping and an explicit typed container for the global cost constraint (`MetricNode("cost")`).

Any claim that the agent is "reasoning over graph topology" is scientifically false for this 729-point benchmark. The agent is reasoning over a **subsystem-partitioned projection with explicit cost attribution**.

---

## 3. Disentangling Causal Attribution: Representation vs. Arithmetic Transparency

A central flaw in naive architectural agent evaluation is conflating **representational abstraction** with **arithmetic transparency**:

```
                         CAUSAL ATTRIBUTION SPLIT
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       ▼                                                         ▼
HYPOTHESIS A: Representational Focus          HYPOTHESIS B: Arithmetic Transparency
- Filtering inactive subsystem tokens         - Providing explicit subsystem cost split:
  reduces attention dispersion and              Core_Cost = 316 KB | Mem_Cost = 1040 KB
  mutation drift.                             - Eliminates the Coupled-Budget Blindspot
- Derived from ADSG component partitioning.     through simple arithmetic clarity.
```

If the candidate agent outperforms Baseline B, **what is the true causal driver?**
1. Is it the **focal subsystem view** (reducing irrelevant cognitive load)?
2. Or is it simply that the agent was given **subsystem cost breakdowns and remaining slack** ($\Delta = 1500 - \text{Cost}$), allowing it to perform arithmetic budgeting without guessing?

### Mandatory Attribution Controls
To prevent false attribution to the ADSG representation:
- We must establish an explicit ablation control: **`Baseline_B_CostBreakdown`**.
- `Baseline_B_CostBreakdown` receives the exact same flat 6-parameter action vector as Baseline B, but is provided with the identical arithmetic cost feedback (`Core_Cost`, `Mem_Cost`, `Slack`).
- **Attribution Decision Rule**:
  - If $\text{ADSG Candidate} > \text{Baseline B}$, but $\text{ADSG Candidate} \approx \text{Baseline B + Cost Breakdown}$, then the entire performance benefit was caused by **arithmetic transparency**, *not* ADSG representation.
  - Only if $\text{ADSG Candidate}$ demonstrates a statistically significant reduction in inactive parameter mutations and steps-to-optimum over **both** Baseline B and Baseline B + Cost Breakdown can we claim evidence for representational focus.

---

## 4. Architectural Principles of the Candidate Interface

To eliminate the failure modes of Baseline D while testing representational focus:

1. **Dual-Channel Action Grammar**:
   - Primary channel: **Focal Subsystem Mutations** (mandatory focus of search reasoning).
   - Secondary channel: **Compensatory Subsystem Adjustments** (optional adjustments to the background subsystem to offset cost or preserve balance).
2. **Atomic 6D Reconstruction**:
   - The action is never evaluated as two sequential steps. The focal and compensatory updates are combined with the previous state into a single 6D vector and resolved atomically by ADSG into a valid `Gem5Configuration`.
3. **Decoupled Focal Scheduling**:
   - The agent should not be forced to waste reasoning tokens selecting which subsystem to target at runtime, which was a proven source of misattribution latency in Baseline D.
