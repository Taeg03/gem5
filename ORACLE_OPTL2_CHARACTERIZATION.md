# Empirical Characterization Report: 972-Point Optional-L2 Design Space

## 1. Executive Summary

This report documents the exhaustive simulation and landscape characterization of the **972-point Optional-L2 Architecture Space**.

Unlike Candidate 1 (where `core_type = DerivO3CPU` strictly dominated 100% of near-optimal regions, leaving conditionality unexercised), the Optional-L2 space exhibits **true workload-dependent structural conditionality**:
- **On Latency**: `has_l2 = True` is strictly mandatory to enter the near-optimal region ($\ge 98\%$ optimum is **$100\%$ with-L2**, $0\%$ No-L2).
- **On Compute**: `has_l2 = False` is strictly optimal to reach the peak region ($\ge 99\%$ optimum is **$100\%$ No-L2**, $0\%$ with-L2).
- **On Concurrency**: `has_l2 = False` is strictly optimal ($\ge 98\%$ optimum is **$100\%$ No-L2**, $0\%$ with-L2).

All 972 configurations were simulated to completion in gem5 across all three neutral workloads:
- `oracle_optl2_compute.csv`
- `oracle_optl2_latency.csv`
- `oracle_optl2_concurrency.csv`
- `oracle_optl2_master.csv`

---

## 2. Verification of the No-L2 gem5 Implementation

### Memory Hierarchy Mechanics
- **With L2 (`has_l2 = True`)**:
  `icache` and `dcache` $\to$ `L2XBar` $\to$ `L2 Cache` (10-cycle tag/data latency) $\to$ `SystemXBar` $\to$ `DDR4`.
- **Without L2 (`has_l2 = False`)**:
  `icache` and `dcache` $\to$ `SystemXBar` $\to$ `DDR4`.
  There is no intermediate `L2XBar` and no `L2 Cache`.

### Microarchitectural Root Causes for IPC Behavior:
1. **Compute Workload**:
   - The active computation loop fits entirely inside L1I (32kB) and L1D (16–64kB).
   - Once cold misses complete, zero demand accesses miss to L2.
   - Without L2, startup cold misses bypass the 10-cycle L2 tag lookup, yielding slightly higher net IPC (**$1.0888$ vs. $1.0733$**), while saving **$512\text{--}2048\text{ KB}$ of area budget**.
2. **Concurrency Workload (8 Parallel Streams across 1.5MB)**:
   - In the with-L2 configuration, 8 concurrent streams accessing 1.5MB through a 1MB L2 cache induce continuous line evictions and buffer contention on the `L2XBar`.
   - Without L2, the 8 streams stream directly to multi-banked `DDR4_2400` via `SystemXBar`, avoiding L2 crossbar serialization. Net IPC increases from **$0.5998$ to $0.6744$**.
3. **Latency Workload (1.5MB Serial Pointer Chase)**:
   - Serial dependent loads ($p = *p$) cannot exploit memory-level parallelism.
   - With a 1MB L2 cache, $\sim 66\%$ of accesses hit in 10 cycles, yielding **$\text{IPC} = 0.5422$**.
   - Without an L2 cache, every missed access must wait for 150ns DDR4 round-trip latency, dropping IPC to **$0.5186$**.

---

## 3. Global Feasibility Analysis under 1500 KB Budget

- **Hardware Budget Constraint**: $B = 1500.0\text{ KB}$.
  $$\text{Core\_Cost} = (IW \times 50) + (ROB \times 1.5) + (MSHR \times 5)$$
  $$\text{Memory\_Cost} = \text{Cap}_{\text{L1D}} \times \left(1 + 0.1 \times \frac{A_{\text{L1D}}}{8}\right) + (\text{Cap}_{\text{L2}} \text{ if } has\_l2 \text{ else } 0)$$

### Empirical Breakdown:
- **Total Configurations**: 972 points.
- **Overall Feasible**: $645 / 972$ configurations (**$66.4\%$**).
- **With L2 (`has_l2 = True`)**: $402 / 729$ feasible (**$55.1\%$**). (All 2MB L2 configurations exceed 1500 KB).
- **Without L2 (`has_l2 = False`)**: $243 / 243$ feasible (**$100.0\%$**). (Maximum No-L2 cost is $702.4\text{ KB} \le 1500\text{ KB}$).

---

## 4. Near-Optimal Composition: True Workload-Dependent Conditionality

| Workload | Constrained Optimum Architecture | Optimum IPC | Feasible Points $\ge 98\%$ | Composition $\ge 98\%$ | Composition $\ge 99\%$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Compute** | `has_l2 = False` (IW:8w, ROB:128, MSHR:8, L1:64kB/8w, L2:inactive) | **$1.0888$** | $133 / 645$ ($20.6\%$) | **$106\text{ No-L2}, 27\text{ With-L2}$** | **$54\text{ No-L2}, 0\text{ With-L2}$** |
| **Latency** | `has_l2 = True` (IW:4w, ROB:128, MSHR:2, L1:64kB/2w, L2:1MB) | **$0.5422$** | $141 / 645$ ($21.9\%$) | **$141\text{ With-L2}, 0\text{ No-L2}$** | **$57\text{ With-L2}, 0\text{ No-L2}$** |
| **Concurrency** | `has_l2 = False` (IW:8w, ROB:128, MSHR:8, L1:64kB/2w, L2:inactive) | **$0.6744$** | $24 / 645$ ($3.7\%$) | **$24\text{ No-L2}, 0\text{ With-L2}$** | **$4\text{ No-L2}, 0\text{ With-L2}$** |

> ### Key Takeaways:
> 1. **On Latency**: The top $2\%$ of designs are **$100\%$ with-L2**. An agent that prunes L2 can never reach the $\ge 98\%$ region.
> 2. **On Compute**: The top $1\%$ of designs are **$100\%$ without L2**. An agent that keeps L2 is locked out of the peak optimum.
> 3. **On Concurrency**: The top $2\%$ of designs are **$100\%$ without L2**.

---

## 5. Representation Challenge: Flat vs. Semantic Conditional

### A. Flat Cartesian Representation
$$\mathbf{x}_{\text{flat}} = (\text{issue\_width}, \text{rob\_size}, \text{l1d\_mshrs}, \text{l1d\_size}, \text{l1d\_assoc}, \text{has\_l2}, \text{l2\_size})$$
- When `has_l2 == False`, the agent is still forced to emit `l2_size` (e.g. `l2_size: "1MB"`).
- Changing `l2_size` produces **zero IPC change and zero cost change** when `has_l2 == False`.
- On Compute and Concurrency (where `has_l2 = False` is optimal), the flat agent wastes token budget and attention exploring mutations on a dead parameter.

### B. ADSG Semantic Conditional Representation
```
                     [CPU]
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
 [Core Engine]                   [Memory System]
       │                               │
       ├─► D[issue_width]              ├─► D[l1d_size]
       ├─► D[rob_size]                 ├─► D[l1d_assoc]
       └─► D[l1d_mshrs]                ├─► S[has_l2]
                                             │
                             ┌───────────────┴───────────────┐
                             ▼                               ▼
                       has_l2: False                   has_l2: True
                       (Direct L1-to-DRAM)              (Dedicated L2)
                             │                               │
                      (l2_size INACTIVE)              └─► D[l2_size]
```
- When `has_l2 == False`, ADSG's `HierarchyAnalyzer` sets $\delta_{\text{l2\_size}} = \text{False}$, removing `l2_size` from the JSON action schema.
- The agent reasons about `l2_size` **only when the L2 cache actually exists**.
