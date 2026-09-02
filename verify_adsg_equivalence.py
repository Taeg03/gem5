"""
verify_adsg_equivalence.py

Exhaustive 729-Point Equivalence Verification Suite
Comparing Reference Direct gem5 Pipeline vs. ADSG Pipeline.

Tests:
1. Design-vector equivalence: ADSG(x) == x for all 729 vectors.
2. Configuration equivalence: ADSG gem5 config == Reference gem5 config.
3. Feasibility equivalence: ADSG feasibility == Reference feasibility under 1500 KB budget.
4. Objective equivalence: ADSG oracle IPC == Reference oracle IPC across all 3 workloads.
5. Completeness: 729/729 resolved, 0 unmapped, 0 unresolved, 0 mismatches.
"""

import itertools
import sys
import time
from typing import List, Dict, Any, Tuple

import adsg_cpu_model as acm
import adsg_translator as at


def run_exhaustive_equivalence_verification(budget_kb: float = 1500.0) -> bool:
    print("=" * 80)
    print("      EXHAUSTIVE 729-POINT ADSG EQUIVALENCE VERIFICATION SUITE")
    print("=" * 80)
    print(f"Hardware Budget Constraint: {budget_kb} KB")
    print("Initializing ADSG Model and Graph Processor...")

    start_time = time.time()
    dsg = acm.build_cpu_adsg()
    gp = acm.get_cpu_graph_processor(dsg)

    n_valid = gp.get_n_valid_designs()
    n_space = gp.get_n_design_space()
    print(f"ADSG Declared Design Space: {n_space}")
    print(f"ADSG Valid Design Space:    {n_valid}")

    if n_valid != 729 or n_space != 729:
        print(f"FATAL: Expected 729 valid and declared points, got valid={n_valid}, declared={n_space}")
        return False

    workloads = ["compute", "latency", "concurrency"]
    # Pre-load oracles to ensure fast lookup
    for wl in workloads:
        at.get_oracle_df(wl)
    print(f"Loaded ground-truth oracles for: {workloads}\n")

    # Metrics counters
    total_vectors = 729
    resolved_count = 0
    unresolved_count = 0
    unmapped_count = 0

    design_vector_equiv_count = 0
    gem5_config_equiv_count = 0
    feasibility_equiv_count = 0
    ipc_equiv_count = {wl: 0 for wl in workloads}

    failures: List[Dict[str, Any]] = []

    # Enumerate all 3^6 = 729 vectors in canonical discrete index order
    all_index_tuples = list(itertools.product([0, 1, 2], repeat=6))
    print(f"Starting exhaustive evaluation of {len(all_index_tuples)} design vectors...")

    for i, idx_tuple in enumerate(all_index_tuples, 1):
        # -------------------------------------------------------------
        # 1. Reference Pipeline
        # -------------------------------------------------------------
        ref_params = acm.decode_indices_to_values(idx_tuple)
        ref_cfg = at.build_gem5_config_from_params(*ref_params, budget_kb=budget_kb)
        ref_ipcs = {wl: at.lookup_oracle_ipc(*ref_params, wl) for wl in workloads}

        # -------------------------------------------------------------
        # 2. ADSG Pipeline
        # -------------------------------------------------------------
        try:
            g_inst, adsg_res_indices, adsg_res_params = acm.resolve_adsg_vector(gp, idx_tuple)
            resolved_count += 1
        except Exception as e:
            unresolved_count += 1
            failures.append({
                "step": i,
                "type": "RESOLUTION_EXCEPTION",
                "original_indices": idx_tuple,
                "reference_params": ref_params,
                "error": str(e)
            })
            continue

        try:
            adsg_cfg = at.adsg_to_gem5_config(g_inst, budget_kb=budget_kb)
            adsg_ipcs = {wl: at.lookup_oracle_ipc(*adsg_res_params, wl) for wl in workloads}
        except Exception as e:
            unmapped_count += 1
            failures.append({
                "step": i,
                "type": "TRANSLATION_EXCEPTION",
                "original_indices": idx_tuple,
                "adsg_params": adsg_res_params,
                "error": str(e)
            })
            continue

        # -------------------------------------------------------------
        # 3. Equivalence Verification Assertions
        # -------------------------------------------------------------
        # A. Design-vector equivalence
        vec_equiv = (list(idx_tuple) == adsg_res_indices) and (ref_params == adsg_res_params)
        if vec_equiv:
            design_vector_equiv_count += 1
        else:
            failures.append({
                "step": i,
                "type": "DESIGN_VECTOR_MISMATCH",
                "original_indices": idx_tuple,
                "resolved_indices": adsg_res_indices,
                "reference_params": ref_params,
                "adsg_params": adsg_res_params,
            })

        # B. Configuration equivalence (all SimObject attributes + CLI args)
        cfg_equiv = (
            (adsg_cfg.as_param_tuple() == ref_cfg.as_param_tuple())
            and (adsg_cfg.simobject_attrs == ref_cfg.simobject_attrs)
            and (adsg_cfg.cli_args == ref_cfg.cli_args)
        )
        if cfg_equiv:
            gem5_config_equiv_count += 1
        else:
            failures.append({
                "step": i,
                "type": "CONFIG_MISMATCH",
                "ref_config": ref_cfg,
                "adsg_config": adsg_cfg,
            })

        # C. Feasibility equivalence (exact cost & feasibility verdict)
        feas_equiv = (
            (abs(adsg_cfg.cost - ref_cfg.cost) < 1e-6)
            and (adsg_cfg.is_feasible == ref_cfg.is_feasible)
        )
        if feas_equiv:
            feasibility_equiv_count += 1
        else:
            failures.append({
                "step": i,
                "type": "FEASIBILITY_MISMATCH",
                "ref_cost": ref_cfg.cost,
                "adsg_cost": adsg_cfg.cost,
                "ref_feasible": ref_cfg.is_feasible,
                "adsg_feasible": adsg_cfg.is_feasible,
            })

        # D. Objective equivalence (all 3 workloads)
        ipc_all_match = True
        for wl in workloads:
            if abs(adsg_ipcs[wl] - ref_ipcs[wl]) < 1e-6:
                ipc_equiv_count[wl] += 1
            else:
                ipc_all_match = False
                failures.append({
                    "step": i,
                    "type": f"IPC_MISMATCH_{wl.upper()}",
                    "ref_ipc": ref_ipcs[wl],
                    "adsg_ipc": adsg_ipcs[wl],
                })

        if i % 100 == 0 or i == total_vectors:
            print(f"  Verified {i:03d} / {total_vectors} design vectors...")

    elapsed = time.time() - start_time
    print(f"\nVerification finished in {elapsed:.2f} seconds.")

    # -------------------------------------------------------------
    # 4. Diagnostics & Reporting
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                    VERIFICATION REPORT SUMMARY")
    print("=" * 80)
    print(f"Total Design Vectors Evaluated:    {total_vectors}")
    print(f"Successfully Resolved Vectors:     {resolved_count} / {total_vectors}")
    print(f"Unresolved States:                 {unresolved_count}")
    print(f"Unmapped States:                   {unmapped_count}")
    print("-" * 80)
    print(f"Design-Vector Equivalence [ADSG(x) = x]:  {design_vector_equiv_count} / {total_vectors}")
    print(f"gem5 Configuration Equivalence:          {gem5_config_equiv_count} / {total_vectors}")
    print(f"Feasibility Equivalence (<= {budget_kb} KB):   {feasibility_equiv_count} / {total_vectors}")
    for wl in workloads:
        print(f"IPC Equivalence [{wl:<11}]:             {ipc_equiv_count[wl]} / {total_vectors}")
    print("=" * 80)

    if failures:
        print(f"\n[!] DETECTED {len(failures)} FAILURES DURING VERIFICATION:")
        for f in failures[:10]:
            print(f"  Failure Details: {f}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more failures.")
        print("\nADSG REPRESENTATION VALIDATED: FAIL")
        return False

    all_ipc_passed = all(count == total_vectors for count in ipc_equiv_count.values())
    all_passed = (
        resolved_count == total_vectors
        and unresolved_count == 0
        and unmapped_count == 0
        and design_vector_equiv_count == total_vectors
        and gem5_config_equiv_count == total_vectors
        and feasibility_equiv_count == total_vectors
        and all_ipc_passed
    )

    if all_passed:
        print("\nADSG REPRESENTATION VALIDATED: PASS")
        return True
    else:
        print("\nADSG REPRESENTATION VALIDATED: FAIL")
        return False


if __name__ == "__main__":
    success = run_exhaustive_equivalence_verification()
    sys.exit(0 if success else 1)
