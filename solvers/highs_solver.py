#!/usr/bin/env python3
# solvers/highs_solver.py
"""
HiGHS solver wrapper for the integrated crew scheduling model.

This wrapper checks availability (tries 'highs' CLI then 'highs_direct'),
fails fast with a clear diagnostic if HiGHS is not available, builds the
model and delegates to model.solve(..., solver_name=chosen_backend).
"""
import sys
import os
import time
import traceback
import shutil
from collections import defaultdict

from .base_solver import BaseSolver

# Ensure repo root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.integrated_csp_model import IntegratedCrewSchedulingModel
from pyomo.opt import SolverFactory


class HiGHSWrapperSolver(BaseSolver):
    """HiGHS solver using the generic Pyomo model."""

    def __init__(self):
        super().__init__("HiGHS_Exact")
        self.solve_time = None
        self.objective_value = None
        self.model = None

    def _check_highs_available(self):
        """Return (available_bool, chosen_backend, diagnostic_message).

        Strategy:
         - Prefer CLI if `shutil.which('highs')` returns a path.
         - Otherwise, attempt to import the Python binding 'highspy' and use 'highs_direct' if present.
         - Avoid blindly calling SolverFactory('highs_direct') when highspy is not installed (that can cause Pyomo to try mapping to an ASL solver).
        """
        attempted = []
        # Prefer CLI if present
        highs_bin = shutil.which('highs')
        if highs_bin:
            # try a light-weight check using SolverFactory but do not rely on available(True) raising
            try:
                s = SolverFactory('highs')
                attempted.append('highs')
                # Return CLI as chosen backend if creation succeeded
                return True, 'highs', f"HiGHS CLI found at {highs_bin}"
            except Exception:
                # If creation unexpectedly fails, continue to binding check
                attempted.append('highs_creation_failed')

        # Try Python binding 'highspy' before calling SolverFactory('highs_direct')
        try:
            import highspy  # type: ignore
            # If import succeeded, attempt to use solver name highs_direct
            try:
                s2 = SolverFactory('highs_direct')
                attempted.append('highs_direct')
                return True, 'highs_direct', "HiGHS Python binding (highspy) is importable; will use highs_direct."
            except Exception:
                attempted.append('highs_direct_creation_failed')
        except Exception:
            attempted.append('highspy_not_installed')

        msg_lines = [
            f"Attempted backends: {attempted}",
            f"shutil.which('highs') -> {highs_bin}",
            "HiGHS is not available to Pyomo in this environment. Install the HiGHS CLI "
            "(conda install -c conda-forge highs / brew install highs) or the Python binding 'highspy' "
            "(pip install highspy) and ensure you are running the same Python environment."
        ]
        return False, None, "\n".join(msg_lines)

    def solve(self, instance_data, time_limit=300.0):
        start_time = time.time()
        try:
            # Quick availability check: fail fast with clear diagnostics if HiGHS is not usable
            avail, backend, diag = self._check_highs_available()
            if not avail:
                print("✗ HiGHS not available:", diag)
                return {
                    "feasibility": "Error",
                    "error_message": diag,
                    "solve_time": 0.0,
                    "uncovered_flights": instance_data.get("flights", []),
                    "instance_info": {
                        "num_flights": len(instance_data.get("flights", [])),
                        "num_crews": len(instance_data.get("crew", [])),
                        "num_days": instance_data.get("parameters", {}).get("num_days", None),
                        "num_cities": len(instance_data.get("cities", []) or [])
                    }
                }

            # Build the Pyomo model
            model = IntegratedCrewSchedulingModel(instance_data)
            self.model = model

            # Solve with HiGHS (use the backend determined by availability check)
            solver_name = 'highs' if backend == 'highs' else 'highs_direct'
            solution = model.solve(time_limit=time_limit, solver_name=solver_name)
            if solution is None:
                print("✗ No solution returned by the solver.")
                return None

            # Check if the solver reported an error or infeasibility
            feasibility = solution.get("feasibility", "Unknown")
            if feasibility in ("Error", "Infeasible", "Unbounded"):
                print(f"✗ Solver finished with status: {feasibility}")
                if "error_message" in solution:
                    print(f"  Error: {solution['error_message']}")
                diagnostics_dir = os.path.join(os.getcwd(), "diagnostics")
                os.makedirs(diagnostics_dir, exist_ok=True)
                model._write_diagnostics(solution, diagnostics_dir, instance_data)
                return None

            self.solve_time = time.time() - start_time
            self.objective_value = solution.get('objective_value')

            # Write diagnostics
            diagnostics_dir = os.path.join(os.getcwd(), "diagnostics")
            os.makedirs(diagnostics_dir, exist_ok=True)
            model._write_diagnostics(solution, diagnostics_dir, instance_data)

            return solution

        except Exception as e:
            print(f"✗ Solver error: {e}")
            traceback.print_exc()
            self.solve_time = time.time() - start_time
            self.objective_value = None
            return None

    def get_solver_stats(self):
        stats = {
            'solver_name': self.name,
            'solve_time_sec': self.solve_time,
            'objective_value': self.objective_value,
            'paper_reference': 'Saemi et al. (2021) Eqs 1-35 - FULL MODEL (Pyomo)',
            'solver_version': 'HiGHS via Pyomo',
        }
        if hasattr(self, 'model') and self.model:
            try:
                stats['variables'] = sum(self.model.var_stats.values())
                stats['var_counts'] = dict(self.model.var_stats)
            except Exception:
                stats['var_counts'] = {}
            try:
                stats['model_constraints'] = sum(self.model.con_stats.values())
                stats['constraint_breakdown'] = dict(self.model.con_stats)
            except Exception:
                stats['constraint_breakdown'] = {}
        return stats

    get_solver_stats = property(get_solver_stats)