#!/usr/bin/env python3
# solvers/highs_solver.py
"""
HiGHS solver wrapper for the crew scheduling model.

Provides:
- robust availability check (prefer CLI /usr/local/bin/highs, fallback to highspy)
- attempts to set executable for Pyomo solver plugin when necessary
- delegates to IntegratedCrewSchedulingModel.solve(...) (which streams solver output)
- returns the model solution (and writes diagnostics prefixed by solver)
"""
import sys
import os
import time
import traceback
import shutil
from collections import defaultdict

from .base_solver import BaseSolver

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.integrated_csp_model import IntegratedCrewSchedulingModel
from pyomo.opt import SolverFactory


class HiGHSWrapperSolver(BaseSolver):
    """HiGHS solver wrapper using Pyomo backends."""

    def __init__(self):
        super().__init__("HiGHS_Exact")
        self.solve_time = None
        self.objective_value = None
        self.model = None

    def _check_highs_available(self):
        """
        Return (available_bool, chosen_backend, diagnostic_message).

        Strategy:
         - If a highs executable is on PATH (shutil.which('highs')), prefer the CLI.
           Try to create SolverFactory('highs'); if not available try to set_executable(..., validate=False).
         - Otherwise, attempt to import highspy; if present, use 'highs_direct'.
         - Avoid calling SolverFactory('highs_direct') blindly when highspy isn't importable.
        """
        attempted = []
        highs_bin = shutil.which('highs')

        # prefer CLI if present
        if highs_bin:
            attempted.append('highs(path_detected)')
            try:
                s = SolverFactory('highs')
                # if plugin not available, try set_executable if method exists
                try:
                    if not s.available(False) and hasattr(s, "set_executable"):
                        try:
                            s.set_executable(highs_bin, validate=False)
                        except Exception:
                            pass
                    if s.available(False):
                        return True, 'highs', f"HiGHS CLI backend available at {highs_bin}"
                except Exception:
                    # if available(False) raised but object created, assume usable when CLI is present
                    return True, 'highs', f"HiGHS CLI found at {highs_bin} (creation succeeded)"
            except Exception:
                pass

        # try python binding if installed
        try:
            import highspy  # type: ignore
            attempted.append('highspy_installed')
            try:
                s2 = SolverFactory('highs_direct')
                try:
                    if s2.available(False):
                        return True, 'highs_direct', "HiGHS python binding available ('highs_direct')."
                except Exception:
                    return True, 'highs_direct', "HiGHS python binding importable; will attempt 'highs_direct'."
            except Exception:
                attempted.append('highs_direct_creation_failed')
        except Exception:
            attempted.append('highspy_not_installed')

        highs_bin = shutil.which('highs')
        msg_lines = [
            f"Attempted backends: {attempted}",
            f"shutil.which('highs') -> {highs_bin}",
            "HiGHS is not available to Pyomo in this environment. Install HiGHS CLI (conda install -c conda-forge highs / brew install highs) "
            "or the Python binding 'highspy' (pip install highspy) and ensure you run from the same Python environment."
        ]
        return False, None, "\n".join(msg_lines)

    def solve(self, instance_data, time_limit=300.0):
        start_time = time.time()
        try:
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

            model = IntegratedCrewSchedulingModel(instance_data)
            self.model = model

            solver_backend = 'highs' if backend == 'highs' else 'highs_direct'
            print(f"Using HiGHS backend: {solver_backend}")

            solution = model.solve(time_limit=time_limit, solver_name=solver_backend)
            if solution is None:
                print("✗ No solution returned by model.solve()")
                return None

            feasibility = solution.get("feasibility", "Unknown")
            if feasibility in ("Error", "Infeasible", "Unbounded"):
                print(f"✗ HiGHS finished with status: {feasibility}")
                if "error_message" in solution:
                    print("  Error:", solution["error_message"])
                diagnostics_dir = os.path.join(os.getcwd(), "diagnostics")
                os.makedirs(diagnostics_dir, exist_ok=True)
                model._write_diagnostics(solution, diagnostics_dir, instance_data)
                return solution

            self.solve_time = time.time() - start_time
            self.objective_value = solution.get("objective_value")
            diagnostics_dir = os.path.join(os.getcwd(), "diagnostics")
            os.makedirs(diagnostics_dir, exist_ok=True)
            model._write_diagnostics(solution, diagnostics_dir, instance_data)
            return solution

        except Exception as e:
            print(f"✗ Exception in HiGHS wrapper: {e}")
            traceback.print_exc()
            return {
                "feasibility": "Error",
                "error_message": str(e),
                "solve_time": time.time() - start_time,
                "uncovered_flights": instance_data.get("flights", []),
                "instance_info": {
                    "num_flights": len(instance_data.get("flights", [])),
                    "num_crews": len(instance_data.get("crew", [])),
                }
            }

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