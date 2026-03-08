#!/usr/bin/env python3
# solvers/mosek_solver.py
import sys
import os
import time
import json
import csv
import traceback
from collections import defaultdict

from .base_solver import BaseSolver

# Ensure repo root is on sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.integrated_csp_model import IntegratedCrewSchedulingModel


class MOSEKSolver(BaseSolver):
    """MOSEK solver using the generic Pyomo model."""

    def __init__(self, license_path=None, log_file=None):
        super().__init__("MOSEK_Exact")
        self.license_path = license_path
        self.log_file = log_file
        self.solve_time = None
        self.objective_value = None
        self.model = None

    def solve(self, instance_data, time_limit=300.0):
        start_time = time.time()
        try:
            # Build the Pyomo model
            model = IntegratedCrewSchedulingModel(instance_data, log_file=self.log_file)
            self.model = model

            # Solve with MOSEK
            solution = model.solve(time_limit=time_limit, solver_name='mosek')
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
            'solver_version': 'MOSEK via Pyomo',
            'log_file': self.log_file
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