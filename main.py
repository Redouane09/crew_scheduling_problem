#!/usr/bin/env python3
"""
Crew Scheduling Solver CLI (updated)

- Adds support for 'mosek' and 'highs' solvers in addition to 'gurobi'.
- Solver backends are imported lazily to avoid hard dependency errors at startup.
- Each solver wrapper is responsible for checking availability (Python API / CLI).
- Keeps existing behavior/output format identical to prior Gurobi path.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import GurobiSolver eagerly (existing code depends on it in many places)
from solvers.gurobi_solver import GurobiSolver


def check_gurobi_license():
    """Verify Gurobi availability (importable)."""
    try:
        import gurobipy as gp  # noqa: F401
        print("✓ Gurobi detected")
        return True
    except Exception as e:
        print(f"✗ Gurobi not available: {str(e)}")
        print("\nTo use Gurobi solver:")
        print("  1. Install with: pip install gurobipy")
        print("  2. Ensure a valid license is available (GRB_LICENSE_FILE or local license)")
        return False


def solve_instance(args):
    """Solve instance with specified solver"""
    try:
        # Load instance
        if not os.path.exists(args.input):
            print(f"✗ Input file not found: {args.input}")
            return 1
        with open(args.input, 'r') as f:
            instance = json.load(f)
        # attach filename so solvers can name saved files
        instance['_instance_filename'] = args.input

        # Basic instance stats (robust)
        num_flights = len(instance.get('flights', []))
        num_days = instance.get('parameters', {}).get('num_days', None)
        num_crew = instance.get('parameters', {}).get('num_crew', None)

        print(f"\nSolving {os.path.basename(args.input)} with {args.solver}...")
        print(f"Instance: {num_flights} flights, "
              f"{num_days if num_days is not None else 'n/a'} days, "
              f"{num_crew if num_crew is not None else 'n/a'} crew members")

        # Select solver lazily
        solver = None
        solver_name = args.solver.lower() if args.solver else "gurobi"

        if solver_name == 'gurobi':
            if not check_gurobi_license():
                print("✗ Aborting: Gurobi not available")
                return 2
            solver = GurobiSolver(license_path=args.license, log_file=args.log_file)

        elif solver_name == 'mosek':
            try:
                from solvers.mosek_solver import MOSEKSolver
            except Exception as e:
                print(f"✗ Could not import MOSEK solver wrapper: {e}")
                print("  Ensure file solvers/mosek_solver.py exists and is importable.")
                return 3
            try:
                solver = MOSEKSolver(license_path=args.license, log_file=args.log_file)
            except Exception as e:
                print(f"✗ Failed to initialize MOSEK solver wrapper: {e}")
                return 3

        elif solver_name in ('highs', 'highs_wrapper', 'highs_solver'):
            try:
                from solvers.highs_solver import HiGHSWrapperSolver
            except Exception as e:
                print(f"✗ Could not import HiGHS solver wrapper: {e}")
                print("  Ensure file solvers/highs_solver.py exists and is importable.")
                return 3
            try:
                solver = HiGHSWrapperSolver()
            except Exception as e:
                print(f"✗ Failed to initialize HiGHS solver wrapper: {e}")
                return 3

        else:
            print(f"✗ Unsupported solver '{args.solver}' in this CLI build. Available: gurobi, mosek, highs")
            return 3

        # Solve
        start_time = time.time()
        solution = solver.solve(instance, time_limit=args.time_limit)
        solve_time = time.time() - start_time

        if not solution:
            print("\n✗ No solution was extracted. Problem may be infeasible or solver encountered an error.")
            return 4

        # Normalize and supply defaults
        solution.setdefault('uncovered_flights', [])
        solution.setdefault('crew_assignments', defaultdict(list))
        solution.setdefault('unused_crew', [])
        solution.setdefault('crew_used', len(solution['crew_assignments']))
        solution.setdefault('hotel_stays', 0)
        solution.setdefault('deadhead_flights', [])
        solution.setdefault('objective_components', {})
        solution.setdefault('instance_info', {})
        solution.setdefault('var_counts', {})
        solution.setdefault('constraint_counts', {})

        # Basic summary
        print("\n" + "="*70)
        print(f"SOLUTION SUMMARY ({args.solver.upper()})")
        print("="*70)
        feasibility = solution.get('feasibility', 'Unknown')
        print(f"Solver Status: {feasibility}")
        if feasibility not in ("Optimal", "Feasible", "TimeLimit"):
            print(f"⚠ WARNING: Solution status is '{feasibility}' — results may not be reliable!")
        obj_val = solution.get('objective_value', solution.get('objective'))
        if obj_val is not None:
            try:
                print(f"Objective Value: ${float(obj_val):,.2f}")
            except Exception:
                print(f"Objective Value: {obj_val}")
        else:
            print("Objective Value: N/A")
        print(f"Solve Time: {solve_time:.2f} seconds")
        total_flights = instance.get('parameters', {}).get('num_flights', num_flights)
        print(f"Flights Covered: {total_flights - len(solution.get('uncovered_flights', []))} / {total_flights}")
        print(f"Crew Used: {solution.get('crew_used', len(solution.get('crew_assignments', {})))} / {instance.get('parameters', {}).get('num_crew', 'n/a')}")
        print(f"Reserve Crew: {len(solution.get('unused_crew', []))}")
        print("="*70)

        # Instance info
        inst = solution.get("instance_info", {})
        if inst:
            print("Instance Info:")
            print(f"  Flights: {inst.get('num_flights', 'n/a')}")
            print(f"  Crews:   {inst.get('num_crews', 'n/a')}")
            print(f"  Days:    {inst.get('num_days', 'n/a')}")
            print(f"  Cities:  {inst.get('num_cities', 'n/a')}")
            # best-effort connection names to avoid KeyError
            same_day_conns = inst.get('same_day_connections', inst.get('same_day_pairs', inst.get('same_day_pairs', 'n/a')))
            next_day_conns = inst.get('next_day_connections', inst.get('next_day_pairs', inst.get('next_day_pairs', 'n/a')))
            print(f"  Connections: same-day={same_day_conns}, next-day={next_day_conns}")
            print("-"*70)

        # Objective breakdown
        obj = solution.get("objective_components", {})
        if obj:
            print("Objective Breakdown:")
            print(f"  Uncover cost: ${obj.get('uncover_cost', 0):,.2f}")
            print(f"  Deadhead cost: ${obj.get('deadhead_cost', 0):,.2f}")
            print(f"  Away-from-home cost: ${obj.get('away_cost', 0):,.2f}")
            print(f"  Crew cost: ${obj.get('crew_cost', 0):,.2f}")
            print(f"  Total objective: ${obj.get('total_objective', obj_val):,.2f}")
            print("-"*70)

        # Model size
        var_counts = solution.get("var_counts", {})
        con_counts = solution.get("constraint_counts", {})
        if var_counts or con_counts:
            print("Model Size:")
            if var_counts:
                try:
                    total_vars = sum(var_counts.values())
                except Exception:
                    total_vars = 'n/a'
                print(f"  Variables (total {total_vars}): {var_counts}")
            if con_counts:
                try:
                    total_cons = sum(con_counts.values())
                except Exception:
                    total_cons = 'n/a'
                print(f"  Constraints (total {total_cons}): {con_counts}")
            print("-"*70)

        # Deadheads and hotels
        print(f"Hotel stays: {solution.get('hotel_stays', 0)}")
        dh = solution.get('deadhead_flights', [])
        print(f"Deadhead flights: {len(dh)}")
        if dh:
            print("  (flight, extra_covers): " + ", ".join([f"({fid}, {cnt})" for fid, cnt in dh]))
        print("-"*70)

        # Crew assignments
        print("Crew Assignments (per crew):")
        crew_assignments = solution.get('crew_assignments', {})
        if isinstance(crew_assignments, dict):
            for crew_id in sorted(crew_assignments.keys()):
                legs = crew_assignments.get(crew_id, [])
                if legs:
                    # Sort by day and round (attempt to normalize day)
                    legs_sorted = sorted(legs, key=lambda x: (int(x.get("day", 0)) if str(x.get("day", "")).isdigit() else 0, x.get("round", 0)))
                    print(f"  Crew {crew_id}:")
                    for leg in legs_sorted:
                        day = leg.get("day", "n/a")
                        round_ = leg.get("round", "n/a")
                        origin = leg.get("origin", leg.get("origin", ""))
                        destination = leg.get("destination", leg.get("destination", ""))
                        depart = leg.get("departure_time", "")
                        print(f"    Day {day}, R{round_}: {origin}->{destination} @ {depart}")
                else:
                    print(f"  Crew {crew_id}: —")
        else:
            print("  (unrecognized crew_assignments format)")
        print("="*70)

        # Optionally save to output file
        if args.output:
            solver_stats = None
            try:
                solver_stats = solver.get_solver_stats
            except Exception:
                solver_stats = None
            with open(args.output, 'w') as f:
                json.dump({
                    'solver': args.solver,
                    'instance': os.path.basename(args.input),
                    'solution': solution,
                    'solver_stats': solver_stats,
                    'solve_time_sec': solve_time
                }, f, indent=2)
            print(f"\n✓ Solution saved to {args.output}")

        return 0

    except KeyError as e:
        print(f"✗ KeyError: Missing key in solution or instance: {e}")
        return 5
    except FileNotFoundError as e:
        print(f"✗ File Not Found: {e}")
        return 6
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 7


def parse_args():
    p = argparse.ArgumentParser(description="Solve crew scheduling instance")
    p.add_argument("solve", nargs='?', help="command (solve)")
    p.add_argument("--solver", default="gurobi", help="solver to use (default: gurobi). Supported: gurobi, mosek, highs")
    p.add_argument("--input", required=True, help="path to instance JSON file")
    p.add_argument("--time-limit", type=float, default=300.0, help="time limit in seconds")
    p.add_argument("--log-file", default=None, help="solver log file path")
    p.add_argument("--license", default=None, help="solver license file (if required)")
    p.add_argument("--output", default=None, help="output file to write solution JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rc = solve_instance(args)
    sys.exit(rc)