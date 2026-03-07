#!/usr/bin/env python3
# model/integrated_csp_model.py
"""
Pyomo implementation of Saemi et al. (2021) integrated crew scheduling model.

This file is a corrected and cleaned-up version of the earlier implementation.
Key fixes:
 - Sanity checks (_run_sanity_checks) are executed on the extracted solution
   and the solution.feasibility is annotated if inconsistencies are found.
 - Diagnostic CSV writers use the model's processed flight data (shifted days,
   departure/arrival), and crew home-base mapping used by the model, so the
   solver summary and the produced CSVs match.
 - Improved robustness in variable index parsing and defensive handling of None
   variable values while extracting solutions.
 - Removed duplicate / misplaced code fragments.
"""

import time
import os
import csv
import json
from collections import defaultdict
from collections.abc import Iterable

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition


class IntegratedCrewSchedulingModel:
    """
    Pyomo implementation of Saemi et al. (2021) integrated crew scheduling model.

    Optional parameter:
      parameters["force_cover_all_flights"] = True/False
        If True, every flight must be covered by at least one crew.
    """

    def __init__(self, instance_data, log_file=None):
        self.instance = instance_data or {}
        self.params = self.instance.get("parameters", {}) or {}
        self.log_file = log_file

        # Basic sets (internal)
        self.num_crew = int(self.params.get("num_crew", 0))
        self.num_days = int(self.params.get("num_days", 0))
        # Crew indices are 0-based
        self.I = list(range(self.num_crew))
        # Days as 1-based list (model uses 1-based days)
        self.D = list(range(1, self.num_days + 1))
        self.J = list(self.instance.get("cities", []))

        # Parameters (defaults recovered when not provided)
        self.M = float(self.params.get("big_m", 10000.0))
        self.epsilon = float(self.params.get("epsilon", 0.001))
        self.min_sit = float(self.params.get("min_sit_time", 30.0))
        self.min_rest = float(self.params.get("min_rest_time", 660.0))
        self.upper1 = float(self.params.get("max_flying_time_duty", 480.0))
        self.upper2 = float(self.params.get("max_flights_horizon", 960.0))
        self.upper3 = float(self.params.get("max_elapsed_time_duty", 840.0))
        self.h = float(self.params.get("hotel_cost", 250.0))

        self.force_cover_all_flights = bool(self.params.get("force_cover_all_flights", False))

        # Crew data (costs and home base)
        self.r_i = {}
        self.o_i = {}
        crew_data = self.instance.get("crew", [])
        for i in self.I:
            if i < len(crew_data):
                # cost may be present in per-crew dictionary, else default
                self.r_i[i] = float(crew_data[i].get("cost", self.params.get("default_crew_cost", 500.0)))
                # home base from instance; fallback to first city if missing
                self.o_i[i] = crew_data[i].get("home_base", self.J[0] if self.J else None)
            else:
                self.r_i[i] = float(self.params.get("default_crew_cost", 100.0))
                self.o_i[i] = self.J[0] if self.J else None

        # Preprocess flights and costs (build flights_by_id, fid_list, etc.)
        self._precompute_flights_and_costs()

        # Determine max_rounds and max_duties (N and F)
        self._determine_rounds_and_duties()

        # Variable and constraint statistics
        self.var_stats = defaultdict(int)
        self.con_stats = defaultdict(int)

        # Variable name map for reconstruction (used by diagnostics)
        self.var_name_map = {}

        # Build the Pyomo model
        self._build_pyomo_model()

        total_vars = sum(self.var_stats.values())
        total_cons = sum(self.con_stats.values())
        print(f"✓ Model created with {total_vars} variables and {total_cons} constraints")

    # ------------------------------------------------------------------
    # Data preprocessing
    # ------------------------------------------------------------------
    def _precompute_flights_and_costs(self):
        """Process instance flights, shift days to 1‑based, build lookups."""
        raw_flights = []
        raw_days = []
        self.flights_by_id = {}
        self.fid_list = []
        self.flight_costs = {}
        self.uncover_costs = {}

        def to_min(t):
            if isinstance(t, (int, float)):
                return float(t)
            # accept either "HH:MM" or already numeric strings
            parts = str(t).split(":")
            if len(parts) == 2:
                h, m = map(int, parts)
                return float(h * 60 + m)
            try:
                return float(t)
            except Exception:
                # fallback: 0
                return 0.0

        # Uniformly accept flight fields with multiple possible keys
        for f in self.instance.get("flights", []):
            fid = f["flight_id"]
            day_raw = int(f["day"])
            raw_days.append(day_raw)
            dep_key = f.get("departure_time", f.get("depart", f.get("de")))
            arr_key = f.get("arrival_time", f.get("arrive", f.get("l")))
            dead_cost = f.get("deadhead_cost", None)
            uncov_cost = f.get("uncover_cost", None)
            raw_flights.append((fid, f.get("origin"), f.get("destination"),
                                day_raw, dep_key, arr_key,
                                dead_cost, uncov_cost))

        # Shift days to make minimum day == 1 if needed
        if raw_days:
            min_raw_day = min(raw_days)
            if min_raw_day < 1:
                shift = 1 - min_raw_day
                print(f"Info: detected flight day indices starting at {min_raw_day}; shifting all flight days by +{shift} to become 1‑based.")
            else:
                shift = 0
        else:
            shift = 0

        for (fid, origin, dest, day_raw, dep_time, arr_time, dead_cost, uncov_cost) in raw_flights:
            day = day_raw + shift
            de = to_min(dep_time)
            ar = to_min(arr_time)
            # normalize keys to consistent internal structure
            self.flights_by_id[fid] = {
                "fid": fid,
                "origin": origin,
                "dest": dest,
                "day": day,
                "de": de,
                "l": ar,
                "de_str": dep_time,
                "arr_str": arr_time,
                "duration": ar - de
            }
            self.fid_list.append(fid)
            self.flight_costs[fid] = float(dead_cost if dead_cost is not None else self.params.get("default_deadhead_cost", 200.0))
            self.uncover_costs[fid] = float(uncov_cost if uncov_cost is not None else self.params.get("default_uncover_cost", 10000.0))

        # Flights by day and by (origin,destination,day)
        self.flights_by_day = defaultdict(list)
        self.flights_by_orig_dest_day = defaultdict(list)
        for fid in self.fid_list:
            f = self.flights_by_id[fid]
            self.flights_by_day[f["day"]].append(fid)
            key = (f["origin"], f["dest"], f["day"])
            self.flights_by_orig_dest_day[key].append(fid)

        # Precompute same‑day and next‑day feasible pairs (for Z)
        self.same_day_pairs = []
        self.next_day_pairs = []
        for fid1 in self.fid_list:
            f1 = self.flights_by_id[fid1]
            for fid2 in self.fid_list:
                if fid1 == fid2:
                    continue
                f2 = self.flights_by_id[fid2]
                if f1["dest"] != f2["origin"]:
                    continue
                if f1["day"] == f2["day"]:
                    self.same_day_pairs.append((fid1, fid2))
                elif f1["day"] + 1 == f2["day"]:
                    self.next_day_pairs.append((fid1, fid2))

        print(f"✓ Precomputed {len(self.fid_list)} flights, {len(self.J)} cities, {len(self.I)} crew")
        print(f"✓ Found {len(self.same_day_pairs)} same‑day pairs and {len(self.next_day_pairs)} next‑day pairs")

    # ------------------------------------------------------------------
    # Determine N and F sizes
    # ------------------------------------------------------------------
    def _determine_rounds_and_duties(self):
        """Determine N (rounds) and F (duties) with caps."""
        MAX_ROUNDS_CAP = int(self.params.get("max_rounds_cap", 50))
        MAX_DUTIES_CAP = int(self.params.get("max_duties_cap", 50))

        try:
            safe_default_rounds = max(1, max(len(self.flights_by_day.get(d, [])) for d in self.D))
        except Exception:
            safe_default_rounds = 10

        raw_max_rounds = self.params.get("max_rounds", None)
        if raw_max_rounds is None:
            max_rounds = safe_default_rounds
        else:
            max_rounds = int(raw_max_rounds)
            if max_rounds < 1:
                raise ValueError("max_rounds must be >= 1")
        if max_rounds > MAX_ROUNDS_CAP:
            print(f"Warning: max_rounds={max_rounds} exceeds cap {MAX_ROUNDS_CAP}; reducing.")
            max_rounds = MAX_ROUNDS_CAP

        raw_max_duties = self.params.get("max_duties", None)
        safe_default_duties = max(1, len(self.D))
        if raw_max_duties is None:
            max_duties = safe_default_duties
        else:
            max_duties = int(raw_max_duties)
            if max_duties < 1:
                raise ValueError("max_duties must be >= 1")
        if max_duties > MAX_DUTIES_CAP:
            print(f"Warning: max_duties={max_duties} exceeds cap {MAX_DUTIES_CAP}; reducing.")
            max_duties = MAX_DUTIES_CAP

        # N and F are 1-based lists
        self.N = list(range(1, max_rounds + 1))
        self.N_minus_1 = list(range(1, max_rounds))  # for Z between n and n+1
        self.F = list(range(1, max_duties + 1))
        print(f"✓ Using max_rounds = {max_rounds}, max_duties = {max_duties}")

    # ------------------------------------------------------------------
    # Pyomo model construction
    # ------------------------------------------------------------------
    def _build_pyomo_model(self):
        """Create Pyomo ConcreteModel with all variables and constraints."""
        self.model = pyo.ConcreteModel()

        # ---------- Sets ----------
        self.model.I = pyo.Set(initialize=self.I, ordered=True)
        self.model.J = pyo.Set(initialize=self.J, ordered=True)
        self.model.D = pyo.Set(initialize=self.D, ordered=True)
        self.model.F = pyo.Set(initialize=self.F, ordered=True)
        self.model.N = pyo.Set(initialize=self.N, ordered=True)
        self.model.N_minus_1 = pyo.Set(initialize=self.N_minus_1, ordered=True)
        self.model.Flights = pyo.Set(initialize=self.fid_list, ordered=True)

        # Pairs for Z (union of same‑day and next‑day)
        self.model.Pairs = pyo.Set(initialize=self.same_day_pairs + self.next_day_pairs,
                                   ordered=True, dimen=2)

        # ---------- Variables ----------
        # x[i,fid,n] binary: crew i does flight fid on round n
        self.model.x = pyo.Var(self.model.I, self.model.Flights, self.model.N,
                               within=pyo.Binary)
        self.var_stats["x"] += sum(1 for _ in self.model.x)

        # Z is defined only for feasible flight pairs (Pairs) and n in N_minus_1
        self.model.Z = pyo.Var(self.model.I, self.model.Pairs, self.model.N_minus_1,
                               within=pyo.Binary)
        self.var_stats["Z"] += sum(1 for _ in self.model.Z)

        # s and w: start and end positions
        self.model.s = pyo.Var(self.model.I, self.model.J, self.model.D,
                               within=pyo.Binary)
        self.var_stats["s"] += sum(1 for _ in self.model.s)

        self.model.w = pyo.Var(self.model.I, self.model.J, self.model.D,
                               within=pyo.Binary)
        self.var_stats["w"] += sum(1 for _ in self.model.w)

        # y: duty assignment (i,d,f)
        self.model.y = pyo.Var(self.model.I, self.model.D, self.model.F,
                               within=pyo.Binary)
        self.var_stats["y"] += sum(1 for _ in self.model.y)

        # SS: crew used flag
        self.model.SS = pyo.Var(self.model.I, within=pyo.Binary)
        self.var_stats["SS"] += sum(1 for _ in self.model.SS)

        # v: covered flight indicator
        self.model.v = pyo.Var(self.model.Flights, within=pyo.Binary)
        self.var_stats["v"] += sum(1 for _ in self.model.v)

        # b: integer deadhead counts
        self.model.b = pyo.Var(self.model.Flights, within=pyo.NonNegativeIntegers)
        self.var_stats["b"] += sum(1 for _ in self.model.b)

        # Build variable name map (for diagnostics)
        self._build_var_name_map()

        # ---------- Constraints ----------
        self._add_constraints()

        # ---------- Objective ----------
        self._set_objective()

    def _build_var_name_map(self):
        """Populate var_name_map with metadata for each variable (useful for debug)."""
        # x
        for i in self.model.I:
            for fid in self.model.Flights:
                for n in self.model.N:
                    var = self.model.x[i, fid, n]
                    self.var_name_map[var.name] = {"kind": "x", "crew": i, "fid": fid, "round": n}
        # s
        for i in self.model.I:
            for j in self.model.J:
                for d in self.model.D:
                    var = self.model.s[i, j, d]
                    self.var_name_map[var.name] = {"kind": "s", "crew": i, "city": j, "day": d}
        # w
        for i in self.model.I:
            for j in self.model.J:
                for d in self.model.D:
                    var = self.model.w[i, j, d]
                    self.var_name_map[var.name] = {"kind": "w", "crew": i, "city": j, "day": d}
        # v and b
        for fid in self.model.Flights:
            self.var_name_map[self.model.v[fid].name] = {"kind": "v", "fid": fid}
            self.var_name_map[self.model.b[fid].name] = {"kind": "b", "fid": fid}

    def _add_constraints(self):
        """Add all constraints (2) to (35)."""
        M = self.M
        eps = self.epsilon
        min_sit = self.min_sit
        min_rest = self.min_rest
        upper1 = self.upper1
        upper2 = self.upper2
        upper3 = self.upper3

        flights_on_day = self.flights_by_day

        # (2) At most one round per flight per crew
        def c2_rule(model, i, fid):
            return sum(model.x[i, fid, n] for n in model.N) <= 1
        self.model.c2 = pyo.Constraint(self.model.I, self.model.Flights, rule=c2_rule)
        self.con_stats["2"] = len(self.model.c2)

        # (3) At most one flight per crew per round
        def c3_rule(model, i, n):
            return sum(model.x[i, fid, n] for fid in model.Flights) <= 1
        self.model.c3 = pyo.Constraint(self.model.I, self.model.N, rule=c3_rule)
        self.con_stats["3"] = len(self.model.c3)

        # (4) Same‑day sit time – BuildAction over pairs
        def c4_build(model):
            for i in model.I:
                for (fid1, fid2) in self.same_day_pairs:
                    f1 = self.flights_by_id[fid1]
                    f2 = self.flights_by_id[fid2]
                    gap = f2["de"] - f1["l"]
                    for n in model.N_minus_1:
                        # gap >= min_sit - M*(2 - x_i_fid2_n+1 - x_i_fid1_n)
                        rhs = min_sit - M * (2 - model.x[i, fid2, n + 1] - model.x[i, fid1, n])
                        model.add_component(f"c4_{i}_{fid1}_{fid2}_{n}", pyo.Constraint(expr=gap >= rhs))
                        self.con_stats["4"] = self.con_stats.get("4", 0) + 1
        self.model.c4 = pyo.BuildAction(rule=c4_build)

        # (5) Next‑day rest – BuildAction over pairs
        def c5_build(model):
            for i in model.I:
                for (fid1, fid2) in self.next_day_pairs:
                    f1 = self.flights_by_id[fid1]
                    f2 = self.flights_by_id[fid2]
                    gap = 1440 + f2["de"] - f1["l"]
                    for n in model.N_minus_1:
                        rhs = min_rest - M * (2 - model.x[i, fid2, n + 1] - model.x[i, fid1, n])
                        model.add_component(f"c5_{i}_{fid1}_{fid2}_{n}", pyo.Constraint(expr=gap >= rhs))
                        self.con_stats["5"] = self.con_stats.get("5", 0) + 1
        self.model.c5 = pyo.BuildAction(rule=c5_build)

        # (6) First duty starts at home
        def c6_rule(model, i, d):
            home = self.o_i.get(i)
            if home is None:
                return pyo.Constraint.Skip
            f = 1
            # s[i,home,d] >= 1 - M*(1 - y[i,d,f])  -> implemented as:
            return model.s[i, home, d] >= -M * (1 - model.y[i, d, f]) + 1
        self.model.c6 = pyo.Constraint(self.model.I, self.model.D, rule=c6_rule)
        self.con_stats["6"] = len(self.model.c6)

        # (7) Last duty ends at home
        def c7_rule(model, i, d, f):
            home = self.o_i.get(i)
            if home is None:
                return pyo.Constraint.Skip
            if f > d:
                return pyo.Constraint.Skip
            # later_sum: sum of duties after (d,f) with index > f
            later_sum = sum(model.y[i, dp, fp]
                            for dp in model.D if dp > d
                            for fp in model.F if fp > f)
            return model.w[i, home, d] >= 1 - M * (1 - model.y[i, d, f]) - M * later_sum
        self.model.c7 = pyo.Constraint(self.model.I, self.model.D, self.model.F, rule=c7_rule)
        self.con_stats["7"] = len(self.model.c7)

        # (8)-(9) Consecutive duties integrity – BuildAction
        def c89_build(model):
            for i in model.I:
                for j in model.J:
                    for d in model.D:
                        for dp in model.D:
                            if dp <= d:
                                continue
                            for f in model.F:
                                if f + 1 not in model.F:
                                    continue
                                expr1 = model.s[i, j, dp] >= model.w[i, j, d] - M * (2 - model.y[i, d, f] - model.y[i, dp, f + 1])
                                expr2 = model.s[i, j, dp] <= model.w[i, j, d] + M * (2 - model.y[i, d, f] - model.y[i, dp, f + 1])
                                model.add_component(f"c8_{i}_{j}_{d}_{dp}_{f}", pyo.Constraint(expr=expr1))
                                model.add_component(f"c9_{i}_{j}_{d}_{dp}_{f}", pyo.Constraint(expr=expr2))
                                self.con_stats["8"] = self.con_stats.get("8", 0) + 1
                                self.con_stats["9"] = self.con_stats.get("9", 0) + 1
        self.model.c89 = pyo.BuildAction(rule=c89_build)

        # (10) Flight integrity on a duty day
        def c10_rule(model, i, j, d):
            incoming = sum(model.x[i, fid, n] for fid in flights_on_day.get(d, [])
                           for n in model.N if self.flights_by_id[fid]["dest"] == j)
            outgoing = sum(model.x[i, fid, n] for fid in flights_on_day.get(d, [])
                           for n in model.N if self.flights_by_id[fid]["origin"] == j)
            return model.s[i, j, d] + incoming == outgoing + model.w[i, j, d]
        self.model.c10 = pyo.Constraint(self.model.I, self.model.J, self.model.D, rule=c10_rule)
        self.con_stats["10"] = len(self.model.c10)

        # (11)-(12) No flights on non‑working days – BuildAction
        def c1112_build(model):
            for i in model.I:
                for d in model.D:
                    x_sum = sum(model.x[i, fid, n] for fid in flights_on_day.get(d, []) for n in model.N)
                    y_sum = sum(model.y[i, d, f] for f in model.F)
                    expr1 = x_sum >= -M * (1 - y_sum) + eps
                    expr2 = x_sum <= M * y_sum
                    model.add_component(f"c11_{i}_{d}", pyo.Constraint(expr=expr1))
                    model.add_component(f"c12_{i}_{d}", pyo.Constraint(expr=expr2))
                    self.con_stats["11"] = self.con_stats.get("11", 0) + 1
                    self.con_stats["12"] = self.con_stats.get("12", 0) + 1
        self.model.c1112 = pyo.BuildAction(rule=c1112_build)

        # (13) At most one duty per day
        def c13_rule(model, i, d):
            return sum(model.y[i, d, f] for f in model.F) <= 1
        self.model.c13 = pyo.Constraint(self.model.I, self.model.D, rule=c13_rule)
        self.con_stats["13"] = len(self.model.c13)

        # (14) At most one day per duty index
        def c14_rule(model, i, f):
            return sum(model.y[i, d, f] for d in model.D) <= 1
        self.model.c14 = pyo.Constraint(self.model.I, self.model.F, rule=c14_rule)
        self.con_stats["14"] = len(self.model.c14)

        # (15) Duty index cannot exceed day
        def c15_rule(model, i, d, f):
            if d < f:
                return model.y[i, d, f] == 0
            else:
                return pyo.Constraint.Skip
        self.model.c15 = pyo.Constraint(self.model.I, self.model.D, self.model.F, rule=c15_rule)
        self.con_stats["15"] = len(self.model.c15)

        # (16) Crew cost if any duty assigned
        def c16_rule(model, i):
            return M * model.SS[i] >= sum(model.y[i, d, f] for d in model.D for f in model.F)
        self.model.c16 = pyo.Constraint(self.model.I, rule=c16_rule)
        self.con_stats["16"] = len(self.model.c16)

        # (17)-(18) First‑day connectivity – BuildAction
        def c1718_build(model):
            for i in model.I:
                for fid in model.Flights:
                    f1 = self.flights_by_id[fid]
                    d = f1["day"]
                    j = f1["origin"]
                    for n in model.N_minus_1:
                        arrivals = sum(model.x[i, fid2, n] for fid2 in flights_on_day.get(d, [])
                                       if self.flights_by_id[fid2]["dest"] == j)
                        f = 1
                        expr1 = arrivals >= 1 - M * (2 - model.y[i, d, f] - model.x[i, fid, n + 1])
                        expr2 = arrivals <= 1 + M * (2 - model.y[i, d, f] - model.x[i, fid, n + 1])
                        model.add_component(f"c17_{i}_{fid}_{n}", pyo.Constraint(expr=expr1))
                        model.add_component(f"c18_{i}_{fid}_{n}", pyo.Constraint(expr=expr2))
                        self.con_stats["17"] = self.con_stats.get("17", 0) + 1
                        self.con_stats["18"] = self.con_stats.get("18", 0) + 1
        self.model.c1718 = pyo.BuildAction(rule=c1718_build)

        # (19)-(20) Inter‑day connectivity – BuildAction
        def c1920_build(model):
            for i in model.I:
                for d in model.D:
                    for dp in model.D:
                        if dp <= d:
                            continue
                        for f in model.F:
                            if f + 1 not in model.F:
                                continue
                            for n in model.N_minus_1:
                                for fid in flights_on_day.get(dp, []):
                                    fdp = self.flights_by_id[fid]
                                    j = fdp["origin"]
                                    arrivals_d = sum(model.x[i, fid2, n] for fid2 in flights_on_day.get(d, [])
                                                     if self.flights_by_id[fid2]["dest"] == j)
                                    arrivals_dp = sum(model.x[i, fid2, n] for fid2 in flights_on_day.get(dp, [])
                                                      if self.flights_by_id[fid2]["dest"] == j)
                                    expr1 = arrivals_d + arrivals_dp >= 1 - M * (3 - model.y[i, dp, f + 1] - model.y[i, d, f] - model.x[i, fid, n + 1])
                                    expr2 = arrivals_d + arrivals_dp <= 1 + M * (3 - model.y[i, dp, f + 1] - model.y[i, d, f] - model.x[i, fid, n + 1])
                                    model.add_component(f"c19_{i}_{fid}_{n}_{d}_{dp}_{f}", pyo.Constraint(expr=expr1))
                                    model.add_component(f"c20_{i}_{fid}_{n}_{d}_{dp}_{f}", pyo.Constraint(expr=expr2))
                                    self.con_stats["19"] = self.con_stats.get("19", 0) + 1
                                    self.con_stats["20"] = self.con_stats.get("20", 0) + 1
        self.model.c1920 = pyo.BuildAction(rule=c1920_build)

        # (21) First flight of first duty starts at home (round n=1)
        def c21_rule(model, i, j, d):
            n = 1
            f = 1
            sum_round1_orig = sum(model.x[i, fid, n] for fid in flights_on_day.get(d, [])
                                   if self.flights_by_id[fid]["origin"] == j)
            return sum_round1_orig >= model.s[i, j, d] - M * (1 - model.y[i, d, f])
        self.model.c21 = pyo.Constraint(self.model.I, self.model.J, self.model.D, rule=c21_rule)
        self.con_stats["21"] = len(self.model.c21)

        # (22) Integrity between two consecutive duties – BuildAction
        def c22_build(model):
            for i in model.I:
                for j in model.J:
                    for d in model.D:
                        for d_prime in model.D:
                            if d_prime >= d:
                                continue
                            for n in model.N:
                                for f in model.F:
                                    if f + 1 not in model.F:
                                        continue
                                    lhs = sum(model.x[i, fid, n] for fid in flights_on_day.get(d, [])
                                              if self.flights_by_id[fid]["origin"] == j)
                                    sum_prev = (sum(model.x[i, fid, n - 1] for fid in flights_on_day.get(d_prime, []))
                                                if (n - 1) in model.N else 0)
                                    sum_curr = sum(model.x[i, fid, n] for fid in flights_on_day.get(d_prime, []))
                                    rhs = (model.s[i, j, d] - M * (2 - model.y[i, d_prime, f] - model.y[i, d, f + 1])
                                           - M * (1 - sum_prev) - M * sum_curr)
                                    # lhs >= rhs
                                    model.add_component(f"c22_{i}_{j}_{d}_{d_prime}_{n}_{f}", pyo.Constraint(expr=lhs >= rhs))
                                    self.con_stats["22"] = self.con_stats.get("22", 0) + 1
        self.model.c22 = pyo.BuildAction(rule=c22_build)

        # (23) Working and non‑working days – BuildAction
        def c23_build(model):
            for i in model.I:
                total_starts = sum(model.s[i, j, d] for j in model.J for d in model.D)
                for d_prime in model.D:
                    for d_doubleprime in model.D:
                        for f_prime in model.F:
                            f = 1
                            later_sum = sum(model.y[i, d_triple, f_prime + 1] for d_triple in model.D if f_prime + 1 in model.F)
                            rhs = (d_doubleprime - d_prime + 1) + M * (2 - model.y[i, d_prime, f] - model.y[i, d_doubleprime, f_prime]) + M * later_sum
                            model.add_component(f"c23_{i}_{d_prime}_{d_doubleprime}_{f_prime}", pyo.Constraint(expr=total_starts <= rhs))
                            self.con_stats["23"] = self.con_stats.get("23", 0) + 1
        self.model.c23 = pyo.BuildAction(rule=c23_build)

        # (24) No starts if no duties
        def c24_rule(model, i):
            lhs = sum(model.s[i, j, d] for j in model.J for d in model.D)
            rhs = M * sum(model.y[i, d, f] for d in model.D for f in model.F)
            return lhs <= rhs
        self.model.c24 = pyo.Constraint(self.model.I, rule=c24_rule)
        self.con_stats["24"] = len(self.model.c24)

        # (25)-(26) Relationship between two consecutive duties – BuildAction
        def c2526_build(model):
            for i in model.I:
                for d in model.D:
                    for f in model.F:
                        if f + 1 not in model.F or f + 1 > d:
                            continue
                        lhs = sum(model.y[i, dp, f] for dp in model.D if dp < d)
                        expr1 = lhs <= 1 + M * (1 - model.y[i, d, f + 1])
                        expr2 = 1 - M * (1 - model.y[i, d, f + 1]) <= lhs + M
                        model.add_component(f"c25_{i}_{d}_{f}", pyo.Constraint(expr=expr1))
                        model.add_component(f"c26_{i}_{d}_{f}", pyo.Constraint(expr=expr2))
                        self.con_stats["25"] = self.con_stats.get("25", 0) + 1
                        self.con_stats["26"] = self.con_stats.get("26", 0) + 1
        self.model.c2526 = pyo.BuildAction(rule=c2526_build)

        # (27)-(30) Flight coverage and deadhead linking
        for fid in self.fid_list:
            x_sum_all = sum(self.model.x[i, fid, n] for i in self.model.I for n in self.model.N)
            if self.force_cover_all_flights:
                self.model.add_component(f"c27_{fid}", pyo.Constraint(expr=x_sum_all >= 1))
            else:
                self.model.add_component(f"c27_{fid}", pyo.Constraint(expr=x_sum_all >= self.model.v[fid]))
            self.model.add_component(f"c28_{fid}", pyo.Constraint(expr=x_sum_all <= M * self.model.v[fid]))
            self.model.add_component(f"c29_{fid}", pyo.Constraint(expr=self.model.b[fid] <= M * self.model.v[fid]))
            self.model.add_component(f"c30_{fid}", pyo.Constraint(expr=self.model.b[fid] >= x_sum_all - M * (1 - self.model.v[fid])))
            self.con_stats["27"] = self.con_stats.get("27", 0) + 1
            self.con_stats["28"] = self.con_stats.get("28", 0) + 1
            self.con_stats["29"] = self.con_stats.get("29", 0) + 1
            self.con_stats["30"] = self.con_stats.get("30", 0) + 1

        # (31) Maximum flying time per duty
        def c31_rule(model, i, d):
            fly_time = sum((self.flights_by_id[fid]["l"] - self.flights_by_id[fid]["de"]) * model.x[i, fid, n]
                           for fid in flights_on_day.get(d, []) for n in model.N)
            return fly_time <= upper1
        self.model.c31 = pyo.Constraint(self.model.I, self.model.D, rule=c31_rule)
        self.con_stats["31"] = len(self.model.c31)

        # (32) Maximum flying time over horizon
        def c32_rule(model, i):
            fly_time = sum((self.flights_by_id[fid]["l"] - self.flights_by_id[fid]["de"]) * model.x[i, fid, n]
                           for fid in model.Flights for n in model.N)
            return fly_time <= upper2
        self.model.c32 = pyo.Constraint(self.model.I, rule=c32_rule)
        self.con_stats["32"] = len(self.model.c32)

        # (33)-(34) AND linearization for Z – BuildAction over pairs
        def c33_build(model):
            for i in model.I:
                for (fid1, fid2) in self.same_day_pairs + self.next_day_pairs:
                    for n in model.N_minus_1:
                        x1 = model.x[i, fid1, n]
                        x2 = model.x[i, fid2, n + 1]
                        z = model.Z[i, (fid1, fid2), n]
                        model.add_component(f"c33_le1_{i}_{fid1}_{fid2}_{n}", pyo.Constraint(expr=z <= x1))
                        model.add_component(f"c33_le2_{i}_{fid1}_{fid2}_{n}", pyo.Constraint(expr=z <= x2))
                        model.add_component(f"c33_ge_{i}_{fid1}_{fid2}_{n}", pyo.Constraint(expr=z >= x1 + x2 - 1))
                        self.con_stats["33_le1"] = self.con_stats.get("33_le1", 0) + 1
                        self.con_stats["33_le2"] = self.con_stats.get("33_le2", 0) + 1
                        self.con_stats["33_ge"] = self.con_stats.get("33_ge", 0) + 1
        self.model.c33 = pyo.BuildAction(rule=c33_build)

        # (35) Maximum elapsed time per duty – BuildAction
        def c35_build(model):
            for i in model.I:
                for d in model.D:
                    fly_terms = sum((self.flights_by_id[fid]["l"] - self.flights_by_id[fid]["de"]) * model.x[i, fid, n]
                                    for fid in flights_on_day.get(d, []) for n in model.N)
                    sit_terms = 0
                    for (fid1, fid2) in self.same_day_pairs + self.next_day_pairs:
                        f1 = self.flights_by_id[fid1]
                        f2 = self.flights_by_id[fid2]
                        if f1["day"] != d:
                            continue
                        if f2["day"] == f1["day"]:
                            sit_gap = f2["de"] - f1["l"]
                        elif f2["day"] == f1["day"] + 1:
                            sit_gap = 1440 + f2["de"] - f1["l"]
                        else:
                            continue
                        sit_terms += sum(sit_gap * model.Z[i, (fid1, fid2), n] for n in model.N_minus_1)
                    model.add_component(f"c35_{i}_{d}", pyo.Constraint(expr=fly_terms + sit_terms <= upper3))
                    self.con_stats["35"] = self.con_stats.get("35", 0) + 1
        self.model.c35 = pyo.BuildAction(rule=c35_build)

    def _set_objective(self):
        """Objective (1)."""
        uncover = sum(self.uncover_costs[fid] * (1 - self.model.v[fid]) for fid in self.model.Flights)
        deadhead = sum(self.flight_costs[fid] * (self.model.b[fid] - self.model.v[fid]) for fid in self.model.Flights)
        away = sum(self.h * self.model.w[i, j, d] for i in self.model.I for j in self.model.J for d in self.model.D
                   if j != self.o_i.get(i))
        crew_cost = sum(self.r_i[i] * self.model.SS[i] for i in self.model.I)
        self.model.obj = pyo.Objective(expr=uncover + deadhead + away + crew_cost, sense=pyo.minimize)

    # ------------------------------------------------------------------
    # Solve and extract solution
    # ------------------------------------------------------------------
    def solve(self, time_limit=300.0, solver_name='gurobi'):
        """
        Solve the model using the specified solver.
        Returns a solution dictionary identical to the original Gurobi version.
        """
        solver = SolverFactory(solver_name)
        if solver is None:
            raise ValueError(f"Solver {solver_name} not available.")

        # Set solver options
        if solver_name in ('gurobi', 'gurobi_direct'):
            solver.options['TimeLimit'] = time_limit
            solver.options['MIPGap'] = float(self.params.get("mip_gap", 0.001))
        elif solver_name == 'mosek':
            solver.options['dparam.optimizer_max_time'] = time_limit
            solver.options['dparam.mio_tol_rel_gap'] = float(self.params.get("mip_gap", 0.001))
        elif solver_name == 'highs':
            solver.options['time_limit'] = time_limit
            solver.options['mip_gap'] = float(self.params.get("mip_gap", 0.001))
        else:
            solver.options['timelimit'] = time_limit
            solver.options['mipgap'] = float(self.params.get("mip_gap", 0.001))

        start = time.time()
        try:
            print(f"\nSolving model with {solver_name} (time limit: {time_limit}s)...")
            results = solver.solve(self.model, tee=True)
            solve_time = time.time() - start
            return self._extract_solution(results, solve_time)
        except Exception as e:
            print(f"✗ Solver error: {e}")
            return {"feasibility": "Error", "error_message": str(e), "solve_time": time.time() - start}

    def _var_index_tuple(self, var):
        """Helper: return index tuple for a VarData, robustly."""
        # Try var.index() first (works for Pyomo VarData)
        try:
            idx = var.index()
            if isinstance(idx, Iterable):
                return tuple(idx)
            else:
                return (idx,)
        except Exception:
            pass
        # Fallback: parse var.name like "x[0,KIH,1]" or "s[0,'KIH',1]"
        name = var.name
        if "[" in name and "]" in name:
            inside = name.split("[", 1)[1].rsplit("]", 1)[0]
            inside_clean = inside.replace("'", "").replace('"', "")
            parts = [p.strip() for p in inside_clean.split(",")]
            def _try_num(s):
                try:
                    if "." in s:
                        return float(s)
                    else:
                        return int(s)
                except Exception:
                    return s
            return tuple(_try_num(p) for p in parts)
        return ()

    def _extract_solution(self, results, solve_time):
        """Extract solution into a dictionary and run sanity checks."""
        sol = {
            "feasibility": "Unknown",
            "objective_value": None,
            "objective_components": {},
            "crew_assignments": defaultdict(list),
            "uncovered_flights": [],
            "deadhead_flights": [],
            "crew_used": 0,
            "hotel_stays": 0,
            "solve_time": solve_time,
            "instance_info": {
                "num_flights": len(self.fid_list),
                "num_crews": len(self.I),
                "num_days": len(self.D),
                "num_cities": len(self.J),
                "same_day_pairs": len(self.same_day_pairs),
                "next_day_pairs": len(self.next_day_pairs),
            },
            "var_counts": dict(self.var_stats),
            "constraint_counts": dict(self.con_stats),
            "raw_vars": {"x": {}, "v": {}, "b": {}, "s": {}, "w": {}},
        }

        # Check solver status / termination condition
        status = results.solver.status
        term_cond = results.solver.termination_condition
        if status == SolverStatus.ok and term_cond == TerminationCondition.optimal:
            sol["feasibility"] = "Optimal"
        elif status == SolverStatus.ok and term_cond == TerminationCondition.feasible:
            sol["feasibility"] = "Feasible"
        elif term_cond == TerminationCondition.infeasible:
            sol["feasibility"] = "Infeasible"
        elif term_cond == TerminationCondition.unbounded:
            sol["feasibility"] = "Unbounded"
        elif term_cond == TerminationCondition.maxTimeLimit:
            sol["feasibility"] = "TimeLimit"
        else:
            sol["feasibility"] = f"Status {status}, {term_cond}"

        # If solution exists (not infeasible/unbounded), extract variable values
        if sol["feasibility"] not in ("Infeasible", "Unbounded", "Error", "Unknown"):
            # objective
            try:
                sol["objective_value"] = float(pyo.value(self.model.obj))
            except Exception:
                sol["objective_value"] = None

            # Raw variable values (for diagnostics)
            for var in self.model.component_data_objects(pyo.Var, descend_into=True):
                val = var.value
                if val is None:
                    continue
                name = var.name
                idx = self._var_index_tuple(var)

                # x: (i, fid, n)
                if name.startswith('x[') and len(idx) >= 3:
                    i, fid, n = idx[:3]
                    key = f"{i},{fid},{n}"
                    sol["raw_vars"]["x"][key] = float(val)
                # v: (fid,)
                elif name.startswith('v[') and len(idx) >= 1:
                    fid = idx[0]
                    key = str(fid)
                    sol["raw_vars"]["v"][key] = float(val)
                # b: (fid,)
                elif name.startswith('b[') and len(idx) >= 1:
                    fid = idx[0]
                    key = str(fid)
                    sol["raw_vars"]["b"][key] = float(val)
                # s: (i, city, d)
                elif name.startswith('s[') and len(idx) >= 3:
                    i, city, d = idx[:3]
                    key = f"{i},{city},{d}"
                    sol["raw_vars"]["s"][key] = float(val)
                # w: (i, city, d)
                elif name.startswith('w[') and len(idx) >= 3:
                    i, city, d = idx[:3]
                    key = f"{i},{city},{d}"
                    sol["raw_vars"]["w"][key] = float(val)
                # ignore Z, y, SS in raw_vars (not useful in CSV diagnostics)

            # Crew assignments from x
            for i in self.model.I:
                for fid in self.model.Flights:
                    for n in self.model.N:
                        try:
                            xval = self.model.x[i, fid, n].value
                            if xval is not None and float(xval) > 0.5:
                                f = self.flights_by_id[fid]
                                sol["crew_assignments"][i].append({
                                    "flight_id": fid,
                                    "day": f["day"],
                                    "round": n,
                                    "origin": f["origin"],
                                    "destination": f["dest"],
                                    "departure_time": f["de_str"],
                                    "arrival_time": f["arr_str"],
                                    "duration": f["duration"],
                                    "x_val": 1.0
                                })
                        except (ValueError, AttributeError, KeyError):
                            continue

            # Sort assignments for each crew
            for i, assigns in sol["crew_assignments"].items():
                assigns.sort(key=lambda a: (a["day"], a["round"], str(a.get("departure_time", ""))))

            sol["crew_used"] = sum(1 for i in self.model.I if any(sol["crew_assignments"].get(i, [])))

            # Uncovered flights and deadhead
            for fid in self.model.Flights:
                vkey = str(fid)
                vval = sol["raw_vars"]["v"].get(vkey, 0.0)
                if vval < 0.5:
                    sol["uncovered_flights"].append(fid)
                bval = sol["raw_vars"]["b"].get(vkey, 0.0)
                if bval > vval + 1e-6:
                    sol["deadhead_flights"].append((fid, bval - vval))

            # Hotel stays count from raw w variables (and crew home base)
            hotel = 0
            for i in self.model.I:
                home = self.o_i.get(i)
                for j in self.model.J:
                    if j == home:
                        continue
                    for d in self.model.D:
                        wkey = f"{i},{j},{d}"
                        wval = sol["raw_vars"]["w"].get(wkey, 0.0)
                        if wval is not None and float(wval) > 0.5:
                            hotel += 1
            sol["hotel_stays"] = hotel

            # Objective components (computed from variable values)
            sol["objective_components"] = self._objective_breakdown()
        # Run sanity checks and attach results (attach regardless; if infeasible, sanity will note)
        try:
            sol = self._run_sanity_checks(sol)
        except Exception as e:
            sol.setdefault("sanity_checks", {})["error"] = str(e)

        return sol

    def _objective_breakdown(self):
        """Compute the four cost components from variable values."""
        uncover_sum = 0.0
        deadhead_sum = 0.0
        away_sum = 0.0
        crew_sum = 0.0
        for fid in self.model.Flights:
            vval = self.model.v[fid].value
            if vval is None:
                vval = 0.0
            uncover_sum += self.uncover_costs[fid] * (1 - float(vval))
            bval = self.model.b[fid].value
            if bval is None:
                bval = 0.0
            deadhead_sum += self.flight_costs[fid] * (float(bval) - float(vval))
        for i in self.model.I:
            for j in self.model.J:
                if j == self.o_i.get(i):
                    continue
                for d in self.model.D:
                    wval = self.model.w[i, j, d].value
                    if wval is None:
                        wval = 0.0
                    away_sum += self.h * float(wval)
            SSval = self.model.SS[i].value
            if SSval is None:
                SSval = 0.0
            crew_sum += self.r_i[i] * float(SSval)
        total = uncover_sum + deadhead_sum + away_sum + crew_sum
        return {
            "uncover_cost": uncover_sum,
            "deadhead_cost": deadhead_sum,
            "away_cost": away_sum,
            "crew_cost": crew_sum,
            "total_objective": total
        }

    # ------------------------------------------------------------------
    # Sanity checks & diagnostics
    # ------------------------------------------------------------------
    def _run_sanity_checks(self, sol, tol=1e-6, max_report=100):
        """
        Sanity checks run after extracting the solver values.
        Populates sol['sanity_checks'] and writes diagnostics/diagnostic_sanity.csv.
        Checks:
          - coverage consistency: v[fid] vs sum_i,n x[i,fid,n]
          - b >= x_sum - v
          - flow balance residuals for eq (10): s + incoming - outgoing - w == 0
          - inferred start/end vs model s/w raw values (per crew/day/city)
        """
        sanity = {
            "coverage_mismatches": [],
            "deadhead_mismatches": [],
            "flow_violations": [],
            "start_end_mismatches": [],
            "summary": {}
        }

        model = self.model
        flights_on_day = self.flights_by_day

        cov_mismatch_count = 0
        dead_mismatch_count = 0

        # Coverage and deadhead checks
        for fid in self.model.Flights:
            try:
                vval = float(model.v[fid].value or 0.0)
            except Exception:
                vval = 0.0
            x_sum = 0
            for i in model.I:
                for n in model.N:
                    xv = model.x[i, fid, n].value
                    if xv is not None and float(xv) > 0.5:
                        x_sum += 1
            # mismatch when x_sum >=1 but v==0 or x_sum==0 but v==1
            if (x_sum >= 1 and vval < 0.5) or (x_sum == 0 and vval > 0.5):
                if cov_mismatch_count < max_report:
                    sanity["coverage_mismatches"].append({"flight": fid, "x_sum": x_sum, "v_val": vval})
                cov_mismatch_count += 1
            # b check: b >= x_sum - v  (if violated -> deadhead mismatch)
            bval = float(model.b[fid].value or 0.0)
            if bval + tol < (x_sum - (1.0 if vval > 0.5 else 0.0)):
                if dead_mismatch_count < max_report:
                    sanity["deadhead_mismatches"].append({"flight": fid, "b_val": bval, "x_sum": x_sum, "v_val": vval})
                dead_mismatch_count += 1

        # Flow balance checks (eq 10)
        flow_violations = []
        for i in model.I:
            for j in model.J:
                for d in model.D:
                    incoming = 0.0
                    outgoing = 0.0
                    for fid in flights_on_day.get(d, []):
                        f = self.flights_by_id[fid]
                        for n in model.N:
                            xv = model.x[i, fid, n].value
                            if xv is None:
                                continue
                            if f["dest"] == j:
                                incoming += float(xv)
                            if f["origin"] == j:
                                outgoing += float(xv)
                    sval = float(model.s[i, j, d].value or 0.0)
                    wval = float(model.w[i, j, d].value or 0.0)
                    residual = sval + incoming - outgoing - wval
                    if abs(residual) > tol:
                        flow_violations.append({
                            "crew": i, "city": j, "day": d, "residual": float(residual),
                            "s": float(sval), "incoming": float(incoming), "outgoing": float(outgoing),
                            "w": float(wval)
                        })
                        if len(flow_violations) >= max_report:
                            break
                if len(flow_violations) >= max_report:
                    break
            if len(flow_violations) >= max_report:
                break

        # Inferred start/end mismatch (based on crew_assignments)
        mismatches = []
        crew_assignments = sol.get("crew_assignments", {})
        raw_s = sol.get("raw_vars", {}).get("s", {})
        raw_w = sol.get("raw_vars", {}).get("w", {})
        for i, assigns in crew_assignments.items():
            per_day = {}
            for a in assigns:
                d = a.get("day")
                per_day.setdefault(d, []).append(a)
            for d, fls in per_day.items():
                fls_sorted = sorted(fls, key=lambda x: (x.get("round", 0), x.get("departure_time", "")))
                first = fls_sorted[0]
                last = fls_sorted[-1]
                inferred_s_city = first.get("origin")
                inferred_w_city = last.get("destination")
                sval_key = f"{i},{inferred_s_city},{d}"
                wval_key = f"{i},{inferred_w_city},{d}"
                raw_s_val = raw_s.get(sval_key, 0)
                raw_w_val = raw_w.get(wval_key, 0)
                if int(raw_s_val) != 1:
                    mismatches.append({"crew": i, "day": d, "city": inferred_s_city, "type": "start", "raw_s": raw_s_val})
                if int(raw_w_val) != 1:
                    mismatches.append({"crew": i, "day": d, "city": inferred_w_city, "type": "end", "raw_w": raw_w_val})
                if len(mismatches) >= max_report:
                    break
            if len(mismatches) >= max_report:
                break

        sanity["summary"] = {
            "coverage_mismatch_count": cov_mismatch_count,
            "deadhead_mismatch_count": dead_mismatch_count,
            "flow_violation_count": len(flow_violations),
            "start_end_mismatch_count": len(mismatches)
        }
        sanity["flow_violations"] = flow_violations
        sanity["start_end_mismatches"] = mismatches

        sol["sanity_checks"] = sanity

        # write CSV diagnostic (sanity)
        diagnostics_dir = os.path.join(os.getcwd(), "diagnostics")
        os.makedirs(diagnostics_dir, exist_ok=True)
        instance_name = self.instance.get('_instance_filename', 'instance')
        sanity_csv = os.path.join(diagnostics_dir, f"{os.path.splitext(os.path.basename(instance_name))[0]}_diagnostic_sanity.csv")
        try:
            with open(sanity_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["check", "detail", "count"])
                w.writerow(["coverage_mismatches", "", cov_mismatch_count])
                w.writerow(["deadhead_mismatches", "", dead_mismatch_count])
                w.writerow(["flow_violations", "", len(flow_violations)])
                w.writerow(["start_end_mismatches", "", len(mismatches)])
                if flow_violations:
                    w.writerow([])
                    w.writerow(["flow_crew", "city", "day", "residual", "s", "incoming", "outgoing", "w"])
                    for fv in flow_violations[:max_report]:
                        w.writerow([fv["crew"], fv["city"], fv["day"], fv["residual"], fv["s"], fv["incoming"], fv["outgoing"], fv["w"]])
            print(f"✓ Sanity CSV written to {sanity_csv}")
        except Exception:
            pass

        # If any serious problems exist, annotate feasibility
        if cov_mismatch_count > 0 or dead_mismatch_count > 0 or len(flow_violations) > 0 or len(mismatches) > 0:
            prev = sol.get("feasibility", "")
            sol["feasibility"] = f"{prev};Inconsistent(sanity_checks_failed)" if prev else "Inconsistent(sanity_checks_failed)"

        return sol

    # ------------------------------------------------------------------
    # Diagnostics writer (now consistent with processed flights and model state)
    # ------------------------------------------------------------------
    def _write_diagnostics(self, solution, diagnostics_dir, instance_data):
        """Write the five diagnostic CSV files using model-processed information."""
        base = os.path.splitext(os.path.basename(instance_data.get('_instance_filename', 'instance')))[0]
        os.makedirs(diagnostics_dir, exist_ok=True)

        # 1) diagnostic_crew.csv
        crew_csv = os.path.join(diagnostics_dir, f"{base}_diagnostic_crew.csv")
        with open(crew_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["crew_id", "home_base", "assignments"])
            # ensure iterate exactly over model crew count
            for i in range(self.num_crew):
                home = self.o_i.get(i, instance_data.get("cities", [""])[0])
                assigns = solution.get("crew_assignments", {}).get(i, [])
                entries = []
                for a in assigns:
                    fid = a.get('flight_id', '')
                    day = a.get('day', '')
                    rnd = a.get('round', '')
                    entries.append(f"{fid}@d{day}@r{rnd}")
                writer.writerow([i, home, ";".join(entries)])
        print(f"✓ Diagnostic crew CSV written to {crew_csv}")

        # 2) diagnostic_flights.csv
        flights_csv = os.path.join(diagnostics_dir, f"{base}_diagnostic_flights.csv")
        with open(flights_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["flight_id", "origin", "destination", "day", "depart", "arrive", "duration", "covered_v", "b"])
            raw = solution.get("raw_vars", {})
            vmap = raw.get("v", {})
            bmap = raw.get("b", {})
            # Use processed flights_by_id values (shifted days)
            for fid in self.fid_list:
                f = self.flights_by_id[fid]
                fid_str = str(fid)
                writer.writerow([
                    fid,
                    f.get("origin", ""),
                    f.get("dest", ""),
                    f.get("day", ""),
                    f.get("de_str", ""),
                    f.get("arr_str", ""),
                    f.get("duration", ""),
                    vmap.get(fid_str, 0),
                    bmap.get(fid_str, 0)
                ])
        print(f"✓ Diagnostic flights CSV written to {flights_csv}")

        # 3) diagnostic_variables.csv
        vars_csv = os.path.join(diagnostics_dir, f"{base}_diagnostic_variables.csv")
        with open(vars_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["var", "value"])
            raw = solution.get("raw_vars", {})
            # x keys "i,fid,n" -> prefix 'x_'
            for k, v in sorted(raw.get("x", {}).items(), key=lambda kv: kv[0]):
                writer.writerow([f"x_{k}", v])
            for k, v in sorted(raw.get("v", {}).items(), key=lambda kv: (int(k) if str(k).isdigit() else k)):
                writer.writerow([f"v_{k}", v])
            for k, v in sorted(raw.get("b", {}).items(), key=lambda kv: (int(k) if str(k).isdigit() else k)):
                writer.writerow([f"b_{k}", v])
            for k, v in sorted(raw.get("s", {}).items(), key=lambda kv: kv[0]):
                writer.writerow([f"s_{k}", v])
            for k, v in sorted(raw.get("w", {}).items(), key=lambda kv: kv[0]):
                writer.writerow([f"w_{k}", v])
        print(f"✓ Diagnostic variables CSV written to {vars_csv}")

        # 4) diagnostic_inference_mismatch.csv
        mismatch_csv = os.path.join(diagnostics_dir, f"{base}_diagnostic_inference_mismatch.csv")
        with open(mismatch_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["crew", "city", "day", "inferred_s", "raw_s", "inferred_w", "raw_w"])
            cities = self.J
            raw = solution.get("raw_vars", {})
            raw_s = raw.get("s", {})
            raw_w = raw.get("w", {})
            crew_assignments = solution.get("crew_assignments", {})
            # Use model num_crew and num_days
            for i in range(self.num_crew):
                assigns = crew_assignments.get(i, [])
                per_day = defaultdict(list)
                for a in assigns:
                    per_day[a.get("day", 0)].append(a)
                for j in cities:
                    for d in range(1, int(self.num_days) + 1):
                        inferred_s = 0
                        inferred_w = 0
                        if per_day.get(d):
                            fls = sorted(per_day[d], key=lambda x: (x.get("round", 0), x.get("departure_time", "")))
                            first = fls[0]
                            last = fls[-1]
                            if first.get("origin") == j:
                                inferred_s = 1
                            if last.get("destination") == j:
                                inferred_w = 1
                        rs = raw_s.get(f"{i},{j},{d}", 0)
                        rw = raw_w.get(f"{i},{j},{d}", 0)
                        writer.writerow([i, j, d, f"{inferred_s:.3f}", f"{int(rs):.3f}", f"{inferred_w:.3f}", f"{int(rw):.3f}"])
        print(f"✓ Diagnostic inference mismatch CSV written to {mismatch_csv}")

        # 5) diagnostic_violations.csv (flow violations)
        viol_csv = os.path.join(diagnostics_dir, f"{base}_diagnostic_violations.csv")
        with open(viol_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["constraint", "message", "residual"])
            violations = solution.get("sanity_checks", {}).get("flow_violations", [])
            for v in violations:
                writer.writerow(["c10", f"crew {v.get('crew')} city {v.get('city')} day {v.get('day')}", v.get('residual')])
        print(f"✓ Diagnostic violations CSV written to {viol_csv}")

    # ------------------------------------------------------------------
    # Convenience method to solve and write diagnostics (as in original)
    # ------------------------------------------------------------------
    def solve_and_save(self, time_limit=300.0, solver_name='gurobi', output_dir='.'):
        """Solve and write solution JSON and diagnostics."""
        solution = self.solve(time_limit=time_limit, solver_name=solver_name)
        if solution:
            diagnostics_dir = os.path.join(output_dir, "diagnostics")
            self._write_diagnostics(solution, diagnostics_dir, self.instance)
            # Write solution JSON
            solutions_dir = os.path.join(output_dir, "solutions")
            os.makedirs(solutions_dir, exist_ok=True)
            instance_fname = self.instance.get('_instance_filename', 'instance')
            base = os.path.splitext(os.path.basename(instance_fname))[0]
            sol_file = os.path.join(solutions_dir, f"{base}_solution_{solver_name}.json")
            try:
                with open(sol_file, 'w') as f:
                    json.dump({
                        'solver': solver_name,
                        'instance': instance_fname,
                        'solution': solution,
                        'solve_time_sec': solution['solve_time']
                    }, f, indent=2)
                print(f"✓ Solution saved to {sol_file}")
            except Exception:
                pass
        return solution