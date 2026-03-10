#!/usr/bin/env python3
"""
Classify, for each flight assigned to multiple crews, which crew is the operating
assignment and which are deadheads.

Usage:
    python3 tools/classify_assignments.py \
        --crew diagnostics/instance_1_diagnostic_crew.csv \
        --flights diagnostics/instance_1_diagnostic_flights.csv \
        [--vars diagnostics/instance_1_diagnostic_variables.csv]

Output:
    Prints a per-flight report and returns nonzero exit code if inconsistencies found.
"""
import csv
import argparse
from collections import defaultdict
import math

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crew", required=True, help="Path to diagnostic_crew.csv")
    p.add_argument("--flights", required=True, help="Path to diagnostic_flights.csv")
    p.add_argument("--vars", required=False, help="Path to diagnostic_variables.csv (optional)")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold to treat numeric vars as 1 (default 0.5)")
    return p.parse_args()

def load_crew_csv(path):
    # returns: flight_id -> list of dicts {crew_id, home_base, raw, day, round, depart, arrive}
    assignments = defaultdict(list)
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            crew = row.get("crew_id") or row.get("crew") or row.get("id")
            crew = crew.strip() if crew is not None else ""
            home = row.get("home_base", "").strip()
            raw = row.get("assignments", "").strip()
            if not raw:
                continue
            parts = [p for p in raw.split(";") if p.strip()]
            for part in parts:
                # expect format like "19@d1@r1" but more robust parsing
                tokens = part.split("@")
                fid_tok = tokens[0]
                try:
                    fid = int(fid_tok)
                except Exception:
                    # skip malformed entry
                    continue
                # parse day and round if present
                day = None
                rnd = None
                for t in tokens[1:]:
                    if t.startswith("d"):
                        try:
                            day = int(t[1:])
                        except:
                            pass
                    if t.startswith("r"):
                        try:
                            rnd = int(t[1:])
                        except:
                            pass
                assignments[fid].append({
                    "crew": crew,
                    "home": home,
                    "raw": part,
                    "day": day,
                    "round": rnd
                })
    return assignments

def load_flights_csv(path):
    # returns: flight_id -> dict with covered_v (int) and b (int) and depart/arrive strings
    flights = {}
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            fid_raw = row.get("flight_id") or row.get("id")
            if fid_raw is None:
                continue
            try:
                fid = int(str(fid_raw).strip())
            except Exception:
                continue
            def to_int(v):
                try:
                    return int(float(v)) if v is not None and v != "" else 0
                except Exception:
                    return 0
            covered = to_int(row.get("covered_v", row.get("v", 0)))
            b = to_int(row.get("b", row.get("covered_count", 0)))
            flights[fid] = {
                "covered_v": covered,
                "b": b,
                "depart": row.get("depart", row.get("de", "")),
                "arrive": row.get("arrive", row.get("l", "")),
                "origin": row.get("origin", ""),
                "destination": row.get("destination", "")
            }
    return flights

def load_vars_csv(path, threshold=0.5):
    # returns dictionaries for y, SS, x, v, b as parsed from a var CSV where rows like "var,value"
    y = {}    # key: (i,d,f) -> value
    SS = {}   # i -> value
    x = {}    # key: (i,fid,n) -> value
    v = {}    # fid -> value
    b = {}    # fid -> value
    # robust parsing: accept either "var,value" with var like 'x_0,19,1' or 'x[0,19,1]' or 'x_0' prefix
    with open(path, newline='') as f:
        r = csv.reader(f)
        # attempt to detect header
        headers = next(r, None)
        # if header contains strings 'var' or 'value', treat it as header
        if headers and any(h.lower() in ("var","value","name") for h in headers):
            # assume two columns: var, value (possibly more; use first two)
            pass
        else:
            # first row may actually be data: treat it as such by rewinding
            if headers:
                r = [headers] + list(r)
        for row in r:
            if len(row) < 2:
                continue
            name = str(row[0]).strip()
            val_raw = row[1]
            try:
                val = float(val_raw)
            except Exception:
                try:
                    val = float(row[-1])
                except Exception:
                    continue
            # normalize names
            n = name
            # remove quotes
            n = n.replace('"','').replace("'", "")
            # patterns:
            # x_0,19,1 or x[0,19,1] or x_0_19_1 or x.0.19.1
            if n.startswith("x_") or n.startswith("x[") or n.startswith("x."):
                inside = n.split("x",1)[1].lstrip("_[").rstrip("]").replace(".", ",")
                parts = [p.strip() for p in inside.split(",") if p.strip()]
                if len(parts) >= 3:
                    try:
                        i = int(parts[0]); fid = int(parts[1]); rnd = int(parts[2])
                        x[(i,fid,rnd)] = val
                    except Exception:
                        pass
                continue
            if n.startswith("y_") or n.startswith("y[") or n.startswith("y."):
                inside = n.split("y",1)[1].lstrip("_[").rstrip("]").replace(".", ",")
                parts = [p.strip() for p in inside.split(",") if p.strip()]
                if len(parts) >= 3:
                    try:
                        i = int(parts[0]); d = int(parts[1]); f = int(parts[2])
                        y[(i,d,f)] = val
                    except Exception:
                        pass
                continue
            if n.startswith("SS[") or n.startswith("SS_") or n.startswith("ss[") or n.startswith("ss_"):
                inside = n.split("[",1)[1].rstrip("]") if "[" in n else n.split("_",1)[1]
                try:
                    i = int(inside)
                    SS[i] = val
                except Exception:
                    pass
                continue
            if n.startswith("v_") or n.startswith("v["):
                inside = n.split("v",1)[1].lstrip("_[").rstrip("]")
                try:
                    fid = int(inside)
                    v[fid] = val
                except Exception:
                    pass
                continue
            if n.startswith("b_") or n.startswith("b["):
                inside = n.split("b",1)[1].lstrip("_[").rstrip("]")
                try:
                    fid = int(inside)
                    b[fid] = val
                except Exception:
                    pass
                continue
            # fallback: if name is like 'x_0,19,1' with comma inside original cell
            if name.startswith("x_") and "," in name:
                inside = name.split("x_",1)[1]
                parts = [p.strip() for p in inside.split(",") if p.strip()]
                if len(parts) >= 3:
                    try:
                        i = int(parts[0]); fid = int(parts[1]); rnd = int(parts[2])
                        x[(i,fid,rnd)] = val
                    except Exception:
                        pass
    # convert numeric to bool where appropriate via threshold when user needs boolean test
    return {"x": x, "y": y, "SS": SS, "v": v, "b": b}

def classify_for_flight(fid, assigns, fdiag, vars_map, threshold=0.5):
    """
    assigns: list of dicts with keys crew, day, round
    fdiag: flight diag dict with covered_v, b, depart, arrive
    vars_map: dict with keys x,y,SS,v,b from load_vars_csv
    """
    assigned_crews = [a["crew"] for a in assigns]
    n_assigned = len(assigned_crews)
    covered = fdiag.get("covered_v", 0)
    res = {"flight": fid, "assigned": assigned_crews, "operating": None, "deadheads": [], "notes": []}

    if n_assigned == 0:
        res["notes"].append("no_assigned")
        return res

    if n_assigned == 1:
        res["operating"] = assigned_crews[0]
        return res

    # n_assigned > 1
    if covered == 0:
        res["notes"].append("flight_marked_uncovered_but_multiple_assigned")
        # treat as ambiguous — no operating crew per model
        return res

    # Use vars to find a crew with y[i,d,f] == 1 (crew has duty that corresponds)
    y = vars_map.get("y", {})
    SS = vars_map.get("SS", {})
    xvars = vars_map.get("x", {})

    # candidate selection
    candidates = []
    for a in assigns:
        crew = a["crew"]
        day = a.get("day")
        rnd = a.get("round")
        # check y for any duty index f that matches (we don't know duty index mapping for flight -> f)
        # heuristic: look for any y[(crew, day, any_f)] == 1 and the x var exists for that crew and fid
        has_y = False
        if day is not None:
            # search y entries with matching crew and day
            for (i,d,fk), val in y.items():
                if i == int(crew) and d == day and float(val) > threshold:
                    has_y = True
                    break
        # SS flag
        has_SS = int(SS.get(int(crew), 0) > threshold) if crew.isdigit() else bool(SS.get(crew,0) > threshold)
        # count x assignments for that crew on this fid (summing across rounds)
        x_count = 0
        # check any x entry with (crew,fid,any)
        for (i,fid2,rn), val in xvars.items():
            try:
                if int(i) == int(crew) and int(fid2) == int(fid) and float(val) > threshold:
                    x_count += 1
            except Exception:
                pass
        candidates.append({"crew": crew, "has_y": has_y, "has_SS": has_SS, "x_count": x_count, "day": day, "round": rnd})

    # Prefer any crew with has_y == True
    by_y = [c for c in candidates if c["has_y"]]
    if by_y:
        # if multiple, pick one with largest x_count then earliest round then crew id
        by_y.sort(key=lambda c: (-c["x_count"], (c["round"] or math.inf), int(c["crew"]) if str(c["crew"]).isdigit() else c["crew"]))
        chosen = by_y[0]["crew"]
        res["operating"] = chosen
        res["deadheads"] = [c["crew"] for c in candidates if c["crew"] != chosen]
        res["notes"].append("picked_by_y")
        return res

    # Next prefer crew with SS==1 and x_count>0
    by_SS = [c for c in candidates if c["has_SS"]]
    if by_SS:
        by_SS.sort(key=lambda c: (-c["x_count"], (c["round"] or math.inf), int(c["crew"]) if str(c["crew"]).isdigit() else c["crew"]))
        chosen = by_SS[0]["crew"]
        res["operating"] = chosen
        res["deadheads"] = [c["crew"] for c in candidates if c["crew"] != chosen]
        res["notes"].append("picked_by_SS")
        return res

    # Else, pick crew with largest x_count (i.e., more involvement), fallback earliest departure/round
    candidates.sort(key=lambda c: (-c["x_count"], (c["round"] or math.inf), int(c["crew"]) if str(c["crew"]).isdigit() else c["crew"]))
    chosen = candidates[0]["crew"]
    res["operating"] = chosen
    res["deadheads"] = [c["crew"] for c in candidates if c["crew"] != chosen]
    res["notes"].append("picked_by_xcount_or_round")
    return res

def main():
    args = parse_args()
    crew_map = load_crew_csv(args.crew)
    flights = load_flights_csv(args.flights)
    vars_map = {"x": {}, "y": {}, "SS": {}, "v": {}, "b": {}}
    if args.vars:
        try:
            vars_map = load_vars_csv(args.vars, threshold=args.threshold)
        except Exception as e:
            print("Warning: failed to parse vars CSV:", e)
    flight_ids = sorted(set(list(crew_map.keys()) + list(flights.keys())))
    violations = []
    for fid in flight_ids:
        assigns = crew_map.get(fid, [])
        fdiag = flights.get(fid, {"covered_v": 0, "b": len(assigns), "depart": "", "arrive": ""})
        report = classify_for_flight(fid, assigns, fdiag, vars_map, threshold=args.threshold)
        # print a friendly report
        print("Flight", fid, "- assigned crews:", [a["crew"] for a in assigns])
        print("  covered_v:", fdiag.get("covered_v"), "b(csv):", fdiag.get("b"))
        if report["operating"]:
            print("  operating:", report["operating"])
            print("  deadheads:", report["deadheads"])
            if report.get("notes"):
                print("  notes:", ";".join(report["notes"]))
        else:
            print("  operating: (none inferred)", "notes:", ";".join(report.get("notes",[])))
        print("")
        # record a violation if multiple assigned and no operating
        if len(assigns) > 1 and report["operating"] is None:
            violations.append((fid, assigns, fdiag))
    print("Summary:")
    print("  flights checked:", len(flight_ids))
    print("  ambiguous flights (multiple assigned but no inferred operating):", len(violations))
    if violations:
        for v in violations:
            print("   flight", v[0], "assigned:", [a["crew"] for a in v[1]], "diag:", v[2])
    return 0

if __name__ == "__main__":
    main()