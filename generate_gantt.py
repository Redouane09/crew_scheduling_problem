#!/usr/bin/env python3
"""
Generate per-solver Gantt charts for crew assignments found under a diagnostics/ directory.

Usage:
  python3 scripts/generate_gantt.py --diagnostics diagnostics --out out

Outputs:
  - out/gantt_<solver>.png  (one PNG per solver found)
  - out/assignment_comparison.txt  (summary comparing assignment sets between solvers)

Assumptions:
  - Crew assignment files: instance_1_diagnostic_crew.csv (baseline), instance_1_mosek_diagnostic_crew.csv,
    instance_1_gurobi_diagnostic_crew.csv, instance_1_highs_diagnostic_crew.csv (if present).
  - Flight files: instance_1_diagnostic_flights.csv, instance_1_mosek_diagnostic_flights.csv, ...
  - Assignment token format: "<flight_id>@dX@rY"  (flight_id is the integer before the first '@').
  - Deadhead detection: flight is deadhead if covered_v == 0 or b == 0 (or b <= 0). Adjust logic if needed.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta

BASE_DATE = datetime(2000, 1, 1)  # arbitrary base date; day offsets are added

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--diagnostics", default="diagnostics", help="Path to diagnostics folder")
    p.add_argument("--out", default="out", help="Output folder for charts and report")
    p.add_argument("--instance_prefix", default="instance_1", help="Instance prefix used in file names")
    return p.parse_args()

def find_files(diagnostics_dir, instance_prefix):
    # We'll search for patterns e.g. instance_1_diagnostic_crew.csv,
    # instance_1_mosek_diagnostic_crew.csv, instance_1_gurobi_diagnostic_crew.csv, ...
    files = {}
    # list known solver tags (include baseline '')
    solver_tags = ["", "_mosek", "_gurobi", "_highs"]
    for tag in solver_tags:
        crew_name = f"{instance_prefix}{tag}_diagnostic_crew.csv"
        flights_name = f"{instance_prefix}{tag}_diagnostic_flights.csv"
        crew_path = Path(diagnostics_dir) / crew_name
        flights_path = Path(diagnostics_dir) / flights_name
        files[tag if tag else "_baseline"] = {
            "crew": crew_path if crew_path.exists() else None,
            "flights": flights_path if flights_path.exists() else None,
            "tag": tag if tag else "_baseline"
        }
    # also include any crew files that match pattern if different naming has been used
    return files

def load_flights(flights_csv):
    df = pd.read_csv(flights_csv)
    # Expect columns: flight_id, day, depart, arrive, covered_v, b
    # Normalize columns
    for c in ["flight_id", "day", "depart", "arrive", "covered_v", "b"]:
        if c not in df.columns:
            raise RuntimeError(f"Expected column '{c}' in {flights_csv}")
    # parse times to datetimes
    def to_dt(row):
        day = int(row["day"])
        dep = str(row["depart"])
        arr = str(row["arrive"])
        fmt = "%H:%M"
        try:
            dep_t = datetime.strptime(dep, fmt)
            arr_t = datetime.strptime(arr, fmt)
        except Exception:
            # sometimes times are HH:MM:SS or other -- try flexible parse
            dep_t = pd.to_datetime(dep).to_pydatetime()
            arr_t = pd.to_datetime(arr).to_pydatetime()
        base = BASE_DATE + timedelta(days=day-1)
        start = base.replace(hour=dep_t.hour, minute=dep_t.minute, second=getattr(dep_t, "second", 0))
        end = base.replace(hour=arr_t.hour, minute=arr_t.minute, second=getattr(arr_t, "second", 0))
        # if arrival earlier than departure, assume arrival next day
        if end <= start:
            end = end + timedelta(days=1)
        return pd.Series({"start": start, "end": end})
    times = df.apply(to_dt, axis=1)
    df = pd.concat([df, times], axis=1)
    df["flight_id"] = df["flight_id"].astype(int)
    return df.set_index("flight_id")

def parse_assignments(assignments_str):
    # input like "4@d1@r1;5@d1@r2;..." or empty
    if pd.isna(assignments_str) or str(assignments_str).strip() == "":
        return []
    parts = str(assignments_str).split(";")
    flight_ids = []
    for p in parts:
        if p.strip() == "":
            continue
        try:
            flight_id = int(p.split("@")[0])
            flight_ids.append(flight_id)
        except Exception:
            # fallback: try parse integer token
            try:
                flight_id = int(p.strip())
                flight_ids.append(flight_id)
            except:
                # ignore unparsable token but record it
                print(f"Warning: couldn't parse assignment token: '{p}'")
    return flight_ids

def is_deadhead(flight_row):
    # default rule: deadhead if covered_v == 0 or b == 0 (or <= 0)
    try:
        covered_v = float(flight_row.get("covered_v", 1.0))
    except:
        covered_v = 1.0
    try:
        b = float(flight_row.get("b", 1.0))
    except:
        b = 1.0
    return (covered_v == 0.0) or (b <= 0.0)

def make_gantt(crew_df, flights_df, out_path, solver_name):
    # crew_df expected columns: crew_id, home_base, assignments
    # Build per-crew segments
    crews = crew_df["crew_id"].tolist()
    crew_idx = {c: i for i, c in enumerate(crews)}
    fig_height = max(4, 0.5 * len(crews))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    yticks = []
    ylabels = []
    colors = {"operational":"#4C72B0", "deadhead":"#DD8452"}
    for i, row in crew_df.iterrows():
        crew = row["crew_id"]
        y = crew_idx[crew]
        yticks.append(y)
        ylabels.append(f"crew_{crew} ({row.get('home_base','')})")
        assignments = parse_assignments(row.get("assignments",""))
        for f in assignments:
            if f not in flights_df.index:
                print(f"[{solver_name}] Warning: flight id {f} assigned to crew {crew} not found in flights CSV")
                continue
            flo = flights_df.loc[f]
            start = flo["start"]
            end = flo["end"]
            dur = (end - start).total_seconds() / 3600.0
            dh = is_deadhead(flo)
            color = colors["deadhead"] if dh else colors["operational"]
            ax.barh(y, dur, left=start, height=0.6, color=color, edgecolor="k", alpha=0.9)
            # annotate flight id
            ax.text(start + (end - start)/2, y, str(f), va="center", ha="center", color="white", fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    # format x axis as dates/times
    ax.xaxis_date()
    plt.xlabel("Time")
    plt.title(f"Gantt: {solver_name}")
    # legend
    op_patch = mpatches.Patch(color=colors["operational"], label="Assigned flight (operational)")
    dh_patch = mpatches.Patch(color=colors["deadhead"], label="Deadhead / reposition")
    plt.legend(handles=[op_patch, dh_patch], loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def compare_assignments(all_assignments):
    # all_assignments: dict solver -> dict crew -> set(flights)
    solvers = list(all_assignments.keys())
    report_lines = []
    report_lines.append("Assignment comparison report\n")
    # list solvers and counts
    for s in solvers:
        total_assignments = sum(len(v) for v in all_assignments[s].values())
        report_lines.append(f"- {s}: crews={len(all_assignments[s])}, total_assigned_flights={total_assignments}")
    report_lines.append("")
    # pairwise comparisons
    for i in range(len(solvers)):
        for j in range(i+1, len(solvers)):
            s1 = solvers[i]; s2 = solvers[j]
            diffs = []
            crews = set(all_assignments[s1].keys()) | set(all_assignments[s2].keys())
            for c in sorted(crews):
                a1 = set(all_assignments[s1].get(c, []))
                a2 = set(all_assignments[s2].get(c, []))
                if a1 != a2:
                    diffs.append((c, a1 - a2, a2 - a1))
            report_lines.append(f"Comparison {s1} vs {s2}: differences in {len(diffs)} crews")
            for c, only1, only2 in diffs[:50]:
                report_lines.append(f"  crew {c}: only_in_{s1}={sorted(list(only1))}, only_in_{s2}={sorted(list(only2))}")
            if len(diffs) > 50:
                report_lines.append(f"  ... ({len(diffs)-50} more crews differ)")
            report_lines.append("")
    return "\n".join(report_lines)

def main():
    args = parse_args()
    diagnostics_dir = Path(args.diagnostics)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_files(diagnostics_dir, args.instance_prefix)

    found_solvers = []
    all_assignments = {}
    for tag, info in files.items():
        crew_f = info["crew"]
        flights_f = info["flights"]
        solver_name = tag.lstrip("_")
        if crew_f is None:
            print(f"Skipping solver {solver_name}: crew file not found")
            continue
        if flights_f is None:
            print(f"Skipping solver {solver_name}: flights file not found")
            continue
        try:
            crew_df = pd.read_csv(crew_f)
            flights_df = load_flights(flights_f)
        except Exception as e:
            print(f"Error loading files for {solver_name}: {e}")
            continue
        # ensure crew_id column exists
        if "crew_id" not in crew_df.columns:
            # try to infer column names or assume first column is crew_id
            crew_df = crew_df.rename(columns={crew_df.columns[0]:"crew_id"})
        # build mapping
        solver_key = solver_name if solver_name else "baseline"
        found_solvers.append(solver_key)
        assign_map = {}
        for _, row in crew_df.iterrows():
            crew = row["crew_id"]
            assigns = parse_assignments(row.get("assignments",""))
            assign_map[crew] = assigns
        all_assignments[solver_key] = assign_map
        # create gantt
        out_png = out_dir / f"gantt_{solver_key}.png"
        make_gantt(crew_df, flights_df, out_png, solver_key)
        print(f"Saved Gantt for {solver_key} -> {out_png}")

    if not found_solvers:
        print("No solver outputs found to produce Gantt charts.")
        return

    # produce comparison report
    report = compare_assignments(all_assignments)
    report_path = out_dir / "assignment_comparison.txt"
    with open(report_path, "w") as fh:
        fh.write(report)
    print(f"Saved assignment comparison -> {report_path}")
    print("Done.")

if __name__ == "__main__":
    main()