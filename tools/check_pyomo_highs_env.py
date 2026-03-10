#!/usr/bin/env python3
import sys, shutil
from pyomo.opt import SolverFactory

print("Python executable:", sys.executable)
print("shutil.which('highs') ->", shutil.which("highs"))

for name in ("highs", "highs_direct"):
    try:
        s = SolverFactory(name)
        print(f"SolverFactory('{name}') ->", s)
        try:
            avail = s.available(True)
            print(f"  available(True) -> {avail}")
        except Exception as e:
            print(f"  available(True) raised: {e}")
        # Show if plugin has set_executable attribute
        has_set_exec = hasattr(s, "set_executable")
        print(f"  has set_executable -> {has_set_exec}")
        if has_set_exec:
            try:
                exec_path = s.executable() if hasattr(s, "executable") else None
            except Exception:
                exec_path = None
            print(f"  solver.executable() -> {exec_path}")
    except Exception as e:
        print(f"SolverFactory('{name}') creation raised: {e}")