"""Build-time: for every sqlglot module P, the modules M whose import (in a
fresh interpreter) loads P, i.e. whose import-time state can depend on P.
A module that fails to import alone depends on everything.
Usage: PYTHONPATH=<sqlglot root> python loaddeps.py <out.json>"""
import json, pkgutil, subprocess, sys

import sqlglot

names = sorted({m.name for m in pkgutil.walk_packages(sqlglot.__path__, "sqlglot.")} | {"sqlglot"})
PROBE = ("import importlib, json, sys; importlib.import_module(sys.argv[1]); "
         "print(json.dumps(sorted(m for m in sys.modules if m == 'sqlglot' or m.startswith('sqlglot.'))))")
loads = {}
for n in names:
    if n == "sqlglot.__main__":
        continue
    r = subprocess.run([sys.executable, "-c", PROBE, n], capture_output=True, text=True)
    loads[n] = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None
rev = {p: set() for p in names}
for m, ls in loads.items():
    for p in (names if ls is None else ls):
        if p in rev and p != m:
            rev[p].add(m)
json.dump({p: sorted(v) for p, v in rev.items()}, open(sys.argv[1], "w"))
print(len(names), "modules;", sum(v is None for v in loads.values()), "failed alone")
