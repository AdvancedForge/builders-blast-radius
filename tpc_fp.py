"""Fingerprint what tests/test_executor.py::test_execute_tpcds computes before
it touches data: per executed TPC-DS query, the DuckDB transpilation the
reference result comes from, the optimized query, the plan and every piece
of Python the executor generates (run over empty tables).

  tpc_fp.py base OUT      write the fingerprints of the tree on sys.path
  tpc_fp.py check BASE    exit 0 if every query matches BASE, 1 at the first
                          difference, 2 on an unexpected error
"""
import hashlib
import json
import re
import pickle
import sys


def fingerprints():
    from sqlglot import parse_one, transpile
    import sqlglot.executor as E
    import sqlglot.planner as P
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

    trace = []
    gen = E.PythonExecutor.generate

    def generate(self, expression):
        trace.append(str(expression))
        return gen(self, expression)

    E.PythonExecutor.generate = generate
    opt = E.optimize

    def optimize(*a, **kw):
        out = opt(*a, **kw)
        trace.append(out.sql())
        return out

    E.optimize = optimize
    plan_init = P.Plan.__init__

    def plan(self, expression):
        plan_init(self, expression)
        # Step order follows object identity, so keep the lines, not their order.
        text = re.sub(r"\(\d{6,}\)", "", repr(self.root)) if hasattr(self, "root") else ""
        trace.append("\n".join(sorted(text.split("\n"))))
        trace.append("|".join(sorted(type(d).__name__ + ":" + str(getattr(d, "name", "")) for d in self.dag)) if hasattr(self, "dag") else "")

    P.Plan.__init__ = plan
    empty = {t: [] for t in TPCDS_SCHEMA}
    for meta, sql, _ in load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"):
        if not meta.get("execute"):
            continue
        trace.clear()
        try:
            trace.append(transpile(sql, write="duckdb")[0])
            expression = pickle.loads(pickle.dumps(parse_one(sql)))
            E.execute(expression, schema=TPCDS_SCHEMA, tables=empty)
        except Exception as e:  # noqa: BLE001 - a new error is a difference
            trace.append(f"{type(e).__name__}: {e}")
        yield hashlib.sha1("\x00".join(sorted(trace)).encode()).hexdigest()


def main():
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "base":
        json.dump(list(fingerprints()), open(path, "w"))
        return 0
    base = json.load(open(path))
    n = 0
    for n, fp in enumerate(fingerprints()):
        if n >= len(base) or fp != base[n]:
            return 1
    return 0 if n + 1 == len(base) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print("tpc_fp error:", type(e).__name__, e, file=sys.stderr)
        sys.exit(2)
