"""Which sqlglot lines each TPC-DS query of test_execute_tpcds runs on the
real data, as {file: {line: bitmask of executed-query indices}}.

Everything that runs outside a query (imports, loading and pickling the
tables, the worker setup) is credited to every query. Run from the sqlglot
checkout with it on sys.path:  python tpcds_exec_lines.py OUT.json.gz
"""
import gzip
import json
import pickle
import sys

import coverage

cov = coverage.Coverage(data_file=None, include=["*/sqlglot/*"])
cov.start()
cov.switch_context("setup")
import tests.test_executor as TE  # noqa: E402
from sqlglot import parse_one  # noqa: E402
from tests.helpers import TPCDS_SCHEMA  # noqa: E402

import sqlglot.executor.python as PX  # noqa: E402

# Names each query's generated code looks up (identifiers and string
# constants, e.g. the comparison a subquery passes by name): which ENV
# entries a query can reach.
NAMES: dict = {}
_gen = PX.PythonExecutor.generate


def _names(code, out):
    out.update(code.co_names)
    for c in code.co_consts:
        if isinstance(c, str):
            out.add(c)
        elif hasattr(c, "co_names"):
            _names(c, out)


def _generate(self, expression):
    code = _gen(self, expression)
    if code is not None:
        _names(code, NAMES.setdefault(q, set()))
    return code


PX.PythonExecutor.generate = _generate
q = -1
TE.TestExecutor.setUpClass()
cls = TE.TestExecutor
tables = pickle.loads(pickle.dumps(cls.tpcds_tables))  # what the spawned pool receives
TE.initializer(TPCDS_SCHEMA, tables)
q = -1
for meta, sql, _ in cls.tpcds_sqls:
    if not meta.get("execute"):
        continue
    q += 1
    cov.switch_context(f"q{q}")
    expression = pickle.loads(pickle.dumps(parse_one(sql)))
    res = TE.mp_execute(expression, meta)
    if res is not None:
        # the pool pickles the result back; the test reads rows and columns
        res = pickle.loads(pickle.dumps(res))
        [tuple(r) for r in res.rows], list(res.columns)
cov.switch_context("setup")
cov.stop()
n = q + 1
data = cov.get_data()
everything = (1 << n) - 1
out = {}
import ast  # noqa: E402


def body_lines(path):
    """Lines inside some function body (what runs when the function is
    called, as opposed to def headers and module or class level code)."""
    out = set()
    for n in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            body = n.body if isinstance(n.body, list) else [n.body]
            out.update(range(body[0].lineno, (n.end_lineno or n.lineno) + 1))
    return out


for f in data.measured_files():
    rel = "sqlglot/" + f.rsplit("/sqlglot/", 1)[1]
    inside = body_lines(f)
    per = {}
    for ln, ctxs in (data.contexts_by_lineno(f) or {}).items():
        mask = 0
        for c in ctxs:
            if c.startswith("q"):
                mask |= 1 << int(c[1:])
            elif ln in inside:
                # a function called while importing or loading the tables:
                # every query depends on what it built
                mask = everything
                break
        if mask:
            per[str(ln)] = mask
    out[rel] = per
names = {}
for qq, ns in NAMES.items():
    if qq < 0:  # generated outside a query: every query may depend on it
        qq = None
    for x in ns:
        names[x] = names.get(x, 0) | (everything if qq is None else 1 << qq)
with gzip.open(sys.argv[1], "wt") as fh:
    json.dump({"queries": n, "lines": out, "names": names}, fh)
print("queries", n, "files", len(out), "names", len(names), "unnamed queries", [i for i in range(n) if i not in NAMES])
