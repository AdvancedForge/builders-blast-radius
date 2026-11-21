"""Blast Radius selector: pick the sqlglot tests a patch can break.

For every changed file the selector works out where the change sits in the
pinned source and asks a map built from that source (maps/map.json.gz) which
tests reach it:

* a line inside a function: the tests that ran that statement, or the header
  of any block around it (a flipped condition changes what runs inside);
* a def line, signature or decorator: every test that ran the function;
* class-level data (constants, dict entries, arg_types): the tests that
  instantiated the class or a subclass, narrowed to tests that built the
  expression types named on the changed lines when there are any;
* module-level names: tests that ran code referring to them;
* imports, syntax errors and anything not understood: every test whose test
  file imports the module, plus every test that ran it;
* a data file (fixtures): the tests that opened it.

A change that leaves a Python file's AST unchanged (comments, blank lines) can
break nothing. The chosen tests are packed onto as many workers as minimise
busiest-worker seconds plus 2 s a worker.
"""

from __future__ import annotations

import ast
import gzip
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("BR_DATA", "/data"))
SRC = Path(os.environ.get("BR_SRC", "/opt/sqlglot"))
MAP = Path(os.environ.get("BR_MAP", HERE / "maps" / "map.json.gz"))
LIVE = os.environ.get("BR_LIVE", "0") == "1"  # live proxies cost more CPU than they save
# Small selections are run live instead of listed: ~0.6-1 s of CPU against the
# 2 s a worker costs, and only tests that really fail are listed.
# Off by default: the scorer's CPU is ~3x slower than a dev box, which makes
# a live run cost about what the worker it saves does.
SMALL = os.environ.get("BR_SMALL", "0") == "1"
# Importing these pulls in DuckDB and pandas and sets up process pools: a live
# run costs 1.3-3 s of CPU, no better than listing.
HEAVY_TEST_FILES = {"tests/test_optimizer.py", "tests/test_executor.py"}
# In-process live runs (fastlive.py): pinned sqlglot and test modules are
# imported once; each commit forks, swaps the changed function bodies and runs
# its candidates, costing about the tests' own time.
FAST = os.environ.get("BR_FAST", "1") == "1"
# Practice board: these cost about as much CPU as they saved there, since
# practice commits mostly break what they touch (run 23bb751: -8.9 s
# simulated, +11.5 s CPU). Off by default; light function-body runs stay.
FAST_DATA = os.environ.get("BR_FAST_DATA", "1") == "1"
FAST_HEAVY = os.environ.get("BR_FAST_HEAVY", "1") == "1"  # the cost estimate prices heavy files
SCORER_CPU = float(os.environ.get("BR_SCORER_CPU", "1.0"))  # board CPU ~= local (practice: 1.5 s)
FAST_MAX_SECONDS = float(os.environ.get("BR_FAST_MAX", "25"))
SMALL_MAX_SECONDS = float(os.environ.get("BR_SMALL_MAX", "3.0"))
MAX_WORKERS = 8
WORKER_COST = 2.0

LIVE_TIMEOUT = 25
DIFF_MAX = int(os.environ.get("BR_DIFF_MAX", "12"))
# The batch must finish in 3 minutes (4.5 in the final); live work stops
# after this much wall time so a hanging patch cannot get the run rejected.
import time as _time
STARTED = _time.monotonic()
LIVE_BUDGET = float(os.environ.get("BR_LIVE_BUDGET", "90"))


def live_allowed() -> bool:
    return _time.monotonic() - STARTED < LIVE_BUDGET
IMPORT_BASE = Path(os.environ.get("BR_IMPORT_BASE", HERE / "import_fp.json"))
TPCDS_EXEC = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
TPCDS_BASE = Path(os.environ.get("BR_TPCDS_BASE", HERE / "tpcds_fp.json"))
FP_ON = os.environ.get("BR_FP", "1") == "1"
TPCDS_LINES = Path(os.environ.get("BR_TPCDS_LINES", HERE / "maps" / "tpcds_lines.json.gz"))
_tpcds_lines = None
# Code that runs on the data itself, which a data-free fingerprint cannot see.
DATA_TIME = ("sqlglot/executor/", "sqlglot/helper.py", "sqlglot/time.py", "sqlglot/schema.py",
             "tests/", "sqlglot/__init__.py", "sqlglot/expressions/datatypes.py")
# Expensive test -> cheap test that runs the same code on the same queries.
PROXIES = {
    "tests/test_executor.py::TestExecutor::test_execute_tpcds": "tests/test_optimizer.py::TestOptimizer::test_tpcds",
    "tests/test_executor.py::TestExecutor::test_execute_tpch": "tests/test_optimizer.py::TestOptimizer::test_tpch",
    "tests/test_executor.py::TestExecutor::test_optimized_tpch": "tests/test_optimizer.py::TestOptimizer::test_tpch",
}

PYTEST_CONFIG = {"setup.cfg", "pyproject.toml", "tox.ini", "pytest.ini", "conftest.py", "tests/conftest.py"}


def log(*args):
    print(*args, file=sys.stderr)


class Everything(Exception):
    """Raised when a change cannot be narrowed: select every test."""


# --------------------------------------------------------------------------- map


class Map:
    def __init__(self, path: Path):
        with gzip.open(path, "rt") as fh:
            m = json.load(fh)
        self.tests: list[str] = m["tests"]
        self.sets: list[list[int]] = m["sets"]
        self.lines: dict[str, dict[str, int]] = m["lines"]
        self.reads: dict[str, int] = m["reads"]
        self.classes: dict[str, int] = m["classes"]
        self.closure: dict[str, list[str] | None] = m["closure"]
        self.import_lines: dict[str, set[int]] = {f: set(v) for f, v in m["import"].items()}
        self.keys: dict[str, int] = m.get("keys", {})
        self.key_attrs = {k.split("|", 1)[0] for k in self.keys}
        self.all = set(range(len(self.tests)))
        self._by_prefix: dict[str, set[int]] = {}

    def s(self, idx: int | None) -> set[int]:
        return set(self.sets[idx]) if idx is not None else set()

    def line(self, path: str, ln: int) -> set[int]:
        return self.s(self.lines.get(path, {}).get(str(ln)))

    def span(self, path: str, a: int, b: int) -> set[int]:
        per = self.lines.get(path, {})
        out: set[int] = set()
        for ln in range(a, b + 1):
            idx = per.get(str(ln))
            if idx is not None:
                out.update(self.sets[idx])
        return out

    def file_any(self, path: str) -> set[int]:
        out: set[int] = set()
        for idx in self.lines.get(path, {}).values():
            out.update(self.sets[idx])
        return out

    def users(self, qualified: str) -> set[int] | None:
        idx = self.classes.get(qualified)
        return set(self.sets[idx]) if idx is not None else None

    def prefix(self, prefix: str) -> set[int]:
        if prefix not in self._by_prefix:
            self._by_prefix[prefix] = {i for i, t in enumerate(self.tests) if t.startswith(prefix)}
        return self._by_prefix[prefix]

    def importers(self, path: str) -> set[int]:
        """Tests that import `path`: their test module imports it, they ran a
        line of it, or they used a class defined in it (dialect modules are
        imported lazily, mid-test)."""
        files = [tf for tf, mods in self.closure.items() if mods is None or path in mods or tf == path]
        out = set()
        for tf in files:
            out |= self.prefix(tf + "::")
        mod = path[:-3].replace("/", ".").removesuffix(".__init__") + ":"
        for q, idx in self.classes.items():
            if q.startswith(mod):
                out.update(self.sets[idx])
        return out | self.file_any(path)


# ------------------------------------------------------------------------- patch


class FilePatch:
    def __init__(self):
        self.old: str | None = None
        self.new: str | None = None
        self.hunks: list[tuple[int, list[tuple[str, str]]]] = []

    @property
    def path(self) -> str:
        return self.old or self.new or ""


def _strip(p: str) -> str | None:
    p = p.split("\t")[0].strip()
    if p == "/dev/null":
        return None
    if p.startswith('"') and p.endswith('"'):
        p = p[1:-1]
    return p[2:] if p[:2] in ("a/", "b/") else p


def parse_patch(text: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    cur: FilePatch | None = None
    hunk: list[tuple[str, str]] | None = None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            cur = FilePatch(); files.append(cur); hunk = None
            m = re.match(r"diff --git (\S+) (\S+)", line)
            if m:
                cur.old, cur.new = _strip(m.group(1)), _strip(m.group(2))
        elif line.startswith("--- ") and (hunk is None or i + 1 < len(lines) and lines[i + 1].startswith("+++ ")):
            if cur is None or cur.hunks:
                cur = FilePatch(); files.append(cur)
            cur.old = _strip(line[4:]); hunk = None
        elif line.startswith("+++ ") and hunk is None and cur is not None:
            cur.new = _strip(line[4:])
        elif line.startswith("rename from ") and cur is not None:
            cur.old = line[len("rename from "):]
        elif line.startswith("rename to ") and cur is not None:
            cur.new = line[len("rename to "):]
        elif line.startswith("deleted file") and cur is not None:
            cur.new = None
        elif line.startswith("new file") and cur is not None:
            cur.old = None
        elif line.startswith("@@") and cur is not None:
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            hunk = []
            cur.hunks.append((int(m.group(1)) if m else 1, hunk))
        elif hunk is not None and line[:1] in (" ", "-", "+"):
            hunk.append((line[0], line[1:]))
        elif hunk is not None and line == "" and i + 1 < len(lines):
            hunk.append((" ", ""))
        i += 1
    return files


def apply(old_text: str, fp: FilePatch) -> tuple[str, list[int], list[tuple[int, int]]]:
    """Return new text, removed old line numbers, and (insertion point, new line) pairs."""
    old = old_text.split("\n")
    out: list[str] = []
    removed: list[int] = []
    added: list[tuple[int, int]] = []
    pos = 1  # next old line (1-based) not yet copied
    for start, body in fp.hunks:
        start = max(start, 1) if any(t != "+" for t, _ in body) else start + 1
        while pos < start and pos <= len(old):
            out.append(old[pos - 1]); pos += 1
        for tag, text in body:
            if tag == " ":
                out.append(old[pos - 1] if pos <= len(old) else text); pos += 1
            elif tag == "-":
                removed.append(pos); pos += 1
            else:
                out.append(text); added.append((pos, len(out)))
    while pos <= len(old):
        out.append(old[pos - 1]); pos += 1
    return "\n".join(out), removed, added


# ---------------------------------------------------------------------- analysis

COMPOUND = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)
if hasattr(ast, "TryStar"):
    COMPOUND = COMPOUND + (ast.TryStar,)
FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


def start(node: ast.AST) -> int:
    decos = getattr(node, "decorator_list", None)
    s = node.lineno
    if decos:
        s = min([s] + [d.lineno for d in decos])
    return s


def chain(tree: ast.AST, line: int) -> list[ast.AST]:
    path: list[ast.AST] = []
    node = tree
    while True:
        nxt = None
        for child in ast.iter_child_nodes(node):
            if hasattr(child, "lineno") and start(child) <= line <= (child.end_lineno or child.lineno):
                nxt = child
                break
        if nxt is None:
            return path
        path.append(nxt)
        node = nxt


def header_end(node: ast.AST) -> int:
    body = getattr(node, "body", None)
    if body and isinstance(body, list) and hasattr(body[0], "lineno"):
        return start(body[0]) - 1
    if isinstance(node, ast.Match) and node.cases:
        return node.cases[0].pattern.lineno - 1
    return node.end_lineno or node.lineno


def in_string(tree: ast.AST, line: int) -> bool:
    """True if `line` lies inside a multi-line string literal (so a line that
    looks like a comment is really string content)."""
    node = tree
    while True:
        nxt = None
        for child in ast.iter_child_nodes(node):
            if hasattr(child, "lineno") and start(child) <= line <= (child.end_lineno or child.lineno):
                nxt = child
                break
        if nxt is None:
            return False
        if isinstance(nxt, (ast.Constant, ast.JoinedStr)) and nxt.lineno < line <= (nxt.end_lineno or nxt.lineno):
            return True
        node = nxt


def is_bare_string(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """`tree` (modified in place) without bare string statements (docstrings).

    Nothing in the suite reads docstrings: the doctests are loaded through
    unittest's load_tests, which pytest does not run."""
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            kept = [x for x in body if not is_bare_string(x)]
            node.body = kept or [ast.Pass()]
    return tree


class _NoneCompare(ast.NodeTransformer):
    """`x == None` behaves as `x is None` for every type sqlglot compares this
    way (Expr.__eq__ checks the type first; dialect __eq__ returns False)."""

    def visit_Compare(self, node):
        self.generic_visit(node)
        ops = []
        for op, right in zip(node.ops, node.comparators):
            if isinstance(right, ast.Constant) and right.value is None:
                op = ast.Is() if isinstance(op, ast.Eq) else ast.IsNot() if isinstance(op, ast.NotEq) else op
            ops.append(op)
        if isinstance(node.left, ast.Constant) and node.left.value is None and len(ops) == 1:
            ops[0] = ast.Is() if isinstance(ops[0], ast.Eq) else ast.IsNot() if isinstance(ops[0], ast.NotEq) else ops[0]
        node.ops = ops
        return node


def normalize(tree: ast.AST, path: str) -> ast.AST:
    """The tree with runtime-irrelevant differences removed: docstrings, and
    == / != None written for is / is not None (outside the executor's
    data-time helpers, whose wrappers define their own __eq__)."""
    tree = strip_docstrings(tree)
    if not path.startswith("sqlglot/executor/env.py"):
        tree = _NoneCompare().visit(tree)
    return tree


def module_name(path: str) -> str:
    return path[:-3].replace("/", ".").removesuffix(".__init__")


EXP_NAME = re.compile(r"\bexp\.([A-Z]\w*)")


class Selector:
    def __init__(self, m: Map):
        self.m = m
        self._texts: dict[str, str] = {}
        self._trees: dict[str, ast.AST] = {}
        self._exp_users: dict[str, set[int] | None] = {}
        self._refs: dict[str, set[int]] = {}
        self._reads_cache: dict[str, set[int] | None] = {}
        self._replacing = False
        self.import_ok = False  # set once a live import check shows import-time state unchanged
        self._lines: dict[str, list[str]] = {}

    def text(self, rel: str) -> str | None:
        if rel not in self._texts:
            p = SRC / rel
            self._texts[rel] = p.read_text("utf-8") if p.is_file() else None
        return self._texts[rel]

    def tree(self, rel: str) -> ast.AST | None:
        if rel not in self._trees:
            t = self.text(rel)
            self._trees[rel] = ast.parse(t) if t is not None else None
        return self._trees[rel]

    # -- helpers

    def exp_users(self, name: str) -> set[int] | None:
        """Tests that built expression class `name` (found in any expressions module)."""
        if name not in self._exp_users:
            hit = None
            for q, idx in self.m.classes.items():
                mod, _, qual = q.partition(":")
                if qual == name and mod.startswith("sqlglot.expressions"):
                    hit = (hit or set()) | self.m.s(idx)
            self._exp_users[name] = hit
        return self._exp_users[name]

    def class_users(self, path: str, quals: list[str]) -> set[int] | None:
        return self.m.users(f"{module_name(path)}:{'.'.join(quals)}")

    def method_reach(self, name: str) -> set[int] | None:
        """Tests a new method `name` can reach: those that ran a same-named
        method it may override, built the expression a `<key>_sql` generator
        method is dispatched for, or ran a line naming it. None if unsure."""
        out = self.runtime_reads(name)
        if out is None:
            return None
        out = set(out)
        for rel in self.m.lines:
            if not rel.startswith("sqlglot/"):
                continue
            text = self.text(rel)
            if text is None or f"def {name}(" not in text:
                continue
            for node in ast.walk(self.tree(rel)):
                if isinstance(node, FUNC) and node.name == name:
                    out |= self.m.span(rel, node.lineno, node.end_lineno)
        if name.endswith("_sql"):
            key = name[:-4]
            hit = False
            for q, idx in self.m.classes.items():
                mod, _, qual = q.partition(":")
                if mod.startswith("sqlglot.expressions") and qual.lower() == key:
                    out |= self.m.s(idx); hit = True
            if not hit and key not in ("", "sql"):
                # a key no traced expression has: could still be dispatched for
                # an expression built only outside the suite's traces
                pass
        return out

    def typing_metadata_users(self, path: str, old_text: str, new_text: str) -> set[int] | None:
        """A typing/<dialect>.py edit that only changes EXPRESSION_METADATA:
        the tests that looked up a changed expression type. sqlglot reads that
        dict only through TypeAnnotator's .get(expr.__class__) (copies aside),
        and the map records each test's lookups. None if not applicable."""
        if not path.startswith("sqlglot/typing/") or "EXPRESSION_METADATA" not in self.m.key_attrs:
            return None
        import fastlive

        ops = fastlive.data_ops(old_text, new_text)
        if not ops or any(op[2] != "dict2" or op[1] != "EXPRESSION_METADATA" or op[0] for op in ops):
            return None
        try:
            fastlive.setup(str(SRC))
            if fastlive.swaps_for(path, old_text, new_text, allow_bodies=True):
                return None  # function bodies changed too
            mod = sys.modules[module_name(path)]
            ns = dict(vars(mod))
            out: set[int] = set()
            for _, _, _, old_src, new_src in ops:
                a, b = eval(old_src, ns), eval(new_src, ns)
                for k in set(a) | set(b):
                    if k in a and k in b and fastlive._safe_eq(a[k], b[k]):
                        continue
                    if not isinstance(k, type):
                        return None
                    idx = self.m.keys.get(f"EXPRESSION_METADATA|t:{k.__qualname__}")
                    if idx is not None:
                        out |= self.m.s(idx)
            # A test reads a dialect's metadata through that dialect, so it
            # must have used one whose family includes this typing module.
            if _ENTRIES is None:
                entry_mask(self, "")
            fams = families() if _ENTRIES else {}
            probe = "sqlglot/dialects/" + os.path.basename(path)
            dialects = {d for d, files in fams.items() if probe in files or path in files}
            if fams and dialects:
                users: set[int] = set()
                for d in dialects:
                    mod = f"sqlglot.dialects.{d}:" if d else None
                    if mod is None:
                        users = None
                        break
                    for q, idx in self.m.classes.items():
                        if q.startswith(mod):
                            users |= self.m.s(idx)
                if users is not None:
                    out &= users
            # No import check: EXPRESSION_METADATA is only copied at import
            # (dialects' .copy()/** merges), never derived from, and a literal
            # that cannot evaluate (a broken import) raised above and fell back.
            self.narrowed_attrs.add("EXPRESSION_METADATA")
            return out
        except Exception as e:  # noqa: BLE001
            log(f"typing narrowing failed: {type(e).__name__}: {e}")
            self.needs_import_check = True  # the module may not even import
            return None

    @staticmethod
    def dict_spans(stmt, ln) -> bool:
        """True if line `ln` of the assignment lies inside a dict literal (the
        dict-entry path handles those)."""
        return stmt.value is not None and any(
            isinstance(n, ast.Dict) and n.lineno <= ln <= (n.end_lineno or n.lineno) for n in ast.walk(stmt.value))

    def mention_reads(self, name: str) -> set[int]:
        """Tests that ran any sqlglot line naming `name` (definitions in class
        bodies excluded). Constants are only read by name: the dynamic
        getattr calls in sqlglot look up *_sql methods."""
        key = "m:" + name
        if key in self._reads_cache:
            return self._reads_cache[key]
        pat = re.compile(r"\b" + re.escape(name) + r"\b")
        out: set[int] = set()
        for rel in self.m.lines:
            if not rel.startswith("sqlglot/"):
                continue
            text = self.text(rel)
            if text is None or name not in text:
                continue
            for i, l in enumerate(self.lines_of(rel), 1):
                if name in l and pat.search(l) and not l.lstrip().startswith("#"):
                    if re.match(r"\s*" + re.escape(name) + r"\s*[:=]", l):
                        continue  # a definition or override
                    out |= self.m.line(rel, i)
        self._reads_cache[key] = out
        return out

    def runtime_reads(self, name: str) -> set[int] | None:
        """Tests that ran a line reading attribute/name `name`, or None if any
        read happens outside a function (class creation, module import), where
        coverage cannot say who depends on it. Class-body assignments of the
        same name (definitions and overrides) are not reads."""
        if name in self._reads_cache:
            return self._reads_cache[name]
        pat = re.compile(r"\b" + re.escape(name) + r"\b")
        out: set[int] | None = set()
        for rel in self.m.lines:
            if not rel.startswith("sqlglot/"):
                continue
            text = self.text(rel)
            if text is None or name not in text:
                continue
            tree = None
            for i, l in enumerate(self.lines_of(rel), 1):
                if name not in l or not pat.search(l) or l.lstrip().startswith("#"):
                    continue
                tree = tree or self.tree(rel)
                ch = chain(tree, i)
                stmts = [n for n in ch if isinstance(n, ast.stmt)]
                in_code = any(isinstance(n, FUNC + (ast.Lambda,)) for n in ch)
                if not in_code:
                    st = stmts[-1] if stmts else None
                    is_def = (isinstance(st, (ast.Assign, ast.AnnAssign)) and name in self.assigned(st)
                              and st.lineno == i and re.match(r"\s*" + re.escape(name) + r"\b", l))
                    if is_def:
                        continue
                    out = None
                    break
                st = stmts[-1]
                if isinstance(st, FUNC) and start(st) <= i <= st.lineno and st.name == name:
                    continue  # a definition, not a read
                if i in self.m.import_lines.get(rel, ()):
                    out = None  # read while modules load (e.g. a metaclass): users unknown
                    break
                out |= self.m.span(rel, start(st), st.end_lineno or st.lineno) | self.m.line(rel, i)
            if out is None:
                break
        self._reads_cache[name] = out
        return out

    def lines_of(self, rel: str) -> list[str]:
        if rel not in self._lines:
            t = self.text(rel)
            self._lines[rel] = t.split("\n") if t is not None else []
        return self._lines[rel]

    def refs(self, names: set[str], depth: int = 0) -> set[int]:
        """Tests that ran code referring to any of `names`, across sqlglot.

        Memoized per name; a name already being resolved contributes nothing
        on the way back in, so module-level names that refer to each other
        cannot recurse without end."""
        out: set[int] = set()
        for name in names:
            if name in self._refs:
                out |= self._refs[name]
                continue
            self._refs[name] = set()  # guard against cycles
            pat = re.compile(r"\b" + re.escape(name) + r"\b")
            found: set[int] = set()
            for rel in self.m.lines:
                if not rel.startswith("sqlglot/"):
                    continue
                text = self.text(rel)
                if text is None or name not in text or not pat.search(text):
                    continue
                tree = self.tree(rel)
                for i, l in enumerate(self.lines_of(rel), 1):
                    if name in l and pat.search(l) and not l.lstrip().startswith("#"):
                        found |= self.at_line(rel, tree, i, set(), depth + 1, names_hint=True)
            self._refs[name] = found
            out |= found
        return out

    # -- core: tests that can notice a change at old line `ln` of `path`

    def at_line(self, path: str, tree: ast.AST, ln: int, extra_names: set[str], depth: int = 0,
                names_hint: bool = False) -> set[int]:
        m = self.m
        ch = chain(tree, ln)
        stmts = [n for n in ch if isinstance(n, ast.stmt)]
        if not stmts:
            return set() if names_hint else m.file_any(path)
        if is_bare_string(stmts[-1]):
            return set()  # a docstring or bare string: no effect at runtime
        funcs = [i for i, n in enumerate(ch) if isinstance(n, FUNC)]
        lambdas = [i for i, n in enumerate(ch) if isinstance(n, ast.Lambda)]
        classes = [n for n in ch if isinstance(n, ast.ClassDef)]
        out: set[int] = set()

        inner_fn = funcs[-1] if funcs else None
        inner_lambda = lambdas[-1] if lambdas else None
        if inner_lambda is not None and (inner_fn is None or inner_lambda > inner_fn):
            lam = ch[inner_lambda]
            if ln > lam.lineno or lam.body.lineno == ln or lam.end_lineno >= ln:
                out |= m.span(path, lam.lineno, lam.end_lineno)
                if out or names_hint:
                    return out
        if inner_fn is not None:
            f = ch[inner_fn]
            body_start = start(f.body[0])
            if ln < body_start:  # decorator, def line, signature
                return m.span(path, f.lineno, f.end_lineno) | m.span(path, start(f), f.lineno)
            inner = [n for n in ch[inner_fn + 1:] if isinstance(n, ast.stmt)]
            imported = m.import_lines.get(path, ())
            if not names_hint and path.startswith("sqlglot/") and imported and not self.import_ok:
                # The changed statement, or a block header around it, runs while
                # sqlglot is imported: breaking it can break every import.
                spots = {ln}
                if inner:
                    last = inner[-1]
                    end = header_end(last) if isinstance(last, COMPOUND) else (last.end_lineno or last.lineno)
                    spots.update(range(start(last), end + 1))
                for n in inner:
                    if isinstance(n, COMPOUND):
                        spots.update(range(n.lineno, header_end(n) + 1))
                if spots & imported:
                    raise Everything(f"import-time code {path}:{ln}")
            for n in inner:
                if isinstance(n, COMPOUND) or isinstance(n, FUNC + (ast.ClassDef,)):
                    out |= m.span(path, n.lineno, max(n.lineno, header_end(n)))
            last = inner[-1] if inner else None
            if last is not None:
                if isinstance(last, COMPOUND) and ln <= header_end(last):
                    out |= m.span(path, last.lineno, header_end(last))
                elif not isinstance(last, COMPOUND):
                    out |= m.span(path, start(last), last.end_lineno or last.lineno)
            out |= m.line(path, ln)
            return out
        if names_hint and depth > 2:
            return set()
        if not names_hint and path.startswith("sqlglot/"):
            # Module or class level code runs at import: if the patched module
            # fails to import, every importer fails; the import check finds out.
            self.needs_import_check = True
        if classes:
            quals: list[str] = []
            for n in ch:
                if isinstance(n, ast.ClassDef):
                    quals.append(n.name)
            users = self.class_users(path, quals)
            stmt = stmts[-1]
            if path.startswith("tests/"):
                return m.prefix(f"{path}::{quals[0]}::") or m.prefix(path + "::")
            if users is None:
                cls = classes[-1]
                names = {cls.name} | self.assigned(stmt) | extra_names
                return m.span(path, cls.lineno, cls.end_lineno) | self.refs(names, depth)
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt in classes[-1].body:
                names = self.assigned(stmt)
                reads = [self.runtime_reads(n) for n in names] if names else [None]
                if all(r is not None for r in reads):
                    return users & set().union(*reads)
                if names and not self.dict_spans(stmt, ln):
                    # Also read while modules load: tests that ran a line naming
                    # it, plus whatever the import check finds derived from it.
                    self.needs_import_check = True
                    self.narrowed_attrs |= names
                    return users & set().union(*(self.mention_reads(n) for n in names))
            ku = self.entry_key_users(path, tree, ch, ln, self.lines_of(path))
            if ku is not None:
                return users & ku
            keys = self.dict_keys(path, ch, ln) | extra_names
            if keys:
                narrowed: set[int] = set()
                for name in keys:
                    eu = self.exp_users(name)
                    if eu is None:
                        return users  # a key we cannot trace: keep every user
                    narrowed |= eu
                return users & narrowed
            return users
        # module level
        stmt = stmts[-1]
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            if names_hint:
                return set()
            return m.importers(path)
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            names = self.assigned(stmt)
            if path.startswith("tests/"):
                return m.importers(path)
            if names_hint:
                return self.refs(names, depth) if depth <= 2 else set()
            return self.refs(names, depth) | m.span(path, stmt.lineno, stmt.end_lineno)
        if names_hint:
            return set()
        return m.importers(path)

    def entry_key_users(self, path, tree, ch, ln, lines) -> set[int] | None:
        """Tests that looked up the key of the class-level dict entry holding
        line `ln` (recorded per test in the map), or None if that cannot be
        relied on: no dict entry, a key that is not a string or enum member,
        a dict never looked up directly, or a multi-word key with no recorded
        lookups (those are matched through tries built at import)."""
        stmt = next((n for n in reversed(ch) if isinstance(n, (ast.Assign, ast.AnnAssign))), None)
        if stmt is None or stmt.value is None:
            return None
        # The outermost dict literal of the assignment that spans the line (the
        # chain can stop at the target when everything is on one line).
        dct = next((n for n in ast.walk(stmt.value) if isinstance(n, ast.Dict)
                    and n.lineno <= ln <= (n.end_lineno or n.lineno)), None)
        if dct is None:
            return None
        names = self.assigned(stmt)
        if len(names) != 1:
            return None
        attr = next(iter(names))
        if attr not in self.m.key_attrs:
            return None
        out: set[int] = set()
        hit = False
        for k, v in zip(dct.keys, dct.values):
            # Every entry on the line: a one-line dict holds several.
            if k is None:
                if v.lineno <= ln <= (v.end_lineno or v.lineno):
                    return None  # a ** splat on the line
                continue
            if not (start(k) <= ln <= (v.end_lineno or v.lineno)):
                continue
            hit = True
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                rep = "s:" + k.value
                multi = " " in k.value
            elif isinstance(k, ast.Attribute) and isinstance(k.value, ast.Name) and k.value.id == "TokenType":
                rep = f"e:TokenType.{k.attr}"
                multi = False
            else:
                return None
            idx = self.m.keys.get(f"{attr}|{rep}")
            if idx is None:
                if multi:
                    return None
                continue
            out |= self.m.s(idx)
        if hit and self.runtime_reads(attr) is None:
            # Also read while modules load (a metaclass deriving other data
            # from it): the live import check must confirm nothing derived
            # at import changed, or add the users of what did.
            self.needs_import_check = True
        if hit:
            self.narrowed_attrs.add(attr)
        return out if hit else None

    def dict_keys(self, path: str, ch: list[ast.AST], ln: int) -> set[str]:
        """exp.X names keying the dict entry that holds line `ln`, if any."""
        for node in reversed(ch):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if k is None:
                        continue
                    if start(k) <= ln <= (v.end_lineno or v.lineno):
                        lines = self.text(path).split("\n")[k.lineno - 1: k.end_lineno]
                        seg = "\n".join(lines)
                        names = set(EXP_NAME.findall(seg))
                        return names if isinstance(k, ast.Attribute) and names else set()
                return set()
        return set()

    @staticmethod
    def key_names(line: str) -> set[str]:
        """exp.X names used as a dict key on a changed line ("exp.X: ...")."""
        return set(re.findall(r"\bexp\.([A-Z]\w*)\s*:", line))

    @staticmethod
    def assigned(stmt: ast.AST) -> set[str]:
        names: set[str] = set()
        targets = []
        if isinstance(stmt, ast.Assign):
            targets = stmt.targets
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
            targets = [stmt.target]
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    names.add(n.id)
                elif isinstance(n, ast.Attribute):
                    names.add(n.attr)
        return names

    # -- per file

    def python_file(self, fp: FilePatch) -> set[int]:
        m = self.m
        path = fp.old or fp.new
        old_text = self.text(fp.old) if fp.old else ""
        if old_text is None:
            old_text = ""
        new_text, removed, added = apply(old_text, fp)
        if fp.new is not None:
            self.patched[fp.new] = new_text
        if fp.new is None or (fp.old and fp.new != fp.old):  # deleted or renamed
            if path.startswith("sqlglot/"):
                raise Everything(f"module removed: {path}")  # dialects load lazily from everywhere
            return m.importers(path)
        try:
            new_tree = ast.parse(new_text.encode("utf-8"))
        except (SyntaxError, ValueError):
            if path.startswith("sqlglot/"):
                raise Everything(f"syntax error in {path}")  # dialects load lazily from everywhere
            return m.importers(path) if fp.old else set()
        if not fp.old or old_text == "":
            return set()  # a new module nothing imports yet
        old_tree = self.tree(fp.old)
        typing_users = self.typing_metadata_users(path, old_text, new_text)
        if typing_users is not None:
            return typing_users
        old_lines = old_text.split("\n")
        new_lines = new_text.split("\n")
        # The whole-module comparison (docstrings, == None) costs two parses
        # and dumps; comment and blank lines are cheaper to rule out per line.
        touched = [old_lines[i - 1] for i in removed] + [new_lines[n - 1] for _, n in added]
        if "None" in "".join(touched) or any('"""' in l or "'''" in l for l in touched):
            if ast.dump(normalize(ast.parse(old_text.encode("utf-8")), path)) == ast.dump(
                    normalize(ast.parse(new_text.encode("utf-8")), path)):
                return set()
        if fp.new != fp.old:
            return m.importers(fp.old)

        old_lines = old_text.split("\n")
        new_lines = new_text.split("\n")
        out: set[int] = set()
        for ln in removed:
            code = old_lines[ln - 1].strip()
            if (not code or code.startswith("#")) and not in_string(old_tree, ln):
                continue  # removing a blank or comment line changes nothing
            loc = self.at_line(path, old_tree, ln, self.key_names(old_lines[ln - 1]))
            self.locs.append(loc); out |= loc
        self._new_lines = new_lines
        self.touched.setdefault(path, set()).update(removed)
        for k, _ in added:
            self.touched[path].update((k - 1, k))
        removed_set = set(removed)
        for k, n in added:
            self._replacing = (k - 1) in removed_set
            loc = self.added_line(path, old_tree, new_tree, k, n, self.key_names(new_lines[n - 1]), old_lines)
            self.locs.append(loc); out |= loc
        return out

    def added_line(self, path, old_tree, new_tree, k, n, names, old_lines) -> set[int]:
        """A line inserted before old line k, at new line n."""
        code = self._new_lines[n - 1].strip()
        if (not code or code.startswith("#")) and not in_string(new_tree, n):
            return set()  # a blank or comment line
        nch = chain(new_tree, n)
        nstmts = [x for x in nch if isinstance(x, ast.stmt)]
        if not nstmts:
            return set()  # blank or comment
        if is_bare_string(nstmts[-1]):
            return set()  # inside a docstring: no effect at runtime
        if path.startswith("sqlglot/") and not any(isinstance(x, FUNC + (ast.Lambda,)) for x in nch):
            self.needs_import_check = True  # new module/class-level code runs at import
        if path.startswith("tests/"):
            # A brand-new test_* method in a test class: not in tests.json, and
            # a new name cannot change what the existing tests run.
            ci = max((i for i, x in enumerate(nch) if isinstance(x, ast.ClassDef)), default=None)
            if ci is not None and ci + 1 < len(nch) and isinstance(nch[ci + 1], FUNC) \
                    and nch[ci + 1].name.startswith("test") \
                    and not self.has_method(old_tree, [x.name for x in nch[:ci + 1] if isinstance(x, ast.ClassDef)],
                                            nch[ci + 1].name):
                return set()
        # Old neighbours that are code: the statement just before and just after.
        # A line that replaces a removed one needs none: the removed line's own
        # analysis covers the statement it was part of.
        out: set[int] = set()
        for cand in (() if self._replacing else (k - 1, k)):
            j = cand
            step = -1 if cand == k - 1 else 1
            while 1 <= j <= len(old_lines) and (not old_lines[j - 1].strip() or old_lines[j - 1].lstrip().startswith("#")):
                j += step
            if 1 <= j <= len(old_lines):
                out |= self.at_line(path, old_tree, j, names)
        # A new or changed entry of a class-level dict: the tests that look up
        # its key, among the class's users.
        quals_n = [x.name for x in nch if isinstance(x, ast.ClassDef)]
        if quals_n and not any(isinstance(x, FUNC + (ast.Lambda,)) for x in nch) and not path.startswith("tests/"):
            users_n = self.class_users(path, quals_n)
            if users_n is not None:
                ku = self.entry_key_users(path, new_tree, nch, n, self._new_lines)
                if ku is not None:
                    return users_n & ku
        # A statement added straight into a class body (a new attribute or a
        # new method, which may override one) changes the class for its users.
        ci = max((i for i, x in enumerate(nch) if isinstance(x, ast.ClassDef)), default=None)
        if ci is not None and ci + 1 < len(nch) and not path.startswith("tests/"):
            member = nch[ci + 1]
            quals = [x.name for x in nch[: ci + 1] if isinstance(x, ast.ClassDef)]
            fn_inside = any(isinstance(x, FUNC) for x in nch[:ci])
            existing = isinstance(member, FUNC) and self.has_method(old_tree, quals, member.name)
            if not fn_inside and not existing and isinstance(member, FUNC) and not path.startswith("tests/"):
                users = self.class_users(path, quals)
                if users is not None:
                    reach = self.method_reach(member.name)
                    if reach is not None:
                        users = users & reach
                    out |= users
            elif not fn_inside and not existing:
                users = self.class_users(path, quals)
                if users is not None:
                    keys = names
                    if keys and all(self.exp_users(x) is not None for x in keys):
                        users = users & set().union(*(self.exp_users(x) for x in keys))
                    elif isinstance(member, (ast.Assign, ast.AnnAssign)) and not any(
                            isinstance(x, ast.Dict) for x in nch[ci + 2:]):
                        names_m = self.assigned(member)
                        reads = [self.runtime_reads(x) for x in names_m] or [None]
                        if all(r is not None for r in reads):
                            users = users & set().union(*reads)
                        elif names_m:
                            self.needs_import_check = True
                            self.narrowed_attrs |= names_m
                            users = users & set().union(*(self.mention_reads(x) for x in names_m))
                    out |= users
        if ci is None and not any(isinstance(x, FUNC) for x in nch):
            stmt = nstmts[-1]
            if isinstance(stmt, (ast.Import, ast.ImportFrom)) or not isinstance(stmt, (FUNC + (ast.ClassDef, ast.Assign, ast.AnnAssign))):
                out |= self.m.importers(path)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                out |= self.refs(self.assigned(stmt))
        return out

    @staticmethod
    def has_method(tree, quals, name) -> bool:
        body = tree.body
        for q in quals:
            cls = next((x for x in body if isinstance(x, ast.ClassDef) and x.name == q), None)
            if cls is None:
                return False
            body = cls.body
        return any(isinstance(x, FUNC) and x.name == name for x in body)

    def data_file(self, fp: FilePatch) -> set[int]:
        m = self.m
        out: set[int] = set()
        if fp.old and fp.new == fp.old and fp.old.startswith("tests/fixtures/") and fp.old.endswith(".sql"):
            # Every .sql fixture is read through tests/helpers._filter_comments,
            # which drops blank lines and "--" comments before parsing.
            old_text = self.text(fp.old)
            if old_text is not None:
                new_text = apply(old_text, fp)[0]

                def filtered(t: str) -> str:
                    return "\n".join(l for l in t.splitlines() if l and not l.startswith("--"))

                if filtered(old_text) == filtered(new_text):
                    return out
        for p in {fp.old, fp.new} - {None}:
            idx = m.reads.get(p)
            if idx is not None:
                readers = m.s(idx)
                # The executor's setUpClass loads both TPC sets for the whole
                # class; only the tests named for a set use it.
                for marker, tag in (("tpc-h", "tpch"), ("tpc-ds", "tpcds")):
                    if f"/{marker}/" in p:
                        readers = {k for k in readers if not m.tests[k].startswith("tests/test_executor.py::")
                                   or m.tests[k].endswith(tag) or m.tests[k].rsplit("_", 1)[-1] == tag}
                self.locs.append(readers)
                out |= readers
            elif p.startswith("tests/"):
                base = os.path.basename(p)
                for tf in m.closure:
                    t = self.text(tf) or ""
                    if base in t or os.path.basename(os.path.dirname(p)) in t:
                        out |= m.prefix(tf + "::")
        return out

    def commit(self, patch_text: str) -> set[int]:
        out: set[int] = set()
        self.locs: list[set[int]] = []
        self.patched: dict[str, str] = {}
        self.deleted: set[str] = set()
        self.touched: dict[str, set[int]] = {}
        self.needs_import_check = False
        self._import_result = _UNSET
        self.import_deep = None
        self.narrowed_attrs: set[str] = set()
        self.whole_suite = False
        for fp in parse_patch(patch_text):
            if fp.old and fp.new is None:
                self.deleted.add(fp.old)
            if fp.old != fp.new and fp.old:
                self.deleted.add(fp.old)
            if fp.new and not fp.new.endswith(".py"):
                base = self.text(fp.old) if fp.old else ""
                self.patched[fp.new] = apply(base or "", fp)[0]
            path = fp.path
            if not path:
                continue
            if path in PYTEST_CONFIG or (fp.new or "") in PYTEST_CONFIG or os.path.basename(path) == "conftest.py":
                if path.endswith(".py") or "pytest" in patch_text:
                    raise Everything(path)
                continue
            if path.endswith(".py"):
                if not (path.startswith("sqlglot/") or path.startswith("tests/")):
                    continue
                out |= self.python_file(fp)
            else:
                out |= self.data_file(fp)
        return out

    # -- live proxies

    def proxy_plan(self, keys: set[int]) -> dict[int, list[int]]:
        """proxy test -> expensive tests it may stand in for in this commit.

        A proxy stands in only if every changed place that reaches the
        expensive test also reaches the proxy, so the proxy runs the change."""
        plan: dict[int, list[int]] = {}
        idx = {t: i for i, t in enumerate(self.m.tests)}
        for exp_id, proxy_id in PROXIES.items():
            x, p = idx.get(exp_id), idx.get(proxy_id)
            if x is None or p is None or x not in keys:
                continue
            if not self.locs or any(x in loc and p not in loc for loc in self.locs):
                continue
            plan.setdefault(p, []).append(x)
        return plan

    def run_live(self, node_ids: list[str]) -> set[str] | None:
        """Run tests on the patched tree; return the ids that passed, or None."""
        import shutil
        import subprocess
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="brt", dir="/tmp"))
        try:
            build_tree(SRC, root, self.patched, self.deleted)
            out = root / ".live.json"
            env = dict(os.environ, PYTHONPATH=f"{root}{os.pathsep}{HERE}", SKIP_INTEGRATION="0",
                       BR_LIVE_OUT=str(out), PYTHONDONTWRITEBYTECODE="1", HOME="/tmp")
            cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "live_plugin",
                   "-o", "addopts=", *node_ids]
            r = subprocess.run(cmd, cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               timeout=LIVE_TIMEOUT)
            if not out.exists():
                log("live run wrote nothing:", r.stderr[-400:].decode(errors="replace"))
                return None
            res = json.loads(out.read_text())
            return {n for n in node_ids if n in res["seen"] and n not in res["failed"]} if not any(
                f == "<collect>" or "::" not in f for f in res["failed"]) else set()
        except Exception as e:  # noqa: BLE001
            log("live run failed:", type(e).__name__, e)
            return None
        finally:
            shutil.rmtree(root, ignore_errors=True)


def fast_live(sel: "Selector", m: "Map", commit: str, chosen: list[dict]) -> list[dict]:
    """Run `chosen` in-process on the patched code if that is cheaper than
    listing it; return the tests to list."""
    files = set(sel.patched) | sel.deleted
    if sel.deleted or not files or not all(f.startswith("sqlglot/") and f.endswith(".py") for f in files):
        return chosen
    # Code that runs at import built data the swap cannot rebuild: such
    # changes need the import check to show no derived state differs.
    at_import = any(lines & m.import_lines.get(path, set()) for path, lines in sel.touched.items())
    live = [t for t in chosen if t["id"] != TPCDS_EXEC]
    rest = [t for t in chosen if t["id"] == TPCDS_EXEC]
    # check_file tests run only the fixture entries that reach the change.
    masks = {}
    for t in live:
        if ENTRY_LIVE and t["file"] == "tests/test_optimizer.py":
            mk = entry_mask(sel, t["id"])
            if mk is not None:
                masks[t["id"]] = mk
    if not FAST_HEAVY and any(t["file"] in HEAVY_TEST_FILES and t["id"] not in masks for t in live):
        log(f"{commit}: fast skip: unmasked heavy " + ",".join(t["id"].split("::")[-1] for t in live
                                                                if t["file"] in HEAVY_TEST_FILES and t["id"] not in masks)[:200])
        return chosen
    if not FAST_DATA and (at_import or class_data_change(sel, files)):
        return chosen
    secs = sum(t["seconds"] for t in live)
    listing = pack_cost([(t["number"], t["seconds"]) for t in chosen])
    after = pack_cost([(t["number"], t["seconds"]) for t in rest])
    # Measured CPU per recorded second: dialect tests ~0.55, optimizer tests
    # ~1.4 (recorded with a parallel pool, run inline here), executor tests
    # ~0.8 plus ~1.0 s of TPC data loading in setUpClass; importing either
    # heavy module costs ~0.25 s once per batch.
    by = {}
    for t in live:
        sec = t["seconds"]
        if t["id"] in masks:
            total = max((v for per in _ENTRIES[t["id"]]["lines"].values() for v in per.values()), default=1)
            sec *= max(bin(masks[t["id"]]).count("1"), 0) / max(total.bit_length() - 1, 1)
        by[t["file"]] = by.get(t["file"], 0.0) + sec
    opt = by.pop("tests/test_optimizer.py", 0.0)
    exe = by.pop("tests/test_executor.py", 0.0)
    mem_dialects = affected_dialects(sel) if any(t["id"] == MEMORY_TEST for t in live) else None
    if mem_dialects is not None:
        fams = families()
        full = by.get("tests/test_generator.py", 0.0)
        mem = next(t["seconds"] for t in live if t["id"] == MEMORY_TEST)
        by["tests/test_generator.py"] = full - mem + mem * max(len(mem_dialects), 1) / max(len(fams), 1)
    est = 0.03 + 0.55 * sum(by.values()) + 1.4 * opt + 0.8 * exe + (1.0 if exe else 0.0)
    import sys as _sys
    for f, mod in (("tests/test_optimizer.py", "tests.test_optimizer"), ("tests/test_executor.py", "tests.test_executor")):
        if (opt if "optimizer" in f else exe) and mod not in _sys.modules:
            est += 0.25
    needs_check = at_import or class_data_change(sel, files)
    if needs_check and getattr(sel, "_import_result", _UNSET) is _UNSET:
        est += 0.75  # the live import check (~0.7 s)
    if not live or secs > FAST_MAX_SECONDS or SCORER_CPU * est >= 0.8 * (listing - after):
        log(f"{commit}: fast skip: cost est {est:.2f} secs {secs:.1f} vs listing {listing - after:.2f}")
        return chosen
    import resource
    import fastlive

    c0 = resource.getrusage(resource.RUSAGE_SELF)
    k0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        fastlive.setup(str(SRC))
        c_setup = resource.getrusage(resource.RUSAGE_SELF)
        swaps = []
        data = []
        for f in files:
            old = sel.text(f)
            if old is None:
                return chosen
            ops = fastlive.data_ops(old, sel.patched[f])
            if ops is None:
                log(f"{commit}: fast live: change outside function bodies and class data")
                return chosen
            sw = fastlive.swaps_for(f, old, sel.patched[f], allow_bodies=bool(ops))
            if sw is None:
                log(f"{commit}: fast live: change not confined to function bodies")
                return chosen
            swaps += sw
            if ops:
                data.append((module_name(f), ops))
        if data or at_import:
            # Data built at import from the changed attributes (tries, copies
            # under other names) cannot be patched in: it must not differ.
            changes = import_changes(sel, deep=True)
            attrs = {op[1] for _, ops in data for op in ops}
            attrs |= {d for a in list(attrs) for d in fastlive.DERIVED.get(a, ())}
            deep = getattr(sel, "import_deep", None) if changes is not None else None
            if deep is None or [c for c in deep if c.rsplit(".", 1)[-1] not in attrs]:
                log(f"{commit}: fast live: class data feeds import-time state")
                return chosen
        if not fastlive.import_tests({t["file"] for t in live}):
            log(f"{commit}: fast live: test import failed")
            return chosen
        c_imp = resource.getrusage(resource.RUSAGE_SELF)
        passed = fastlive.run([t["id"] for t in live], swaps, LIVE_TIMEOUT, data, defer=HEAVY_TEST_FILES,
                              masks=masks, dialects=mem_dialects)
    except Exception as e:  # noqa: BLE001
        log(f"{commit}: fast live failed: {type(e).__name__}: {e}")
        return chosen
    c1 = resource.getrusage(resource.RUSAGE_SELF)
    k1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    _u = lambda a, b: b.ru_utime + b.ru_stime - a.ru_utime - a.ru_stime
    log(f"{commit}: est {est:.2f} listing {listing - after:.2f} masks "
        + ",".join(f"{k.split('::')[-1]}:{bin(v).count('1')}" for k, v in masks.items()))
    log(f"{commit}: fast cpu setup {_u(c0, c_setup):.2f} imports {_u(c_setup, c_imp):.2f} "
        f"swap+fork {_u(c_imp, c1):.2f} child {_u(k0, k1):.2f}")
    if passed is None:
        log(f"{commit}: fast live: run failed")
        return chosen
    kept = [t for t in live if t["id"] not in passed] + rest
    log(f"{commit}: fast live ran {len(live)} ({secs:.2f}s recorded), {len(live) - len(passed)} failed, "
        f"parent cpu {c1.ru_utime + c1.ru_stime - c0.ru_utime - c0.ru_stime:.2f}")
    return kept


_ENTRIES = None
# Off by default: neutral on the labelled set and sample (the big savings
# are blocked by setUp-level code, executor tests or data edits).
ENTRY_LIVE = os.environ.get("BR_ENTRY_LIVE", "1") == "1"
ENTRIES = Path(os.environ.get("BR_ENTRIES", HERE / "maps" / "entries.json.gz"))


def entry_mask(sel: "Selector", test: str) -> int | None:
    """Fixture entries of a check_file test that run a changed function, as a
    bitmask (entries numbered from 1), or None when every entry must run: a
    change outside a function, outside sqlglot, or in code the test runs
    outside its entry loop."""
    global _ENTRIES
    if _ENTRIES is None:
        try:
            with gzip.open(ENTRIES, "rt") as fh:
                _ENTRIES = json.load(fh)
        except OSError:
            _ENTRIES = {}
    e = _ENTRIES.get(test)
    if e is None or not sel.touched:
        return None
    mask = 0
    for path, lines in sel.touched.items():
        if not path.startswith("sqlglot/"):
            return None
        tree = sel.tree(path)
        per = e["lines"].get(path, {})
        outside = set(e["outside"].get(path, ()))
        setup = set(e.get("setup", {}).get(path, ()))
        for ln in lines:
            ch = chain(tree, ln)
            funcs = [n for n in ch if isinstance(n, FUNC)]
            if not funcs:
                if [n for n in ch if isinstance(n, ast.stmt)]:
                    # Class or module data: only entries whose dialect is
                    # built from this file can see it.
                    dm = dialect_entries(test, path)
                    if dm is None:
                        return None
                    mask |= dm
                continue
            f = funcs[-1]
            span = range(start(f), (f.end_lineno or f.lineno) + 1)
            if any(x in setup for x in span):
                return None  # setUp builds data every entry consumes (self.schema)
            if any(x in outside for x in span) and not data_pure_test(sel, test):
                return None  # outside code may compute data the entries consume
            for x in span:
                mask |= per.get(str(x), 0)
    return mask


_PURE: dict[str, bool] = {}


def data_pure_test(sel: "Selector", test: str) -> bool:
    """True if a check_file-style test passes only literal data into its
    fixture entries: every local assignment in its body is a literal and
    every check_file argument is a literal, such a local, or a reference
    (e.g. optimizer.optimize). Its other code then runs in full in a scoped
    run and cannot change what unaffected entries compute."""
    if test in _PURE:
        return _PURE[test]
    ok = False
    try:
        rel, cls, meth = test.split("::")
        tree = sel.tree(rel)
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
        fn = next(n for n in node.body if isinstance(n, FUNC) and n.name == meth)

        def literal(e) -> bool:
            try:
                ast.literal_eval(e)
                return True
            except Exception:  # noqa: BLE001
                return False

        locals_ok = set()
        ok = True
        for n in ast.walk(fn):
            if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if all(isinstance(t, ast.Name) for t in targets) and literal(n.value):
                    locals_ok |= {t.id for t in targets}
                elif any(isinstance(t, ast.Name) for t in targets):
                    # a computed local: fine only if not passed to check_file
                    pass
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "check_file":
                for a in list(n.args) + [k.value for k in n.keywords]:
                    if not (literal(a) or isinstance(a, ast.Attribute)
                            or (isinstance(a, ast.Name) and a.id in locals_ok)):
                        ok = False
        # Own fixture loops (no check_file): every local read in the loop
        # body that is assigned before it must be a literal.
        for n in ast.walk(fn):
            if isinstance(n, ast.For) and "load_sql_fixture" in ast.dump(n.iter):
                assigned_before = {t.id for st in fn.body if st.lineno < n.lineno and isinstance(st, ast.Assign)
                                   for t in st.targets if isinstance(t, ast.Name)}
                used = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
                if (used & assigned_before) - locals_ok:
                    ok = False
    except Exception:  # noqa: BLE001
        ok = False
    _PURE[test] = ok
    return ok


DIALECT_SPECIFIC = ("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")


_FAMILIES: dict[str, set[str]] | None = None


def families() -> dict[str, set[str]]:
    """Dialect -> files its behaviour can come from: the files of its classes
    (from the map) plus every dialect-specific module those files import,
    transitively (dialects compose others: Athena uses Hive and Trino)."""
    global _FAMILIES
    if _FAMILIES is not None:
        return _FAMILIES
    base = (_ENTRIES or {}).get("__families__", {})
    imports: dict[str, set[str]] = {}

    def deps(rel: str) -> set[str]:
        if rel in imports:
            return imports[rel]
        imports[rel] = set()
        try:
            tree = ast.parse((SRC / rel).read_text("utf-8"))
        except (OSError, SyntaxError):
            return imports[rel]
        found = set()
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("sqlglot"):
                mods = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            elif isinstance(n, ast.Import):
                mods = [a.name for a in n.names if a.name.startswith("sqlglot")]
            for mname in mods:
                for cand in (mname.replace(".", "/") + ".py", mname.replace(".", "/") + "/__init__.py"):
                    if (SRC / cand).is_file() and cand.startswith(DIALECT_SPECIFIC) \
                            and os.path.basename(cand) not in ("__init__.py", "dialect.py", "base.py"):
                        found.add(cand)
        imports[rel] = found
        for f in list(found):
            imports[rel] |= deps(f)
        return imports[rel]

    out = {}
    for d, files in base.items():
        fam = set(files)
        for f in list(files):
            fam |= deps(f)
        out[d] = fam
    _FAMILIES = out
    return out


def dialect_entries(test: str, path: str) -> int | None:
    """Entries of `test` whose dialect's classes come from `path` (a
    dialect-specific module; typing/<x>.py counts as dialects/<x>.py), or
    None for shared modules."""
    base = os.path.basename(path)
    if not path.startswith(DIALECT_SPECIFIC) or base in ("__init__.py", "dialect.py", "base.py"):
        return None
    fams = families()
    ents = _ENTRIES.get("__dialects__", {}).get(test)
    if not fams or ents is None:
        return None
    probe = "sqlglot/dialects/" + base if path.startswith("sqlglot/typing/") else path
    hit = {d for d, files in fams.items() if probe in files}
    if path.startswith("sqlglot/typing/"):
        hit |= {d for d, files in fams.items() if path in files}
    mask = 0
    for i, ds in ents.items():
        if any(d in hit for d in ds):
            mask |= 1 << int(i)
        elif any(d not in fams for d in ds):
            return None  # an entry dialect we cannot place
    return mask


MEMORY_TEST = "tests/test_generator.py::TestGenerator::test_generator_memory"


def affected_dialects(sel: "Selector") -> list[str] | None:
    """Dialects (Dialects values) built from the changed files, or None when a
    change is in shared code (every dialect)."""
    if _ENTRIES is None:
        entry_mask(sel, "")  # loads the map
    fams = families() if _ENTRIES else {}
    if not fams or not sel.touched:
        return None
    hit: set[str] = set()
    for path in set(sel.patched) | sel.deleted:
        if not path.endswith(".py"):
            continue
        base = os.path.basename(path)
        if not path.startswith(DIALECT_SPECIFIC) or base in ("__init__.py", "dialect.py", "base.py"):
            return None
        probe = "sqlglot/dialects/" + base if path.startswith("sqlglot/typing/") else path
        hit |= {d for d, files in fams.items() if probe in files or path in files}
    return sorted(hit)


def class_data_change(sel: "Selector", files: set[str]) -> bool:
    """True if some file's change is outside function bodies (cheap AST test)."""
    import fastlive

    for f in files:
        old = sel.text(f)
        if old is None:
            return True
        ops = fastlive.data_ops(old, sel.patched[f])
        if ops is None or ops:
            return True
    return False


def fixture_pairs(text: str) -> list[tuple]:
    """tests/helpers.load_sql_fixture_pairs on a text."""
    filtered = "\n".join(l for l in text.splitlines() if l and not l.startswith("--"))
    st = filtered.split(";")
    return [(st[i].strip(), st[i + 1].strip()) for i in range(0, len(st), 2) if i + 1 < len(st)]


def fixture_live(sel: "Selector", m: "Map", commit: str, chosen: list[dict]) -> list[dict]:
    """A commit that only edits optimizer fixture entries: run the optimizer
    tests that read them on just the changed entries, with the patched text."""
    import fastlive

    files = set(sel.patched) | sel.deleted
    if sel.deleted or not files or not all(
            f.startswith("tests/fixtures/optimizer/") and f.endswith(".sql") for f in files):
        return chosen
    if not all(t["file"] == "tests/test_optimizer.py" for t in chosen) or not chosen:
        return chosen
    texts, mask = {}, 0
    for f in files:
        old, new = sel.text(f), sel.patched.get(f)
        if old is None or new is None:
            return chosen
        a, b = fixture_pairs(old), fixture_pairs(new)
        if len(a) != len(b):
            return chosen  # entries added or removed: numbering shifts
        for i, (x, y) in enumerate(zip(a, b), start=1):
            if x != y:
                mask |= 1 << i
        texts[f[len("tests/fixtures/"):]] = new
    if not mask:
        return []
    listing = pack_cost([(t["number"], t["seconds"]) for t in chosen])
    est = 0.3 + 0.05 * bin(mask).count("1")
    if SCORER_CPU * est >= 0.8 * listing:
        return chosen
    try:
        fastlive.setup(str(SRC))
        if not fastlive.import_tests({"tests/test_optimizer.py"}):
            return chosen
        masks = {t["id"]: mask for t in chosen}
        passed = fastlive.run([t["id"] for t in chosen], [], LIVE_TIMEOUT, masks=masks, fixtures=texts)
    except Exception as e:  # noqa: BLE001
        log(f"{commit}: fixture live failed: {type(e).__name__}: {e}")
        return chosen
    if passed is None:
        return chosen
    log(f"{commit}: fixture live ran {len(chosen)} on {bin(mask).count('1')} entries, "
        f"{len(chosen) - len(passed)} failed")
    return [t for t in chosen if t["id"] not in passed]


def pack_cost(tests: list[tuple[int, float]]) -> float:
    workers = pack(tests)
    if not workers:
        return 0.0
    sec = dict(tests)
    return max(sum(sec[n] for n in w) for w in workers) + WORKER_COST * len(workers)


def tpcds_queries(sel: "Selector") -> int | None:
    """Bitmask of TPC-DS queries whose processing runs a function the patch
    changes, or None if some change is outside a function (then every query
    may be affected)."""
    global _tpcds_lines
    if _tpcds_lines is None:
        with gzip.open(TPCDS_LINES, "rt") as fh:
            _tpcds_lines = json.load(fh)
    mask = 0
    for path, lines in sel.touched.items():
        tree = sel.tree(path)
        per = _tpcds_lines.get(path, {})
        for ln in lines:
            ch = chain(tree, ln)
            funcs = [n for n in ch if isinstance(n, FUNC)]
            if not funcs:
                if not [n for n in ch if isinstance(n, ast.stmt)]:
                    continue  # outside any statement: blank, comment
                return None
            f = funcs[-1]
            for x in range(start(f), (f.end_lineno or f.lineno) + 1):
                mask |= per.get(str(x), 0)
    return mask


def fingerprint_diff(sel: "Selector", files: set[str]) -> list[int] | None:
    """TPC-DS queries (indices among executed ones) whose transpilation,
    optimized form, plan or generated executor code the patch changes, up to
    DIFF_MAX + 1 of them; None if the check cannot be made."""
    import shutil
    import subprocess
    import tempfile

    if not TPCDS_BASE.exists() or any(f.startswith(DATA_TIME) or not f.endswith(".py") for f in files):
        return None
    try:
        queries = tpcds_queries(sel)
    except Exception as e:  # noqa: BLE001
        log("tpcds query map failed:", type(e).__name__, e)
        queries = None
    if queries == 0:
        return []  # no TPC-DS query runs the changed code
    root = Path(tempfile.mkdtemp(prefix="brf", dir="/tmp"))
    try:
        build_tree(SRC, root, sel.patched, sel.deleted)
        env = dict(os.environ, PYTHONPATH=str(root), PYTHONHASHSEED="0", SKIP_INTEGRATION="0",
                   PYTHONDONTWRITEBYTECODE="1", HOME="/tmp")
        if queries is not None:
            env["BR_QUERIES"] = ",".join(str(i) for i in range(queries.bit_length()) if queries >> i & 1)
        env["BR_DIFF_MAX"] = str(DIFF_MAX)
        r = subprocess.run([sys.executable, str(HERE / "tpc_fp.py"), "diff", str(TPCDS_BASE)], cwd=root, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=LIVE_TIMEOUT)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout.decode().strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        log("fingerprint failed:", type(e).__name__, e)
        return None
    finally:
        shutil.rmtree(root, ignore_errors=True)


_UNSET = object()
_REVDEPS = None


def deep_modules(sel: "Selector") -> list[str]:
    """Patched modules plus every module that imports them (transitively):
    copies of their data and state derived from it live there."""
    global _REVDEPS
    if _REVDEPS is None:
        try:
            _REVDEPS = json.load(open(HERE / "revdeps.json"))
        except OSError:
            _REVDEPS = {}
    mods = {module_name(f) for f in set(sel.patched) | sel.deleted if f.endswith(".py")}
    out = set(mods)
    for mname in mods:
        out |= set(_REVDEPS.get(mname, ()))
    return sorted(out)


def import_changes(sel: "Selector", deep: bool = False) -> list[str] | None:
    """Import patched sqlglot, load every dialect and compare the SQL objects
    and dict shapes built at import with the pinned commit. Returns the
    "module:Class.attr" entries that differ, or None if the import fails."""
    import shutil
    import subprocess
    import tempfile

    if not IMPORT_BASE.exists():
        return None
    if getattr(sel, "_import_result", _UNSET) is not _UNSET:
        return sel._import_result  # already checked for this commit
    sel._import_result = None
    sel.import_deep = None
    root = Path(tempfile.mkdtemp(prefix="bri", dir="/tmp"))
    try:
        build_tree(SRC, root, sel.patched, sel.deleted)
        env = dict(os.environ, PYTHONPATH=str(root), PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", HOME="/tmp",
                   BR_IMPORT_DEEP="1",
                   BR_DEEP_MODULES=",".join(deep_modules(sel)))
        r = subprocess.run([sys.executable, str(HERE / "import_fp.py"), "check", str(IMPORT_BASE)], cwd=root,
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=LIVE_TIMEOUT)
        if r.returncode != 0:
            return None
        lines = [l for l in r.stdout.decode().split("\n") if l]
        sel.import_deep = [l[5:] for l in lines if l.startswith("deep|")]
        sel._import_result = [l for l in lines if not l.startswith("deep|")]
        return sel._import_result
    except Exception as e:  # noqa: BLE001
        log("import check failed:", type(e).__name__, e)
        return None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def import_change_users(m: "Map", changes: list[str]) -> set[int] | None:
    """Tests that use what changed at import: users of each changed class."""
    out: set[int] = set()
    for entry in changes:
        mod, _, rest = entry.partition(":")
        owner = rest.rsplit(".", 1)[0]
        if not owner:  # a module-level object: every importer of the module
            path = mod.replace(".", "/") + ".py"
            if path not in m.lines:
                path = mod.replace(".", "/") + "/__init__.py"
            out |= m.importers(path)
            continue
        users = m.users(f"{mod}:{owner}")
        if users is None:
            return None
        out |= users
    return out


def tpcds_queries_pass(sel: "Selector", commit: str, diff: list[int]) -> bool:
    """Run the TPC-DS executor test on only the queries whose fingerprint
    changed, with the patch swapped in; every other query computes exactly
    what it did at the pinned commit, so the test passes iff these do."""
    import fastlive

    files = set(sel.patched) | sel.deleted
    try:
        if sel.deleted or not all(f.startswith("sqlglot/") and f.endswith(".py") for f in files):
            return False
        fastlive.setup(str(SRC))
        swaps, data = [], []
        for f in files:
            old = sel.text(f)
            ops = fastlive.data_ops(old, sel.patched[f]) if old is not None else None
            if ops is None:
                return False
            sw = fastlive.swaps_for(f, old, sel.patched[f], allow_bodies=bool(ops))
            if sw is None:
                return False
            swaps += sw
            if ops:
                data.append((module_name(f), ops))
        if data:
            changes = import_changes(sel, deep=True)
            attrs = {op[1] for _, ops in data for op in ops}
            attrs |= {d for a in list(attrs) for d in fastlive.DERIVED.get(a, ())}
            deep = getattr(sel, "import_deep", None) if changes is not None else None
            if deep is None or [c for c in deep if c.rsplit(".", 1)[-1] not in attrs]:
                return False
        if not fastlive.import_tests({"tests/test_executor.py"}):
            return False
        passed = fastlive.run([TPCDS_EXEC], swaps, LIVE_TIMEOUT, data, tpcds=diff)
        return passed is not None and TPCDS_EXEC in passed
    except Exception as e:  # noqa: BLE001
        log(f"{commit}: tpcds verify failed: {type(e).__name__}: {e}")
        return False


def build_tree(src: Path, dst: Path, patched: dict[str, str], deleted: set[str]) -> None:
    """Mirror src into dst with symlinks, writing patched files for real."""
    touched = set(patched) | set(deleted)
    dirs = {str(Path(p).parent) for p in touched}
    need = set()
    for d in dirs:
        parts = Path(d).parts
        for i in range(len(parts) + 1):
            need.add(str(Path(*parts[:i])) if i else ".")

    def walk(rel: str):
        s = src / rel if rel != "." else src
        d = dst / rel if rel != "." else dst
        d.mkdir(exist_ok=True)
        names = set(os.listdir(s)) if s.is_dir() else set()
        prefix = "" if rel == "." else rel + "/"
        names |= {p[len(prefix):].split("/")[0] for p in patched if p.startswith(prefix)}
        for name in names:
            r = prefix + name
            if r in deleted and r not in patched:
                continue
            if r in patched:
                (d / name).write_text(patched[r], "utf-8")
            elif r in need or (r.rstrip("/") in need):
                walk(r)
            else:
                os.symlink(s / name, d / name)

    walk(".")


# ------------------------------------------------------------------------ output


def pack(tests: list[tuple[int, float]]) -> list[list[int]]:
    if not tests:
        return []
    order = sorted(tests, key=lambda t: (-t[1], t[0]))
    best, best_cost = None, None
    for k in range(1, MAX_WORKERS + 1):
        if k > len(order):
            break
        workers = [[] for _ in range(k)]
        loads = [0.0] * k
        for num, sec in order:
            i = loads.index(min(loads))
            workers[i].append(num)
            loads[i] += sec
        cost = max(loads) + WORKER_COST * k
        if best_cost is None or cost < best_cost - 1e-9:
            best, best_cost = workers, cost
    return best


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text("utf-8"))["tests"]
    by_id = {t["id"]: t for t in tests}
    m = Map(MAP)
    sel = Selector(m)
    every = sorted(by_id.values(), key=lambda t: t["number"])
    import resource as _res
    def _cpu():
        a, b = _res.getrusage(_res.RUSAGE_SELF), _res.getrusage(_res.RUSAGE_CHILDREN)
        return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime
    for patch in sorted((DATA / "commits").glob("*.patch")):
        commit = patch.stem
        _c0 = _cpu()
        try:
            text = patch.read_text("utf-8", errors="replace")
            sel.import_ok = False
            try:
                keys = sel.commit(text)
            except Everything as e:
                if not str(e).startswith("import-time") or not live_allowed():
                    raise
                changes = import_changes(sel)
                extra = import_change_users(m, changes) if changes is not None else None
                if extra is None:
                    raise
                log(f"{commit}: import-time changes {changes[:5]}{'...' if len(changes) > 5 else ''}")
                sel.import_ok = True
                keys = sel.commit(text) | extra
            if sel.needs_import_check:
                changes = import_changes(sel) if live_allowed() else None
                if changes is None:
                    raise Everything("import check failed or unavailable for a dict read at import")
                # The narrowed dict itself differs in every class that copies
                # it ({**Parent.FUNCTIONS, ...}); its lookups are already
                # covered by key. Only other import-time state is derived.
                changes = [c for c in changes if c.rsplit(".", 1)[-1] not in sel.narrowed_attrs]
                extra = import_change_users(m, changes)
                if extra is None:
                    raise Everything("import-time changes to an untraced class")
                if changes:
                    log(f"{commit}: dict entry also changes import-time state {changes[:5]}")
                keys |= extra
            plan = sel.proxy_plan(keys) if LIVE else {}
            if plan:
                ids = [m.tests[p] for p in plan]
                passed = sel.run_live(ids)
                log(f"{commit}: proxies {ids} passed={passed}")
                for p, xs in plan.items():
                    if passed is not None and m.tests[p] in passed:
                        keys -= {p, *xs}
            tp = next((k for k in keys if m.tests[k] == TPCDS_EXEC), None)
            if FP_ON and tp is not None and live_allowed():
                files = set(sel.patched) | sel.deleted
                diff = fingerprint_diff(sel, files)
                if diff == []:
                    keys.discard(tp)
                    log(f"{commit}: TPC-DS fingerprint unchanged, dropping {TPCDS_EXEC}")
                elif diff is not None and len(diff) <= DIFF_MAX and tpcds_queries_pass(sel, commit, diff):
                    keys.discard(tp)
                    log(f"{commit}: TPC-DS {len(diff)} changed queries pass for real, dropping {TPCDS_EXEC}")
            chosen = [by_id[m.tests[k]] for k in keys if m.tests[k] in by_id]
            code_change = any(f.endswith(".py") for f in set(sel.patched) | sel.deleted)
            if FAST and chosen and live_allowed():
                if all(f.startswith("tests/fixtures/") for f in set(sel.patched) | sel.deleted):
                    chosen = fixture_live(sel, m, commit, chosen)
                else:
                    chosen = fast_live(sel, m, commit, chosen)
            heavy = any(t["file"] in HEAVY_TEST_FILES for t in chosen)
            if SMALL and chosen and code_change and not heavy and live_allowed() and sum(t["seconds"] for t in chosen) <= SMALL_MAX_SECONDS:
                import resource
                c0 = resource.getrusage(resource.RUSAGE_CHILDREN)
                passed = sel.run_live([t["id"] for t in chosen])
                c1 = resource.getrusage(resource.RUSAGE_CHILDREN)
                if passed is not None:
                    log(f"{commit}: ran {len(chosen)} live, {len(chosen) - len(passed)} failed, "
                        f"cpu {c1.ru_utime + c1.ru_stime - c0.ru_utime - c0.ru_stime:.2f} "
                        f"files {sorted({t['file'] for t in chosen})[:4]}")
                    chosen = [t for t in chosen if t["id"] not in passed]
        except Everything as e:
            log(f"{commit}: everything ({e})")
            chosen = every
        except Exception as e:  # noqa: BLE001 - never lose a commit to a bug
            log(f"{commit}: error {type(e).__name__}: {e}; selecting everything")
            chosen = every
        workers = pack([(t["number"], t["seconds"]) for t in chosen])
        print(json.dumps({"commit": commit, "workers": workers}), flush=True)
        log(f"{commit}: commit cpu {_cpu() - _c0:.2f}")
        log(f"{commit}: {len(chosen)} tests, {len(workers)} workers")


if __name__ == "__main__":
    main()
