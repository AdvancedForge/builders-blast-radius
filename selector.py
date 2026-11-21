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
MAX_WORKERS = 8
WORKER_COST = 2.0

LIVE_TIMEOUT = 60
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


def is_bare_string(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """A copy of `tree` without bare string statements (docstrings).

    Nothing in the suite reads docstrings: the doctests are loaded through
    unittest's load_tests, which pytest does not run."""
    import copy
    tree = copy.deepcopy(tree)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            kept = [x for x in body if not is_bare_string(x)]
            node.body = kept or [ast.Pass()]
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
        if fp.new is None:  # deleted
            return m.importers(path)
        try:
            new_tree = ast.parse(new_text.encode("utf-8"))
        except (SyntaxError, ValueError):
            return m.importers(path) if fp.old else set()
        if not fp.old or old_text == "":
            return set()  # a new module nothing imports yet
        old_tree = self.tree(fp.old)
        if ast.dump(strip_docstrings(old_tree)) == ast.dump(strip_docstrings(new_tree)):
            return set()
        if fp.new != fp.old:
            return m.importers(fp.old)

        old_lines = old_text.split("\n")
        new_lines = new_text.split("\n")
        out: set[int] = set()
        for ln in removed:
            code = old_lines[ln - 1].strip()
            if not code or code.startswith("#"):
                continue  # removing a blank or comment line changes nothing
            loc = self.at_line(path, old_tree, ln, self.key_names(old_lines[ln - 1]))
            self.locs.append(loc); out |= loc
        for k, n in added:
            loc = self.added_line(path, old_tree, new_tree, k, n, self.key_names(new_lines[n - 1]), old_lines)
            self.locs.append(loc); out |= loc
        return out

    def added_line(self, path, old_tree, new_tree, k, n, names, old_lines) -> set[int]:
        """A line inserted before old line k, at new line n."""
        nch = chain(new_tree, n)
        nstmts = [x for x in nch if isinstance(x, ast.stmt)]
        if not nstmts:
            return set()  # blank or comment
        # Old neighbours that are code: the statement just before and just after.
        out: set[int] = set()
        for cand in (k - 1, k):
            j = cand
            step = -1 if cand == k - 1 else 1
            while 1 <= j <= len(old_lines) and (not old_lines[j - 1].strip() or old_lines[j - 1].lstrip().startswith("#")):
                j += step
            if 1 <= j <= len(old_lines):
                out |= self.at_line(path, old_tree, j, names)
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
                        reads = [self.runtime_reads(x) for x in self.assigned(member)] or [None]
                        if all(r is not None for r in reads):
                            users = users & set().union(*reads)
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
    for patch in sorted((DATA / "commits").glob("*.patch")):
        commit = patch.stem
        try:
            keys = sel.commit(patch.read_text("utf-8", errors="replace"))
            plan = sel.proxy_plan(keys) if LIVE else {}
            if plan:
                ids = [m.tests[p] for p in plan]
                passed = sel.run_live(ids)
                log(f"{commit}: proxies {ids} passed={passed}")
                for p, xs in plan.items():
                    if passed is not None and m.tests[p] in passed:
                        keys -= {p, *xs}
            chosen = [by_id[m.tests[k]] for k in keys if m.tests[k] in by_id]
        except Everything as e:
            log(f"{commit}: everything ({e})")
            chosen = every
        except Exception as e:  # noqa: BLE001 - never lose a commit to a bug
            log(f"{commit}: error {type(e).__name__}: {e}; selecting everything")
            chosen = every
        workers = pack([(t["number"], t["seconds"]) for t in chosen])
        print(json.dumps({"commit": commit, "workers": workers}), flush=True)
        log(f"{commit}: {len(chosen)} tests, {len(workers)} workers")


if __name__ == "__main__":
    main()
