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
MAX_WORKERS = 8
WORKER_COST = 2.0

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
        files = [tf for tf, mods in self.closure.items() if mods is None or path in mods or tf == path]
        out = set()
        for tf in files:
            out |= self.prefix(tf + "::")
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


def module_name(path: str) -> str:
    return path[:-3].replace("/", ".").removesuffix(".__init__")


EXP_NAME = re.compile(r"\bexp\.([A-Z]\w*)")


class Selector:
    def __init__(self, m: Map):
        self.m = m
        self._texts: dict[str, str] = {}
        self._trees: dict[str, ast.AST] = {}
        self._exp_users: dict[str, set[int] | None] = {}

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

    def refs(self, names: set[str], depth: int = 0) -> set[int]:
        """Tests that ran code referring to any of `names`, across sqlglot."""
        out: set[int] = set()
        if not names:
            return out
        pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(names))) + r")\b")
        for rel in self.m.lines:
            if not rel.startswith("sqlglot/"):
                continue
            text = self.text(rel)
            if text is None or not pat.search(text):
                continue
            tree = self.tree(rel)
            for i, l in enumerate(text.split("\n"), 1):
                if pat.search(l) and not l.lstrip().startswith("#"):
                    out |= self.at_line(rel, tree, i, set(), depth + 1, names_hint=True)
        return out

    # -- core: tests that can notice a change at old line `ln` of `path`

    def at_line(self, path: str, tree: ast.AST, ln: int, extra_names: set[str], depth: int = 0,
                names_hint: bool = False) -> set[int]:
        m = self.m
        ch = chain(tree, ln)
        stmts = [n for n in ch if isinstance(n, ast.stmt)]
        if not stmts:
            return set() if names_hint else m.file_any(path)
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
        if fp.new is None:  # deleted
            return m.importers(path)
        try:
            new_tree = ast.parse(new_text.encode("utf-8"))
        except (SyntaxError, ValueError):
            return m.importers(path) if fp.old else set()
        if not fp.old or old_text == "":
            return set()  # a new module nothing imports yet
        old_tree = self.tree(fp.old)
        if ast.dump(old_tree) == ast.dump(new_tree):
            return set()
        if fp.new != fp.old:
            return m.importers(fp.old)

        old_lines = old_text.split("\n")
        new_lines = new_text.split("\n")
        out: set[int] = set()
        for ln in removed:
            out |= self.at_line(path, old_tree, ln, self.key_names(old_lines[ln - 1]))
        for k, n in added:
            out |= self.added_line(path, old_tree, new_tree, k, n, self.key_names(new_lines[n - 1]), old_lines)
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
            if not fn_inside and not existing:
                users = self.class_users(path, quals)
                if users is not None:
                    keys = names
                    if keys and all(self.exp_users(x) is not None for x in keys):
                        users = users & set().union(*(self.exp_users(x) for x in keys))
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
                out |= m.s(idx)
            elif p.startswith("tests/"):
                base = os.path.basename(p)
                for tf in m.closure:
                    t = self.text(tf) or ""
                    if base in t or os.path.basename(os.path.dirname(p)) in t:
                        out |= m.prefix(tf + "::")
        return out

    def commit(self, patch_text: str) -> set[int]:
        out: set[int] = set()
        for fp in parse_patch(patch_text):
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
