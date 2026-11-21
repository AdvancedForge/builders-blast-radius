"""Persistent in-process test runner for small selections.

The selector imports pinned sqlglot (and the test modules it needs) once.
For a commit whose changes are confined to function bodies, it compiles the
patched module, pairs its code objects with the pinned ones, forks, swaps the
changed functions' __code__ in the child, runs the candidate tests with
unittest and reports which failed. Per commit this costs about the tests'
own time: no interpreter start, no imports.

Anything the swap cannot represent exactly (a change to a module or class
body, a decorator or default argument, a changed function shape) returns
None, and the caller keeps the static selection.
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import pickle
import sys
import time
import types
import unittest

_READY = False
_LIVE: dict[tuple[str, str, int], list[types.FunctionType]] | None = None
_SRC = ""


def _inline_pools():
    import concurrent.futures as cf

    def roundtrip(obj):
        return pickle.loads(pickle.dumps(obj))

    class InlineExecutor(cf.Executor):
        def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
            if initializer is not None:
                initializer(*roundtrip(initargs))

        def submit(self, fn, /, *args, **kwargs):
            fut = cf.Future()
            try:
                fn, args, kwargs = roundtrip((fn, args, kwargs))
                fut.set_result(roundtrip(fn(*args, **kwargs)))
            except BaseException as e:  # noqa: BLE001
                fut.set_exception(e)
            return fut

        def shutdown(self, wait=True, *, cancel_futures=False):
            pass

    cf.ProcessPoolExecutor = InlineExecutor


def setup(src: str) -> None:
    """Import pinned sqlglot and every module and dialect, once."""
    global _READY, _SRC
    if _READY:
        return
    _SRC = os.path.abspath(src)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    sys.dont_write_bytecode = True
    sys.path.insert(0, _SRC)
    os.chdir(_SRC)
    _inline_pools()
    import importlib
    import pkgutil

    import sqlglot
    from sqlglot.dialects.dialect import Dialect
    import sqlglot.dialects as d

    for m in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if m.name != "sqlglot.__main__":
            try:
                importlib.import_module(m.name)
            except Exception:  # noqa: BLE001
                pass
    for name in getattr(d, "DIALECTS", []):
        try:
            Dialect.get_or_raise(name.lower())
        except Exception:  # noqa: BLE001
            pass
    _READY = True


def import_tests(files: set[str]) -> bool:
    """Import the test modules for `files` in the parent (pinned code)."""
    import importlib

    for f in sorted(files):
        name = f[:-3].replace("/", ".")
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            print("fastlive import:", name, type(e).__name__, e, file=sys.stderr)
            return False
    return True


def _live_index() -> dict:
    global _LIVE
    if _LIVE is None:
        idx: dict = {}
        prefix = os.path.join(_SRC, "sqlglot") + os.sep
        for obj in gc.get_objects():
            if isinstance(obj, types.FunctionType):
                c = obj.__code__
                if c.co_filename.startswith(prefix):
                    idx.setdefault((c.co_filename, c.co_qualname, c.co_firstlineno), []).append(obj)
        _LIVE = idx
    return _LIVE


def _codes(code: types.CodeType, parent: int, out: list) -> None:
    me = len(out)
    out.append((code, parent))
    for k in code.co_consts:
        if isinstance(k, types.CodeType):
            _codes(k, me, out)


def _own(code: types.CodeType):
    """What a code object does, apart from nested code and line numbers."""
    consts = tuple(("<code>", k.co_qualname) if isinstance(k, types.CodeType) else (type(k).__name__, repr(k))
                   for k in code.co_consts)
    return (code.co_code, consts, code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
            code.co_flags, code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_exceptiontable)


def data_ops(old_text: str, new_text: str):
    """Class-level data changes between two versions of a module, as
    [(class path, attr, op, key source, value source)], or None if anything
    else changed outside function bodies. op is "set" (a constant), or
    "put"/"del" (one dict entry). Function body changes are allowed here;
    swaps_for handles them."""
    import ast

    try:
        a, b = ast.parse(old_text), ast.parse(new_text)
    except SyntaxError:
        return None
    seg_new = new_text
    ops = []

    def same(x, y):
        return ast.dump(x) == ast.dump(y)

    def func_shape(f):
        return ast.dump(ast.Module(body=[], type_ignores=[])) + ast.dump(f.args) + "".join(
            ast.dump(d) for d in f.decorator_list) + (ast.dump(f.returns) if f.returns else "") + f.name

    def body(xs, ys, path):
        if len(xs) != len(ys):
            return False
        for x, y in zip(xs, ys):
            if same(x, y):
                continue
            if type(x) is not type(y):
                return False
            if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if func_shape(x) != func_shape(y):
                    return False  # signature, defaults or decorators: evaluated at import
                continue
            if isinstance(x, ast.ClassDef):
                if path is None and False:
                    return False
                if x.name != y.name or not all(same(p, q) for p, q in zip(x.bases + x.keywords + x.decorator_list,
                                                                        y.bases + y.keywords + y.decorator_list)) \
                        or len(x.bases + x.keywords + x.decorator_list) != len(y.bases + y.keywords + y.decorator_list):
                    return False
                if not body(x.body, y.body, (path or ()) + (x.name,)):
                    return False
                continue
            if path and isinstance(x, (ast.Assign, ast.AnnAssign)):
                tx = x.targets if isinstance(x, ast.Assign) else [x.target]
                ty = y.targets if isinstance(y, ast.Assign) else [y.target]
                if len(tx) != 1 or len(ty) != 1 or not same(tx[0], ty[0]) or not isinstance(tx[0], ast.Name):
                    return False
                if x.value is None or y.value is None:
                    return False
                attr = tx[0].id
                if isinstance(x.value, ast.Dict) and isinstance(y.value, ast.Dict):
                    if not dict_ops(x.value, y.value, path, attr):
                        return False
                else:
                    ops.append((path, attr, "set", None, ast.get_source_segment(seg_new, y.value)))
                continue
            return False
        return True

    def dict_ops(dx, dy, path, attr):
        def entries(d):
            out = {}
            for k, v in zip(d.keys, d.values):
                if k is None:
                    out[("**", ast.dump(v))] = None
                    continue
                if not (isinstance(k, ast.Constant) or isinstance(k, ast.Attribute)):
                    return None
                out[ast.dump(k)] = (k, v)
            return out

        ex, ey = entries(dx), entries(dy)
        if ex is None or ey is None:
            return False
        if {k for k in ex if isinstance(k, tuple)} != {k for k in ey if isinstance(k, tuple)}:
            return False  # a ** splat changed
        keys = []
        for k in ex.keys() | ey.keys():
            if isinstance(k, tuple):
                continue
            if k in ex and k not in ey:
                keys.append(ast.get_source_segment(old_text, ex[k][0]))
            elif k in ey and k in ex and isinstance(ex[k][1], ast.Lambda) and isinstance(ey[k][1], ast.Lambda) \
                    and same(ex[k][1].args, ey[k][1].args):
                continue  # a lambda body: a code swap, not data
            elif k in ey and (k not in ex or not same(ex[k][1], ey[k][1])):
                keys.append(ast.get_source_segment(seg_new, ey[k][0]))
        if keys:
            # The whole new literal is evaluated in the child: a removed entry
            # may uncover a value from an earlier ** splat.
            ops.append((path, attr, "dict", keys, ast.get_source_segment(seg_new, dy)))
        return True

    if not body(a.body, b.body, None):
        return None
    return ops


def _apply_data(module, ops):
    """In the child: apply data ops to the live classes (and to subclass
    copies of a changed dict that still hold the old entry)."""
    for path, attr, op, ksrc, vsrc in ops:
        cls = module
        for name in path:
            cls = getattr(cls, name)
        ns = dict(vars(module))
        ns.update({k: v for k, v in vars(cls).items()})
        if op == "set":
            setattr(cls, attr, eval(vsrc, ns))
            continue
        d = cls.__dict__.get(attr)
        if type(d) is not dict:
            raise RuntimeError(f"{attr} is not a plain dict")
        fresh = eval(vsrc, ns)
        for key_src in ksrc:
            key = eval(key_src, ns)
            before = dict(d)
            old = before.get(key, _MISSING)
            new = fresh.get(key, _MISSING)
            targets = [d]
            stack = list(cls.__subclasses__())
            seen = set()
            while stack:
                sc = stack.pop()
                if id(sc) in seen:
                    continue
                seen.add(id(sc))
                stack.extend(sc.__subclasses__())
                sd = sc.__dict__.get(attr)
                if type(sd) is dict and sd is not d:
                    if old is not _MISSING and sd.get(key, _MISSING) is old:
                        targets.append(sd)
                    elif old is _MISSING and key not in sd and set(before) <= set(sd):
                        targets.append(sd)  # a {**Parent.X, ...} copy gains the new key
            for t in targets:
                if new is _MISSING:
                    t.pop(key, None)
                else:
                    t[key] = new


_MISSING = object()


def swaps_for(rel: str, old_text: str, new_text: str, allow_bodies: bool = False):
    """[(function objects, new code)] that turn pinned `rel` into the patch,
    or None if the change is not confined to function bodies (class bodies
    may differ when allow_bodies is set: data_ops covers them)."""
    path = os.path.join(_SRC, rel)
    try:
        old = compile(old_text, path, "exec", dont_inherit=True)
        new = compile(new_text, path, "exec", dont_inherit=True)
    except SyntaxError:
        return None
    a: list = []
    b: list = []
    _codes(old, -1, a)
    _codes(new, -1, b)
    if len(a) != len(b):
        return None
    changed = set()
    for i, ((oc, op), (nc, np_)) in enumerate(zip(a, b)):
        if op != np_ or oc.co_qualname != nc.co_qualname:
            return None
        if _own(oc) != _own(nc):
            if not (oc.co_flags & inspect.CO_NEWLOCALS):
                if allow_bodies and op >= 0:
                    continue  # a class body (not the module): data_ops applies it
                return None  # a module or class body: data built at import
            changed.add(i)
    # A function embeds its nested functions' code: swap every enclosing function too.
    todo = set(changed)
    for i in changed:
        p = a[i][1]
        while p >= 0 and a[p][0].co_flags & inspect.CO_NEWLOCALS:
            todo.add(p)
            p = a[p][1]
    live = _live_index()
    out = []
    for i in sorted(todo):
        oc, nc = a[i][0], b[i][0]
        funcs = live.get((path, oc.co_qualname, oc.co_firstlineno), [])
        for f in funcs:
            if len(f.__code__.co_freevars) != len(nc.co_freevars):
                return None
        out.append((funcs, nc))
    return out


class _Result(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.failed: set[str] = set()

    def _mark(self, test):
        tid = getattr(test, "id", lambda: str(test))()
        self.failed.add(tid)

    def addError(self, test, err):
        self._mark(test)

    def addFailure(self, test, err):
        self._mark(test)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self._mark(test)

    def addUnexpectedSuccess(self, test):
        self._mark(test)


def _to_unittest(node_id: str) -> str:
    f, cls, meth = node_id.split("::")
    return f"{f[:-3].replace('/', '.')}.{cls}.{meth}"


def _restrict_entries(masks: dict[str, int]) -> None:
    """In the child: fixture loops in TestOptimizer tests (check_file or their
    own loops over load_sql_fixture_pairs) see only the entries in the
    running test's mask (entries numbered from 1 per file)."""
    import tests.test_optimizer as TO

    cur = [None]
    orig_setup = TO.TestOptimizer.setUp
    orig_pairs = TO.load_sql_fixture_pairs

    def setUp(self):
        _, cls, meth = self.id().rsplit(".", 2)
        cur[0] = masks.get(f"tests/test_optimizer.py::{cls}::{meth}")
        return orig_setup(self)

    def pairs(filename):
        for i, item in enumerate(orig_pairs(filename), start=1):
            if cur[0] is None or cur[0] >> i & 1:
                yield item

    TO.TestOptimizer.setUp = setUp
    TO.load_sql_fixture_pairs = pairs


def _restrict_tpcds(queries: list[int]) -> None:
    """In the child: TestExecutor's TPC-DS test runs only these queries
    (indices among the queries it executes, as tpc_fp numbers them)."""
    import tests.test_executor as TE

    orig = TE.TestExecutor.setUpClass.__func__

    def setUpClass(cls):
        orig(cls)
        executed = [i for i, (meta, _, _) in enumerate(cls.tpcds_sqls) if meta.get("execute")]
        cls.tpcds_sqls = [cls.tpcds_sqls[executed[q]] for q in queries]

    TE.TestExecutor.setUpClass = classmethod(setUpClass)


def run(node_ids: list[str], swaps: list, timeout: float, data: list | None = None,
        defer: set[str] | None = None, masks: dict[str, int] | None = None,
        tpcds: list[int] | None = None) -> set[str] | None:
    """Fork, apply swaps, run the tests; return the node ids that passed."""
    names = {_to_unittest(n): n for n in node_ids}
    r, w = os.pipe()
    gc.collect()
    gc.freeze()
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            os.close(r)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            for funcs, nc in swaps:
                for f in funcs:
                    f.__code__ = nc
            for modname, ops in (data or []):
                _apply_data(sys.modules[modname], ops)
            if masks:
                _restrict_entries(masks)
            if tpcds is not None:
                _restrict_tpcds(tpcds)
            loader = unittest.TestLoader()
            first, second = unittest.TestSuite(), unittest.TestSuite()
            for name in names:
                nid = names[name]
                later = nid.split("::")[0] in (defer or ()) and nid not in (masks or {})
                (second if later else first).addTests(loader.loadTestsFromName(name))
            planned = {t.id() for t in _iter(first)}  # unittest empties the suite as it runs
            later = {t.id() for t in _iter(second)}
            result = _Result()
            first.run(result)
            # Once something fails a worker is paid for anyway, and listing a
            # deferred (optimizer/executor) test costs no more than running it.
            if not result.failed:
                planned |= later
                second.run(result)
            # A failure that is not a candidate test (setUpClass, a module
            # import) fails every candidate of its class, or all of them.
            failed = set()
            for tid in result.failed:
                if tid in names:
                    failed.add(names[tid])
                elif "(" in tid and ")" in tid:
                    cls = tid[tid.index("(") + 1:tid.rindex(")")]
                    hit = {n for u, n in names.items() if u.startswith(cls + ".") or u.startswith(cls)}
                    failed |= hit or set(names.values())
                else:
                    failed |= set(names.values())
            ran = {names[t] for t in planned if t in names}
            os.write(w, json.dumps({"failed": sorted(failed), "ran": sorted(ran),
                                    "errors": result.testsRun}).encode())
            code = 0
        except BaseException as e:  # noqa: BLE001
            try:
                os.write(w, json.dumps({"crash": f"{type(e).__name__}: {e}"}).encode())
            except Exception:  # noqa: BLE001
                pass
            code = 2
        finally:
            os._exit(code)
    os.close(w)
    deadline = time.monotonic() + timeout
    status = None
    while True:
        done, st = os.waitpid(pid, os.WNOHANG)
        if done:
            status = st
            break
        if time.monotonic() > deadline:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            os.close(r)
            return None
        time.sleep(0.002)
    data = b""
    while True:
        chunk = os.read(r, 65536)
        if not chunk:
            break
        data += chunk
    os.close(r)
    if not data:
        return None
    res = json.loads(data)
    if "crash" in res:
        print("fastlive child:", res["crash"], file=sys.stderr)
        return None
    return {n for n in res["ran"] if n not in set(res["failed"])}


def _iter(suite):
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from _iter(t)
        else:
            yield t
