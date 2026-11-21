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


def swaps_for(rel: str, old_text: str, new_text: str):
    """[(function objects, new code)] that turn pinned `rel` into the patch,
    or None if the change is not confined to function bodies."""
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


def run(node_ids: list[str], swaps: list, timeout: float) -> set[str] | None:
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
            loader = unittest.TestLoader()
            suite = unittest.TestSuite()
            for name in names:
                suite.addTests(loader.loadTestsFromName(name))
            planned = {t.id() for t in _iter(suite)}  # unittest empties the suite as it runs
            result = _Result()
            suite.run(result)
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
