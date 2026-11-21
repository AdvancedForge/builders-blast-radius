"""pytest plugin: record which non-.py repo files each test reads.

Reads during a test's call phase go to its node id. Reads during setup (where
setUpClass runs) go to the class, and reads outside any test (imports and
collection) go to the test file that triggered them, found on the stack.
"""
import json, os, sys

_key = [None]
_reads: dict = {}
_root = os.getcwd()


def _stack_test_file():
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if fn.startswith(_root + os.sep):
            rel = os.path.relpath(fn, _root)
            if rel.startswith("tests/") and os.path.basename(rel).startswith("test_"):
                return "FILE:" + rel
        f = f.f_back
    return "FILE:?"


def _hook(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    try:
        p = os.path.abspath(os.fsdecode(args[0]))
    except Exception:
        return
    if not p.startswith(_root + os.sep) or p.endswith((".py", ".pyc")) or "__pycache__" in p:
        return
    key = _key[0] or _stack_test_file()
    _reads.setdefault(key, set()).add(os.path.relpath(p, _root))


sys.addaudithook(_hook)


def pytest_runtest_setup(item):
    _key[0] = "CLASS:" + item.nodeid.rsplit("::", 1)[0]


def pytest_runtest_call(item):
    _key[0] = item.nodeid


def pytest_runtest_teardown(item):
    _key[0] = "CLASS:" + item.nodeid.rsplit("::", 1)[0]


def pytest_runtest_logfinish(nodeid, location):
    _key[0] = None


def pytest_sessionfinish(session):
    out = os.environ.get("TRACE_OUT")
    if out:
        wid = os.environ.get("PYTEST_XDIST_WORKER", "main")
        with open(f"{out}.{wid}.json", "w") as f:
            json.dump({k: sorted(v) for k, v in _reads.items()}, f)
        with open(f"{out}.types.{wid}.json", "w") as f:
            json.dump(_dump_types(), f)
        _dump_keys(out, wid)


# Optimizer and executor tests farm work out to spawned process pools, which
# coverage cannot attribute to a test. For the map run only, run that work
# in-process so every line lands on the test that caused it.
import concurrent.futures as _cf
import pickle


def _roundtrip(obj):
    # A spawned pool pickles arguments and results; sqlglot expressions
    # pickle through serde, so keep that path in the map.
    return pickle.loads(pickle.dumps(obj))


class _InlineExecutor(_cf.Executor):
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        if initializer is not None:
            initializer(*_roundtrip(initargs))

    def submit(self, fn, /, *args, **kwargs):
        fut = _cf.Future()
        try:
            fn, args, kwargs = _roundtrip((fn, args, kwargs))
            fut.set_result(_roundtrip(fn(*args, **kwargs)))
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)
        return fut

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass


_cf.ProcessPoolExecutor = _InlineExecutor


# Which sqlglot classes each test instantiates (dialects, tokenizers, parsers,
# generators, expressions and every subclass), so a change to class-level data
# such as a constant or a dict entry can be traced to the tests that use it.
_types: dict = {}


def _record_type(obj):
    key = _key[0] or _stack_test_file()
    _types.setdefault(key, set()).add(type(obj))


def _wrap_init(cls):
    orig = cls.__dict__["__init__"]

    def __init__(self, *a, **kw):
        if type(self) is cls or "__init__" not in type(self).__dict__:
            _record_type(self)
        return orig(self, *a, **kw)

    __init__.__wrapped__ = orig
    cls.__init__ = __init__


def pytest_configure(config):
    import sqlglot  # noqa: F401
    from sqlglot import tokens, parser, generator
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.expressions.core import Expr
    import sqlglot.dialects as d
    for name in getattr(d, "DIALECTS", []):
        try:
            Dialect.get_or_raise(name.lower())
        except Exception:
            pass
    import importlib, pkgutil
    for m in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        try:
            importlib.import_module(m.name)
        except Exception:
            pass
    seen = set()
    stack = [tokens.Tokenizer, parser.Parser, generator.Generator, Dialect, Expr]
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        stack.extend(c.__subclasses__())
    # A class without its own __init__ is caught by the nearest wrapped ancestor.
    for c in seen:
        if "__init__" in c.__dict__:
            _wrap_init(c)
    _wrap_dialect_lookups(Dialect)


def _record_dialect_class(cls):
    """A dialect used without an instance (Dialect["x"].format_time): count
    the test as a user of the dialect and its tokenizer/parser/generator."""
    key = _key[0] or _stack_test_file()
    bucket = _types.setdefault(key, set())
    for c in (cls, getattr(cls, "tokenizer_class", None), getattr(cls, "parser_class", None),
              getattr(cls, "generator_class", None)):
        if isinstance(c, type):
            bucket.add(c)


def _wrap_dialect_lookups(Dialect):
    meta = type(Dialect)
    for name in ("__getitem__", "get"):
        orig = meta.__dict__.get(name)
        if orig is None:
            continue

        def make(orig):
            is_cm = isinstance(orig, classmethod)
            fn = orig.__func__ if is_cm else orig

            def wrapper(cls, *a, **kw):
                res = fn(cls, *a, **kw)
                if isinstance(res, type):
                    _record_dialect_class(res)
                return res
            return classmethod(wrapper) if is_cm else wrapper
        setattr(meta, name, make(orig))
    orig_gor = Dialect.__dict__["get_or_raise"].__func__

    def get_or_raise(cls, *a, **kw):
        res = orig_gor(cls, *a, **kw)
        _record_dialect_class(res if isinstance(res, type) else type(res))
        return res
    Dialect.get_or_raise = classmethod(get_or_raise)


def _dump_types():
    out = {}
    for k, ts in _types.items():
        names = set()
        for t in ts:
            for m in t.__mro__:
                if m.__module__.startswith("sqlglot"):
                    names.add(f"{m.__module__}:{m.__qualname__}")
        out[k] = sorted(names)
    return out


# Code run while pytest imports a test module (module-level parse_one calls,
# class bodies) belongs to that file's tests, not to sqlglot's own import.
def _cov():
    try:
        import coverage
        return coverage.Coverage.current()
    except Exception:
        return None


def pytest_collectstart(collector):
    import _pytest.python as P
    if isinstance(collector, P.Module):
        c = _cov()
        if c is not None:
            c.switch_context("COLLECT:" + collector.nodeid)


def pytest_collectreport(report):
    c = _cov()
    if c is not None and report.nodeid.endswith(".py"):
        c.switch_context("")


# Which keys each test looks up in class-level dicts (FUNCTIONS, KEYWORDS,
# TRANSFORMS, arg_types, ...), so a change to one dict entry can be traced
# to the tests that use that key.
_keys: dict = {}


def _keyrepr(k):
    import enum
    if isinstance(k, str):
        return "s:" + k
    if isinstance(k, type):
        return f"t:{k.__qualname__}"
    if isinstance(k, enum.Enum):
        return f"e:{type(k).__name__}.{k.name}"
    return None


class _RecDict(dict):
    __slots__ = ("_br_attr",)

    def _note(self, k):
        r = _keyrepr(k)
        if r is not None:
            key = _key[0] or "FILE:?"
            _keys.setdefault(key, set()).add((self._br_attr, r))

    def __getitem__(self, k):
        self._note(k)
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        self._note(k)
        return dict.get(self, k, default)

    def __contains__(self, k):
        self._note(k)
        return dict.__contains__(self, k)


def _wrap_dicts():
    import inspect, sys as _sys
    seen = set()
    for mod in list(_sys.modules.values()):
        if not getattr(mod, "__name__", "").startswith("sqlglot"):
            continue
        stack = [v for v in vars(mod).values() if inspect.isclass(v) and v.__module__ == mod.__name__]
        while stack:
            c = stack.pop()
            if id(c) in seen:
                continue
            seen.add(id(c))
            for name, v in list(vars(c).items()):
                if name.startswith("__"):
                    continue
                if type(v) is dict:
                    w = _RecDict(v)
                    w._br_attr = name
                    try:
                        setattr(c, name, w)
                    except Exception:
                        pass
                elif inspect.isclass(v) and v.__module__ == c.__module__:
                    stack.append(v)


_orig_configure = pytest_configure


def pytest_configure(config):  # noqa: F811
    _orig_configure(config)
    _wrap_dicts()


def _dump_keys(out, wid):
    with open(f"{out}.keys.{wid}.json", "w") as f:
        json.dump({k: sorted(f"{a}|{r}" for a, r in v) for k, v in _keys.items()}, f)
