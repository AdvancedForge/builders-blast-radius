"""pytest plugin: per fixture-entry line coverage for tests built on
TestOptimizer.check_file. Contexts are "<nodeid>#<entry indices>"; lines run
outside any entry stay on "<nodeid>". Pool work runs inline (pickled)."""
import concurrent.futures as _cf
import contextlib
import os
import pickle

import coverage

_cov = coverage.Coverage(data_file=os.environ["ENTRY_COV"], include=["*/sqlglot/*"])
_node = [None]
_titles = {}
_dialects = {}


def _rt(o):
    return pickle.loads(pickle.dumps(o))


class _Inline(_cf.Executor):
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        if initializer is not None:
            initializer(*_rt(initargs))

    def submit(self, fn, /, *args, **kwargs):
        f = _cf.Future()
        try:
            fn, args, kwargs = _rt((fn, args, kwargs))
            f.set_result(_rt(fn(*args, **kwargs)))
        except BaseException as e:  # noqa: BLE001
            f.set_exception(e)
        return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass


_cf.ProcessPoolExecutor = _Inline


def pytest_configure(config):
    _cov.start()
    import tests.test_optimizer as T

    orig_pairs = T.load_sql_fixture_pairs

    def pairs(filename):
        for i, (meta, sql, expected) in enumerate(orig_pairs(filename), start=1):
            title = meta.get("title") or f"{i}, {sql}"
            _titles.setdefault((_node[0], title), []).append(i)
            _dialects.setdefault(_node[0], {}).setdefault(i, set()).add(meta.get("dialect") or "")
            _cov.switch_context(f"{_node[0]}#{i}")
            yield meta, sql, expected
            _cov.switch_context(_node[0])

    T.load_sql_fixture_pairs = pairs
    orig_sub = T.TestOptimizer.subTest

    @contextlib.contextmanager
    def subtest(self, msg=None, **params):
        idx = _titles.get((_node[0], msg))
        if idx:
            _cov.switch_context(f"{_node[0]}#{'+'.join(map(str, idx))}")
        try:
            with orig_sub(self, msg, **params):
                yield
        finally:
            _cov.switch_context(_node[0])

    T.TestOptimizer.subTest = subtest
    orig_setup = T.TestOptimizer.setUp

    def setUp(self):
        _cov.switch_context(f"{_node[0]}@setup")
        try:
            return orig_setup(self)
        finally:
            _cov.switch_context(_node[0])

    T.TestOptimizer.setUp = setUp


def pytest_runtest_call(item):
    _node[0] = item.nodeid
    _cov.switch_context(item.nodeid)


def pytest_runtest_teardown(item):
    _cov.switch_context("")


def pytest_sessionfinish(session):
    _cov.stop()
    _cov.save()
    import json
    # Which dialect each entry uses, and which source files each dialect's
    # classes (dialect, tokenizer, parser, generator, and their bases) live in.
    from sqlglot.dialects.dialect import Dialect, Dialects
    fam = {}
    for d in [""] + [x.value for x in Dialects]:
        try:
            dc = type(Dialect.get_or_raise(d or None))
        except Exception:
            continue
        files = set()
        for c in (dc, dc.tokenizer_class, dc.parser_class, dc.generator_class):
            for k in c.__mro__:
                f = getattr(__import__(k.__module__, fromlist=["x"]), "__file__", "") or ""
                if "/sqlglot/" in f:
                    files.add(f[f.rindex("/sqlglot/") + 1:])
        fam[d] = sorted(files)
    json.dump({"entries": {n: {str(i): sorted(v) for i, v in m.items()} for n, m in _dialects.items()},
               "families": fam}, open(os.environ["ENTRY_COV"] + ".dialects.json", "w"))
