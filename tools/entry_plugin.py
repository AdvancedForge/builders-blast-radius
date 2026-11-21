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


def pytest_runtest_call(item):
    _node[0] = item.nodeid
    _cov.switch_context(item.nodeid)


def pytest_runtest_teardown(item):
    _cov.switch_context("")


def pytest_sessionfinish(session):
    _cov.stop()
    _cov.save()
