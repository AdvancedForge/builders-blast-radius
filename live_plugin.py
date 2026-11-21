"""pytest plugin for live proxy runs inside the scoring container.

Runs process-pool work in-process (pickling arguments and results the way a
spawned pool would, since sqlglot expressions pickle through serde) so a run
uses no extra interpreters, and writes the node ids that failed."""
import concurrent.futures as _cf
import json
import os
import pickle


def _roundtrip(obj):
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

_failed = set()
_seen = set()


def pytest_runtest_logreport(report):
    _seen.add(report.nodeid)
    if report.failed:
        _failed.add(report.nodeid)


def pytest_collectreport(report):
    if report.failed:
        _failed.add(report.nodeid or "<collect>")


def pytest_sessionfinish(session):
    with open(os.environ["BR_LIVE_OUT"], "w") as f:
        json.dump({"failed": sorted(_failed), "seen": sorted(_seen)}, f)
