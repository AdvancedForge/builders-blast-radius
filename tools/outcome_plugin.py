"""pytest plugin: write the node ids that failed (any phase, any subtest) or
whose file failed to import, the way the track labels a failure."""
import json, os

_failed = set()
_collect_errors = set()


def pytest_runtest_logreport(report):
    if report.failed:
        _failed.add(report.nodeid)


def pytest_collectreport(report):
    if report.failed:
        _collect_errors.add(report.nodeid)


def pytest_sessionfinish(session):
    out = os.environ.get("OUTCOME_OUT")
    if out:
        wid = os.environ.get("PYTEST_XDIST_WORKER", "main")
        with open(f"{out}.{wid}.json", "w") as f:
            json.dump({"failed": sorted(_failed), "collect_errors": sorted(_collect_errors)}, f)
