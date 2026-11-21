"""Runtime hooks for the instrumented sqlglot copy.

masks[(test, site)] has bit i set if alternative operator i could have given
a different result than the real operator on some evaluation in that test.
Anything uncertain (non-bool results, side effects) counts as different."""
import operator, os

ALT_CMP = ["==", "!=", "<", "<=", ">", ">=", "is", "is not", "in", "not in"]
_F = {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt,
      ">=": operator.ge, "is": operator.is_, "is not": operator.is_not,
      "in": lambda a, b: a in b, "not in": lambda a, b: a not in b}
ALL = (1 << len(ALT_CMP)) - 1
SAFE_CONTAINERS = (list, tuple, set, frozenset, dict, str, bytes, range, type({}.keys()))
_sites = None
current = ["<import>"]
masks = {}


def _load():
    global _sites
    import json
    _sites = json.load(open(os.environ["BR_SITES"]))


def _note(sid, m):
    if m:
        key = (current[0], sid)
        masks[key] = masks.get(key, 0) | m


def _br_cmp(sid, a, b):
    if _sites is None:
        _load()
    op = _sites[sid][4]
    real = _F[op](a, b)
    if type(real) is not bool:
        _note(sid, ALL)  # e.g. Expr operator overloading builds an expression
        return real
    m = 0
    for i, alt in enumerate(ALT_CMP):
        if alt == op:
            continue
        if alt in ("in", "not in") and not isinstance(b, SAFE_CONTAINERS):
            m |= 1 << i
            continue
        try:
            v = _F[alt](a, b)
        except Exception:
            m |= 1 << i
            continue
        if type(v) is not bool or v != real:
            m |= 1 << i
    _note(sid, m)
    return real


def _br_bool(sid, left, right_thunk):
    if _sites is None:
        _load()
    op = _sites[sid][4]
    if op == "and":
        if not left:
            _note(sid, 1)  # 'or' would have evaluated the right side
            return left
        right = right_thunk()
        if right is not left:  # 'or' would have returned left
            _note(sid, 1)
        return right
    if left:
        _note(sid, 1)  # 'and' would have evaluated the right side
        return left
    right = right_thunk()
    if right is not left:  # 'and' would have returned left
        _note(sid, 1)
    return right


# pytest plugin side: switch the current test, dump masks at the end.
def pytest_runtest_setup(item):
    current[0] = "S:" + item.nodeid.rsplit("::", 1)[0]


def pytest_runtest_call(item):
    current[0] = item.nodeid


def pytest_runtest_teardown(item):
    current[0] = "S:" + item.nodeid.rsplit("::", 1)[0]


def pytest_collectstart(collector):
    if collector.nodeid.endswith(".py"):
        current[0] = "C:" + collector.nodeid


def pytest_sessionfinish(session):
    import json
    out = os.environ.get("BR_MASKS")
    if out:
        wid = os.environ.get("PYTEST_XDIST_WORKER", "main")
        agg = {}
        for (t, sid), m in masks.items():
            agg.setdefault(t, {})[sid] = m
        json.dump(agg, open(f"{out}.{wid}.json", "w"))
