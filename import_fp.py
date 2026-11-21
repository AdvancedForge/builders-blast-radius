"""Fingerprint sqlglot's import-time state: import every module, load every
dialect, and hash every SQL expression object held at module or class level
(templates parsed at import) plus the keys of class-level dicts.

  import_fp.py base OUT     write the fingerprint of the tree on sys.path
  import_fp.py check BASE   print the "module:Class.attr" entries that differ
                            (one per line); exit 1 if the import fails
"""
import hashlib
import importlib
import inspect
import json
import os
import pkgutil
import sys


DEEP = os.environ.get("BR_IMPORT_DEEP", "1") == "1"
VALS = os.environ.get("BR_IMPORT_VALS", "1") == "1"  # plain values (val| entries)
# Deep hashing only for these modules' globals and classes, and classes that
# inherit from them (where state derived from a change can live); all if unset.
FAILED: list = []
ONLY = set(filter(None, os.environ.get("BR_IMPORT_ONLY", "").split(","))) or None
DEEP_MODULES = set(filter(None, os.environ.get("BR_DEEP_MODULES", "").split(","))) or None


def fingerprint():
    import sqlglot
    from sqlglot.expressions.core import Expr
    from sqlglot.dialects.dialect import Dialect
    import sqlglot.dialects as d

    mods = []
    # Only modules a patch can affect (it and its importers) need importing;
    # every other module keeps its pinned import-time state.
    only = ONLY
    for m in sorted(pkgutil.walk_packages(sqlglot.__path__, "sqlglot."), key=lambda m: m.name):
        if m.name == "sqlglot.__main__" or (only is not None and m.name not in only):
            continue
        try:
            mods.append(importlib.import_module(m.name))
        except ImportError as e:
            if "No module named" not in str(e):
                raise
            FAILED.append(m.name)  # optional dependency missing (compared with the base)
    if only is None:
        for name in getattr(d, "DIALECTS", []):
            Dialect.get_or_raise(name.lower())

    out = {}

    import enum

    def canon(v, depth=0):
        """A stable text for data built at import: primitives, enum members,
        classes and expressions by value; containers by content."""
        if depth > 8:
            return "..."
        if v is None or isinstance(v, (bool, int, float, str, bytes)):
            return repr(v)
        if isinstance(v, enum.Enum):
            return f"{type(v).__name__}.{v.name}"
        if isinstance(v, type):
            return f"<{v.__module__}.{v.__qualname__}>"
        if isinstance(v, Expr):
            return f"{type(v).__name__}:{v.sql()}"
        if isinstance(v, dict):
            return "{" + ",".join(sorted(f"{canon(k, depth + 1)}:{canon(x, depth + 1)}" for k, x in v.items())) + "}"
        if isinstance(v, (set, frozenset)):
            return "{" + ",".join(sorted(canon(x, depth + 1) for x in v)) + "}"
        if isinstance(v, (list, tuple)):
            items = [canon(x, depth + 1) for x in v]
            if v and all(isinstance(x, type) for x in v):
                items.sort()  # a tuple of classes for isinstance: often built from a set
            return "[" + ",".join(items) + "]"
        if callable(v):
            return f"<fn {getattr(v, '__qualname__', type(v).__name__)}>"
        return f"<{type(v).__name__}>"

    def note(where, value):
        # Shallow entries (SQL objects, dict sizes) drive selection; "deep|"
        # entries (full contents) only gate live data patches.
        if isinstance(value, Expr):
            out[where] = f"{type(value).__name__}:{value.sql()}"
        elif isinstance(value, dict):
            exprs = [f"{k!r}:{v.sql()}" for k, v in value.items() if isinstance(v, Expr)]
            out[where] = f"keys={len(value)}" + ("|" + "|".join(sorted(exprs)) if exprs else "")
        elif VALS and (isinstance(value, (bool, int, float, str, bytes, tuple, frozenset, enum.Enum)) or value is None):
            out["val|" + where] = canon(value)  # plain values: only gate import-time live runs
        if DEEP and deep_ok[0] and isinstance(value, (dict, set, frozenset, list, tuple)):
            out["deep|" + where] = canon(value)

    deep_ok = [True]

    def related(obj) -> bool:
        if DEEP_MODULES is None:
            return True
        if inspect.isclass(obj):
            return any(k.__module__ in DEEP_MODULES for k in obj.__mro__)
        return getattr(obj, "__name__", None) in DEEP_MODULES

    seen = set()
    for mod in mods:
        for k, v in sorted(vars(mod).items()):
            if not k.startswith("__"):
                deep_ok[0] = related(mod)
                note(f"{mod.__name__}:.{k}", v)
            if inspect.isclass(v) and v.__module__ == mod.__name__ and id(v) not in seen:
                seen.add(id(v))
                stack = [v]
                while stack:
                    c = stack.pop()
                    deep_ok[0] = related(c)
                    for ck, cv in sorted(vars(c).items()):
                        if not ck.startswith("__"):
                            note(f"{c.__module__}:{c.__qualname__}.{ck}", cv)
                        if inspect.isclass(cv) and cv.__module__ == mod.__name__ and id(cv) not in seen:
                            seen.add(id(cv)); stack.append(cv)
    return {k: hashlib.sha1(v.encode()).hexdigest()[:16] for k, v in out.items()}


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    try:
        fp = fingerprint()
    except BaseException as e:  # noqa: BLE001 - a failing import is a difference
        print("import_fp:", type(e).__name__, e, file=sys.stderr)
        sys.exit(1)
    if mode == "base":
        fp["__failed__"] = sorted(FAILED)
        json.dump(fp, open(path, "w"))
        sys.exit(0)
    base = json.load(open(path))
    new_failures = set(FAILED) - set(base.pop("__failed__", []))
    if new_failures:
        print("import_fp: newly failing imports:", sorted(new_failures), file=sys.stderr)
        sys.exit(1)
    if ONLY is not None:
        def mod_of(k):
            return k.removeprefix("deep|").removeprefix("val|").split(":", 1)[0]
        base = {k: v for k, v in base.items() if mod_of(k) in ONLY}
    if not DEEP:
        base = {k: v for k, v in base.items() if not k.startswith("deep|")}
    if not VALS:
        base = {k: v for k, v in base.items() if not k.startswith("val|")}
    elif DEEP_MODULES is not None:
        base = {k: v for k, v in base.items() if not k.startswith("deep|") or k in fp}
    for k in sorted(set(base) | set(fp)):
        if base.get(k) != fp.get(k):
            print(k)
    sys.exit(0)
