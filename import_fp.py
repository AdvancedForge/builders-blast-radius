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
import pkgutil
import sys


def fingerprint():
    import sqlglot
    from sqlglot.expressions.core import Expr
    from sqlglot.dialects.dialect import Dialect
    import sqlglot.dialects as d

    mods = []
    for m in sorted(pkgutil.walk_packages(sqlglot.__path__, "sqlglot."), key=lambda m: m.name):
        if m.name == "sqlglot.__main__":
            continue
        try:
            mods.append(importlib.import_module(m.name))
        except ImportError as e:
            if "No module named" not in str(e):
                raise
    for name in getattr(d, "DIALECTS", []):
        Dialect.get_or_raise(name.lower())

    out = {}

    def note(where, value):
        if isinstance(value, Expr):
            out[where] = f"{type(value).__name__}:{value.sql()}"
        elif isinstance(value, dict):
            exprs = [f"{k!r}:{v.sql()}" for k, v in value.items() if isinstance(v, Expr)]
            out[where] = f"keys={len(value)}" + ("|" + "|".join(sorted(exprs)) if exprs else "")

    seen = set()
    for mod in mods:
        for k, v in sorted(vars(mod).items()):
            if not k.startswith("__"):
                note(f"{mod.__name__}:.{k}", v)
            if inspect.isclass(v) and v.__module__ == mod.__name__ and id(v) not in seen:
                seen.add(id(v))
                stack = [v]
                while stack:
                    c = stack.pop()
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
        json.dump(fp, open(path, "w"))
        sys.exit(0)
    base = json.load(open(path))
    for k in sorted(set(base) | set(fp)):
        if base.get(k) != fp.get(k):
            print(k)
    sys.exit(0)
