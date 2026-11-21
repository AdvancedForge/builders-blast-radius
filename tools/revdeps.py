"""Build-time: for every sqlglot module, the sqlglot modules that import it
(transitively). Usage: python revdeps.py <sqlglot root> <out.json>"""
import ast, json, os, sys

root, out = sys.argv[1], sys.argv[2]
mods = {}
for d, _, fs in os.walk(os.path.join(root, "sqlglot")):
    for f in fs:
        if f.endswith(".py"):
            rel = os.path.relpath(os.path.join(d, f), root)
            name = rel[:-3].replace("/", ".").removesuffix(".__init__")
            mods[name] = rel
deps = {m: set() for m in mods}
for m, rel in mods.items():
    try:
        tree = ast.parse(open(os.path.join(root, rel)).read())
    except SyntaxError:
        continue
    for n in ast.walk(tree):
        names = []
        if isinstance(n, ast.ImportFrom) and n.module:
            base = n.module
            if n.level:
                pkg = m.split(".")[: len(m.split(".")) - n.level + (1 if rel.endswith("__init__.py") else 0)]
                base = ".".join(pkg + ([n.module] if n.module else []))
            names = [base] + [f"{base}.{a.name}" for a in n.names]
        elif isinstance(n, ast.Import):
            names = [a.name for a in n.names]
        for x in names:
            if x in mods and x != m:
                deps[m].add(x)
rev = {m: set() for m in mods}
for m, ds in deps.items():
    for x in ds:
        rev[x].add(m)
closure = {}
for m in mods:
    seen, stack = set(), [m]
    while stack:
        x = stack.pop()
        for y in rev.get(x, ()):
            if y not in seen:
                seen.add(y); stack.append(y)
    closure[m] = sorted(seen)
json.dump(closure, open(out, "w"))
print(len(closure), "modules; typing.hive importers:", closure.get("sqlglot.typing.hive"))
