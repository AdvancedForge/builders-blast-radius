"""Turn a per-test coverage run into maps/map.json.gz.

Usage (inside br-analysis): python build_map.py /out /out/map.json.gz
Map layout:
  tests:  [nodeid, ...]                    index = test key used below
  sets:   [[test key, ...], ...]           deduplicated test sets
  lines:  {file: {line: set index}}        tests that ran the line
  import: {file: [line, ...]}              lines run only at import (no test context)
  reads:  {data file: set index}           tests that open a non-.py file
"""
import coverage, glob, gzip, json, sys

out_dir, dest = sys.argv[1], sys.argv[2]
data = coverage.CoverageData(f"{out_dir}/.coverage"); data.read()

tests, tindex = [], {}
def tkey(nodeid):
    if nodeid not in tindex:
        tindex[nodeid] = len(tests); tests.append(nodeid)
    return tindex[nodeid]

sets, sindex = [], {}
def skey(keys):
    t = tuple(sorted(set(keys)))
    if t not in sindex:
        sindex[t] = len(sets); sets.append(list(t))
    return sindex[t]

all_nodeids = sorted({c.split("|")[0] for c in data.measured_contexts() if c and not c.startswith("COLLECT:")})
for t in all_nodeids:
    tkey(t)
lines, imports = {}, {}
for path in sorted(data.measured_files()):
    rel = path.removeprefix("/src/")
    per = {}
    imp = []
    for ln, ctxs in (data.contexts_by_lineno(path) or {}).items():
        ids = []
        for c in ctxs:
            if c.startswith("COLLECT:"):
                ids += [tkey(t) for t in all_nodeids if t.startswith(c[8:] + "::")]
            elif c:
                nodeid, _, phase = c.partition("|")
                if phase in ("setup", "teardown"):
                    # setUpClass/tearDownClass run in the first/last test's
                    # setup/teardown but serve every test in the class.
                    cls = nodeid.rsplit("::", 1)[0] + "::"
                    ids += [tkey(t) for t in all_nodeids if t.startswith(cls)]
                else:
                    ids.append(tkey(nodeid))
        if ids:
            per[str(ln)] = skey(ids)
        if "" in ctxs:
            imp.append(ln)
    lines[rel] = per
    imports[rel] = sorted(imp)

all_ids = list(tindex)
def expand(k):
    if k.startswith("CLASS:"):
        pre = k[6:] + "::"; return [t for t in all_ids if t.startswith(pre)]
    if k.startswith("FILE:"):
        f = k[5:]; return [t for t in all_ids if t.startswith(f + "::")] if f != "?" else []
    return [k]

def invert(pattern):
    raw = {}
    for p in glob.glob(f"{out_dir}/{pattern}"):
        for k, v in json.load(open(p)).items():
            raw.setdefault(k, set()).update(v)
    inv = {}
    for k, items in raw.items():
        ids = [tkey(i) for i in expand(k)]
        for x in items:
            inv.setdefault(x, set()).update(ids)
    return {x: skey(v) for x, v in inv.items() if v}

reads = invert("reads.gw*.json") | invert("reads.main.json")
classes = invert("reads.types.*.json")

# Modules each test file imports, from a fresh interpreter per file.
import os, subprocess
closure = {}
for tf in sorted(set(t.split("::")[0] for t in all_ids)):
    mod = tf[:-3].replace("/", ".")
    code = (f"import sys, importlib; importlib.import_module({mod!r}); "
            "print('\\n'.join(sorted(m.__file__ for m in list(sys.modules.values()) "
            "if getattr(m, '__file__', None) and m.__file__.startswith('/src/'))))")
    r = subprocess.run([sys.executable, "-c", code], cwd="/src", capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH="/src", SKIP_INTEGRATION="0"))
    closure[tf] = [f.removeprefix("/src/") for f in r.stdout.split()] if r.returncode == 0 else None
    if r.returncode: print("import failed", tf, r.stderr[-300:])

with gzip.open(dest, "wt") as fh:
    json.dump({"tests": tests, "sets": sets, "lines": lines, "import": imports, "reads": reads, "classes": classes, "closure": closure}, fh, separators=(",", ":"))
print(len(tests), "tests", len(sets), "sets", sum(map(len, lines.values())), "lines")
