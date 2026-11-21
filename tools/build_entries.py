"""Per fixture-entry map from the entry_plugin coverage run.
Output {test nodeid: {"lines": {file: {line: entry bitmask}},
                      "outside": {file: [lines run outside any entry]}}}"""
import coverage, gzip, json, sys
d = coverage.CoverageData(sys.argv[1]); d.read()
out = {}
for path in d.measured_files():
    rel = path[path.rindex("/sqlglot/") + 1:]
    for ln, ctxs in (d.contexts_by_lineno(path) or {}).items():
        for c in ctxs:
            if not c:
                continue
            node, _, ent = c.partition("#")
            setup = node.endswith("@setup")
            node = node.removesuffix("@setup")
            t = out.setdefault(node, {"lines": {}, "outside": {}, "setup": {}})
            if setup:
                t["setup"].setdefault(rel, []).append(ln)
            elif ent:
                mask = 0
                for i in ent.split("+"):
                    mask |= 1 << int(i)
                per = t["lines"].setdefault(rel, {})
                per[str(ln)] = per.get(str(ln), 0) | mask
            else:
                t["outside"].setdefault(rel, []).append(ln)
# keep only tests that have entries
out = {k: v for k, v in out.items() if v["lines"]}
import os
dj = sys.argv[1] + ".dialects.json"
if os.path.exists(dj):
    info = json.load(open(dj))
    out["__dialects__"] = {t: {i: sorted({x.strip() for d in ds for x in d.split(",")}) for i, ds in m.items()}
                           for t, m in info["entries"].items()}
    out["__families__"] = info["families"]
with gzip.open(sys.argv[2], "wt") as fh:
    json.dump(out, fh)
print(len(out), "tests with entries")
