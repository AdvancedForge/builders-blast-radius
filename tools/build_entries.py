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
            t = out.setdefault(node, {"lines": {}, "outside": {}})
            if ent:
                mask = 0
                for i in ent.split("+"):
                    mask |= 1 << int(i)
                per = t["lines"].setdefault(rel, {})
                per[str(ln)] = per.get(str(ln), 0) | mask
            else:
                t["outside"].setdefault(rel, []).append(ln)
# keep only tests that have entries
out = {k: v for k, v in out.items() if v["lines"]}
with gzip.open(sys.argv[2], "wt") as fh:
    json.dump(out, fh)
print(len(out), "tests with entries")
