"""Build-time: dialect families (selector.families) into a JSON file.
Usage: BR_SRC=<sqlglot root> python families.py <out.json>  (run where selector.py lives)"""
import gzip, json, os, sys
sys.path.insert(0, "/app")
import selector as S

S._ENTRIES = json.load(gzip.open(S.ENTRIES, "rt"))
S._FAMILIES = None
out = sys.argv[1]
if os.path.exists(S.HERE / "families.json"):
    raise SystemExit("families.json already exists; build it before installing")
fam = S.families()
json.dump({d: sorted(v) for d, v in fam.items()}, open(out, "w"))
print(len(fam), "dialect families")
