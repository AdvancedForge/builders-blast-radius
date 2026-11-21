"""Score the selector on labeled mutants: misses and simulated cost.

Usage: BR_SRC=work/sqlglot python tools/eval_mutants.py work/mut
"""
import json, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import selector as S

root = sys.argv[1]
tests = json.load(open("sample/tests.json"))["tests"]
by_id = {t["id"]: t for t in tests}
m = S.Map(S.MAP); sel = S.Selector(m)
every = list(by_id.values())
total = misses = n = 0
rows = []
for lp in sorted(glob.glob(f"{root}/labels/*.json")):
    pid = os.path.basename(lp)[:-5]
    lab = json.load(open(lp))
    if lab.get("apply_failed") or not lab.get("ran"):
        continue
    fail = {by_id[i]["number"] for i in lab["failed"] if i in by_id}
    for ce in lab["collect_errors"]:
        fail |= {t["number"] for t in tests if t["id"].startswith(ce)}
    text = open(f"{root}/patches/{pid}.patch").read()
    kind = open(f"{root}/patches/{pid}.kind").read() if os.path.exists(f"{root}/patches/{pid}.kind") else "?"
    try:
        keys = sel.commit(text); chosen = [by_id[m.tests[k]] for k in keys if m.tests[k] in by_id]
    except S.Everything:
        chosen = every
    workers = S.pack([(t["number"], t["seconds"]) for t in chosen])
    secs = {t["number"]: t["seconds"] for t in chosen}
    cost = (max(sum(secs[x] for x in w) for w in workers) + 2 * len(workers)) if workers else 0
    missed = fail - set(secs)
    n += 1; total += cost; misses += bool(missed)
    rows.append((pid, kind, len(fail), len(secs), round(cost, 1), sorted(missed)))
for r in rows:
    if r[5] or "-v" in sys.argv:
        print(*r[:5], "MISS" if r[5] else "", [by_id_t for by_id_t in (tests[i]["id"] for i in r[5][:4])])
print(f"{n} mutants, {misses} missed commits, simulated {total:.1f} s, mean {total / max(n, 1):.2f} s")
