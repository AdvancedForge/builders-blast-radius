"""Make random single-site mutations of sqlglot, shaped like the sample commits.

Usage: python mutate.py <sqlglot checkout> <out dir> <count> <seed>
Writes <id>.patch files (unified diff against the checkout).
"""
import os, random, re, subprocess, sys, difflib

src, out, count, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rng = random.Random(seed)
py = [os.path.join(r, f)[len(src) + 1:] for r, _, fs in os.walk(os.path.join(src, "sqlglot"))
      for f in fs if f.endswith(".py") and "__pycache__" not in r]
fixtures = [os.path.join(r, f)[len(src) + 1:] for r, _, fs in os.walk(os.path.join(src, "tests/fixtures"))
            for f in fs if f.endswith(".sql")]
tests = [os.path.join(r, f)[len(src) + 1:] for r, _, fs in os.walk(os.path.join(src, "tests"))
         for f in fs if f.endswith(".py") and "__pycache__" not in r]

SWAPS = [(" == ", " != "), (" != ", " == "), (" < ", " <= "), (" > ", " >= "), (" <= ", " < "),
         (" and ", " or "), (" or ", " and "), ("True", "False"), ("False", "True"),
         ("is not None", "is None"), ("if not ", "if "), (" in ", " not in "), ("+ 1", "- 1"),
         ("[0]", "[1]"), ("[1]", "[0]"), ("[-1]", "[0]"), ("return True", "return False"),
         ("return False", "return True"), ("upper()", "lower()"), ("startswith", "endswith")]


def mutate_py(lines):
    kind = rng.random()
    if kind < 0.12:  # comment only
        i = rng.randrange(len(lines))
        indent = re.match(r"\s*", lines[i]).group(0)
        lines.insert(i, f"{indent}# {rng.choice(['TODO: revisit', 'Cheap check first', 'Kept for compatibility', 'NOTE'])}\n")
        return "comment"
    if kind < 0.2:  # integer literal change
        cands = [i for i, l in enumerate(lines) if re.search(r"\b\d+\b", l) and not l.lstrip().startswith("#")]
        if not cands: return None
        i = rng.choice(cands)
        lines[i] = re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) + rng.choice([1, -1, 2])), lines[i], count=1)
        return "const"
    if kind < 0.28:  # string literal change
        cands = [i for i, l in enumerate(lines) if re.search(r'"[A-Za-z_]{2,}"', l) and not l.lstrip().startswith("#")]
        if not cands: return None
        i = rng.choice(cands)
        lines[i] = re.sub(r'"([A-Za-z_]{2,})"', lambda m: '"' + m.group(1)[:-1] + '"', lines[i], count=1)
        return "string"
    if kind < 0.33:  # drop a statement line
        cands = [i for i, l in enumerate(lines) if re.match(r"\s+(return|[a-z_]+\s*=|self\.)", l) and l.rstrip().endswith((")", "]")) is False and l.strip()]
        if not cands: return None
        i = rng.choice(cands)
        indent = re.match(r"\s*", lines[i]).group(0)
        lines[i] = indent + "pass\n"
        return "drop"
    cands = [(i, a, b) for i, l in enumerate(lines) for a, b in SWAPS if a in l and not l.lstrip().startswith("#")]
    if not cands: return None
    i, a, b = rng.choice(cands)
    lines[i] = lines[i].replace(a, b, 1)
    return "code"


def mutate_fixture(lines):
    cands = [i for i, l in enumerate(lines) if l.strip() and not l.startswith("#")]
    if not cands: return None
    i = rng.choice(cands)
    words = lines[i].split(" ")
    j = rng.randrange(len(words))
    words.insert(j, rng.choice(["1", "x", "AS", "a", ","]))
    lines[i] = " ".join(words)
    return "fixture"


made = 0
while made < count:
    r = rng.random()
    path = rng.choice(fixtures) if r < 0.25 else (rng.choice(tests) if r < 0.3 else rng.choice(py))
    old = open(os.path.join(src, path)).readlines()
    if not old: continue
    new = list(old)
    kind = mutate_fixture(new) if path.endswith(".sql") else mutate_py(new)
    if kind is None or new == old: continue
    diff = "".join(difflib.unified_diff(old, new, f"a/{path}", f"b/{path}", n=3))
    pid = f"m{seed}_{made:03d}"
    with open(os.path.join(out, pid + ".patch"), "w") as f:
        f.write(f"diff --git a/{path} b/{path}\n" + diff)
    with open(os.path.join(out, pid + ".kind"), "w") as f:
        f.write(kind)
    made += 1
