"""Rewrite a sqlglot checkout so every single comparison and every and/or
records, per test, which alternative operators would have changed its result.

Usage: python instrument.py <src root> <dst root>
Writes an instrumented copy of <src>/sqlglot into <dst>/sqlglot and a site
table <dst>/sites.json: site id -> [file, line, col, kind, op].
"""
import ast, json, os, shutil, sys

src, dst = sys.argv[1], sys.argv[2]
CMP = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
       ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in"}
sites = []


class T(ast.NodeTransformer):
    def __init__(self, rel):
        self.rel = rel

    def site(self, node, kind, op):
        sites.append([self.rel, node.lineno, node.col_offset, kind, op])
        return len(sites) - 1

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) != 1 or type(node.ops[0]) not in CMP:
            return node
        sid = self.site(node, "cmp", CMP[type(node.ops[0])])
        call = ast.Call(func=ast.Name("_br_cmp", ast.Load()),
                        args=[ast.Constant(sid), node.left, node.comparators[0]], keywords=[])
        return ast.copy_location(call, node)

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if len(node.values) != 2:
            return node
        op = "and" if isinstance(node.op, ast.And) else "or"
        sid = self.site(node, "bool", op)
        # Evaluate lazily, as Python does: the right side is a lambda.
        call = ast.Call(func=ast.Name("_br_bool", ast.Load()),
                        args=[ast.Constant(sid), node.values[0],
                              ast.Lambda(ast.arguments([], [], None, [], [], None, []), node.values[1])],
                        keywords=[])
        return ast.copy_location(call, node)


def walk_lambda_safe(tree):
    # Lambdas cannot hold walrus targets that must bind in the enclosing scope.
    for n in ast.walk(tree):
        if isinstance(n, ast.BoolOp) and any(isinstance(x, ast.NamedExpr) for x in ast.walk(n.values[1])):
            n._skip = True


if os.path.exists(os.path.join(dst, "sqlglot")):
    shutil.rmtree(os.path.join(dst, "sqlglot"))
shutil.copytree(os.path.join(src, "sqlglot"), os.path.join(dst, "sqlglot"),
                ignore=shutil.ignore_patterns("__pycache__"))
for root, _, files in os.walk(os.path.join(dst, "sqlglot")):
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(root, f)
        rel = os.path.relpath(p, dst)
        tree = ast.parse(open(p).read())
        walk_lambda_safe(tree)

        class T2(T):
            def visit_BoolOp(self, node):
                if getattr(node, "_skip", False):
                    self.generic_visit(node)
                    return node
                return super().visit_BoolOp(node)

        tree = T2(rel).visit(tree)
        # Put the hooks in scope after any __future__ import.
        body = tree.body
        i = 0
        while i < len(body) and (isinstance(body[i], ast.ImportFrom) and body[i].module == "__future__"
                                 or (i == 0 and isinstance(body[i], ast.Expr) and isinstance(getattr(body[i], "value", None), ast.Constant))):
            i += 1
        hook = ast.parse("from br_hooks import _br_cmp, _br_bool").body[0]
        body.insert(i, hook)
        ast.fix_missing_locations(tree)
        open(p, "w").write(ast.unparse(tree))
json.dump(sites, open(os.path.join(dst, "sites.json"), "w"))
print(len(sites), "sites")
