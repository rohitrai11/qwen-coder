#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_math_dsl_dataset.py

Generate a balanced training set for English → algebra DSL code.
Outputs chat-style JSONL where each line has:
{"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}

DSL conventions (kept consistent with your seed examples):
- Fields: QQ (rationals), ZZ (integers), GF(p)
- Rings:  R, x = polynomial_ring(QQ, "x")
          R, (x, y) = polynomial_ring(ZZ, ["x","y"])
- Matrices: A = matrix(ZZ, 3, 3, [1 2 3; 4 5 6; 7 8 9])
- Ops: det(A), eigenvalues(A), factor(f), roots(f), gcd(f,g)

Usage:
  python make_math_dsl_dataset.py --n 5000 --out train.jsonl --seed 42
"""

import argparse
import json
import random
import string
from typing import List, Tuple, Dict

SYSTEM_PROMPT = "You translate English math tasks into valid algebra DSL code. Only output code."

# -----------------------
# Helpers
# -----------------------

def pick(seq):
    return random.choice(seq)

def rand_name(base_choices=("x","y","z","u","v","w")):
    base = pick(base_choices)
    # 50% attach small index to diversify names
    return base if random.random() < 0.5 else f"{base}{random.randint(0,3)}"

def rand_names(k: int) -> List[str]:
    out = []
    while len(out) < k:
        n = rand_name()
        if n not in out:
            out.append(n)
    return out

def field_symbol() -> str:
    # Choose QQ/ZZ/GF(p) with bias towards QQ (common in your seeds)
    r = random.random()
    if r < 0.55:
        return "QQ"
    elif r < 0.75:
        return "ZZ"
    else:
        p = pick([3,5,7,11,13,17,19])
        return f"GF({p})"

def gf_prime() -> int:
    return pick([3,5,7,11,13,17,19,23,29])

def coeff(min_c=-5, max_c=5):
    c = 0
    # avoid zero too often to keep expressions interesting
    while c == 0:
        c = random.randint(min_c, max_c)
    return c

def poly_expr(var: str, deg_min=2, deg_max=5) -> str:
    deg = random.randint(deg_min, deg_max)
    # leading coefficient nonzero
    terms = []
    for p in range(deg, -1, -1):
        c = coeff()
        if p == 0:
            terms.append(f"{c}")
        elif p == 1:
            terms.append(f"{c}*{var}")
        else:
            terms.append(f"{c}*{var}^{p}")
    # join with ' + ' then replace '+ -' with '- '
    expr = " + ".join(terms)
    expr = expr.replace("+ -", "- ")
    return expr

def poly_two_vars_expr(vars: Tuple[str,str], deg=2) -> str:
    x, y = vars
    # construct a small bivariate with mixed terms
    terms = []
    # x^2, y^2, x*y, x, y, const
    for t in [(2,0),(0,2),(1,1),(1,0),(0,1),(0,0)]:
        cx = coeff(-3,3)
        if t == (0,0):
            terms.append(f"{cx}")
        elif t == (1,0):
            terms.append(f"{cx}*{x}")
        elif t == (0,1):
            terms.append(f"{cx}*{y}")
        elif t == (1,1):
            terms.append(f"{cx}*{x}*{y}")
        elif t == (2,0):
            terms.append(f"{cx}*{x}^{2}")
        elif t == (0,2):
            terms.append(f"{cx}*{y}^{2}")
    expr = " + ".join(terms).replace("+ -", "- ")
    return expr

def ring_def(field: str, vars_list: List[str]) -> str:
    if len(vars_list) == 1:
        v = vars_list[0]
        return f'R, {v} = polynomial_ring({field}, "{v}")'
    else:
        vars_tuple = "(" + ", ".join(vars_list) + ")"
        arr = "[\"" + "\", \"".join(vars_list) + "\"]"
        return f"R, {vars_tuple} = polynomial_ring({field}, {arr})"

def matrix_entries(rows: int, cols: int, lo=-5, hi=9) -> str:
    # produce "[a b c; d e f; ...]" form
    all_rows = []
    for _ in range(rows):
        row = [str(random.randint(lo, hi)) for _ in range(cols)]
        all_rows.append(" ".join(row))
    return "[" + "; ".join(all_rows) + "]"

def diag_entries(diag: List[int]) -> str:
    n = len(diag)
    rows = []
    for i in range(n):
        row = [str(0)] * n
        row[i] = str(diag[i])
        rows.append(" ".join(row))
    return "[" + "; ".join(rows) + "]"

def chat_ex(user: str, assist: str) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assist}
        ]
    }

# -----------------------
# Templates
# -----------------------

RING_TEMPLATES = [
    "Define a polynomial ring in variable {v} over {fld}.",
    "Construct a polynomial ring over {fld} with a single variable {v}.",
    "Create the polynomial ring R in {v} over {fld}.",
    "Define the ring R = {fld}[{v}].",
]

RING2_TEMPLATES = [
    "Define a polynomial ring in variables {v1} and {v2} over {fld}.",
    "Construct the multivariate polynomial ring over {fld} with variables {v1}, {v2}.",
    "Create the polynomial ring R in {v1}, {v2} over {fld}.",
    "Define the ring R = {fld}[{v1}, {v2}].",
]

FIELD_TEMPLATES = [
    "Define a finite field with {p} elements.",
    "Create the field GF({p}).",
    "Construct the finite field of order {p}.",
]

RING_OVER_GF_TEMPLATES = [
    "Define a polynomial ring over GF({p}) in variable {v}.",
    "Create R = GF({p})[{v}].",
    "Construct a univariate polynomial ring in {v} over the finite field with {p} elements.",
]

POLY_DEF_TEMPLATES = [
    "Define the polynomial f = {expr} over the rationals.",
    "Over QQ, set f to {expr}.",
    "In a polynomial ring over QQ, define f = {expr}.",
]

POLY_FACTOR_TEMPLATES = [
    "Factor the polynomial {expr} over the rationals.",
    "Compute the factorization of {expr} in QQ[x].",
    "Over QQ, factor {expr}.",
]

POLY_ROOTS_TEMPLATES = [
    "Find the roots of the polynomial {expr}.",
    "Compute the zeros of {expr} over the rationals.",
    "Solve {expr} = 0 over QQ.",
]

POLY_GCD_TEMPLATES = [
    "Compute the gcd of polynomials f = {f} and g = {g} over QQ.",
    "Over the rationals, find gcd(f, g) for f = {f} and g = {g}.",
    "Let f = {f} and g = {g} in QQ[x]; compute their gcd.",
]

POLY_EVAL_TEMPLATES = [
    "Evaluate f = {f} at {v} = {val}.",
    "Let f = {f}. Compute f({val}).",
    "Given f = {f}, evaluate at {v} = {val}.",
]

MATRIX_DEF_TEMPLATES = [
    "Define a {r}x{c} matrix A with integer entries.",
    "Create the integer matrix A of size {r} by {c}.",
    "Construct A ∈ M_{{{r}×{c}}}(ZZ).",            # ← braces escaped
]

MATRIX_DIAG_TEMPLATES = [
    "Define a {n}x{n} diagonal matrix D over QQ with diagonal entries {diag}.",
    "Create a diagonal matrix D ∈ M_{{{n}}}(QQ) having diagonal {diag}.",   # ← braces escaped
]


MATRIX_DET_TEMPLATES = [
    "Compute the determinant of matrix A.",
    "Find det(A).",
    "Calculate the determinant of A.",
]

MATRIX_EIG_TEMPLATES = [
    "Find the eigenvalues of matrix A.",
    "Compute the spectrum of A.",
    "Determine eigenvalues(A).",
]

MATRIX_MULT_TEMPLATES = [
    "Multiply the matrices A and B.",
    "Compute the product A * B.",
    "Find A times B.",
]

MIXED_TEMPLATES = [
    # ring -> poly -> factor
    "Over {fld}, define a polynomial ring in {v}. Then set f = {expr}. Factor f.",
    # GF(p) -> ring -> roots
    "Work in GF({p}); define a polynomial ring in {v}, then solve f = {expr} for its roots.",
    # Two polynomials -> gcd
    "In QQ, define f = {f} and g = {g}; compute gcd(f, g).",
    # matrix -> det
    "Define an integer matrix A of size {n} and then compute det(A).",
]

# -----------------------
# Generators per category
# -----------------------

def _format_sanity_check():
    probe = dict(
        r=3, c=4, n=3, diag=[1,2,3],
        fld="QQ", v="x", v1="x", v2="y",
        p=7, expr="x^2 - 5*x + 6", f="x^3 - x", g="x^2 - 1", val=5
    )

    template_groups = {
        "RING_TEMPLATES": RING_TEMPLATES,
        "RING2_TEMPLATES": RING2_TEMPLATES,
        "FIELD_TEMPLATES": FIELD_TEMPLATES,
        "RING_OVER_GF_TEMPLATES": RING_OVER_GF_TEMPLATES,
        "POLY_DEF_TEMPLATES": POLY_DEF_TEMPLATES,
        "POLY_FACTOR_TEMPLATES": POLY_FACTOR_TEMPLATES,
        "POLY_ROOTS_TEMPLATES": POLY_ROOTS_TEMPLATES,
        "POLY_GCD_TEMPLATES": POLY_GCD_TEMPLATES,
        "POLY_EVAL_TEMPLATES": POLY_EVAL_TEMPLATES,
        "MATRIX_DEF_TEMPLATES": MATRIX_DEF_TEMPLATES,
        "MATRIX_DIAG_TEMPLATES": MATRIX_DIAG_TEMPLATES,
        "MATRIX_DET_TEMPLATES": MATRIX_DET_TEMPLATES,
        "MATRIX_EIG_TEMPLATES": MATRIX_EIG_TEMPLATES,
        "MATRIX_MULT_TEMPLATES": MATRIX_MULT_TEMPLATES,
        "MIXED_TEMPLATES": MIXED_TEMPLATES,
    }

    for name, arr in template_groups.items():
        for i, t in enumerate(arr):
            try:
                _ = t.format(**probe)
            except Exception as e:
                raise RuntimeError(f"Template error in {name}[{i}]: {t}\n{e}")
    print("All templates format cleanly ✅")


def gen_ring() -> Dict:
    fld = field_symbol()
    v = rand_name()
    user = pick(RING_TEMPLATES).format(v=v, fld=fld)
    code = ring_def(fld, [v])
    return chat_ex(user, code)

def gen_ring2() -> Dict:
    fld = field_symbol()
    v1, v2 = rand_names(2)
    user = pick(RING2_TEMPLATES).format(v1=v1, v2=v2, fld=fld)
    code = ring_def(fld, [v1, v2])
    return chat_ex(user, code)

def gen_field() -> Dict:
    p = gf_prime()
    user = pick(FIELD_TEMPLATES).format(p=p)
    code = f"F = GF({p})"
    return chat_ex(user, code)

def gen_ring_over_gf() -> Dict:
    p = gf_prime()
    v = rand_name()
    user = pick(RING_OVER_GF_TEMPLATES).format(p=p, v=v)
    code = ring_def(f"GF({p})", [v])
    return chat_ex(user, code)

def gen_poly_def() -> Dict:
    v = rand_name()
    expr = poly_expr(v)
    user = pick(POLY_DEF_TEMPLATES).format(expr=expr)
    code = f'R, {v} = polynomial_ring(QQ, "{v}");\n' \
           f"f = {expr}"
    return chat_ex(user, code)

def gen_poly_factor() -> Dict:
    v = rand_name()
    expr = poly_expr(v)
    user = pick(POLY_FACTOR_TEMPLATES).format(expr=expr)
    code = f'R, {v} = polynomial_ring(QQ, "{v}");\n' \
           f"factor({expr})"
    return chat_ex(user, code)

def gen_poly_roots() -> Dict:
    v = rand_name()
    expr = poly_expr(v)
    user = pick(POLY_ROOTS_TEMPLATES).format(expr=expr)
    code = f'R, {v} = polynomial_ring(QQ, "{v}");\n' \
           f"roots({expr})"
    return chat_ex(user, code)

def gen_poly_gcd() -> Dict:
    v = rand_name()
    f_expr = poly_expr(v, 2, 4)
    g_expr = poly_expr(v, 2, 4)
    user = pick(POLY_GCD_TEMPLATES).format(f=f_expr, g=g_expr)
    code = f'R, {v} = polynomial_ring(QQ, "{v}");\n' \
           f"f = {f_expr}; g = {g_expr};\n" \
           f"gcd(f, g)"
    return chat_ex(user, code)

def gen_poly_eval() -> Dict:
    v = rand_name()
    f_expr = poly_expr(v, 2, 4)
    val = random.randint(-5, 5)
    user = pick(POLY_EVAL_TEMPLATES).format(f=f_expr, v=v, val=val)
    code = f'R, {v} = polynomial_ring(QQ, "{v}");\n' \
           f"f = {f_expr};\n" \
           f"f({val})"
    return chat_ex(user, code)

def gen_poly_bivariate() -> Dict:
    # define bivariate polynomial and maybe factor
    v1, v2 = rand_names(2)
    fld = pick(["QQ", "ZZ", f"GF({gf_prime()})"])
    expr = poly_two_vars_expr((v1, v2))
    action = pick(["define", "factor"])
    if action == "define":
        user = f"Define a bivariate polynomial in {v1} and {v2} over {fld}."
        code = f"{ring_def(fld, [v1, v2])};\nF = {expr}\n"
    else:
        user = f"Over {fld}, factor the bivariate polynomial {expr}."
        code = f"{ring_def(fld, [v1, v2])};\n" \
               f"factor({expr})"
    return chat_ex(user, code.strip())

def gen_matrix_def() -> Dict:
    r = random.randint(2, 4)
    c = random.randint(2, 4)
    user = pick(MATRIX_DEF_TEMPLATES).format(r=r, c=c)
    code = f"A = matrix(ZZ, {r}, {c}, {matrix_entries(r, c)})"
    return chat_ex(user, code)

def gen_matrix_diag() -> Dict:
    n = random.randint(2, 5)
    diag = [random.randint(1, 7) for _ in range(n)]
    user = pick(MATRIX_DIAG_TEMPLATES).format(n=n, diag=str(diag))
    code = f"D = matrix(QQ, {n}, {n}, {diag_entries(diag)})"
    return chat_ex(user, code)

def gen_matrix_det() -> Dict:
    user = pick(MATRIX_DET_TEMPLATES)
    code = "det(A)"
    return chat_ex(user, code)

def gen_matrix_eig() -> Dict:
    user = pick(MATRIX_EIG_TEMPLATES)
    code = "eigenvalues(A)"
    return chat_ex(user, code)

def gen_matrix_mult() -> Dict:
    r = random.randint(2, 4)
    k = random.randint(2, 4)
    c = random.randint(2, 4)
    user = pick(MATRIX_MULT_TEMPLATES)
    code = f"A = matrix(ZZ, {r}, {k}, {matrix_entries(r, k)});\n" \
           f"B = matrix(ZZ, {k}, {c}, {matrix_entries(k, c)});\n" \
           f"A * B"
    return chat_ex(user, code)

def gen_mixed() -> Dict:
    choice = pick([0,1,2,3])
    if choice == 0:
        fld = field_symbol()
        v = rand_name()
        expr = poly_expr(v)
        user = pick(MIXED_TEMPLATES[0:1]).format(fld=fld, v=v, expr=expr)
        code = f"{ring_def(fld, [v])};\n" \
               f"f = {expr};\n" \
               f"factor(f)"
    elif choice == 1:
        p = gf_prime()
        v = rand_name()
        expr = poly_expr(v)
        user = pick(MIXED_TEMPLATES[1:2]).format(p=p, v=v, expr=expr)
        code = f'{ring_def(f"GF({p})", [v])};\n' \
               f"roots({expr})"
    elif choice == 2:
        v = rand_name()
        f_expr = poly_expr(v, 2, 4)
        g_expr = poly_expr(v, 2, 4)
        user = pick(MIXED_TEMPLATES[2:3]).format(f=f_expr, g=g_expr)
        code = f'R, {v} = polynomial_ring(QQ, "{v}");\n' \
               f"f = {f_expr}; g = {g_expr};\n" \
               f"gcd(f, g)"
    else:
        n = random.randint(2, 4)
        user = pick(MIXED_TEMPLATES[3:4]).format(n=f"{n}×{n}")
        code = f"A = matrix(ZZ, {n}, {n}, {matrix_entries(n, n)});\n" \
               f"det(A)"
    return chat_ex(user, code)

# -----------------------
# Sampler / Main
# -----------------------

CATEGORIES = {
    # name: (generator_fn, weight %)
    "ring1":   (gen_ring,            12),
    "ring2":   (gen_ring2,            8),
    "field":   (gen_field,            6),
    "ring_gf": (gen_ring_over_gf,     6),
    "polydef": (gen_poly_def,        12),
    "polyfac": (gen_poly_factor,     10),
    "polyroot":(gen_poly_roots,       8),
    "polygcd": (gen_poly_gcd,         8),
    "polyeval":(gen_poly_eval,        6),
    "poly2v":  (gen_poly_bivariate,   5),
    "matdef":  (gen_matrix_def,      10),
    "matdiag": (gen_matrix_diag,      3),
    "matdet":  (gen_matrix_det,       4),
    "mateig":  (gen_matrix_eig,       3),
    "matmul":  (gen_matrix_mult,      5),
    "mixed":   (gen_mixed,            6),
}

def compute_counts(n_total: int) -> Dict[str,int]:
    # proportional counts that sum to n_total
    weights = {k:w for k,(_,w) in CATEGORIES.items()}
    total_w = sum(weights.values())
    raw = {k: n_total * w / total_w for k,w in weights.items()}
    # round and adjust to make sum equal
    rounded = {k: int(round(v)) for k,v in raw.items()}
    diff = n_total - sum(rounded.values())
    # fix rounding drift
    keys = list(rounded.keys())
    i = 0
    while diff != 0:
        k = keys[i % len(keys)]
        rounded[k] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        i += 1
    return rounded

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="Number of samples")
    ap.add_argument("--out", type=str, default="train.jsonl", help="Output JSONL path")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = ap.parse_args()

    random.seed(args.seed)

    _format_sanity_check()

    counts = compute_counts(args.n)
    gens = {k: CATEGORIES[k][0] for k in CATEGORIES.keys()}

    all_rows = []
    for name, cnt in counts.items():
        g = gens[name]
        for _ in range(max(cnt, 0)):
            ex = g()
            all_rows.append(ex)

    # Shuffle for better mixing
    random.shuffle(all_rows)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Small summary
    print(f"Wrote {len(all_rows)} samples to {args.out}")
    print("Category counts:")
    for k in sorted(counts.keys()):
        print(f"  {k:8s}: {counts[k]}")

if __name__ == "__main__":
    main()