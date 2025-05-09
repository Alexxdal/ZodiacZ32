#!/usr/bin/env python3
"""Z32 Solver – Genetic Algorithm version

This script applies a basic GA to the Z32 cipher using the two most
promising transposition routes discovered so far (row_serp / col_serp).

Key features
------------
* **Population** size configurable (default 400).
* **Tournament selection** with size 5.
* **Single‑point crossover** on the mapping genome.
* **Mutations**: 80 % swap of two symbols, 20 % 3‑cycle rotation.
* Live CSV logging **every generation**; sorted report at the end.

Quick start
-----------
```bash
python z32_ga_solver.py --route col_serp \
                       --pop 400 --gens 1500 \
                       --seed 42 --log ga_col_1500.csv
```
The script prints the best score/plaintext per generation and saves the
log (plus a final *_sorted.csv*).
"""

import argparse
import csv
import math
import pathlib
import random
import string
from collections import defaultdict

# ------------------------------------------------------------------
# 1. Cipher matrix & routes (same as in z32_solver.py)
# ------------------------------------------------------------------
TOKENS = [
    "C","I","F","E","L","N","I","O",
    "W","H","D","A","Ω","N","G","O",
    "A","O","E","S","N","B","X","□",
    "T","C","E","T","D","I","E","I",
]
WIDTH, HEIGHT = 4, 8
COL_PERM = [1, 2, 3, 0]

rows = [TOKENS[i : i + WIDTH] for i in range(0, len(TOKENS), WIDTH)]
if len(rows) < HEIGHT:
    rows += [["?"] * WIDTH] * (HEIGHT - len(rows))

MATRIX = [[rows[r][c] for c in COL_PERM] for r in range(HEIGHT)]

NULLS = {"?", "□"}


def route_row_serp():
    out = []
    for r in range(HEIGHT):
        rng = range(WIDTH) if r % 2 == 0 else range(WIDTH - 1, -1, -1)
        for c in rng:
            tok = MATRIX[r][c]
            if tok not in NULLS:
                out.append(tok)
    return out


def route_col_serp():
    out = []
    for idx, c in enumerate(range(WIDTH)):
        rng = range(HEIGHT) if idx % 2 == 0 else range(HEIGHT - 1, -1, -1)
        for r in rng:
            tok = MATRIX[r][c]
            if tok not in NULLS:
                out.append(tok)
    return out

ROUTES = {
    "row_serp": route_row_serp,
    "col_serp": route_col_serp,
}

SYMBOLS = sorted(set(route_row_serp()))  # 20 distinct cipher symbols
SYM_IDX = {s: i for i, s in enumerate(SYMBOLS)}

# ------------------------------------------------------------------
# 2. Fitness (quadgram + crib bonus)
# ------------------------------------------------------------------
QGRAM = {
    "TION": 126024,
    "THER": 113290,
    "HERE": 96550,
    "WITH": 70160,
    "IGHT": 77290,
}
QTOT = sum(QGRAM.values())
QLOG = {k: math.log10(v / QTOT) for k, v in QGRAM.items()}
QFLOOR = math.log10(0.01 / QTOT)

CRIBS = {"STATION": 50, "THERE": 30, "WITH": 25, "HERE": 20, "THREE": 20}


def quad_score(txt: str) -> float:
    clean = "".join(ch for ch in txt.upper() if ch.isalpha())
    return sum(QLOG.get(clean[i : i + 4], QFLOOR) for i in range(len(clean) - 3))


def fitness(txt: str) -> float:
    score = quad_score(txt)
    up = txt.upper()
    for w, bonus in CRIBS.items():
        if w in up:
            score += bonus
    return score

# ------------------------------------------------------------------
# 3. GA primitives
# ------------------------------------------------------------------
LETTERS = list(string.ascii_uppercase)


def random_genome():  # permutation of letters (len == len(SYMBOLS))
    g = LETTERS.copy()
    random.shuffle(g)
    return g[: len(SYMBOLS)]


def genome_to_plain(seq, genome):
    mapping = {s: genome[SYM_IDX[s]] for s in SYMBOLS}
    return "".join(mapping[t] for t in seq)


def mutate(g):
    g = g.copy()
    if random.random() < 0.8:  # swap 2
        a, b = random.sample(range(len(g)), 2)
        g[a], g[b] = g[b], g[a]
    else:  # 3‑cycle
        x, y, z = random.sample(range(len(g)), 3)
        g[x], g[y], g[z] = g[y], g[z], g[x]
    return g


def crossover(p1, p2):
    """single-point crossover – sempre produce permutazione valida."""
    L = len(p1)
    cut = random.randint(1, L - 2)
    head = p1[:cut]
    tail = [l for l in p2 if l not in head]
    return head + tail


def tournament(pop, scores, k=5):
    best = None
    for _ in range(k):
        cand = random.choice(pop)
        if best is None or scores[id(cand)] > scores[id(best)]:
            best = cand
    return best

# ------------------------------------------------------------------
# 4. GA runner
# ------------------------------------------------------------------

def run_ga(seq, pop_size, generations, log_writer):
    # --- init population ---
    pop = [random_genome() for _ in range(pop_size)]
    scores = {}

    def eval_genome(g):
        if id(g) not in scores:
            scores[id(g)] = fitness(genome_to_plain(seq, g))
        return scores[id(g)]

    best_g = max(pop, key=eval_genome)
    best_score = eval_genome(best_g)

    for gen in range(generations):
        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = tournament(pop, scores), tournament(pop, scores)
            child = crossover(p1, p2)
            if random.random() < 0.3:
                child = mutate(child)
            new_pop.append(child)
        pop = new_pop
        # evaluate best
        cur_best = max(pop, key=eval_genome)
        cur_score = eval_genome(cur_best)
        if cur_score > best_score:
            best_g, best_score = cur_best, cur_score
        # log generation best
        log_writer.writerow([gen, best_score, genome_to_plain(seq, best_g)])
    return genome_to_plain(seq, best_g), best_score

# ------------------------------------------------------------------
# 5. CLI
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=ROUTES.keys(), default="col_serp")
    ap.add_argument("--pop", type=int, default=400, help="Population size")
    ap.add_argument("--gens", type=int, default=1500, help="Generations")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", default="ga_results.csv")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    seq = ROUTES[args.route]()

    log_path = pathlib.Path(args.log).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "score", "plaintext"])
        best_plain, best_score = run_ga(seq, args.pop, args.gens, writer)

    # sort & save
    sorted_path = log_path.with_name(log_path.stem + "_sorted.csv")
    with log_path.open() as src, sorted_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        rows = list(reader)
        rows.sort(key=lambda r: float(r["score"]), reverse=True)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader(); writer.writerows(rows)

    print("Best score", best_score)
    print("Best plaintext", best_plain)
    print("Sorted log saved to", sorted_path)


if __name__ == "__main__":
    main()