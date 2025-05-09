#!/usr/bin/env python3
"""Z32 Solver – Unified Advanced Toolkit (SA + GA)

Adds **seeded‑GA** capability: you can pass a plaintext produced by a
previous SA run and the script will inject the corresponding mapping as
an elite individual in the GA population, so evolution starts from your
best known score instead of random noise.

New CLI flags (only used in `ga` mode)
-------------------------------------
--seed-plain "<plaintext>"   Plaintext string (32 chars) obtained from a
good SA candidate – must correspond to the
route you are running.
--elite        N            Number of elite genomes copied to next gen
(default 5 % of population).

Examples
--------
```bash
# 1) Generate seed with SA first
python z32_solver_advanced.py sa_cycle --route col_serp \
       --steps 300000 --restarts 1 --log tmp.csv
# Suppose the best plaintext printed is:  "EPRNCQEOZHXDBQLOQRKXOVMNRTTHERE"

# 2) Run GA seeded with that plaintext
python z32_solver_advanced.py ga --route col_serp \
       --pop 600 --gens 4000 --seed-plain "EPRNCQEOZHXDBQLOQRKXOVMNRTTHERE" \
       --elite 100 --log ga_seeded.csv
```"""

import argparse
import csv
import math
import pathlib
import random
import string
from typing import List, Dict

# ------------------------------------------------------------------
# 1. Cipher matrix & routes
# ------------------------------------------------------------------
TOKENS = [
    "C","I","F","E","L","N","I","O","W","H","D","A","Ω","N","G","O",
    "A","O","E","S","N","B","X","□","T","C","E","T","D","I","E","I",
]
WIDTH, HEIGHT = 4, 8
COL_PERM = [1, 2, 3, 0]
rows = [TOKENS[i : i + WIDTH] for i in range(0, len(TOKENS), WIDTH)]
if len(rows) < HEIGHT:
    rows += [["?"] * WIDTH] * (HEIGHT - len(rows))
MATRIX = [[rows[r][c] for c in COL_PERM] for r in range(HEIGHT)]
NULLS = {"?", "□"}

def route_row_serp() -> List[str]:
    seq = []
    for r in range(HEIGHT):
        rng = range(WIDTH) if r % 2 == 0 else range(WIDTH - 1, -1, -1)
        for c in rng:
            tok = MATRIX[r][c]
            if tok not in NULLS:
                seq.append(tok)
    return seq

def route_col_serp() -> List[str]:
    seq = []
    for idx, c in enumerate(range(WIDTH)):
        rng = range(HEIGHT) if idx % 2 == 0 else range(HEIGHT - 1, -1, -1)
        for r in rng:
            tok = MATRIX[r][c]
            if tok not in NULLS:
                seq.append(tok)
    return seq

ROUTES = {"row_serp": route_row_serp, "col_serp": route_col_serp}
SYMBOLS = sorted(set(route_row_serp()))
LETTERS = list(string.ascii_uppercase)

# ------------------------------------------------------------------
# 2. Fitness (quadgram + crib bonus)
# ------------------------------------------------------------------
QGRAM = {"TION":126024, "THER":113290, "HERE":96550, "WITH":70160, "IGHT":77290}
QTOT = sum(QGRAM.values())
QLOG = {k: math.log10(v/QTOT) for k, v in QGRAM.items()}
QFLOOR = math.log10(0.01/QTOT)
CRIBS = {"STATION":50, "THERE":30, "WITH":25, "HERE":20, "THREE":20}

def quad_score(text: str) -> float:
    clean = "".join(ch for ch in text.upper() if ch.isalpha())
    return sum(QLOG.get(clean[i:i+4], QFLOOR) for i in range(len(clean)-3))

def fitness(text: str) -> float:
    score = quad_score(text)
    up = text.upper()
    for w, b in CRIBS.items():
        if w in up:
            score += b
    return score

# ------------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------------

def random_mapping() -> Dict[str,str]:
    letters = LETTERS.copy()
    random.shuffle(letters)
    return {s: letters[i] for i, s in enumerate(SYMBOLS)}

def apply_map(seq: List[str], mp: Dict[str,str]) -> str:
    return ''.join(mp[t] for t in seq)

# ------------------------------------------------------------------
# 4-A. Simulated Annealing (with optional 3-cycle)
# ------------------------------------------------------------------

def sa_once(seq: List[str], steps: int, cycle_prob: float, T0=30.0, alpha=0.9993):
    mp = random_mapping()
    best = current = apply_map(seq, mp)
    best_s = current_s = fitness(current)
    T = T0
    for _ in range(steps):
        if random.random() > cycle_prob:
            a, b = random.sample(SYMBOLS, 2)
            mp[a], mp[b] = mp[b], mp[a]
            undo = (a, b, None)
        else:
            x, y, z = random.sample(SYMBOLS, 3)
            mp[x], mp[y], mp[z] = mp[y], mp[z], mp[x]
            undo = (x, y, z)
        cand = apply_map(seq, mp)
        cand_s = fitness(cand)
        if cand_s > current_s or random.random() < math.exp((cand_s-current_s)/T):
            current, current_s = cand, cand_s
            if cand_s > best_s:
                best, best_s = cand, cand_s
        else:
            if undo[2] is None:
                a, b = undo[:2]
                mp[a], mp[b] = mp[b], mp[a]
            else:
                x, y, z = undo
                mp[x], mp[y], mp[z] = mp[z], mp[x], mp[y]
        T *= alpha
    return best, best_s

def run_sa(seq: List[str], steps: int, restarts: int, cycle: bool=False, writer=None):
    cp = 0.2 if cycle else 0.0
    best_p, best_s = '', -1e9
    for r in range(restarts):
        p, s = sa_once(seq, steps, cp)
        if writer:
            writer.writerow([r, s, p])
        if s > best_s:
            best_p, best_s = p, s
    return best_p, best_s

# ------------------------------------------------------------------
# 4-B. Genetic Algorithm with seed & elitism
# ------------------------------------------------------------------

def derive_seed_genome(seq: List[str], plain: str) -> List[str]:
    """Build a GA genome from a seed plaintext. Assumes `plain` maps one-to-one to `seq`"""
    if len(seq) != len(plain):
        raise ValueError("seed plaintext length mismatch")
    # Build mapping symbol -> letter
    mapping: Dict[str,str] = {sym: ch for sym, ch in zip(seq, plain.upper())}
    # Construct genome ordering by SYMBOLS
    genome: List[str] = [mapping[s] for s in SYMBOLS]
    # Ensure it's a permutation (no duplicates)
    if len(set(genome)) != len(SYMBOLS):
        raise ValueError("Seed plaintext leads to duplicate letters in genome")
    return genome

def random_genome() -> List[str]:
    g = LETTERS.copy()
    random.shuffle(g)
    return g[:len(SYMBOLS)]

def genome_to_plain(seq: List[str], genome: List[str]) -> str:
    mp = {s: genome[i] for i, s in enumerate(SYMBOLS)}
    return ''.join(mp[t] for t in seq)

def mutate(genome: List[str]) -> List[str]:
    g = genome.copy()
    if random.random() < 0.6:
        a, b = random.sample(range(len(g)), 2)
        g[a], g[b] = g[b], g[a]
    else:
        x, y, z = random.sample(range(len(g)), 3)
        g[x], g[y], g[z] = g[y], g[z], g[x]
    return g

def crossover(p1: List[str], p2: List[str]) -> List[str]:
    L = len(p1)
    cut = random.randint(1, L-2)
    head = p1[:cut]
    tail = [l for l in p2 if l not in head]
    return head + tail

def tournament(pop: List[List[str]], scores: Dict[int, float], k: int=5) -> List[str]:
    best = None
    for _ in range(k):
        cand = random.choice(pop)
        if best is None or scores[id(cand)] > scores[id(best)]:
            best = cand
    return best

def run_ga(seq: List[str], pop_size: int, generations: int, seed_genome: List[str], elite_n: int, writer) -> (str, float):
    population = [seed_genome] + [random_genome() for _ in range(pop_size-1)]
    scores = {}
    def eval_g(g):
        if id(g) not in scores:
            scores[id(g)] = fitness(genome_to_plain(seq, g))
        return scores[id(g)]
    best_g = max(population, key=eval_g)
    best_s = eval_g(best_g)
    for gen in range(generations):
        elite = sorted(population, key=eval_g, reverse=True)[:elite_n]
        new_pop = elite.copy()
        while len(new_pop) < pop_size:
            p1 = tournament(population, scores)
            p2 = tournament(population, scores)
            child = crossover(p1, p2)
            if random.random() < 0.5:
                child = mutate(child)
            new_pop.append(child)
        population = new_pop
        cur_best = max(population, key=eval_g)
        cur_s = eval_g(cur_best)
        if cur_s > best_s:
            best_g, best_s = cur_best, cur_s
        writer.writerow([gen, best_s, genome_to_plain(seq, best_g)])
    return genome_to_plain(seq, best_g), best_s

# ------------------------------------------------------------------
# 5. CLI
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sa", "sa_cycle", "ga"], help="Algorithm mode")
    ap.add_argument("--route", choices=ROUTES.keys(), default="col_serp")
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--restarts", type=int, default=300)
    ap.add_argument("--pop", type=int, default=400)
    ap.add_argument("--gens", type=int, default=3000)
    ap.add_argument("--seed-plain", help="Seed plaintext for GA")
    ap.add_argument("--elite", type=int, default=None, help="Elite count for GA")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--log", default="run_log.csv")
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    seq = ROUTES[args.route]()
    log_path = pathlib.Path(args.log).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "sa": ["restart","score","plaintext"],
        "sa_cycle": ["restart","score","plaintext"],
        "ga": ["generation","score","plaintext"]
    }[args.mode]
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        if args.mode in ("sa", "sa_cycle"):
            best_p, best_s = run_sa(seq, args.steps, args.restarts, args.mode=="sa_cycle", writer)
        else:
            elite_n = args.elite if args.elite is not None else max(1, args.pop//20)
            if args.seed_plain:
                seed_genome = derive_seed_genome(seq, args.seed_plain)
            else:
                print("[GA] No seed_plain provided – starting random")
                seed_genome = random_genome()
            best_p, best_s = run_ga(seq, args.pop, args.gens, seed_genome, elite_n, writer)
    sorted_path = log_path.with_name(log_path.stem + "_sorted.csv")
    with log_path.open() as src, sorted_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        rows = list(reader)
        rows.sort(key=lambda r: float(r["score"]), reverse=True)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Best score:", best_s)
    print("Best plaintext:", best_p)
    print("Sorted log →", sorted_path)

if __name__ == "__main__":
    main()
