#!/usr/bin/env python3
"""Z32 Solver – Genetic Algorithm (GA) con diversità dinamica e fitness linguistica realistica."""

import argparse
import csv
import math
import pathlib
import random
import string

# ------------------------------------------------------------------
# 1. Cipher matrix & routes
# ------------------------------------------------------------------
TOKENS = [
    "C", "I", "F", "E", "L", "N", "I", "O",
    "W", "H", "D", "A", "Ω", "N", "G", "O",
    "A", "O", "E", "S", "N", "B", "X", "□",
    "T", "C", "E", "T", "D", "I", "E", "I",
]
WIDTH, HEIGHT = 4, 8
COL_PERM = [1, 2, 3, 0]
NULLS = {"?", "□"}

rows = [TOKENS[i:i + WIDTH] for i in range(0, len(TOKENS), WIDTH)]
if len(rows) < HEIGHT:
    rows += [["?"] * WIDTH] * (HEIGHT - len(rows))
MATRIX = [[rows[r][c] for c in COL_PERM] for r in range(HEIGHT)]

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

SYMBOLS = sorted(set(route_row_serp()))
SYM_IDX = {s: i for i, s in enumerate(SYMBOLS)}

# ------------------------------------------------------------------
# 2. Fitness (quadgrams + cribs)
# ------------------------------------------------------------------
def load_quadgrams(path="english_quadgrams.txt"):
    quad_freqs = {}
    total = 0
    with open(path, "r") as f:
        for line in f:
            quad, freq = line.strip().split()
            quad_freqs[quad] = int(freq)
            total += int(freq)
    qlog = {k: math.log10(v / total) for k, v in quad_freqs.items()}
    qfloor = math.log10(0.01 / total)
    return qlog, qfloor

QLOG, QFLOOR = load_quadgrams()

CRIBS = {
    "STATION": 50, "THERE": 30, "WITH": 25,
    "HERE": 20, "THREE": 20, "KILL": 15, "ZODIAC": 40,
}

def quad_score(txt: str) -> float:
    clean = "".join(ch for ch in txt.upper() if ch.isalpha())
    return sum(QLOG.get(clean[i:i + 4], QFLOOR) for i in range(len(clean) - 3))

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

def random_genome():
    g = LETTERS.copy()
    random.shuffle(g)
    return g[:len(SYMBOLS)]

def genome_to_plain(seq, genome):
    mapping = {s: genome[SYM_IDX[s]] for s in SYMBOLS}
    return "".join(mapping[t] for t in seq)

def mutate(g):
    g = g.copy()
    if random.random() < 0.6:
        a, b = random.sample(range(len(g)), 2)
        g[a], g[b] = g[b], g[a]
    else:
        x, y, z = random.sample(range(len(g)), 3)
        g[x], g[y], g[z] = g[y], g[z], g[x]
    return g

def crossover(p1, p2):
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
    pop = [random_genome() for _ in range(pop_size)]
    scores = {}

    def eval_genome(g):
        if id(g) not in scores:
            scores[id(g)] = fitness(genome_to_plain(seq, g))
        return scores[id(g)]

    best_g = max(pop, key=eval_genome)
    best_score = eval_genome(best_g)
    print("Initial unique genomes:", len({tuple(g) for g in pop}))

    for gen in range(generations):
        elite_n = max(1, pop_size // 20)
        elite = sorted(pop, key=eval_genome, reverse=True)[:elite_n]
        new_pop = elite.copy()

        while len(new_pop) < pop_size:
            p1, p2 = tournament(pop, scores), tournament(pop, scores)
            child = crossover(p1, p2)
            if random.random() < 0.5:
                child = mutate(child)
            new_pop.append(child)

        # --- Iniezione di diversità se necessaria ---
        diversity = len({tuple(g) for g in new_pop})
        if diversity < pop_size * 0.5:
            print(f"[Gen {gen}] ⚠️ Bassa diversità ({diversity}). Re-iniezione...")
            inject_n = pop_size // 10
            for _ in range(inject_n):
                new_pop[random.randint(0, pop_size - 1)] = random_genome()

            # Forza mutazioni del best genome
            for _ in range(3):
                mutant = mutate(best_g)
                new_pop[random.randint(0, pop_size - 1)] = mutant

        pop = new_pop

        cur_best = max(pop, key=eval_genome)
        cur_score = eval_genome(cur_best)
        if cur_score > best_score:
            best_g, best_score = cur_best, cur_score

        log_writer.writerow([gen, best_score, genome_to_plain(seq, best_g)])

        if gen % 100 == 0:
            print(f"Gen {gen} | Score: {best_score:.2f} | Unique: {diversity}")

    return genome_to_plain(seq, best_g), best_score

# ------------------------------------------------------------------
# 5. CLI
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=ROUTES.keys(), default="col_serp")
    ap.add_argument("--pop", type=int, default=400)
    ap.add_argument("--gens", type=int, default=5000)
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

    sorted_path = log_path.with_name(log_path.stem + "_sorted.csv")
    with log_path.open() as src, sorted_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        rows = list(reader)
        rows.sort(key=lambda r: float(r["score"]), reverse=True)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Best score:", best_score)
    print("Best plaintext:", best_plain)
    print("Sorted log saved to:", sorted_path)

if __name__ == "__main__":
    main()
