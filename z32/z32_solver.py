#!/usr/bin/env python3
"""Z32 Solver – Simulated Annealing Batch (improved logging)

Usage examples
--------------
Run the standard row‑serpentine route with a big sweep and keep logs::

    python z32_solver.py --route row_serp --steps 100000 --restarts 500 \
                         --seed 42 --log results_live.csv

The script now **flushes each result to disk immediately** so you can tail/grep
while the batch is running.  When all restarts finish, it automatically
produces a **sorted copy** named ``<LOG>_sorted.csv`` with the candidates
ordered by descending score.

Routes available
----------------
``row_serp``
    Row‑wise serpentine reading of the 4×8 matrix (perm 1‑2‑3‑0).
``col_serp``
    Column‑wise serpentine reading of the same matrix.
"""

import random
import math
import string
import argparse
import csv
import pathlib

# ------------------------------------------------------------
# Cipher setup (matrix 4×8, column permutation 1‑2‑3‑0)
# ------------------------------------------------------------
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

# ---------------- Route generators ---------------------------

def route_row_serp():
    """Row‑wise serpentine (L→R then R→L)."""
    seq = []
    for r in range(HEIGHT):
        rng = range(WIDTH) if r % 2 == 0 else range(WIDTH - 1, -1, -1)
        for c in rng:
            tok = MATRIX[r][c]
            if tok not in {"?", "□"}:
                seq.append(tok)
    return seq


def route_col_serp():
    """Column‑wise serpentine (top‑down then bottom‑up)."""
    seq = []
    for idx, c in enumerate(range(WIDTH)):
        rng = range(HEIGHT) if idx % 2 == 0 else range(HEIGHT - 1, -1, -1)
        for r in rng:
            tok = MATRIX[r][c]
            if tok not in {"?", "□"}:
                seq.append(tok)
    return seq

ROUTES = {
    "row_serp": route_row_serp,
    "col_serp": route_col_serp,
}

# ---------------- Scoring ---------------------------
QGRAM = {
    "TION": 126024,
    "THER": 113290,
    "HERE": 96550,
    "WITH": 70160,
    "IGHT": 77290,
}
QTOTAL = sum(QGRAM.values())
QLOG = {k: math.log10(v / QTOTAL) for k, v in QGRAM.items()}
QFLOOR = math.log10(0.01 / QTOTAL)

CRIBS = {
    "STATION": 50,
    "THERE": 30,
    "WITH": 25,
    "HERE": 20,
    "THREE": 20,
}


def quad_score(text: str) -> float:
    t = "".join(ch for ch in text.upper() if ch.isalpha())
    return sum(QLOG.get(t[i : i + 4], QFLOOR) for i in range(len(t) - 3))


def fitness(text: str) -> float:
    score = quad_score(text)
    up = text.upper()
    for w, bonus in CRIBS.items():
        if w in up:
            score += bonus
    return score

# ---------------- Simulated annealing ------------------------

def random_mapping(symbols):
    letters = list(string.ascii_uppercase)
    random.shuffle(letters)
    return {s: letters[i] for i, s in enumerate(symbols)}


def apply_map(seq, mapping):
    return "".join(mapping[t] for t in seq)


def anneal(seq, symbols, steps, T0=40.0, alpha=0.9993):
    mp = random_mapping(symbols)
    best = current = apply_map(seq, mp)
    best_s = current_s = fitness(current)
    T = T0

    for _ in range(steps):
        use_cycle = random.random() >= 0.8    # 20 % dei casi
        if not use_cycle:
            # ---------- swap a 2 ----------
            a, b = random.sample(symbols, 2)
            mp[a], mp[b] = mp[b], mp[a]
        else:
            # ---------- rotazione a 3 ----------
            x, y, z = random.sample(symbols, 3)
            mp[x], mp[y], mp[z] = mp[y], mp[z], mp[x]

        candidate = apply_map(seq, mp)
        cand_s = fitness(candidate)

        accept = cand_s > current_s or random.random() < math.exp((cand_s - current_s) / T)
        if accept:
            current, current_s = candidate, cand_s
            if cand_s > best_s:
                best, best_s = candidate, cand_s
        else:
            # ---- undo senza pop ----
            if not use_cycle:
                mp[a], mp[b] = mp[b], mp[a]
            else:
                mp[x], mp[y], mp[z] = mp[z], mp[x], mp[y]

        T *= alpha
    return best, best_s

# ---------------- Main CLI wrapper ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=ROUTES.keys(), default="row_serp")
    ap.add_argument("--steps", type=int, default=50000, help="Annealing steps per restart")
    ap.add_argument("--restarts", type=int, default=200, help="Number of random restarts")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", default="results.csv", help="CSV file for live logging")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    seq = ROUTES[args.route]()
    symbols = sorted(set(seq))

    log_path = pathlib.Path(args.log).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_plain, best_score = "", -1e9

    # Live logging (append mode) ------------------------------------
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["restart", "score", "plaintext"])
        for r in range(args.restarts):
            plain, score = anneal(seq, symbols, steps=args.steps)
            writer.writerow([r, score, plain])
            f.flush()  # <-- immediate flush for live monitoring
            if score > best_score:
                best_plain, best_score = plain, score

    # Post‑sort into a separate file --------------------------------
    sorted_path = log_path.with_name(log_path.stem + "_sorted.csv")
    with log_path.open() as src, sorted_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        rows = list(reader)
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nBest score:", best_score)
    print("Best plaintext:", best_plain)
    print("Sorted log saved to", sorted_path)


if __name__ == "__main__":
    main()