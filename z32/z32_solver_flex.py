#!/usr/bin/env python3
"""Z32 Solver – Flexible Transposition Explorer

Extends the original SA solver to sweep multiple grid dimensions (W×H)
and column permutations. Logs the best result for each configuration,
so you can identify which layout yields the highest fitness.

Usage:
    python z32_solver_flex.py --dims 4x8 8x4 \
        --restarts 200 --steps 100000 --seed 42 \
        --log flex_results.csv
"""
import argparse
import csv
import itertools
import math
import random
import string
import pathlib
# (Assuming TOKENS length is 32)
from typing import List, Tuple

# Base tokens (Z32)
TOKENS = [
    "C","I","F","E","L","N","I","O",
    "W","H","D","A","Ω","N","G","O",
    "A","O","E","S","N","B","X","□",
    "T","C","E","T","D","I","E","I",
]
LETTERS = list(string.ascii_uppercase)
CRIBS = {"STATION":50,"THERE":30,"WITH":25,"HERE":20,"THREE":20}
# Quadgram as before
QGRAM = {"TION":126024,"THER":113290,"HERE":96550,"WITH":70160,"IGHT":77290}
QTOT = sum(QGRAM.values())
QLOG = {k: math.log10(v/QTOT) for k,v in QGRAM.items()}
QFLOOR = math.log10(0.01/QTOT)

# Scoring functions
def quad_score(txt: str) -> float:
    clean = ''.join(ch for ch in txt.upper() if ch.isalpha())
    return sum(QLOG.get(clean[i:i+4], QFLOOR) for i in range(len(clean)-3))

def fitness(txt: str) -> float:
    s = quad_score(txt)
    up = txt.upper()
    for w,b in CRIBS.items():
        if w in up: s += b
    return s

# Generate sequence from matrix dims and col_perm
def make_routes(tokens: List[str], W: int, H: int, col_perm: Tuple[int]) -> List[str]:
    # build rows
    rows = [tokens[i*W:(i+1)*W] for i in range(H)]
    # pad if needed
    if len(rows[-1]) < W:
        rows[-1] += ['?']*(W-len(rows[-1]))
    # apply col_perm
    M = [[rows[r][c] for c in col_perm] for r in range(H)]
    # column-wise serp
    seq = []
    for idx,c in enumerate(range(W)):
        rng = range(H) if idx%2==0 else range(H-1,-1,-1)
        for r in rng:
            tok = M[r][c]
            if tok not in {'?', '□'}:
                seq.append(tok)
    return seq

# Simulated annealing per-sequence
def anneal(seq: List[str], steps: int, T0=40.0, alpha=0.9993) -> Tuple[str,float]:
    symbols = sorted(set(seq))
    mp = {s: random.choice(LETTERS) for s in symbols}
    best = cur = ''.join(mp[t] for t in seq)
    best_s = cur_s = fitness(cur)
    T = T0
    for _ in range(steps):
        a,b = random.sample(symbols,2)
        mp[a],mp[b] = mp[b],mp[a]
        cand = ''.join(mp[t] for t in seq)
        s = fitness(cand)
        if s>cur_s or random.random()<math.exp((s-cur_s)/T):
            cur,cur_s = cand,s
            if s>best_s: best,best_s = cand,s
        else:
            mp[a],mp[b] = mp[b],mp[a]
        T *= alpha
    return best,best_s

# Main
if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dims', nargs='+', required=True,
        help='Grid dimensions WxH, e.g. 4x8 8x4')
    ap.add_argument('--restarts', type=int, default=100)
    ap.add_argument('--steps', type=int, default=50000)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--log', default='flex_results.csv')
    args = ap.parse_args()
    if args.seed is not None: random.seed(args.seed)

    # prepare log
    logp = pathlib.Path(args.log)
    with logp.open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['W','H','col_perm','best_score','best_plain'])
        # iterate dims and perms
        for d in args.dims:
            W,H = map(int,d.split('x'))
            # all column perms of range(W)
            for col_perm in itertools.permutations(range(W)):
                seq = make_routes(TOKENS, W,H,col_perm)
                best_score,best_plain = -1e9,'')
                # multi-restart SA
                for r in range(args.restarts):
                    p,s = anneal(seq,args.steps)
                    if s>best_score:
                        best_score,best_plain = s,p
                writer.writerow([W,H,col_perm,best_score,best_plain])
    print('Sweep complete; results in',args.log)