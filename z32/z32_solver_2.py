#!/usr/bin/env python3  # Shebang line: tells the system to run this file with Python 3
""" z32_solver.py – Automatic exhaustive solver for Zodiac Z32 cipher

Usage: python z32_solver.py

The script:

1. Loads Z32 cipher and previous ciphers (Z13, Z408, Z340).
2. Defines internally a list of grid dimensions and column permutations.
3. Computes preliminary stats: length, entropy, IC, symbol freqs, repeated n-grams.
4. Runs exhaustive grid-search on two algorithms:
   • Simulated Annealing over parameter grid
   • Genetic Algorithm over parameter grid
5. Uses combined fitness: quadgram chi-square + crib bonuses.
6. Logs all results to all_results.csv, then produces sorted and overview CSVs.

Dependencies: Python stdlib, numpy, pandas. """

import itertools         # for creating permutations and Cartesian products
import math              # for mathematical functions (log, exp, etc.)
import multiprocessing as mp  # for parallel processing (imported but unused here)
import random            # for randomness in GA and SA
import string            # for uppercase letters
from collections import Counter, defaultdict  # for counting and default dicts
import csv               # for CSV writing
import pandas as pd      # for DataFrame operations and CSV I/O
import numpy as np       # for numerical operations (imported but unused here)

# ZODIAC SIMBOLS
SHADE_TRIANGLE = '1'    # triangolo vuoto
FULL_SQUARE = '■'       #  quadrato pieno
INVERSE_K = '3'         # K specchiata
ANCORA = '4'            # symbol di un ancora
INVERSE_F = '5'         # F specchiata
FULL_TRIANGLE = '▲'     # simbolo triangolo riempito
OMEGA = '7'             # Omega
O_WITH_DOT = '0'        # Carattere O con puntino al centro
INVERSE_J = '8'         # J specchiata
HALF_SHADE_SQUARE = '9' # Quadrato pieno con un angolo bianco
ZODIAC_SYMBOL = '2'     # Simbolo di zodiac cerchio con croce
OTTO_IN_ZERO = '6'      # Simbolo usato in Z13 che sembra un 8 dentro uno zero oppure segno zodiacale cancro o ariete?

Z13 = "AEN" + ZODIAC_SYMBOL + OTTO_IN_ZERO + "K" + OTTO_IN_ZERO + "M" + OTTO_IN_ZERO + ANCORA + "NAM"
Z32 = "C" + SHADE_TRIANGLE + "JI" + FULL_SQUARE + "O" + INVERSE_K + ANCORA + "AM" + INVERSE_F + FULL_TRIANGLE + OMEGA + "ORTGX" + O_WITH_DOT + "FDV" + INVERSE_J + HALF_SHADE_SQUARE + "HCEL" + ZODIAC_SYMBOL + "PW" + SHADE_TRIANGLE

# 1. Load ciphers
previous_ciphers = {
    'Z13': Z13,  # placeholder for the Z13 cipher text
    'Z408': "...", # placeholder for the Z408 cipher text
    'Z340': "...", # placeholder for the Z340 cipher text
    'Z32': Z32  # the Z32 cipher to solve
}
CIPHER = previous_ciphers['Z32']  # select the Z32 ciphertext for processing

# 2. Define grid dims and column permutations
GRIDS_DIMENSION = [(32, 1), (4, 8), (8, 4)]  # possible grid shapes: width x height

# 3. Scoring data: quadgrams and cribs
QUADGRAMS = {}
with open('quadgrams.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2:
            quad, cnt = parts
            QUADGRAMS[quad] = int(cnt)

TOTAL_QUADGRAMS = sum(QUADGRAMS.values())  # total quadgram counts
QLOG = {k: math.log10(v / TOTAL_QUADGRAMS) for k, v in QUADGRAMS.items()}  # log frequencies
QFLOOR = math.log10(0.01 / TOTAL_QUADGRAMS)  # floor log score for unseen quadgrams
CRIBS = {'INCHES': 50, 'RADIANS': 50}  # bonus scores for known words


# Compute preliminary statistics on the ciphertext
def compute_stats(text):
    text_len = len(text)
    symbol_frequency = Counter(text)
    # Shannon entropy: -sum(p*log2(p))
    ent = -sum((c/text_len) * math.log2(c/text_len) for c in symbol_frequency.values())
    # Index of coincidence: measures text's letter repetition
    index_of_coincidence = sum(c*(c-1) for c in symbol_frequency.values())/(text_len*(text_len-1))
    # absolute and relative frequencies
    abs_freq = dict(symbol_frequency)
    rel_freq = {ch: c/text_len for ch, c in symbol_frequency.items()}
    # repeated n-grams: bigrams and trigrams occurring more than once

    def rep_ngram(k):
        cnt = Counter(text[i:i+k] for i in range(text_len-k+1))
        return {g: c for g, c in cnt.items() if c > 1}
    reps = {'bigrams': rep_ngram(2), 'trigrams': rep_ngram(3), 'quadgrams:': rep_ngram(4)}
    return {'length': text_len, 'entropy': ent, 'ic': index_of_coincidence, 'abs_freq': abs_freq,
            'rel_freq': rel_freq, 'reps': reps}


# Score plaintext candidate using quadgrams
def quad_score(pt):
    txt = ''.join(ch for ch in pt.upper() if ch.isalpha())
    # sum log probabilities for each quadgram
    return sum(QLOG.get(txt[i:i+4], QFLOOR) for i in range(len(txt)-3))


# Combined fitness: quadgram score + crib bonuses
def fitness(pt):
    score = quad_score(pt)
    up = pt.upper()
    for w, b in CRIBS.items():  # add bonuses if crib words appear
        if w in up:
            score += b
    return score


# Generate reading routes through a transposition grid
def generate_routes(cipher, W, H, colperm):
    tokens = list(cipher)
    rows = [tokens[i*W:(i+1)*W] for i in range(H)]  # split into rows
    # pad last row with placeholders if short
    if len(rows[-1])<W:
        rows[-1] += ['?']*(W-len(rows[-1]))
    # apply column permutation
    mat = [[rows[r][c] for c in colperm] for r in range(H)]
    seqs = {}
    # row-wise serpentine reading
    out = []
    for r in range(H):
        rng = range(W) if r%2==0 else range(W-1,-1,-1)
        for c in rng:
            if mat[r][c] not in ['?','□']:
                out.append(mat[r][c])
    seqs['row-serp'] = out
    # column-wise serpentine reading
    out = []
    for c in range(W):
        rng = range(H) if c%2==0 else range(H-1,-1,-1)
        for r in rng:
            if mat[r][c] not in ['?','□']:
                out.append(mat[r][c])
    seqs['col-serp'] = out
    # mixed route: half row-serp then half col-serp
    row_seq = seqs['row-serp']
    col_seq = seqs['col-serp']
    mid = len(row_seq)//2
    mixed_seq = row_seq[:mid] + col_seq[mid:]
    seqs['mixed'] = mixed_seq
    return seqs


# Simulated Annealing algorithm for substitution solving
def anneal(seq, steps, T0, alpha, cycle_prob):
    symbols = list(set(seq))  # unique cipher symbols
    # random initial mapping cipher->plaintext
    mapping = {s: random.choice(string.ascii_uppercase) for s in symbols}
    best = ''.join(mapping[s] for s in seq)  # initial plaintext
    best_s = fitness(best)
    cur, cur_s = best, best_s
    T = T0
    # hill-climb with occasional uphill moves
    for _ in range(steps):
        a,b = random.sample(symbols,2)
        mapping[a],mapping[b] = mapping[b],mapping[a]  # swap two mappings
        cand = ''.join(mapping[s] for s in seq)
        s = fitness(cand)
        # accept if better or with prob e^(delta/T)
        if s>cur_s or random.random()<math.exp((s-cur_s)/T):
            cur,cur_s = cand,s
        else:
            mapping[a],mapping[b] = mapping[b],mapping[a]  # revert swap
        T *= alpha  # cool down
    return cur,cur_s, mapping


def apply_mapping(cipher_text, mapping):
    """
    Applica il dizionario mapping al testo cifrato, sostituendo
    ogni simbolo con la corrispondente lettera in chiaro o '?' se non mappato.
    """
    return ''.join(mapping.get(sym, '?') for sym in cipher_text)


# Genetic Algorithm solver
def run_ga(seq, pop, gens, elite_frac, mut_rate, seed_plain=None):
    symbols = seq
    # optional seeded initial individual
    if seed_plain:
        if len(seed_plain)!=len(seq): raise ValueError
        seed_map = dict(zip(seq, seed_plain))
        seed_gen = [seed_map[s] for s in seq]
    # helper to generate random genome

    def rand_gen():
        let = list(string.ascii_uppercase)
        random.shuffle(let)
        return let[:len(seq)]
    popu = ([seed_gen] if seed_plain else []) + [rand_gen() for _ in range(pop - (seed_plain is not None))]
    # decode genome to text
    def decode(gen): return ''.join(gen[i] for i in range(len(seq)))
    def score_gen(gen): return fitness(decode(gen))
    best_gen = max(popu, key=score_gen)
    best_s = score_gen(best_gen)
    elite_n = max(1, int(pop*elite_frac))
    for _ in range(gens):
        popu = sorted(popu, key=score_gen, reverse=True)[:elite_n]  # select elites
        while len(popu)<pop:
            p1,p2 = random.choice(popu), random.choice(popu)
            cut = random.randint(1,len(seq)-2)
            child = p1[:cut] + [c for c in p2 if c not in p1[:cut]]  # crossover
            if random.random()<mut_rate:
                i,j = random.sample(range(len(seq)),2)
                child[i],child[j] = child[j],child[i]  # mutation
            popu.append(child)
        cur_best = max(popu, key=score_gen)
        cur_s = score_gen(cur_best)
        if cur_s>best_s:
            best_gen,best_s = cur_best,cur_s
    return decode(best_gen), best_s


# Main driver: iterate over all grid configs and parameters
def main():
    stats = compute_stats(CIPHER)
    print("Preliminary stats:", stats)  # show stats
    records = []
    # parameter grids for SA and GA
    sa_params = list(itertools.product([50000, 100000, 200000],  # steps
                                       [100, 300, 500],           # restarts
                                       [10, 30, 50],              # T0
                                       [0.9990, 0.9993, 0.9995],   # alpha
                                       [0, 0.1, 0.2]))            # cycle_prob
    ga_params = list(itertools.product([200, 400, 600],           # population
                                       [1000, 3000, 5000],       # generations
                                       [0.01, 0.05, 0.10],        # elite_frac
                                       [0.3, 0.5, 0.7]))          # mutation rate
    for GRID_W, GRID_H in GRIDS_DIMENSION:
        for colperm in itertools.permutations(range(GRID_W)):
            # generate each reading route
            for name, seq in generate_routes(CIPHER, GRID_W, GRID_H, colperm).items():
                print(f"[CONFIG] W={GRID_W}, H={GRID_H}, route={name}, colperm={colperm}")
                # SA grid search
                for steps, restarts, T0, alpha, cp in sa_params:
                    best_s = -1e9
                    best_p = ""
                    for r in range(restarts):
                        p, s, mapping = anneal(seq, steps, T0, alpha, cp)
                        if s > best_s:
                            best_s, best_p = s, p
                            compare = apply_mapping(previous_ciphers['Z13'], mapping)
                            print(f"[BEST] score={best_s}, text={best_p},Z13={compare}, map= {mapping}")
                    records.append(['SA', GRID_W, GRID_H, colperm, name, steps, restarts, T0, alpha, cp, best_s, best_p])
                # GA grid search
                for pop, gens, ef, mr in ga_params:
                    p, s = run_ga(seq, pop, gens, ef, mr)
                    records.append(['GA', GRID_W, GRID_H, colperm, name, pop, gens, ef, mr, s, p])

    # save results
    cols = ['alg','W','H','colperm','route'] + ['p1','p2','p3','p4','p5'] + ['score','plaintext']
    df = pd.DataFrame(records, columns=cols)
    df.to_csv('all_results.csv', index=False)
    df.sort_values('score', ascending=False).to_csv('all_results_sorted.csv', index=False)
    # overview top results
    top_sa = df[df.alg=='SA'].head(10)
    top_ga = df[df.alg=='GA'].head(10)
    pd.concat([top_sa, top_ga]).to_csv('best_overview.csv', index=False)
    print("Done. See all_results.csv, all_results_sorted.csv, best_overview.csv")

if __name__ == '__main__':
    main()  # entry point: run main() when executed as script
