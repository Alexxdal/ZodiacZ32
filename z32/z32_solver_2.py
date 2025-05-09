#!/usr/bin/env python3
"""
z32_solver.py – Automatic exhaustive solver for Zodiac Z32 cipher

Usage:
  python z32_solver.py

The script:
 1. Loads Z32 cipher and previous ciphers (Z13, Z408, Z340).
 2. Defines internally a list of grid dimensions and column permutations.
 3. Computes preliminary stats: length, entropy, IC, symbol freqs, repeated n-grams.
 4. Runs exhaustive grid-search on two algorithms:
     • Simulated Annealing over parameter grid
     • Genetic Algorithm over parameter grid
 5. Uses combined fitness: quadgram chi-square + crib bonuses.
 6. Logs all results to all_results.csv, then produces sorted and overview CSVs.

Dependencies: Python stdlib, numpy, pandas.
"""
import itertools
import math
import multiprocessing as mp
import random
import string
from collections import Counter, defaultdict
import csv
import pandas as pd
import numpy as np

# 1. Load ciphers
previous_ciphers = {
    'Z13': "...",  # fill as needed
    'Z408': "...",
    'Z340': "...",
    'Z32': "CIFELNIOWHDAΩNGOAOESNBX□TCETDIEI"
}
CIPHER = previous_ciphers['Z32']

# 2. Define dimensions and column perms
# Example dims list; adjust if needed
DIMS = [(4,8), (8,4)]

# 3. Scoring data
QGRAM = {"TION":126024, "THER":113290, "HERE":96550, "WITH":70160, "IGHT":77290}
QTOT = sum(QGRAM.values())
QLOG = {k: math.log10(v/QTOT) for k,v in QGRAM.items()}
QFLOOR = math.log10(0.01/QTOT)
CRIBS = {'STATION':50, 'THERE':30, 'WITH':25, 'HERE':20, 'THREE':20}

# Preliminary statistics
def compute_stats(text):
    n = len(text)
    # Shannon entropy
    freqs = Counter(text)
    ent = -sum((c/n)*math.log2(c/n) for c in freqs.values())
    # Index of coincidence
    ic = sum(c*(c-1) for c in freqs.values())/(n*(n-1))
    # frequencies
    abs_freq = dict(freqs)
    rel_freq = {ch: c/n for ch,c in freqs.items()}
    # repeated ngrams
    def rep_ngram(k):
        cnt = Counter(text[i:i+k] for i in range(n-k+1))
        return {g:c for g,c in cnt.items() if c>1}
    reps = {'bigrams': rep_ngram(2), 'trigrams': rep_ngram(3)}
    return {'length':n, 'entropy':ent, 'ic':ic,
            'abs_freq':abs_freq, 'rel_freq':rel_freq, 'reps':reps}

# Fitness
def quad_score(pt):
    txt = ''.join(ch for ch in pt.upper() if ch.isalpha())
    return sum(QLOG.get(txt[i:i+4], QFLOOR) for i in range(len(txt)-3))

def fitness(pt):
    score = quad_score(pt)
    up = pt.upper()
    for w,b in CRIBS.items():
        if w in up: score += b
    return score

# Route generator
def generate_routes(cipher, W, H, colperm):
    tokens = list(cipher)
    rows = [tokens[i*W:(i+1)*W] for i in range(H)]
    if len(rows[-1])<W: rows[-1] += ['?']*(W-len(rows[-1]))
    mat = [[rows[r][c] for c in colperm] for r in range(H)]
    # both row and col serpentine
    seqs = {}
    # row-serp
    out=[]
    for r in range(H):
        rng = range(W) if r%2==0 else range(W-1,-1,-1)
        for c in rng:
            if mat[r][c] not in ['?','□']: out.append(mat[r][c])
    seqs['row-serp'] = out
    # col-serp
    out=[]
    for c in range(W):
        rng = range(H) if c%2==0 else range(H-1,-1,-1)
        for r in rng:
            if mat[r][c] not in ['?','□']: out.append(mat[r][c])
    seqs['col-serp'] = out
    return seqs

# Simulated annealing
def anneal(seq, steps, T0, alpha, cycle_prob):
    symbols = list(set(seq))
    mapping = {s:random.choice(string.ascii_uppercase) for s in symbols}
    best = ''.join(mapping[s] for s in seq)
    best_s = fitness(best)
    cur = best; cur_s = best_s; T=T0
    for _ in range(steps):
        a,b = random.sample(symbols,2)
        mapping[a],mapping[b] = mapping[b],mapping[a]
        cand = ''.join(mapping[s] for s in seq)
        s = fitness(cand)
        if s>cur_s or random.random()<math.exp((s-cur_s)/T): cur,cur_s=cand,s
        else: mapping[a],mapping[b]=mapping[b],mapping[a]
        T *= alpha
    return cur,cur_s

# Genetic algorithm
def run_ga(seq, pop, gens, elite_frac, mut_rate, seed_plain=None):
    symbols = seq
    # decode seed
    if seed_plain:
        if len(seed_plain)!=len(seq): raise ValueError
        seed_map = dict(zip(seq, seed_plain))
        seed_gen = [seed_map[s] for s in seq]
    # population as list of genomes (list of letters)
    def rand_gen():
        let=list(string.ascii_uppercase); random.shuffle(let)
        return let[:len(seq)]
    popu = ([seed_gen] if seed_plain else []) + [rand_gen() for _ in range(pop - (seed_plain is not None))]
    def decode(gen): return ''.join(gen[i] for i in range(len(seq)))
    def score_gen(gen): return fitness(decode(gen))
    best_gen = max(popu, key=score_gen); best_s=score_gen(best_gen)
    elite_n = max(1, int(pop*elite_frac))
    for g in range(gens):
        # selection+elitism
        popu = sorted(popu, key=score_gen, reverse=True)[:elite_n]
        # fill
        while len(popu)<pop:
            p1,p2 = random.choice(popu), random.choice(popu)
            cut=random.randint(1,len(seq)-2)
            child = p1[:cut] + [c for c in p2 if c not in p1[:cut]]
            if random.random()<mut_rate:
                i,j=random.sample(range(len(seq)),2); child[i],child[j]=child[j],child[i]
            popu.append(child)
        cur_best = max(popu, key=score_gen); cur_s=score_gen(cur_best)
        if cur_s>best_s: best_gen,best_s=cur_best,cur_s
    return decode(best_gen), best_s

# Main grid search
def main():
    stats = compute_stats(CIPHER)
    print("Preliminary stats:", stats)
    records=[]
    # param grids
    sa_params = list(itertools.product([50000,100000,200000],[100,300,500],[10,30,50],[0.9990,0.9993,0.9995],[0,0.1,0.2]))
    ga_params = list(itertools.product([200,400,600],[1000,3000,5000],[0.01,0.05,0.10],[0.3,0.5,0.7]))
    for W,H in DIMS:
        # iterate over each grid configuration
        for colperm in itertools.permutations(range(W)):
            for name, seq in generate_routes(CIPHER, W, H, colperm).items():
                print(f"[CONFIG] W={W}, H={H}, route={name}, colperm={colperm}")
                # SA grid

                for steps,restarts,T0,alpha,cp in sa_params:
                    best_s = -1e9; best_p=""
                    for r in range(restarts):
                        p,s=anneal(seq,steps,T0,alpha,cp)
                        if s>best_s: best_s, best_p = s,p
                    print(f"SA W={W}, H={H}, colperm={colperm}, name={name}, restarts={restarts}, T0={T0}, alpha={alpha}, cp={cp}, best_s={best_s}, best_p={best_p}")
                    records.append(['SA',W,H,colperm,name,steps,restarts,T0,alpha,cp,best_s,best_p])
                # GA grid
                for pop,gens,ef, mr in ga_params:
                    p,s = run_ga(seq,pop,gens,ef,mr)
                    records.append(['GA',W,H,colperm,name,pop,gens,ef,mr,s,p])
    # save
    cols = ['alg','W','H','colperm','route'] + ['p1','p2','p3','p4','p5'] + ['score','plaintext']
    df=pd.DataFrame(records, columns=cols)
    df.to_csv('all_results.csv', index=False)
    df.sort_values('score', ascending=False).to_csv('all_results_sorted.csv', index=False)
    # overview
    top_sa = df[df.alg=='SA'].head(10)
    top_ga = df[df.alg=='GA'].head(10)
    pd.concat([top_sa, top_ga]).to_csv('best_overview.csv', index=False)
    print("Done. See all_results.csv, all_results_sorted.csv, best_overview.csv")

if __name__=='__main__':
    main()