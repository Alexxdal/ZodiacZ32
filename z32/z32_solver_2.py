#!/usr/bin/env python3
"""
z32_solver.py – Enhanced solver for Zodiac Z32 cipher with debug fix

Features:
  • Parametric route generators: serpentine, rail-fence(N), spiral (cw/ccw), diagonals, zigzag blocks
  • Combinatorial mixing of route patterns
  • CLI with argparse: cipher input, grids, SA/GA params, cribs, early-stop, hill-climb, seeding, ensemble
  • Parallel grid search via concurrent.futures
  • Early-stopping in SA
  • Optional hill-climbing post-processing, n-gram based seeding, ensemble voting
  • Debug logging with --debug flag

Dependencies: Python 3.7+, numpy, pandas, tqdm
"""
import argparse
import math
import random
import string
import time
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------------- N-gram Loaders -----------------------------
def load_ngrams(path, k):
    freqs = {}
    with open(path, 'r') as f:
        for line in f:
            gram, count = line.split()
            if len(gram) == k:
                freqs[gram] = int(count)
    total = sum(freqs.values())
    logp = {g: math.log10(v/total) for g,v in freqs.items()}
    floor = math.log10(0.01/total)
    return logp, floor

# ----------------------------- Statistics -----------------------------
def compute_stats(text):
    n = len(text)
    freqs = Counter(text)
    ent = -sum((c/n) * math.log2(c/n) for c in freqs.values())
    ic = sum(c*(c-1) for c in freqs.values())/(n*(n-1)) if n>1 else 0
    return {'length':n,'entropy':ent,'ic':ic}

# ----------------------------- Route Generators -----------------------------
def route_serpentine(cipher, width, height, colperm):
    tokens = list(cipher)
    grid = [tokens[i*width:(i+1)*width] for i in range(height)]
    if len(grid[-1])<width: grid[-1] += ['?']*(width-len(grid[-1]))
    mat = np.array(grid)[:, list(colperm)]
    seqs = {}
    # row-serp
    out=[]
    for r in range(height):
        cols = range(width) if r%2==0 else range(width-1,-1,-1)
        for c in cols:
            ch=mat[r,c]
            if ch not in('?', '□'): out.append(ch)
    seqs['serp_row']=out
    # col-serp
    out=[]
    for c in range(width):
        rows = range(height) if c%2==0 else range(height-1,-1,-1)
        for r in rows:
            ch=mat[r,c]
            if ch not in('?', '□'): out.append(ch)
    seqs['serp_col']=out
    return seqs

def route_rail(cipher, width, height, colperm, rails):
    seqs={}
    fence=['']*rails
    rail,dir=0,1
    for ch in cipher:
        if ch in('?', '□'): continue
        fence[rail]+=ch
        rail+=dir
        if rail==rails-1 or rail==0: dir*=-1
    seqs[f'rail_{rails}']=list(''.join(fence))
    return seqs

def route_spiral(cipher, width, height, colperm, clockwise=True):
    tokens=list(cipher)
    grid=[tokens[i*width:(i+1)*width] for i in range(height)]
    if len(grid[-1])<width: grid[-1]+=['?']*(width-len(grid[-1]))
    mat=np.array(grid)
    seq=[]
    visited=np.zeros(mat.shape, bool)
    dirs = [(0,1),(1,0),(0,-1),(-1,0)] if clockwise else [(1,0),(0,1),(-1,0),(0,-1)]
    r,c,di=0,0,0
    for _ in range(width*height):
        if not visited[r,c] and mat[r,c] not in('?', '□'): seq.append(mat[r,c])
        visited[r,c]=True
        nr,nc=r+dirs[di][0],c+dirs[di][1]
        if nr<0 or nr>=height or nc<0 or nc>=width or visited[nr,nc]:
            di=(di+1)%4
            nr,nc=r+dirs[di][0],c+dirs[di][1]
        r,c=nr,nc
    return {f"spiral_{'cw' if clockwise else 'ccw'}":seq}

def route_diagonal(cipher, width, height, colperm, down_right=True):
    tokens=list(cipher)
    grid=[tokens[i*width:(i+1)*width] for i in range(height)]
    if len(grid[-1])<width: grid[-1]+=['?']*(width-len(grid[-1]))
    mat=np.array(grid)
    seq=[]
    if down_right:
        for s in range(width+height-1):
            for r in range(height):
                c=s-r
                if 0<=c<width and mat[r,c] not in('?', '□'):
                    seq.append(mat[r,c])
    else:
        for s in range(width+height-1):
            for r in range(height-1,-1,-1):
                c=s-(height-1-r)
                if 0<=c<width and mat[r,c] not in('?', '□'):
                    seq.append(mat[r,c])
    key=f"diag_{'dr' if down_right else 'ul'}"
    return {key:seq}

def route_zigzag_blocks(cipher, width, height, colperm, block_h, block_w):
    tokens=list(cipher)
    grid=[tokens[i*width:(i+1)*width] for i in range(height)]
    if len(grid[-1])<width: grid[-1]+=['?']*(width-len(grid[-1]))
    seq=[]
    for br in range(0,height,block_h):
        for bc in range(0,width,block_w):
            for r in range(br, min(br+block_h,height)):
                row_range=range(bc, min(bc+block_w,width)) if (r-br)%2==0 else range(min(bc+block_w,width)-1, bc-1, -1)
                for c in row_range:
                    ch=grid[r][c]
                    if ch not in('?', '□'): seq.append(ch)
    return {f"zigzag_{block_h}x{block_w}":seq}

def mix_routes(seq1, seq2):
    mid=len(seq1)//2
    return seq1[:mid]+seq2[mid:]

# ----------------------------- Fitness -----------------------------
def score_quad(text, qlog, floor):
    txt=''.join(ch for ch in text.upper() if ch.isalpha())
    return sum(qlog.get(txt[i:i+4], floor) for i in range(len(txt)-3))

def fitness(text, qlog, floor, cribs):
    s=score_quad(text, qlog, floor)
    up=text.upper()
    for w,b in cribs.items():
        if w in up: s+=b
    return s

# ----------------------------- Optimizers -----------------------------
def simulated_annealing(seq, steps, restarts, T0, alpha, pc, early_stop, qlog, floor, cribs, seed_map):
    best_p,best_s='',float('-inf')
    for r in range(int(restarts)):
        symbols=list(set(seq))
        mapping=seed_map.copy() if seed_map else {s:random.choice(string.ascii_uppercase) for s in symbols}
        cur=''.join(mapping[s] for s in seq)
        cur_s=fitness(cur,qlog,floor,cribs)
        T=T0; no_imp=0; local_best,local_best_s=cur,cur_s
        for i in range(int(steps)):
            a,b=random.sample(symbols,2)
            mapping[a],mapping[b]=mapping[b],mapping[a]
            cand=''.join(mapping[s] for s in seq)
            s=fitness(cand,qlog,floor,cribs)
            if s>cur_s or random.random()<math.exp((s-cur_s)/T):
                cur,cur_s=cand,s; no_imp=0
                if s>local_best_s:
                    local_best,local_best_s=cand,s
                    logging.debug(f"New local best SA r={r} it={i}: s={s}")
            else:
                mapping[a],mapping[b]=mapping[b],mapping[a]
                no_imp+=1
            T*=alpha
            if early_stop and no_imp>early_stop:
                logging.debug(f"Early stop SA r={r} at iter={i}")
                break
            if i%max(1,int(steps/10))==0:
                logging.debug(f"SA progress r={r} it={i}/{steps} cur_s={cur_s}")
        if local_best_s>best_s:
            best_p,best_s=local_best,local_best_s
    return ('SA',steps,restarts,T0,alpha,pc,best_s,best_p)

def genetic_algorithm(seq, population, gens, ef, mr, qlog, floor, cribs, seed_map):
    def make_gen():
        return [seed_map[s] for s in seq] if seed_map else random.sample(list(string.ascii_uppercase), len(seq))
    pop=[make_gen()] + [make_gen() for _ in range(int(population)-1)]
    def score_gen(gen): return fitness(''.join(gen),qlog,floor,cribs)
    elite_n=max(1,int(population*ef))
    best_gen=max(pop,key=score_gen); best_s=score_gen(best_gen)
    for g in range(int(gens)):
        pop=sorted(pop,key=score_gen,reverse=True)[:elite_n]
        offspring=[]
        while len(offspring)<population-elite_n:
            p1,p2=random.sample(pop,2)
            cut=random.randint(1,len(seq)-2)
            child=p1[:cut] + [c for c in p2 if c not in p1[:cut]]
            if random.random()<mr:
                i,j=random.sample(range(len(seq)),2); child[i],child[j]=child[j],child[i]
            offspring.append(child)
        pop+=offspring
        curr_best=max(pop,key=score_gen); curr_s=score_gen(curr_best)
        if curr_s>best_s:
            best_gen,best_s=curr_best,curr_s
            logging.debug(f"New best GA gen={g}: s={best_s}")
        if g%max(1,int(gens/10))==0:
            logging.debug(f"GA progress gen={g}/{gens} best_s={best_s}")
    return ('GA',population,gens,ef,mr,best_s,''.join(best_gen))

# Hill climbing post-process
def hill_climb(text,qlog,floor,cribs,steps=1000):
    best=text; best_s=fitness(text,qlog,floor,cribs)
    for _ in range(steps):
        i,j=random.sample(range(len(text)),2)
        cand=list(best)
        cand[i],cand[j]=cand[j],cand[i]
        s=fitness(''.join(cand),qlog,floor,cribs)
        if s>best_s: best,best_s=''.join(cand),s
    return best,best_s

# Ensemble

def ensemble_plaintexts(texts,qlog,floor,cribs):
    L=len(texts[0])
    vote=[]
    for i in range(L):
        freqs=Counter(t[i] for t in texts)
        vote.append(freqs.most_common(1)[0][0])
    pt=''.join(vote)
    return pt,fitness(pt,qlog,floor,cribs)

# Worker wrappers

def worker_sa(task): return simulated_annealing(*task)
def worker_ga(task): return genetic_algorithm(*task)

# Main

def main():
    parser=argparse.ArgumentParser(description="Enhanced Z32 solver")
    parser.add_argument('--cipher',required=True,help="Ciphertext or file path")
    parser.add_argument('--dims',nargs='+',default=['4,8','8,4'],help="Grid dims W,H")
    parser.add_argument('--sa-params',nargs='+',default=['50000,10,100,0.9993,0.1'],help="steps,restarts,T0,alpha,cycle_prob")
    parser.add_argument('--ga-params',nargs='+',default=['400,3000,0.5,0.05'],help="pop,gens,elite_frac,mut_rate")
    parser.add_argument('--cribs',nargs='*',default=[],help="Cribs WORD:BONUS")
    parser.add_argument('--quad-file',required=True,help="Quadgram file path")
    parser.add_argument('--tri-file',help="Trigram file path (optional)")
    parser.add_argument('--seed-ngram',help="N-gram file for seeding mappings")
    parser.add_argument('--early-stop',type=int,default=0,help="Early-stop SA if no improve k iters")
    parser.add_argument('--hill-climb',action='store_true',help="Enable hill climbing post")
    parser.add_argument('--ensemble',action='store_true',help="Enable ensemble of top plaintexts")
    parser.add_argument('--top',type=int,default=10,help="Top N to output")
    parser.add_argument('--debug',action='store_true',help="Enable debug logging")
    parser.add_argument('--output',default='results.csv',help="Output CSV")
    args=parser.parse_args()

    log_level=logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level,format='%(asctime)s %(levelname)s: %(message)s')

    # Load cipher
    txt=args.cipher
    if len(txt)>1 and not txt.isalpha():
        with open(txt) as f: txt=f.read().strip()

    # Load ngrams
    qlog,qfloor=load_ngrams(args.quad_file,4)
    trilog,tri_floor=(load_ngrams(args.tri_file,3) if args.tri_file else (None,None))
    seed_map=None
    if args.seed_ngram:
        # placeholder seeding implementation
        seed_map={s:random.choice(string.ascii_uppercase) for s in set(txt)}

    # Cribs
    cribs={w.upper():float(b) for w,b in (c.split(':') for c in args.cribs)}

    stats=compute_stats(txt)
    logging.info(f"Stats len={stats['length']} ent={stats['entropy']:.3f} IC={stats['ic']:.4f}")

    # Build sequences
    dims=[tuple(map(int,d.split(','))) for d in args.dims]
    seqs={}
    for W,H in dims:
        perms=set(__import__('itertools').permutations(range(W)))
        for perm in perms:
            seqs.update(route_serpentine(txt,W,H,perm))
            seqs.update(route_spiral(txt,W,H,perm,True))
            seqs.update(route_spiral(txt,W,H,perm,False))
            seqs.update(route_diagonal(txt,W,H,perm,True))
            seqs.update(route_diagonal(txt,W,H,perm,False))
            seqs.update(route_zigzag_blocks(txt,W,H,perm,2,2))
            seqs.update(route_zigzag_blocks(txt,W,H,perm,3,3))
            seqs.update(route_rail(txt,W,H,perm,H))
    names=list(seqs.keys())
    # Mix first few pairs
    for i in range(min(5,len(names))):
        for j in range(i+1,min(5,len(names))):
            seqs[f"mix_{names[i]}_{names[j]}"]=mix_routes(seqs[names[i]],seqs[names[j]])

    # Prepare tasks
    sa_tasks=[]; ga_tasks=[]
    for name,seq in seqs.items():
        for p in args.sa_params:
            steps,restarts,T0,alpha,pc=map(float,p.split(','))
            sa_tasks.append((seq,steps,restarts,T0,alpha,pc,args.early_stop,qlog,qfloor,cribs,seed_map))
        for p in args.ga_params:
            pop,gens,ef,mr=map(float,p.split(','))
            ga_tasks.append((seq,pop,gens,ef,mr,qlog,qfloor,cribs,seed_map))

    # Execute
    start=time.time(); results=[]
    with ProcessPoolExecutor() as exe:
        futures=[exe.submit(worker_sa,t) for t in sa_tasks] + [exe.submit(worker_ga,t) for t in ga_tasks]
        for f in tqdm(as_completed(futures),total=len(futures)):
            results.append(f.result())
    logging.info(f"Search done in {time.time()-start:.1f}s")

    # Compile results
    df=pd.DataFrame(results,columns=['alg','p1','p2','p3','p4','p5','score','plaintext'])
    df.sort_values('score',ascending=False,inplace=True)
    top_df=df.head(args.top)
    if args.hill_climb:
        refined=[{'alg':'HC','p1':None,'p2':None,'p3':None,'p4':None,'p5':None,'score':s,'plaintext':pt}
                 for pt,(pt,s) in [(pt, hill_climb(pt,qlog,qfloor,cribs)) for pt in top_df['plaintext']]]
        top_df=pd.concat([top_df,pd.DataFrame(refined)],ignore_index=True).sort_values('score',ascending=False)
    if args.ensemble:
        ep,es=ensemble_plaintexts(list(top_df['plaintext'][:args.top]),qlog,qfloor,cribs)
        top_df=top_df.append({'alg':'ENS','p1':None,'p2':None,'p3':None,'p4':None,'p5':None,'score':es,'plaintext':ep},ignore_index=True)
    top_df.to_csv(args.output,index=False)
    logging.info(f"Top results saved to {args.output}")

if __name__=='__main__':
    main()
