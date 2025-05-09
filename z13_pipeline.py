# z13_pipeline_two_pass.py  –  full script con step di raffinamento
import re, math, random, csv, os, multiprocessing as mp
from pathlib import Path
from functools import partial
from tqdm import tqdm
from route_cipher import route_cipher
from simanneal import Annealer

# ---------------- CONFIG ---------------------------------------------------------
GRID_HEIGHTS   = [20, 24, 22, 21, 19, 17, 15, 13, 11]
STEPS          = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 5)]

# --- passaggio 1 (ampio) ---
RUNS_PER_SEG_1   = 25
ANNEAL_STEPS_1   = 15_000
EARLY_SCORE_1    = 35

# --- passaggio 2 (rifinitura) ---
RUNS_PER_SEG_2   = 60      # molti più semi
ANNEAL_STEPS_2   = 40_000  # catene lunghe
EARLY_SCORE_2    = 30
TOP_N            = 50      # quanti segmenti tenere per il secondo giro

SCORE_THRESHOLD  = 48       # filtro per csv
CACHE_FILE       = "z13_segments_cache.txt"

# ---------------- QUADGRAM TABLE -------------------------------------------------
QUAD = {}
with open("english_quadgrams.txt") as fh:
    total = 0
    for line in fh:
        q, cnt = line.split()
        total += int(cnt); QUAD[q] = int(cnt)
for q in QUAD:
    QUAD[q] = math.log10(QUAD[q] / total)

def quad_score(txt: str) -> float:
    s = -sum(QUAD.get(txt[i:i+4], -8.0) for i in range(len(txt)-3))
    if "NAME"   in txt: s -= 2
    if "MYNAME" in txt: s -= 2
    return s

# ---------------- CORPUS ---------------------------------------------------------
def load_corpus(folder="corpus"):
    txt = []
    for f in Path(folder).glob("*.txt"):
        txt += Path(f).read_text(encoding="utf8", errors="ignore").splitlines()
    return [re.sub('[^A-Za-z ]+', ' ', t).upper().strip()
            for t in txt if 40 <= len(t) <= 120]

# ---------------- PATTERN (0-1-2 spazi, no aaa / no ⌴⌴⌴) -------------------------
def matches_pattern(seg: str) -> bool:
    if not (seg[0]==seg[11] and seg[2]==seg[10] and seg[3]==seg[8] and seg[7]==seg[12]):
        return False
    mid = seg[4:7]
    if mid == '   ':                     # tre spazi
        return False
    if mid[0] == mid[1] == mid[2]:       # tre uguali
        return False
    return True

# ---------- Segment extraction (prima esecuzione) --------------------------------
def build_segment_cache():
    corpus  = load_corpus()
    seen    = set()                              # <<-- nuovo
    with open(CACHE_FILE, "w", newline="") as fh:
        w = csv.writer(fh, delimiter='\t')
        for line in tqdm(corpus, desc="building segment cache"):
            for h in GRID_HEIGHTS:
                for step in STEPS:
                    cipher = route_cipher(line, h, step)
                    for i in range(len(cipher) - 12):
                        seg = cipher[i:i+13]
                        tup = (seg, h, step)     # chiave univoca
                        if tup in seen:          # già visto -> skip
                            continue
                        if matches_pattern(seg) and len(set(seg)) >= 2:
                            seen.add(tup)
                            w.writerow([seg, h, step])

def load_cached_segments():
    segs = []
    with open(CACHE_FILE, newline="") as fh:
        r = csv.reader(fh, delimiter='\t')
        for seg, h, step in r:
            segs.append((seg, int(h), eval(step)))
    return segs

def get_segments():
    if not os.path.exists(CACHE_FILE):
        build_segment_cache()
    return load_cached_segments()

# ---------------- SIMULATED ANNEALING -------------------------------------------
class HomoState:
    def __init__(self, seg: str):
        self.seg = seg
        self.sym = list(dict.fromkeys(seg))
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        random.shuffle(letters)
        self.key = {s: letters.pop() for s in self.sym}

    def move(self):
        a, b = random.sample(self.sym, 2)
        self.key[a], self.key[b] = self.key[b], self.key[a]

    def energy(self):
        return quad_score(''.join(self.key[c] for c in self.seg))

class HomoSolver(Annealer):
    move   = lambda self: self.state.move()
    energy = lambda self: self.state.energy()

def solve_segment(seg: str, runs, steps, early):
    best_state, best_e = None, float('inf')
    for _ in range(runs):
        st = HomoState(seg)
        sl = HomoSolver(st); sl.Tmax=10; sl.Tmin=0.1; sl.steps=steps
        st_best, e = sl.anneal()
        if e < best_e:
            best_state, best_e = st_best, e
        if best_e < early:
            break
    pt = ''.join(best_state.key[c] for c in seg)
    return pt, best_e, best_state.key

# ---------------- Worker ---------------------------------------------------------
def worker(item, runs, steps, early):
    seg, h, step = item
    pt, e, key = solve_segment(seg, runs, steps, early)
    if e < SCORE_THRESHOLD:
        return (e, pt, seg, f"{h}x?", step, key)
    return None

# ---------------- MAIN -----------------------------------------------------------
def main():
    segments = get_segments()
    cpu = mp.cpu_count()
    ctx  = mp.get_context("spawn")

    # ---- passaggio 1 ------------------------------------------------------------
    print("=== PASSO 1: solver rapido su tutti i segmenti ===")
    results = []
    w1 = partial(worker, runs=RUNS_PER_SEG_1, steps=ANNEAL_STEPS_1, early=EARLY_SCORE_1)
    with ctx.Pool(cpu) as pool:
        for out in tqdm(pool.imap_unordered(w1, segments), total=len(segments)):
            if out: results.append(out)

    # ordina e salva risultati grezzi
    results.sort(key=lambda x: x[0])

    # ---- passaggio 2 ------------------------------------------------------------
    print(f"=== PASSO 2: raffinatura sui migliori {TOP_N} segmenti ===")
    top_segments = [(seg, int(h[:-2]), step)           # h era "11x?"
                    for _,_,seg,h,step,_ in results[:TOP_N]]

    w2 = partial(worker, runs=RUNS_PER_SEG_2, steps=ANNEAL_STEPS_2, early=EARLY_SCORE_2)
    refined = []
    with ctx.Pool(cpu) as pool:
        for out in tqdm(pool.imap_unordered(w2, top_segments), total=len(top_segments)):
            if out: refined.append(out)

    # unisci e riordina
    results.extend(refined)
    results.sort(key=lambda x: x[0])

    # ---- scrivi CSV finale ------------------------------------------------------
    with open("z13_candidates.csv", "w", encoding="utf8") as fh:
        fh.write("score,plaintext,segment,h,step,key\n")
        for e, pt, seg, h, step, key in results[:300]:
            fh.write(f"{e:.2f},{pt},{seg},{h},{step},{key}\n")

    print("Salvato z13_candidates.csv con risultati ordinati.")

if __name__ == "__main__":
    main()