"""Addendum 59/61 (main-book mode when K > 7: A61_MINW, A61_GOLDCAP, A61_RSNAME env) — compact-book climb with stressed-deck weight.
Score = (1-a)*U1g3(base deck) + a*U1g3(stressed deck), fast E1 evaluator; compact rules (<=K holdings, each >=5%,
ETF cap 20% (30% if K<=5), single-holding RS <=33% (45% if K<=5), standing blocks, gates incl. stressed floor).
Run: source env57.sh; SINGLES_MAX=99 BRKB_CAP=0.15 MIN_HOLD=0.05 python3 search_a59.py K alpha out.json
"""
import sys, json, itertools, numpy as np, os
import standing as ST, uncertainty as U
from core import wvec, IDX, UNIV, DECKM

K = int(sys.argv[1]); A = float(sys.argv[2]); OUT = sys.argv[3]
MINW = float(os.environ.get("A61_MINW", "0.02")); GOLDCAP = float(os.environ.get("A61_GOLDCAP", "1.0"))
GOLDS = ["GLDM", "FNV", "WPM", "GDX", "SIL", "PSLV"]
if os.environ.get("A61_RSNAME"): ST.RS_NAME = float(os.environ["A61_RSNAME"])
POOL = [t for t in os.environ.get("A61_EXTRA_POOL","").split(",") if t] + ["GLDM", "SHLD", "CIBR", "DBMF", "KMLM", "BRKB", "MSCI", "UTHR", "CME", "URNM", "XAR", "SMH", "XLE", "NUKZ", "CTA", "GRID", "SPMO", "FNV", "SOXX", "XSD"]
import report_v3 as R
DK = R.M0   # DECK_SHIFT-aware deck (A61 fix: fast evaluator previously ignored DECK_SHIFT)
NOGATES = os.environ.get('A61_NOGATES', '0') == '1'
CEK = os.environ.get('A61_CE', 'ce3')   # A62: objective gamma (ce15 / ce2 / ce3)
D = U.Draws(int(os.environ.get("NPATHS", "6000")), 31415)
DS = ST.DECK_STRESSED
if os.environ.get('A61_ALT_SHIFT'):
    import report_v3 as _R; DS = _R.shifted(DECKM, json.load(open(os.environ['A61_ALT_SHIFT'])))   # A61: alternative deck for the blended score
cache = {}

def score(d):
    key = tuple(sorted((t, round(v, 2)) for t, v in d.items() if v > 1e-9))
    if key in cache: return cache[key]
    w = wvec(dict(key))
    res = None
    if sum(dict(key).get(t, 0) for t in GOLDS) > GOLDCAP + 1e-9: res = None
    elif not ST.violations(w):
        f = ST.evaluate_fast(w, deck=DK)
        if NOGATES or not f["fails"]:
            ub = U.evaluate_u(w, D, 1.0, deck=DK)[CEK]; us = U.evaluate_u(w, D, 1.0, deck=DS)[CEK] if A > 0 else ub
            if A > 0 and NOGATES is False and float((DS.T @ w - float(__import__('core').ERV @ w)).min()) < 0: pass
            T = float(os.environ.get("ENGINEV2_T", "50"))
            gb, gs = (ub ** (1 / T) - 1) * 100, (us ** (1 / T) - 1) * 100
            res = ((1 - A) * gb + A * gs, gb, gs, f["geo"])
    cache[key] = res
    return res

def neighbours(d, step):
    names = list(d)
    for a, b in itertools.permutations(names, 2):          # shift weight
        if d[a] - step >= MINW - 1e-9:
            n = dict(d); n[a] -= step; n[b] += step; yield n
    if len(names) < K:                                     # add a name
        for t in POOL:
            if t in d: continue
            for a in names:
                if d[a] - MINW >= MINW - 1e-9:
                    n = dict(d); n[a] -= MINW; n[t] = MINW; yield n
    for a in names:                                        # swap a name
        for t in POOL:
            if t in d: continue
            n = dict(d); n[t] = n.pop(a); yield n

def climb(seed):
    cur = dict(seed); s = score(cur)
    if s is None: return None
    for step in (0.05, 0.02, 0.01):
        improved = True
        while improved:
            improved = False
            best = (s[0], cur)
            for n in neighbours(cur, step):
                v = score(n)
                if v and v[0] > best[0] + 1e-4: best = (v[0], n)
            if best[1] is not cur:
                cur = {t: round(x, 2) for t, x in best[1].items() if x > 1e-9}; s = score(cur); improved = True
    return cur, s

SEEDS_MAIN = [dict(DBMF=.20, SHLD=.20, SMH=.12, BRKB=.11, NUKZ=.10, URNM=.06, CME=.05, MSCI=.05, CIBR=.03, UTHR=.02, KMLM=.02, XAR=.02, XLE=.02),
          dict(SHLD=.17, KMLM=.16, DBMF=.14, SMH=.12, NUKZ=.12, BRKB=.11, CME=.05, CIBR=.03, MSCI=.03, UTHR=.03, URNM=.02, GLDM=.02),
          dict(GLDM=.10, DBMF=.20, SHLD=.15, CIBR=.10, SMH=.05, URNM=.05, CME=.05, MSCI=.05, UTHR=.05, BRKB=.10, KMLM=.08, XAR=.02),
          dict(GLDM=.20, DBMF=.20, SHLD=.15, CIBR=.10, SMH=.05, URNM=.05, CME=.05, MSCI=.05, UTHR=.05, BRKB=.05, KMLM=.03, XAR=.02),
          dict(SHLD=.16, GLDM=.15, CIBR=.15, DBMF=.15, NUKZ=.08, BRKB=.05, CME=.05, MSCI=.05, URNM=.05, UTHR=.05, XLE=.02, XAR=.02, SMH=.02)]
SEEDS7 = [dict(DBMF=.20, SHLD=.20, CIBR=.19, BRKB=.15, URNM=.11, KMLM=.10, MSCI=.05),
          dict(GLDM=.20, SHLD=.20, CIBR=.20, DBMF=.20, URNM=.10, MSCI=.05, UTHR=.05),
          dict(DBMF=.20, GLDM=.19, SHLD=.19, CIBR=.17, BRKB=.15, MSCI=.05, UTHR=.05),
          dict(SHLD=.20, CIBR=.20, DBMF=.20, GLDM=.15, URNM=.10, MSCI=.05, UTHR=.05, KMLM=.05),
          dict(DBMF=.20, KMLM=.15, SHLD=.20, CIBR=.15, BRKB=.15, GLDM=.10, UTHR=.05)]
SEEDS5 = [dict(SHLD=.30, DBMF=.29, BRKB=.15, URNM=.14, SMH=.12),
          dict(GLDM=.30, SHLD=.23, CIBR=.20, DBMF=.17, URNM=.10),
          dict(DBMF=.30, SHLD=.23, CIBR=.20, GLDM=.17, URNM=.10),
          dict(DBMF=.25, SHLD=.25, CIBR=.20, GLDM=.15, BRKB=.15)]
out = {}
for i, sd in enumerate(SEEDS_MAIN if K > 7 else (SEEDS7 if K > 5 else SEEDS5)):
    sd = {t: v for t, v in sd.items()}
    if len(sd) > K: continue
    r = climb(sd)
    if r is None:
        print("seed", i, "inadmissible", flush=True); continue
    b, s = r
    out[os.environ.get("A61_TAG", "A61") + f"_s{i}"] = b
    print(i, {t: b[t] for t in sorted(b, key=lambda t: -b[t])}, "score %.3f base %.3f stressed %.3f geo %.2f" % s, flush=True)
json.dump(out, open(OUT, "w"), indent=0)
