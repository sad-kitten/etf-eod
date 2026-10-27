"""Search on the ratified reporting basis (B): real 3% spending, guardrail on real wealth, Social Security
(placeholder $45k = 2.25% of W0) offsetting the corpus draw from year 11; taxes omitted in the search (they are
book-independent to within 0.02 W0) and applied at verification with policy_engine2.
Objective: maximin over the R8b deck and the C-haircut deck of geo-PathCE(B) (CRN FastEval, seed 777).
Constraints: standing feasibility (core.feasible) + at most K single names; gate battery vs P1 with the tail gates
(P10/P5/ruin) re-based to basis (B) on the R8/HCC/LS decks; closed-form gates unchanged (worst, RCEW40, R3AF, DRB40@6%).
Run: R7=1 R8=1 DECK_FILE=Deck_Recal_R8b_20260903.csv SCEN_FILE=scenarios_24_R6_20260831.csv NPATHS=4000 python3 search_B.py
"""
import sys, json, time, numpy as np, pandas as pd
import lib_pathce as L
from core import UNIV, N, IDX, DECKM, ERV, sigma_of, wvec, P1v, CONVS, FE, feasible, SINGLES

grades = pd.read_csv("out/Row_Grades_R8b_20260903.csv").set_index("Ticker")["Grade"]
colmed = np.median(DECKM, axis=0); HCC = DECKM.copy()
for i, t in enumerate(UNIV):
    if grades.get(t, "B") == "C": HCC[i] = 0.75 * DECKM[i] + 0.25 * colmed
INFL = 0.025; SS = 45000.0 / 1999798.78; T = L.T

def run_B(r_net, pin, conv, sigma):
    s = FE.S[pin]; mu = np.log1p(r_net)[s]
    if conv == "G": mu = mu - sigma * sigma / 2
    X = np.exp(mu + sigma * FE.Z) / (1.0 + INFL)
    W = np.ones(FE.npaths); red = np.zeros(FE.npaths, bool); ru = np.zeros(FE.npaths, bool)
    for t in range(1, T + 1):
        W = W * X[:, t - 1]
        red = np.where(W < 0.80, True, np.where(W >= 1.0, False, red))
        draw = np.maximum(0.03 * np.where(red, 0.75, 1.0) - (SS if t >= 11 else 0.0), 0.0)
        W = W - draw; ru |= W <= 0; W = np.maximum(W, 1e-9)
    lg = np.log(W)
    return dict(PathCE=float(np.exp(lg.mean())), P5=float(np.quantile(W, .05)), P10=float(np.quantile(W, .10)), ruin=int(ru.sum()))

def geoB(w, M):
    r = M.T @ w - float(ERV @ w); sig = sigma_of(w)
    res = {c: run_B(r, c[0], c[1], sig) for c in CONVS}
    return float(np.mean([np.log(res[c]["PathCE"]) for c in CONVS])), res

def objective(w):   # maximin log geo-PathCE(B) over the R8b deck and the C-haircut deck
    a, resA = geoB(w, DECKM); b, resH = geoB(w, HCC)
    return min(a, b), a, b, resA

# ---- standing gate battery (nominal basis, vs P1) ----
def ev(w, M, shift=0.0):
    """Gate evaluation: tails (P10/P5/ruin) on the ratified basis (B); closed-form scenario gates unchanged."""
    r_net = M.T @ w - float(ERV @ w) + shift; sig = sigma_of(w)
    res = {(p, c): run_B(r_net, p, c, sig) for p, c in CONVS}
    return dict(res=res, worst=float(r_net.min()), RCEW=L.rcew40(r_net)["robust"], DRB6=L.drb40(r_net, h=0.06),
                R3AF=L.r3af(r_net, L.prow(L.EVG11) - L.wer_evg11()))
DECKS = {"R8": (DECKM, 0.0), "HCC": (HCC, 0.0), "LS": (DECKM, -0.02)}
BASES = {k: ev(P1v, M, s) for k, (M, s) in DECKS.items()}
def fails_vs(e, b):
    f = []
    for c in CONVS:
        if e["res"][c]["P10"] < b["res"][c]["P10"] - 1e-9: f.append("P10")
        if e["res"][c]["P5"] < b["res"][c]["P5"] - 1e-9: f.append("P5")
        if e["res"][c]["ruin"] > b["res"][c]["ruin"]: f.append("ruin")
    if e["worst"] < b["worst"] - 1e-12: f.append("worst")
    if e["RCEW"] < b["RCEW"] - 1e-9: f.append("RCEW")
    if e["R3AF"] < b["R3AF"] - 0.002: f.append("R3AF")
    if e["DRB6"] > b["DRB6"] + 1e-12: f.append("DRB6")
    return f
def cf_fails(w):
    """closed-form gates only (instant): worst, RCEW40, R3AF, DRB6 on each deck"""
    out = []
    for k, (M, s) in DECKS.items():
        r_net = M.T @ w - float(ERV @ w) + s; b = BASES[k]
        if r_net.min() < b["worst"] - 1e-12: out.append(f"{k}:worst")
        if L.rcew40(r_net)["robust"] < b["RCEW"] - 1e-9: out.append(f"{k}:RCEW")
        if L.r3af(r_net, L.prow(L.EVG11) - L.wer_evg11()) < b["R3AF"] - 0.002: out.append(f"{k}:R3AF")
        if L.drb40(r_net, h=0.06) > b["DRB6"] + 1e-12: out.append(f"{k}:DRB6")
    return out
def gate_fails(w):
    out = cf_fails(w)
    if out: return out
    for k, (M, s) in DECKS.items():
        out += [f"{k}:{f}" for f in fails_vs(ev(w, M, s), BASES[k])]
    return out
def gates_ok(w): return not gate_fails(w)

def n_singles(w): return int(sum(w[IDX[t]] > 1e-9 for t in SINGLES if t in IDX))
def held(w): return [t for t in UNIV if w[IDX[t]] > 1e-9]
def show(w): return " ".join(f"{t}{int(round(w[IDX[t]]*100))}" for t in sorted(held(w), key=lambda t: -w[IDX[t]]))

POOL = [t for t in UNIV if t not in SINGLES and t != "QTUM"]
import os as _os
XLE_CAP = float(_os.environ.get("XLE_CAP", "1.0"))   # optional judgment cap on the energy sector ETF
S1 = [t for t in SINGLES if t in IDX]

def climb(seed, K, tag, max_steps=60, shortlist=32):
    w = wvec(seed); assert feasible(w) and n_singles(w) <= K, (tag, feasible(w), n_singles(w))
    cur, a, b, res = objective(w)
    print(f"[{tag}] seed {show(w)} | maximin {np.exp(cur):.3f} (R8 {np.exp(a):.3f} HCC {np.exp(b):.3f}) P5 {res[('robust','G')]['P5']:.2f} gates {gates_ok(w)}", flush=True)
    # marginal screen: +3% of each pool name funded pro rata -> shortlist
    sc = {}
    for j in POOL + [t for t in S1 if K > 0]:
        v = w * 0.97; v[IDX[j]] += 0.03; v = v / v.sum()
        sc[j] = geoB(v, DECKM)[0]
    short = sorted(sc, key=lambda j: -sc[j])[:shortlist]
    pool = sorted(set(short) | set(held(w)))
    t0 = time.time()
    def moves(w):
        cands = []
        for i in held(w):
            wi = w[IDX[i]]
            for j in pool:
                if j == i: continue
                wj = w[IDX[j]]
                if wj > 1e-9: sizes = [0.01] if wi >= 0.03 - 1e-9 else [wi]
                else: sizes = [0.02] if wi >= 0.04 - 1e-9 else ([wi] if abs(wi - 0.02) < 1e-9 else [])
                for d in sizes:
                    v = w.copy(); v[IDX[i]] -= d; v[IDX[j]] += d; v[np.abs(v) < 1e-12] = 0.0
                    if not feasible(v) or n_singles(v) > K or v[IDX['XLE']] > XLE_CAP + 1e-9: continue
                    cands.append((v, i, j, d))
        return cands
    CF = ("worst", "RCEW", "R3AF", "DRB6")
    key = lambda f: (sum(x.split(":")[1] in CF for x in f), len(f))     # closed-form fails dominate, then count
    fl = gate_fails(w)
    while fl:
        print(f"[{tag}] repair: seed fails {fl}", flush=True)
        best = None
        for v, i, j, d in moves(w):
            f = gate_fails(v)
            if best is None or key(f) < key(best[0]): best = (f, v, i, j, d)
        if best is None or key(best[0]) >= key(fl): print(f"[{tag}] repair failed; stopping"); return w, objective(w)[0]
        fl, w = best[0], best[1]; cur = objective(w)[0]
        print(f"[{tag}] repair move {best[2]}->{best[3]} {best[4]*100:.0f}% | fails now {fl} | maximin {np.exp(cur):.3f} | {show(w)}", flush=True)
    for step in range(max_steps):
        cands = moves(w)
        scored = []
        for v, i, j, d in cands:
            if cf_fails(v): continue
            o, a, b, res = objective(v)
            if o > cur + 1e-6: scored.append((o, a, b, res, v, i, j, d))
        scored.sort(key=lambda x: -x[0])
        moved = False
        for o, a, b, res, v, i, j, d in scored:
            if gates_ok(v):
                w, cur = v, o; moved = True
                print(f"[{tag}] step {step+1}: {i}->{j} {d*100:.0f}% | maximin {np.exp(o):.3f} (R8 {np.exp(a):.3f} HCC {np.exp(b):.3f}) P5 {res[('robust','G')]['P5']:.2f} | {show(w)} | {time.time()-t0:.0f}s", flush=True)
                break
        if not moved: break
    return w, cur

SEEDS = {
  "K0_a": (0, {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.15,'CIBR':.10,'GRID':.10,'CTA':.05,'XAR':.05,'URNM':.05,'ROKT':.05}),
  "K0_b": (0, {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.12,'CIBR':.10,'GRID':.05,'CTA':.05,'XAR':.05,'URNM':.06,'XLE':.05,'SMH':.04,'KMLM':.03}),
  "K0_c": (0, {'GLDM':.15,'SHLD':.15,'XLE':.15,'DBMF':.14,'CIBR':.12,'NUKZ':.12,'URNM':.08,'SMH':.03,'PSLV':.03,'XAR':.03}),
  "K1":   (1, {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.13,'CIBR':.08,'GRID':.05,'CTA':.05,'XAR':.06,'URNM':.10,'CEG':.03,'XLE':.05}),
  "K2":   (2, {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.15,'CIBR':.08,'GRID':.05,'CME':.05,'CTA':.05,'MSCI':.03,'XAR':.06,'URNM':.08}),
}
REF = {"R3b": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.12,'CIBR':.08,'GRID':.05,'CME':.05,'BRKB':.04,'CTA':.05,'URNM':.03,'XAR':.03,'CEG':.03,'FNV':.04,'MSCI':.03},
       "R1b_final": {'GLDM':.15,'SHLD':.14,'DBMF':.15,'NUKZ':.12,'CIBR':.10,'CME':.05,'MSCI':.05,'XLE':.05,'XAR':.04,'CEG':.03,'URNM':.03,'CBOE':.03,'GRID':.03,'CTA':.03},
       "R2_final": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.13,'CIBR':.09,'GRID':.05,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.04,'URNM':.03,'XAR':.03,'CEG':.03},
       "P2": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.15,'CIBR':.10,'GRID':.10,'CTA':.05,'CME':.05,'XAR':.05,'MSCI':.05}}
if __name__ == "__main__":
    if sys.argv[1:] == ["ref"]:
        for nm, d in REF.items():
            w = wvec(d); o, a, b, res = objective(w)
            print(f"{nm}: maximin {np.exp(o):.3f} (R8 {np.exp(a):.3f} HCC {np.exp(b):.3f}) P5 {res[('robust','G')]['P5']:.2f} singles {n_singles(w)} gates {gate_fails(w)}", flush=True)
        sys.exit()
    which = sys.argv[1:] or list(SEEDS)
    out = {}
    for tag in which:
        K, seed = SEEDS[tag]
        w, cur = climb(seed, K, tag)
        out[tag] = {t: round(float(w[IDX[t]]), 2) for t in held(w)}
        print(f"[{tag}] FINAL maximin {np.exp(cur):.3f} singles {n_singles(w)} | {show(w)}", flush=True)
        json.dump(out, open(f"out/searchB_{'_'.join(which)}{'_xle%d' % round(XLE_CAP*100) if XLE_CAP < 1 else ''}.json", "w"), indent=1)
