"""Search on basis (B) for the joint objective: higher geo-PathCE with higher surprise-adjusted CE (Addendum 11 rev. 2).
Objective = ln SACE_B = ln geo_B(R8b deck) - [0.9*slope + 0.8*theme_wt*slope + S4(w) + 0.25*single_wt],
  slope = (ln geo_B - ln geo_B at -200bp)/2 (same recipe as Addenda 4-10); theme set widened to every thematic sleeve
  in the universe; single set = all single names; S4 by rule: 0.06 + 0.30*max(0, C-row weight - 0.20)
  + 0.25*max(0, weight outside P1's names - 0.05)  (reproduces the assigned charges: P2 .06, R2 .09, R3b/R1b .10).
Constraints: standing feasibility, at most K singles (0/1/2), XLE <= XLE_CAP (judgment, default 5%), tail gates on
basis (B) vs P1 on the R8/HCC/LS decks, closed-form gates unchanged.
Run: R7=1 R8=1 DECK_FILE=Deck_Recal_R8b_20260903.csv SCEN_FILE=scenarios_24_R6_20260831.csv NPATHS=4000 XLE_CAP=0.05 python3 search_SACE.py K1
"""
import sys, json, time, numpy as np
import lib_pathce as L
from core import UNIV, IDX, DECKM, ERV, sigma_of, wvec, CONVS, feasible, SINGLES, DEF, NUC, SEMIS
import search_B as S

THEM2 = sorted(set(DEF + NUC + SEMIS + ['CIBR','HACK','IHAK','QTUM','PAVE','AIRR','XT','KOMP','BOTZ','ROBO','PRNT','IBIT','COPX','REMX','AQWA']))
P1N = set(L.P1.keys())
CW = np.array([1.0 if S.grades.get(t, "B") == "C" else 0.0 for t in UNIV])
NW = np.array([0.0 if t in P1N else 1.0 for t in UNIV])
TW = np.array([1.0 if t in THEM2 else 0.0 for t in UNIV])
SW = np.array([1.0 if t in SINGLES else 0.0 for t in UNIV])

def s4(w): return 0.06 + 0.30 * max(0.0, float(CW @ w) - 0.20) + 0.25 * max(0.0, float(NW @ w) - 0.05)

def objective(w):
    """returns (ln SACE, ln geo, ln geo_LS, P5 robust-G, pen)"""
    r = DECKM.T @ w - float(ERV @ w); sig = sigma_of(w)
    res = {c: S.run_B(r, c[0], c[1], sig) for c in CONVS}
    resL = {c: S.run_B(r - 0.02, c[0], c[1], sig) for c in CONVS}
    g = float(np.mean([np.log(res[c]["PathCE"]) for c in CONVS])); gL = float(np.mean([np.log(resL[c]["PathCE"]) for c in CONVS]))
    slope = (g - gL) / 2
    pen = 0.9 * slope + 0.8 * float(TW @ w) * slope + s4(w) + 0.25 * float(SW @ w)
    return g - pen, g, gL, res[("robust", "G")]["P5"], pen

def climb(seed, K, tag, max_steps=80, shortlist=32):
    w = wvec(seed); assert feasible(w) and S.n_singles(w) <= K, (tag, feasible(w), S.n_singles(w))
    cur, g, gL, p5, pen = objective(w)
    print(f"[{tag}] seed {S.show(w)} | SACE {np.exp(cur):.3f} geo {np.exp(g):.3f} pen {pen:.3f} P5 {p5:.2f} fails {S.gate_fails(w)}", flush=True)
    sc = {}
    for j in S.POOL + [t for t in S.S1 if K > 0]:
        v = w * 0.97; v[IDX[j]] += 0.03; v = v / v.sum()
        sc[j] = objective(v)[0]
    short = sorted(sc, key=lambda j: -sc[j])[:shortlist]
    pool = sorted(set(short) | set(S.held(w)))
    def moves(w):
        cands = []
        for i in S.held(w):
            wi = w[IDX[i]]
            for j in pool:
                if j == i: continue
                wj = w[IDX[j]]
                if wj > 1e-9: sizes = [0.01] if wi >= 0.03 - 1e-9 else [wi]
                else: sizes = [0.02] if wi >= 0.04 - 1e-9 else ([wi] if abs(wi - 0.02) < 1e-9 else [])
                for d in sizes:
                    v = w.copy(); v[IDX[i]] -= d; v[IDX[j]] += d; v[np.abs(v) < 1e-12] = 0.0
                    if not feasible(v) or S.n_singles(v) > K or v[IDX['XLE']] > S.XLE_CAP + 1e-9: continue
                    cands.append((v, i, j, d))
        return cands
    CF = ("worst", "RCEW", "R3AF", "DRB6"); key = lambda f: (sum(x.split(":")[1] in CF for x in f), len(f))
    fl = S.gate_fails(w); t0 = time.time()
    while fl:
        best = None
        for v, i, j, d in moves(w):
            f = S.gate_fails(v)
            if best is None or key(f) < key(best[0]): best = (f, v, i, j, d)
        if best is None or key(best[0]) >= key(fl): print(f"[{tag}] repair failed: {fl}"); return w, objective(w)[0]
        fl, w = best[0], best[1]; cur = objective(w)[0]
        print(f"[{tag}] repair {best[2]}->{best[3]} {best[4]*100:.0f}% | fails {fl} | {S.show(w)}", flush=True)
    for step in range(max_steps):
        scored = []
        for v, i, j, d in moves(w):
            if S.cf_fails(v): continue
            o, g, gL, p5, pen = objective(v)
            if o > cur + 1e-6: scored.append((o, g, p5, pen, v, i, j, d))
        scored.sort(key=lambda x: -x[0])
        moved = False
        for o, g, p5, pen, v, i, j, d in scored:
            if S.gates_ok(v):
                w, cur = v, o; moved = True
                print(f"[{tag}] step {step+1}: {i}->{j} {d*100:.0f}% | SACE {np.exp(o):.3f} geo {np.exp(g):.3f} pen {pen:.3f} P5 {p5:.2f} | {S.show(w)} | {time.time()-t0:.0f}s", flush=True)
                break
        if not moved: break
    return w, cur

SEEDS = {
  "K0": (0, {"NUKZ":.15,"GLDM":.15,"SHLD":.15,"DBMF":.15,"CIBR":.14,"URNM":.06,"XLE":.05,"PSLV":.04,"XAR":.03,"SMH":.02,"CTA":.02,"XLF":.02,"XLK":.02}),
  "K0p": (0, {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.15,'CIBR':.10,'GRID':.10,'CTA':.05,'XAR':.05,'URNM':.05,'ROKT':.05}),
  "K1": (1, {"GLDM":.15,"SHLD":.15,"DBMF":.15,"NUKZ":.12,"CIBR":.12,"URNM":.06,"CEG":.05,"XLE":.05,"XAR":.05,"CTA":.03,"PSLV":.03,"SMH":.02,"XLK":.02}),
  "K2": (2, {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.15,'CIBR':.08,'GRID':.05,'CME':.05,'CTA':.05,'MSCI':.03,'XAR':.06,'URNM':.08}),
  "K3": (3, {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.13,'CIBR':.09,'GRID':.05,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.04,'URNM':.03,'XAR':.03,'CEG':.03}),
  "K2r": (2, {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.13,'CIBR':.09,'GRID':.05,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.04,'URNM':.03,'XAR':.03,'CEG':.03}),
}
REF = {"R3b": S.REF["R3b"], "R1b_final": S.REF["R1b_final"], "R2_final": S.REF["R2_final"], "P2": S.REF["P2"]}
if __name__ == "__main__":
    if sys.argv[1:] == ["ref"]:
        for nm, d in REF.items():
            w = wvec(d); o, g, gL, p5, pen = objective(w)
            print(f"{nm}: SACE {np.exp(o):.3f} geo {np.exp(g):.3f} pen {pen:.3f} (S4 {s4(w):.3f} theme {float(TW@w):.2f} singles {float(SW@w):.2f}) P5 {p5:.2f}", flush=True)
        sys.exit()
    which = sys.argv[1:] or list(SEEDS); out = {}
    for tag in which:
        K, seed = SEEDS[tag]
        w, cur = climb(seed, K, tag)
        out[tag] = {t: round(float(w[IDX[t]]), 2) for t in S.held(w)}
        o, g, gL, p5, pen = objective(w)
        print(f"[{tag}] FINAL SACE {np.exp(o):.3f} geo {np.exp(g):.3f} pen {pen:.3f} S4 {s4(w):.3f} singles {S.n_singles(w)} | {S.show(w)}", flush=True)
        json.dump(out, open(f"out/searchSACE_{'_'.join(which)}.json", "w"), indent=1)
