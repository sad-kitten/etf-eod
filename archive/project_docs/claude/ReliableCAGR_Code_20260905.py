"""Rank books by RELIABLE, CONSISTENT retained CAGR rather than by the certainty-equivalent alone.

geo-PathCE and SACE are already multiples of the initial corpus: geo 5.78 means the log-average path
ends at 5.78x the starting corpus in real terms AFTER forty years of the 3% real draw. The implied
retained CAGR is geo^(1/40)-1. This script keeps that unit and adds the two things a certainty
equivalent hides:
  (a) the DISTRIBUTION of realised 40-year retained CAGR across paths -- P5/P25/median/P75, pooled
      over all four conventions (4 x 60k = 240k paths), so the reliability floor is read directly;
  (b) WITHIN-PATH consistency -- the standard deviation of a path's four decade CAGRs. A book that
      compounds 4.5%/yr steadily is more reliable than one that does nothing for two decades and
      then 9%, even at identical terminal wealth.
Also carried: the regime floor (worst of the 24 deck scenarios) and the fraction of scenarios below 6%.
"""
import numpy as np, pandas as pd, json, sys, os, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L
import policy_engine2 as E
import search_SACE as SS
from core import DECKM, ERV, sigma_of, wvec

CONV = [("robust","G"),("robust","L"),("canonical","G"),("canonical","L")]

def profile(d):
    w = wvec({a:b for a,b in d.items() if b > 0})
    r = DECKM.T @ w - float(ERV @ w); sig = sigma_of(w)
    mc = E.suite(r, sig, E.POL); mcL = E.suite(r - 0.02, sig, E.POL)
    Ws, D = [], []
    for c in CONV:
        run = mc[c]; W = np.maximum(run["W"], 1e-6); s = run["snaps"]
        Ws.append(W)
        prev = np.ones_like(W); dec = []
        for t in (10,20,30,40):
            cur = np.maximum(s[t], 1e-6); dec.append((cur/prev)**0.1 - 1.0); prev = cur
        D.append(np.std(np.vstack(dec), axis=0, ddof=1))
    W = np.concatenate(Ws); dsd = np.concatenate(D)
    g = W**(1/40.0) - 1.0
    geo = E.geo(mc); geoL = E.geo(mcL)
    q = lambda p: float(np.quantile(g, p))
    def ce(gam):
        if abs(gam - 1.0) < 1e-9: return float(np.exp(np.log(W).mean()))
        return float(np.mean(W**(1.0-gam))**(1.0/(1.0-gam)))
    ce1, ce3, ce5 = ce(1.0), ce(3.0), ce(5.0)
    return dict(
        geo_x=round(geo, 3), geo_cagr=round((geo**(1/40.)-1)*100, 2),
        ce1_cagr=round((ce1**(1/40.)-1)*100, 2), ce3_x=round(ce3, 3), ce3_cagr=round((ce3**(1/40.)-1)*100, 2),
        ce5_x=round(ce5, 3), ce5_cagr=round((ce5**(1/40.)-1)*100, 2),
        SACE_x=round(geo*np.exp(-(0.9*((np.log(geo)-np.log(geoL))/2) + 0.8*float(SS.TW@w)*((np.log(geo)-np.log(geoL))/2) + SS.s4(w) + 0.25*float(SS.SW@w))), 3),
        cagr_P5=round(q(.05)*100, 2), cagr_P25=round(q(.25)*100, 2), cagr_med=round(q(.50)*100, 2),
        cagr_P75=round(q(.75)*100, 2), cagr_P95=round(q(.95)*100, 2),
        spread_P5_P95=round((q(.95)-q(.05))*100, 2), IQR=round((q(.75)-q(.25))*100, 2),
        p_ge3=round(float((g >= 0.03).mean())*100, 1), p_ge4=round(float((g >= 0.04).mean())*100, 1),
        decade_sd=round(float(dsd.mean())*100, 2),
        worst_scen=round(float(r.min())*100, 2), scen_lt6=int((r < 0.06).sum()),
        scen_sd=round(float(np.std(r))*100, 2),
        ls_geo_x=round(geoL, 3), vol48=round(sig*100, 2))

if __name__ == "__main__":
    BK = json.load(open(sys.argv[1]))
    only = os.environ.get("ONLY","").split(",") if os.environ.get("ONLY") else None
    rows = []
    for nm, d in BK.items():
        if only and nm not in only: continue
        p = profile(d); p["Port"] = nm; rows.append(p); print(nm, p, flush=True)
    df = pd.DataFrame(rows).set_index("Port")
    df.to_csv(os.environ.get("OUT","out/Reliable_CAGR_20260905.csv"))
    pd.set_option("display.width", 400); print(df.to_string())
"""Search for books with the highest RELIABLE CONSISTENT retained CAGR.

Objective: R25 = exp( mean over the four conventions of ln P25(terminal wealth) ) on basis (B), i.e. the
25th-percentile multiple of the initial corpus after 40 years of the 3% real draw. CAGR = R25^(1/40)-1.
P25 rather than the log-mean because a certainty equivalent pays for upside the owner cannot spend; P25
rather than P5 because the 5th percentile is a tail statistic, not a description of ordinary bad luck.
Maximin over the R8 deck and the C-haircut deck, as elsewhere. Standing gates, risk caps, K singles.
"""
import sys, json, time, os, numpy as np
import lib_pathce as L
from core import UNIV, IDX, DECKM, ERV, sigma_of, wvec, CONVS, feasible, SINGLES
import search_B as S
import search_SACE as SS

def r25_on(w, M):
    r = M.T @ w - float(ERV @ w); sig = sigma_of(w)
    K = os.environ.get("RCOBJ","P25")
    return float(np.mean([np.log(S.run_B(r, c[0], c[1], sig)[K]) for c in CONVS]))

def obj(w):
    a = r25_on(w, DECKM); b = r25_on(w, S.HCC)
    return min(a, b), a

def rep(w):
    r = DECKM.T @ w - float(ERV @ w); sig = sigma_of(w)
    res = {c: S.run_B(r, c[0], c[1], sig) for c in CONVS}
    g = float(np.mean([np.log(res[c]["PathCE"]) for c in CONVS]))
    p25 = float(np.mean([np.log(res[c][os.environ.get("RCOBJ","P25")]) for c in CONVS]))
    return np.exp(p25), np.exp(g), res[("robust","G")]["P5"]

KMAX = int(os.environ.get("KMAX","2"))
def climb(seed, tag, max_steps=90):
    w = wvec(seed); assert abs(w.sum()-1) < 1e-9 and feasible(w), tag
    cur, _ = obj(w)
    pool = sorted(set(S.POOL) | {"BRKB"} | set(S.S1))
    def moves(w):
        out = []
        for i in S.held(w):
            wi = w[IDX[i]]
            for j in pool:
                if j == i: continue
                wj = w[IDX[j]]
                if wj > 1e-9: sizes = [0.01] if wi >= 0.03 - 1e-9 else [wi]
                else: sizes = [0.02] if wi >= 0.04 - 1e-9 else ([wi] if abs(wi - 0.02) < 1e-9 else [])
                for d in sizes:
                    v = w.copy(); v[IDX[i]] -= d; v[IDX[j]] += d; v[np.abs(v) < 1e-12] = 0.0
                    if not feasible(v) or S.n_singles(v) > KMAX: continue
                    if v[IDX['XLE']] > S.XLE_CAP + 1e-9 or not S.nuc_ok(v) or not S.risk_ok(v): continue
                    out.append((v, i, j, d))
        return out
    CF = ("worst","RCEW","R3AF","DRB6"); key = lambda f: (sum(x.split(":")[1] in CF for x in f), len(f))
    fl = S.gate_fails(w); t0 = time.time()
    while fl:
        best = None
        for v, i, j, d in moves(w):
            f = S.gate_fails(v)
            if best is None or key(f) < key(best[0]): best = (f, v, i, j, d)
        if best is None or key(best[0]) >= key(fl): print(f"[{tag}] repair failed {fl}", flush=True); return w
        fl, w = best[0], best[1]
        print(f"[{tag}] repair {best[2]}->{best[3]} | {fl} | {S.show(w)}", flush=True)
    for step in range(max_steps):
        scored = []
        for v, i, j, d in moves(w):
            if S.cf_fails(v): continue
            o, _ = obj(v)
            if o > cur + 1e-6: scored.append((o, v, i, j, d))
        scored.sort(key=lambda x: -x[0]); moved = False
        for o, v, i, j, d in scored:
            if S.gates_ok(v):
                w, cur = v, o; moved = True
                p25, g, p5 = rep(w)
                print(f"[{tag}] {step+1}: {i}->{j} {d*100:.0f}% | R25 {p25:.3f} ({(p25**(1/40.)-1)*100:.2f}%/yr) geo {g:.3f} P5 {p5:.2f} | {S.show(w)} | {time.time()-t0:.0f}s", flush=True)
                break
        if not moved: break
    return w

SEEDS = {
 "G3_BAL":  {'GLDM':.15,'DBMF':.15,'SHLD':.15,'CIBR':.13,'NUKZ':.10,'CTA':.05,'CME':.05,'MSCI':.05,'XLE':.05,'BRKB':.05,'FNV':.04,'URNM':.03},
 "G3_R7a":  {'GLDM':.15,'SHLD':.15,'CIBR':.15,'DBMF':.12,'NUKZ':.11,'XLE':.08,'URNM':.07,'CME':.05,'MSCI':.05,'BRKB':.05,'XLI':.02},
 "C_BAL2":  {'GLDM':.15,'DBMF':.15,'SHLD':.15,'CIBR':.13,'NUKZ':.10,'CME':.05,'MSCI':.05,'XLE':.05,'CTA':.05,'BRKB':.05,'PSLV':.04,'URNM':.03},
 "C_BAL3":  {'GLDM':.15,'DBMF':.15,'SHLD':.15,'CIBR':.13,'NUKZ':.10,'CTA':.05,'CME':.05,'MSCI':.05,'XLE':.05,'BRKB':.05,'FNV':.04,'URNM':.03},
 "C_TAIL3": {'GLDM':.15,'DBMF':.15,'CTA':.10,'SHLD':.15,'CIBR':.12,'NUKZ':.10,'BRKB':.05,'CME':.05,'MSCI':.05,'FNV':.05,'XLE':.03},
 "C_R7a":   {'GLDM':.15,'SHLD':.15,'CIBR':.15,'DBMF':.12,'NUKZ':.11,'XLE':.08,'URNM':.07,'CME':.05,'MSCI':.05,'BRKB':.05,'XLI':.02},
 "C_XAR":   {'GLDM':.15,'SHLD':.15,'CIBR':.15,'DBMF':.12,'NUKZ':.11,'XLE':.08,'URNM':.07,'CME':.05,'MSCI':.05,'BRKB':.05,'XAR':.02},
 "C_R3bB5": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.11,'CIBR':.08,'GRID':.05,'CME':.05,'BRKB':.05,'CTA':.05,'URNM':.03,'XAR':.03,'CEG':.03,'FNV':.04,'MSCI':.03},
 "C_BAL":   {'GLDM':.15,'DBMF':.15,'SHLD':.15,'CIBR':.13,'NUKZ':.10,'BRKB':.05,'CME':.05,'MSCI':.05,'XLE':.05,'CTA':.05,'FNV':.04,'URNM':.03},
 "C_TAIL":  {'GLDM':.15,'DBMF':.15,'CTA':.10,'SHLD':.15,'CIBR':.12,'NUKZ':.10,'BRKB':.05,'CME':.05,'MSCI':.05,'FNV':.05,'XLE':.03},
}
if __name__ == "__main__":
    which = sys.argv[1:] or list(SEEDS); out = {}
    for tag in which:
        w = climb(SEEDS[tag], tag)
        out[tag] = {t: round(float(w[IDX[t]]), 2) for t in S.held(w)}
        p25, g, p5 = rep(w)
        print(f"[{tag}] FINAL R25 {p25:.3f} = {(p25**(1/40.)-1)*100:.2f}%/yr | geo {g:.3f} | {S.show(w)}", flush=True)
        json.dump(out, open("out/searchR25.json","w"), indent=1)
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, search_B as S
from core import DECKM, ERV, wvec
V = pd.read_csv("out/C_Verify_R8g_20260905.csv").set_index("Port")
BK = json.load(open("out/cverify.json"))
mr = L.monthly_returns(); win = mr.loc[pd.Period("2022-09","M"):pd.Period("2026-08","M")]
rows = []
for nm, d in BK.items():
    w = wvec(d); v = V.loc[nm]
    sub = win[list(d)]; wv = np.array([d[t] for t in sub.columns]); live = sub.notna().values
    Wm = live*wv; Wm = Wm/np.maximum(Wm.sum(1, keepdims=True), 1e-12)
    ps = pd.Series(np.nansum(sub.values*Wm, 1), index=sub.index)
    dn = ps[ps < 0]; sortino = (ps.mean()*12)/(dn.std(ddof=1)*np.sqrt(12))
    cum = (1+ps).cumprod(); mdd = (cum/cum.cummax()-1).min()
    ns, mx = S.risk_shares(w); fails = S.gate_fails(w)
    rows.append(dict(Port=nm, gates="PASS" if not fails else ";".join(fails), P5=v.P5, ruin=int(v.ruin),
                     DRB6=v.DRB6, R3AF=round(L.r3af(DECKM.T@w - float(ERV@w), L.prow(L.EVG11)-L.wer_evg11()), 4),
                     Sortino48=round(float(sortino),2), maxDD48=round(float(mdd)*100,1),
                     nuc_rs=round(ns*100), name_rs=round(mx*100), holdings=len(d),
                     singles=int(v.singles), ER=round(float(ERV@w)*100,3), tax=v.tax))
df = pd.DataFrame(rows).set_index("Port"); df.to_csv("out/C_Extra_20260905.csv")
pd.set_option("display.width",300); print(df.to_string())
