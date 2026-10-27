"""The feasible neighbourhood of the recommended book R7a: every gate-passing one-trade variant
(1/2/3 pp moved between a held name and any admissible name), scored on the same two ratified
criteria. Purpose: identify the books whose performance is indistinguishable from R7a's, so the
choice among them can be made on grounds the objective does not see."""
import numpy as np, json, itertools, time, os, sys
import lib_pathce as L
from core import UNIV, IDX, DECKM, ERV, sigma_of, wvec, CONVS, feasible, SINGLES
import search_B as S
import search_SACE as SS

R7A = {'GLDM':.15,'SHLD':.15,'CIBR':.15,'DBMF':.12,'NUKZ':.11,'XLE':.08,'URNM':.07,'CME':.05,'MSCI':.05,'BRKB':.05,'XLI':.02}
w0 = wvec(R7A); assert abs(w0.sum()-1) < 1e-9 and feasible(w0)
KMAX = 2   # R7a holds 2 single names (BRKB excluded); do not widen the single-name count

pool = sorted(set(S.POOL) | {"BRKB"} | set(S.S1))
cands = {}
for i in S.held(w0):
    wi = w0[IDX[i]]
    for j in pool:
        if j == i: continue
        for d in (0.01, 0.02, 0.03):
            if d > wi + 1e-9: continue
            v = w0.copy(); v[IDX[i]] -= d; v[IDX[j]] += d
            v[np.abs(v) < 1e-12] = 0.0
            if abs(v.sum() - 1.0) > 1e-9: continue
            if not feasible(v) or S.n_singles(v) > KMAX: continue
            if v[IDX['XLE']] > S.XLE_CAP + 1e-9 or not S.nuc_ok(v) or not S.risk_ok(v): continue
            cands[(i, j, d)] = v
print(f"feasible one-trade neighbours: {len(cands)}", flush=True)

t0 = time.time(); rows = []
for k, v in cands.items():
    if S.cf_fails(v): continue          # closed-form gates first (cheap)
    f = S.gate_fails(v)
    o, g, gL, p5, pen = SS.objective(v)
    ns, mx = S.risk_shares(v)
    rows.append(dict(move=f"{k[0]}->{k[1]} {int(k[2]*100)}", geo=float(np.exp(g)), SACE=float(np.exp(o)),
                     P5=float(p5), pen=float(pen), nuc_rs=round(ns,3), name_rs=round(mx,3), fails=";".join(f), w={t: round(float(v[IDX[t]]),2) for t in S.held(v)}))
print(f"scored {len(rows)} gate-eligible neighbours in {time.time()-t0:.0f}s", flush=True)
o0, g0, gL0, p50, pen0 = SS.objective(w0)
base = dict(move="R7a (base)", geo=float(np.exp(g0)), SACE=float(np.exp(o0)), P5=float(p50), pen=float(pen0), fails="", w=R7A)
rows.sort(key=lambda r: -r["geo"])
json.dump(dict(base=base, rows=rows), open("out/nbhd_R7a.json","w"), indent=1)
print(f"BASE geo {base['geo']:.3f} SACE {base['SACE']:.3f} P5 {base['P5']:.2f}")
for r in rows[:25]:
    print(f"  {r['move']:18s} geo {r['geo']:.3f} ({r['geo']-base['geo']:+.3f}) SACE {r['SACE']:.3f} ({r['SACE']-base['SACE']:+.3f}) P5 {r['P5']:.2f} {'PASS' if not r['fails'] else 'FAIL '+r['fails']}")
"""MC noise floor: re-run the recommended book and two neighbours under different RNG seeds so
'very close' can be defined by the engine's own resolution rather than by eye."""
import numpy as np, json, sys
import lib_pathce as L
import policy_engine2 as E
import search_SACE as SS
from core import DECKM, ERV, sigma_of, wvec

def suite_seed(r, sig, seed):
    out = {}
    for pin, pi in (("robust", L.PI_ROBUST), ("canonical", L.PI_CANON)):
        rng = np.random.default_rng(seed)
        for conv in ("G", "L"):
            out[(pin, conv)] = E.run(rng, r, pi, conv, sig, E.POL)
    return out

BK = {"R7a": {'GLDM':.15,'SHLD':.15,'CIBR':.15,'DBMF':.12,'NUKZ':.11,'XLE':.08,'URNM':.07,'CME':.05,'MSCI':.05,'BRKB':.05,'XLI':.02},
      "R5b_D1": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'CIBR':.15,'NUKZ':.11,'XLE':.08,'URNM':.07,'CME':.05,'MSCI':.05,'SMH':.02,'PSLV':.02}}
SEEDS = [42, 7, 101, 2024, 555, 31337]
for nm, d in BK.items():
    w = wvec(d); assert abs(w.sum()-1) < 1e-9
    r = DECKM.T @ w - float(ERV @ w); sig = sigma_of(w)
    G, P, S = [], [], []
    for s in SEEDS:
        mc = suite_seed(r, sig, s); mcL = suite_seed(r - 0.02, sig, s)
        slope = (np.log(E.geo(mc)) - np.log(E.geo(mcL))) / 2
        pen = 0.9*slope + 0.8*float(SS.TW @ w)*slope + SS.s4(w) + 0.25*float(SS.SW @ w)
        G.append(E.geo(mc)); P.append(mc[("robust","G")]["P5"]); S.append(E.geo(mc)*np.exp(-pen))
    G, P, S = map(np.array, (G, P, S))
    print(f"{nm}: geo {G.mean():.4f} sd {G.std(ddof=1):.4f} range {G.min():.3f}-{G.max():.3f}")
    print(f"      P5 {P.mean():.4f} sd {P.std(ddof=1):.4f} range {P.min():.3f}-{P.max():.3f}")
    print(f"    SACE {S.mean():.4f} sd {S.std(ddof=1):.4f} range {S.min():.3f}-{S.max():.3f}", flush=True)
"""How much of the ordering inside the tie group is real? Perturb each deck ROW LEVEL by a
grade-scaled shock (A 0.4pp, B 0.7pp, C 1.5pp sd -- the campaign's own realised corrections were
CEG -2.5, URNM -0.6, RPRX -3.1, so this is if anything optimistic), recompute every book's
scenario-return vector, and rank by RCEW40 (closed-form CE, monotone proxy for geo-PathCE).
Reports how often each book is #1 and its mean rank."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L
from core import DECKM, ERV, wvec, UNIV, IDX
import search_B as S

BK = json.load(open("out/tiegroup.json"))
SD = {"A": 0.004, "B": 0.007, "C": 0.015}
sd = np.array([SD.get(S.grades.get(t, "B"), 0.007) for t in UNIV])
W = {nm: wvec({a: b for a, b in d.items() if b > 0}) for nm, d in BK.items()}
names = list(W)
rng = np.random.default_rng(2026); N = 2000
rank1 = {n: 0 for n in names}; ranks = {n: [] for n in names}
for _ in range(N):
    shock = rng.standard_normal(len(UNIV)) * sd
    D = DECKM + shock[:, None]
    sc = {n: L.rcew40(D.T @ w - float(ERV @ w))["robust"] for n, w in W.items()}
    order = sorted(names, key=lambda n: -sc[n])
    rank1[order[0]] += 1
    for i, n in enumerate(order): ranks[n].append(i + 1)
base = {n: L.rcew40(DECKM.T @ w - float(ERV @ w))["robust"] for n, w in W.items()}
out = pd.DataFrame([dict(Port=n, RCEW40=round(base[n], 2), rank_base=0,
                         pct_best=round(100 * rank1[n] / N, 1),
                         mean_rank=round(float(np.mean(ranks[n])), 1),
                         p10_rank=int(np.percentile(ranks[n], 10)), p90_rank=int(np.percentile(ranks[n], 90)))
                    for n in names])
out = out.sort_values("RCEW40", ascending=False).reset_index(drop=True)
out["rank_base"] = out.index + 1
out.to_csv("out/Tie_Sensitivity_20260905.csv", index=False)
print(out.to_string(index=False))
"""Full metrics for the books whose performance is indistinguishable from the recommended R7a.
Engine columns from Tie_Verify_R8g_20260905.csv (60k engine-exact, basis B, deck R8g)."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, search_B as S
from core import DECKM, ERV, wvec
V = pd.read_csv("out/Tie_Verify_R8g_20260905.csv").set_index("Port")
SENS = pd.read_csv("out/Tie_Sensitivity_20260905.csv").set_index("Port")
BOOKS = json.load(open("out/tiegroup.json"))
ORDER = ["R7a","N_XLI_XAR2","N_DBMF_XAR2","N_CIBR_XLI1","N_XLI_IAI2","R5b","R5b_GX3","R7r_ROKT4","R7j_JQUA5","R5b_D1","R5c2"]
mr = L.monthly_returns(); win = mr.loc[pd.Period("2022-09","M"):pd.Period("2026-08","M")]
ivv_full = mr.loc[pd.Period("2018-11","M"):pd.Period("2026-08","M")]["IVV"]; bad = ivv_full <= ivv_full.quantile(0.10)
bench = L.prow(L.EVG11) - L.wer_evg11()
rows = {}
for nm in ORDER:
    d = {a: b for a, b in BOOKS[nm].items() if b > 0}; w = wvec(d)
    r = DECKM.T @ w - float(ERV @ w); v = V.loc[nm]
    sub = win[list(d)]; wv = np.array([d[t] for t in sub.columns]); live = sub.notna().values
    Wm = live*wv; Wm = Wm/np.maximum(Wm.sum(1, keepdims=True), 1e-12)
    ps48 = pd.Series(np.nansum(sub.values*Wm, 1), index=sub.index)
    dn = ps48[ps48 < 0]; sortino = (ps48.mean()*12)/(dn.std(ddof=1)*np.sqrt(12))
    cum = (1+ps48).cumprod(); mdd = (cum/cum.cummax()-1).min()
    sub2 = mr.loc[pd.Period("2018-11","M"):pd.Period("2026-08","M")][list(d)]
    live2 = sub2.notna().values; W2 = live2*wv; W2 = W2/np.maximum(W2.sum(1, keepdims=True), 1e-12)
    ps = pd.Series(np.nansum(sub2.values*W2, 1), index=sub2.index)
    ns, mx = S.risk_shares(w); sn = SENS.loc[nm]
    rows[nm] = {
      "geo-PathCE (real, x corpus)":        v.geo,
      "  geo vs R7a":                       f"{v.geo-V.loc['R7a'].geo:+.2f}",
      "surprise-adjusted CE":               v.SACE,
      "  SACE vs R7a":                      f"{v.SACE-V.loc['R7a'].SACE:+.2f}",
      "P5 (5th pct terminal, real)":        v.P5,
      "ruin paths / 60k":                   int(v.ruin),
      "-200bp: geo / P5":                   f"{v.LS_geo:.2f} / {v.LS_P5:.2f}",
      "stress-vol: geo / P5":               f"{v.SV_geo:.2f} / {v.SV_P5:.2f}",
      "worst net CAGR %":                   v.worst_pct,
      "scenarios < 6% net (of 24)":         int((r < 0.06).sum()),
      "RCEW40 robust":                      round(L.rcew40(r)["robust"], 2),
      "R3AF vs EVG11":                      round(L.r3af(r, bench), 4),
      "DRB40 @6% (lower better)":           v.DRB6,
      "vol48 %":                            v.vol48,
      "Sortino 48m":                        round(float(sortino), 2),
      "maxDD 48m %":                        round(float(mdd)*100, 1),
      "IVV worst-decile: book mean %":      round(float(ps[bad.reindex(ps.index)].mean()*100), 1),
      "corr to IVV (2018-11..)":            round(float(ps.corr(mr["IVV"].reindex(ps.index))), 2),
      "nuclear complex risk share %":       round(ns*100),
      "largest name risk share %":          round(mx*100),
      "single names (BRKB excl.)":          int(v.singles),
      "holdings":                           len(d),
      "weighted ER %":                      round(float(ERV @ w)*100, 3),
      "lifetime tax (x corpus)":            v.tax,
      "% best under row noise":             sn.pct_best,
      "rank band under row noise (p10-p90)":f"{int(sn.p10_rank)}-{int(sn.p90_rank)} of 22",
    }
df = pd.DataFrame(rows)[ORDER]
df.to_csv("out/Tie_Metrics_R8g_20260905.csv")
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 40)
print(df.to_string())
print()
for nm in ORDER:
    print(f"{nm:14s} " + " ".join(f"{t}{int(round(x*100))}" for t, x in sorted(BOOKS[nm].items(), key=lambda kv: -kv[1])))
