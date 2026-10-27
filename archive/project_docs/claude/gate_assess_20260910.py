"""Gate retention assessment: for every gate (and the legacy gates that were dropped), on the whole candidate pool —
fail counts, UNIQUE fail counts, overlap, margins, knife-edge sensitivity to 1% trades, and fast-vs-exact offset."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, standing as ST, search_B as S
from core import DECKM, ERV, wvec, IDX, UNIV, feasible
pool = json.load(open("out/gate_pool.json"))
bench = L.prow(L.EVG11) - L.wer_evg11()
rows = []
for nm, d in pool.items():
    w = wvec(d)
    if not feasible(w): continue
    r = DECKM.T @ w - float(ERV @ w)
    e = ST.evaluate_fast(w)
    sigv, _ = ST.sigma_vec(w, "robust")
    base = ST.run_fast(r, sigv, "robust", "G"); ls = ST.run_fast(r - 0.02, sigv, "robust", "G"); hcc = ST.run_fast(S.HCC.T @ w - float(ERV @ w), sigv, "robust", "G")
    rows.append(dict(Port=nm, geo=e["geo"], ce3=e["ce3"], P5=base["P5"], P10=base["P10"], ruin=base["ruin"], worst=float(r.min()),
                     R3AF=L.r3af(r, bench), LS_P5=ls["P5"], LS_ruin=ls["ruin"], HCC_P5=hcc["P5"], HCC_ruin=hcc["ruin"],
                     DRB6=L.drb40(r, h=0.06), RCEW=L.rcew40(r)["robust"], sf6=ST.shortfall6(r), viol=len(ST.violations(w))))
df = pd.DataFrame(rows).set_index("Port"); df.to_csv("out/Gate_Pool_Stats_20260910.csv")
G = ST.GATES
fails = pd.DataFrame({
    "ruin": df.ruin > G["ruin"], "P5>=1.00": df.P5 < G["P5"], "worst>=5.5": df.worst < G["worst"], "R3AF": df.R3AF < G["R3AF"],
    "LS ruin": df.LS_ruin > 0, "LS P5>=0.45": df.LS_P5 < G["LS_P5"], "HCC ruin": df.HCC_ruin > 0, "HCC P5>=0.85": df.HCC_P5 < G["HCC_P5"],
    "(legacy) P10>=P1": df.P10 < df.loc["P1", "P10"] if "P1" in df.index else False,
    "(legacy) DRB6<=P1": df.DRB6 > df.loc["P1", "DRB6"] if "P1" in df.index else False,
    "(legacy) RCEW>=P1": df.RCEW < df.loc["P1", "RCEW"] if "P1" in df.index else False,
    "(dropped) RCEW>=14": df.RCEW < 14.0, "(dropped) DRB6<=0.03": df.DRB6 > 0.03})
active = ["ruin","P5>=1.00","worst>=5.5","R3AF","LS ruin","LS P5>=0.45","HCC ruin","HCC P5>=0.85"]
n = len(df); print(f"pool: {n} feasible books\n")
print(f"{'gate':22s} {'fails':>5s} {'unique':>6s} {'fails|constraints ok':>20s}  margin: min / p10 / median")
for g in fails.columns:
    f = fails[g]; others = fails[[x for x in active if x != g]].any(axis=1) if g in active else fails[active].any(axis=1)
    uniq = int((f & ~others).sum()) if g in active else int((f & ~others).sum())
    ok = int((f & (df.viol == 0)).sum())
    col = {"P5>=1.00": df.P5 - 1.0, "worst>=5.5": (df.worst - 0.055) * 100, "R3AF": df.R3AF + 0.002, "LS P5>=0.45": df.LS_P5 - 0.45,
           "HCC P5>=0.85": df.HCC_P5 - 0.85, "(dropped) RCEW>=14": df.RCEW - 14, "(dropped) DRB6<=0.03": 0.03 - df.DRB6}.get(g)
    m = f"{col.min():+.3f} / {col.quantile(.1):+.3f} / {col.median():+.3f}" if col is not None else ""
    print(f"{g:22s} {int(f.sum()):5d} {uniq:6d} {ok:20d}  {m}")
print("\ncorrelation of gate statistics across the pool:")
print(df[["P5","P10","LS_P5","HCC_P5","worst","R3AF","DRB6","RCEW","sf6","ce3","geo"]].corr().round(2).to_string())
# knife-edge: fraction of 1% one-trade neighbours of the frontier books that flip each gate
print("\nknife-edge: share of admissible 1%-trade neighbours whose gate status differs from the book's")
FR = {k: pool[k] for k in ["V3_geoX3c","X3d","X2_F2_DBMF15","C5_bal","V3_reliab"] if k in pool}
pool_t = sorted(set(S.POOL) | {"BRKB"} | set(S.S1))
for nm, d in FR.items():
    w0 = wvec(d); e0 = set(x.split(" ")[0] for x in ST.evaluate_fast(w0)["fails"]); flips = {}; cnt = 0
    for i in S.held(w0):
        for j in pool_t:
            if j == i: continue
            v = w0.copy(); v[IDX[i]] -= 0.01; v[IDX[j]] += 0.01; v[np.abs(v) < 1e-12] = 0
            if v[IDX[i]] > 0 and v[IDX[i]] < 0.02 - 1e-9: continue
            if v[IDX[j]] < 0.02 - 1e-9 or ST.violations(v): continue
            cnt += 1; e = set(x.split(" ")[0] for x in ST.evaluate_fast(v)["fails"])
            for g in (e ^ e0): flips[g] = flips.get(g, 0) + 1
    print(f"  {nm:13s} {cnt:4d} neighbours: " + (", ".join(f"{g} {c/cnt*100:.0f}%" for g, c in sorted(flips.items(), key=lambda kv: -kv[1])) or "no gate flips"))
