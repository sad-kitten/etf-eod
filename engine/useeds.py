"""Seed-robustness check for E1 U1/U2 (Addendum 61). Same pipeline as report_u.py, estimation-error draws re-seeded.
Run: source env57.sh; python3 useeds.py books.json Book seed1,seed2,... out.csv"""
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import standing as ST, report_v3 as R, uncertainty as U, lib_pathce as L
from core import wvec, ERV
BK = json.load(open(sys.argv[1])); nm = sys.argv[2]; seeds = [int(s) for s in sys.argv[3].split(",")]
d = {t: v for t, v in BK[nm].items() if v > 0}; w = wvec(d); r = R.M0.T @ w - float(ERV @ w)
sig = {p: ST.sigma_vec(w, p)[0] for p in ("robust", "canonical")}
rows = []
for sd in seeds:
    DX = U.Draws(L.NP_, sd)
    for tag, k in (("U1", 1.0), ("U2", 2.0)):
        dn, ds, m = DX.book(w, k, True); mc = U.suite_u(r, sig, dn, ds, m); W = R.pooled(mc)
        rows.append(dict(Port=nm, seed=sd, tag=tag, g3=R.ce_cagr(W, 3)[1], geo=R.ce_cagr(W, 1)[0],
                         P5=mc[("robust", "G")]["P5"], P1=float(np.quantile(W, .01)), depleted_pct=float((W <= 1e-8).mean() * 100)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(sys.argv[4], index=False)
