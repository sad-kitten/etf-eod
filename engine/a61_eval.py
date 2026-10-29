"""Addendum 61 — assumption sensitivity: U1 at gamma 1/2/3/5 for books under env-defined variants.
Variant knobs (env): ENGINEV2_PANEL (trend-2022-neutral panel), UNC_THEM/UNC_BLOCK (thematic error), DECK_SHIFT (gold rows), ENGINEV2_LAM.
Run: source env57.sh; [knobs] python3 a61_eval.py books.json Name1,Name2 out.csv tag"""
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import standing as ST, report_v3 as R, uncertainty as U, lib_pathce as L
from core import wvec, ERV
BK = json.load(open(sys.argv[1])); names = sys.argv[2].split(","); tag = sys.argv[4]
DX = U.Draws(L.NP_, 31415); rows = []
for nm in names:
    d = {t: v for t, v in BK[nm].items() if v > 0}; w = wvec(d); r = R.M0.T @ w - float(ERV @ w)
    sig = {p: ST.sigma_vec(w, p)[0] for p in ("robust", "canonical")}
    dn, ds, m = DX.book(w, 1.0, True); mc = U.suite_u(r, sig, dn, ds, m); W = R.pooled(mc)
    row = dict(variant=tag, Port=nm, **{f"U1 g{g}": round(R.ce_cagr(W, g)[1], 2) for g in (1, 1.5, 2, 3, 5)},
               **{"U1 P5": round(mc[("robust", "G")]["P5"], 3), "U1 geo x": round(R.ce_cagr(W, 1)[0], 2),
                  "worst %": round(float(r.min()) * 100, 2), "sigma0 %": round(ST.sigma0_of(w) * 100, 2)})
    rows.append(row); print(row, flush=True)
pd.DataFrame(rows).to_csv(sys.argv[3], index=False)
