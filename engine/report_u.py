"""Engine-exact report (Addendum 56 report_u.py, verbatim logic). Run: python3 report_u.py books.json out.csv [names]"""
import sys, os, json, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import standing as ST, report_v3 as R, uncertainty as U, policy_engine2 as E, lib_pathce as L
from core import wvec, ERV, IDX
DX = U.Draws(L.NP_, 31415)
BK = json.load(open(sys.argv[1])); keep = sys.argv[3].split(",") if len(sys.argv) > 3 else list(BK)
rows = []
for nm in keep:
    d = {t: v for t, v in BK[nm].items() if v > 0}
    t0 = time.time(); row = R.profile(nm, d)
    w = wvec(d); r = R.M0.T @ w - float(ERV @ w); sig = {p: ST.sigma_vec(w, p)[0] for p in ("robust", "canonical")}
    for tag, k in (("U1", 1.0), ("Uhalf", 0.5), ("U2", 2.0)):
        dn, ds, m = DX.book(w, k, True); mc = U.suite_u(r, sig, dn, ds, m); W = R.pooled(mc); rg = mc[("robust", "G")]
        pc = W ** (1 / float(L.T)) - 1
        row.update({f"{tag} geo x": round(R.ce_cagr(W, 1)[0], 3), f"{tag} real CAGR %": round(R.ce_cagr(W, 1)[1], 2),
                    f"{tag} g3 CE-CAGR %": round(R.ce_cagr(W, 3)[1], 2), f"{tag} P5 x": round(rg["P5"], 3),
                    f"{tag} P5 path CAGR %": round(float(np.quantile(pc, .05)) * 100, 2),
                    f"{tag} P50 x": round(float(np.median(W)), 2), f"{tag} P(W<1) %": round(float((W < 1).mean()) * 100, 1)})
    sdn, sds = DX.book_sd(w); ts = U.tail_stats(r)
    row.update({"err sd pp": round(sdn * 100, 2), "CVaR10 %": round(ts["cvar10"] * 100, 2)})
    row["secs total"] = round(time.time() - t0); rows.append(row)
    print(f"{nm:18s} geo {row['geo x']:.2f} g3 {row['g3 CE-CAGR %']:.2f} P5 {row['P5 terminal x']:.2f} | U1 geo {row['U1 geo x']:.2f} g3 {row['U1 g3 CE-CAGR %']:.2f} P5 {row['U1 P5 x']:.2f} | U2 g3 {row['U2 g3 CE-CAGR %']:.2f} | {row['gates']} | {row['constraints']} ({row['secs total']}s)", flush=True)
pd.DataFrame(rows).set_index("Port").to_csv(sys.argv[2])
