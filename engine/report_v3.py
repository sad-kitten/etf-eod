"""report_v3 (v3.4 bundle, engine-exact 60k x 4 conventions, regime sigma), T-generic (50-y basis via ENGINEV2_T).
Verbatim from R8n_Code_20261024.py. Usage: python report_v3.py books.json out.csv"""
import sys, os, json, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, policy_engine2 as E, standing as ST, search_B as S
from core import DECKM, ERV, wvec, UNIV, IDX
CONV = [("robust","G"),("robust","L"),("canonical","G"),("canonical","L")]
T = L.T
SEVv = L.SEV
def shifted(M, spec):
    M = M.copy()
    for t, v in (spec or {}).items():
        if t not in IDX: continue
        if isinstance(v, dict):
            if "all" in v: M[IDX[t]] += v["all"] / 100.0
            if "sev45" in v: M[IDX[t], SEVv >= 4] += v["sev45"] / 100.0
        else: M[IDX[t]] += v / 100.0
    return M
DSHIFT = json.loads(os.environ["DECK_SHIFT"]) if os.environ.get("DECK_SHIFT") else {}
M0 = shifted(DECKM, DSHIFT); H0 = shifted(S.HCC, DSHIFT); MS = ST.stressed_deck(M0)

def ce_cagr(W, gam):
    ce = float(np.exp(np.log(W).mean())) if gam == 1 else float(np.mean(W ** (1 - gam)) ** (1 / (1 - gam)))
    return ce, (ce ** (1 / T) - 1) * 100
def pooled(mc, key="W"): return np.concatenate([np.maximum(mc[c][key], 1e-9) for c in CONV])
def profile(nm, d):
    t0 = time.time()
    w = wvec(d); wer = float(ERV @ w); r = M0.T @ w - wer; rs = MS.T @ w - wer
    sig = {p: ST.sigma_vec(w, p)[0] for p in ("robust", "canonical")}
    mc = E.suite(r, sig, E.POL); W = pooled(mc); rg = mc[("robust", "G")]
    m1 = E.suite(r - 0.01, sig, E.POL); m2 = E.suite(r - 0.02, sig, E.POL); mh = E.suite(H0.T @ w - wer, sig, E.POL)
    g1x, g1 = ce_cagr(W, 1); g3x, g3 = ce_cagr(W, 3); g5x, g5 = ce_cagr(W, 5)
    pc = W ** (1 / T) - 1
    p5_mean = None
    if abs(rg["P5"] - ST.GATES["P5"]) < ST.P5_BAND:
        p5_mean = float(np.mean([E.run(np.random.default_rng(s), r, L.PI_ROBUST, "G", sig["robust"], E.POL)["P5"] for s in ST.P5_SEEDS]))
    fails = ST.gate_fails_abs(rg, m2[("robust", "G")], mh[("robust", "G")], r, p5_mean=p5_mean)
    sfail = float(rs.min()) < ST.STRESS_FLOOR - 1e-12
    if sfail: fails.append(f"stressed worst {rs.min()*100:.2f}<{ST.STRESS_FLOOR*100:.2f}")
    pim = (np.array(L.PI_ROBUST) + np.array(L.PI_CANON)) / 2
    row = {"Port": nm, "geo x": round(g1x, 3), "real CAGR %": round(g1, 2), "nominal CAGR %": round(((1 + g1/100) * 1.025 - 1) * 100, 2),
        "g3 CE-CAGR %": round(g3, 2), "g5 CE-CAGR %": round(g5, 2), "P5 terminal x": round(rg["P5"], 3),
        "P5 path CAGR %": round(float(np.quantile(pc, .05)) * 100, 2), "P25 path CAGR %": round(float(np.quantile(pc, .25)) * 100, 2),
        "P50 terminal x": round(float(np.median(W)), 2), "P(W<1) %": round(float((W < 1.0).mean()) * 100, 1),
        "-100bp CAGR %": round(ce_cagr(pooled(m1), 1)[1], 2), "-200bp geo / P5": f"{E.geo(m2):.2f} / {m2[('robust','G')]['P5']:.2f}",
        "HCC geo / P5": f"{E.geo(mh):.2f} / {mh[('robust','G')]['P5']:.2f}",
        "pi-mean net %": round(float(pim @ r) * 100, 2),
        "worst scen %": round(float(r.min()) * 100, 2), "worst scen": L.SC[int(np.argmin(r))][:24],
        "stressed worst %": round(float(rs.min()) * 100, 2), "R3AF": round(ST.r3af_std(r), 4),
        "ruin": int(rg["ruin"]), "gates": "PASS" if not fails else "; ".join(fails),
        "constraints": "ok" if not ST.violations(w) else "; ".join(ST.violations(w)),
        "lifetime tax x": round(rg["tax_paid"], 3), "yrs reduced %": round(rg["yrs_red"] * 100, 1), "secs": round(time.time() - t0)}
    row.update(ST.report_structure(w, d)); row["weights"] = " ".join(f"{t}{int(round(x*100))}" for t, x in sorted(d.items(), key=lambda kv: -kv[1]))
    return row
if __name__ == "__main__":
    BK = json.load(open(sys.argv[1])); rows = []
    for nm, d in BK.items():
        d = {t: v for t, v in d.items() if v > 0}; rows.append(profile(nm, d))
        x = rows[-1]; print(f"{nm:14s} geo {x['geo x']:.2f} real {x['real CAGR %']:.2f} g3 {x['g3 CE-CAGR %']:.2f} P5 {x['P5 terminal x']:.2f} worst {x['worst scen %']:.2f} | {x['gates']} | {x['constraints']} ({x['secs']}s)", flush=True)
    pd.DataFrame(rows).set_index("Port").to_csv(sys.argv[2])
