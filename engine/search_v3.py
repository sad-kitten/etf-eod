"""search_v3 (v3.4 bundle, verbatim logic from R8n_Code_20261024.py)."""
import sys, os, json, time, numpy as np, warnings; warnings.filterwarnings('ignore')
import standing as ST, search_B as S
from core import UNIV, IDX, DECKM, ERV, wvec, CONVS
OBJ = os.environ.get("OBJ", "CE3"); KEY = {"CE3": "CE3", "GEO": "PathCE", "CE5": "CE5"}[OBJ]
SIZES = [0.01, 0.02, 0.03]; MAXSTEPS = int(os.environ.get("MAXSTEPS", "40")); SHORTLIST = int(os.environ.get("SHORTLIST", "30"))
EXCL = set(os.environ.get("EXCL", "DBMFh,RSSTh,QTUM").split(","))
def objective(w):
    r = DECKM.T @ w - float(ERV @ w); vals = []
    for pin in ("robust", "canonical"):
        sigv, _ = ST.sigma_vec(w, pin)
        for conv in ("G", "L"): vals.append(np.log(ST.run_fast(r, sigv, pin, conv)[KEY]))
    return float(np.mean(vals))
def admissible(v): return not ST.violations(v)
def gates_ok(v): return not ST.evaluate_fast(v)["fails"]
def deficit(v):
    e = ST.evaluate_fast(v); r = DECKM.T @ v - float(ERV @ v); d = 0.0
    d += max(0.0, ST.GATES["worst"] - float(r.min())) * 100
    d += max(0.0, ST.STRESS_FLOOR - ST.stressed_worst(v)[0]) * 100
    d += max(0.0, ST.FAST_P5 - e["P5"]); d += max(0.0, ST.GATES["R3AF"] - ST.r3af_std(r)) * 10
    d += 1.0 * (e["ruin"] > 0) + 1.0 * (ST.run_fast(r - 0.02, ST.sigma_vec(v, "robust")[0], "robust", "G")["ruin"] > 0)
    return d
def neighbours(w, pool):
    out = []
    for i in S.held(w):
        wi = w[IDX[i]]
        for j in pool:
            if j == i: continue
            wj = w[IDX[j]]; sizes = SIZES if wj > 1e-9 else [s for s in SIZES if s >= 0.02]
            for d in sizes:
                if d > wi + 1e-9 or (0 < wi - d < 0.02 - 1e-9): continue
                v = w.copy(); v[IDX[i]] -= d; v[IDX[j]] += d; v[np.abs(v) < 1e-12] = 0.0
                if admissible(v): out.append((v, i, j, d))
    return out
