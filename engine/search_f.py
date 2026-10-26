"""Addendum 56 climb (verbatim logic from Factor_Code_20261025.py search_f.py): E1-F U1 gamma-3 objective, dual-sigma gates,
LOCK floors, ADD_POOL, EXCL. 2026-10-26: repair phase also accepts PREF-violating seeds (deficit adds gold-over-cap)."""
import os, sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
import standing as ST, search_B as S, uncertainty as U, search_v3 as SV
from core import DECKM, ERV, IDX, UNIV
D = U.Draws(S.FE.npaths, 2718)
def objective(w):
    r = DECKM.T @ w - float(ERV @ w); dn, ds, m = D.book(w, 1.0, True); vals = []
    for pin in ("robust", "canonical"):
        sigv, _ = ST.sigma_vec(w, pin)
        for conv in ("G", "L"): vals.append(np.log(U.run_fast_u(r, sigv, pin, conv, dn, ds, m)["CE3"]))
    return float(np.mean(vals))
SV.objective = objective
_g, _d = SV.gates_ok, SV.deficit
def _m1(f, v):
    s = ST.SIGMA_LEVEL; ST.SIGMA_LEVEL = "cov120"
    try: return f(v)
    finally: ST.SIGMA_LEVEL = s
SV.gates_ok = lambda v: _g(v) and _m1(_g, v)
SV.deficit = lambda v: _d(v) + _m1(_d, v)
LOCK = {k: float(x) for k, x in (p.split(":") for p in os.environ.get("LOCK", "").split(",") if p)}
ADD = [t for t in os.environ.get("ADD_POOL", "").split(",") if t]
EXCL = set(os.environ.get("EXCL", "").split(","))
def climb(seed, tag):
    import time
    w = SV.wvec(seed) if hasattr(SV, "wvec") else None
    from core import wvec; w = wvec(seed); t0 = time.time()
    pool_all = sorted((set(S.POOL) | {"BRKB"} | set(S.S1)) - EXCL)
    sc = {}
    for j in pool_all:
        v = w * 0.97; v[IDX[j]] += 0.03; v /= v.sum(); sc[j] = objective(v)
    pool = sorted((set(sorted(sc, key=lambda j: -sc[j])[:SV.SHORTLIST]) | set(S.held(w)) | set(ADD)) - EXCL)
    print(f"[{tag}] pool {pool}", flush=True)
    def nb(w):
        out = []
        for v, i, j, d in SV.neighbours(w, pool):
            if i in LOCK and v[IDX[i]] < LOCK[i] - 1e-9: continue
            out.append((v, i, j, d))
        return out
    cur = objective(w); print(f"[{tag}] start {S.show(w)} | U1ce3 {np.exp(cur):.3f} | viol {ST.violations(w)} | fails {ST.evaluate_fast(w)['fails']}", flush=True)
    for step in range(SV.MAXSTEPS):
        cands = [(o, v, i, j, d) for v, i, j, d in nb(w) for o in [objective(v)] if o > cur + 1e-6]
        if not cands: break
        cands.sort(key=lambda x: -x[0]); acc = None
        for c in cands[:20]:
            if SV.gates_ok(c[1]): acc = c; break
        if acc is None: print(f"[{tag}] no improving move clears gates", flush=True); break
        o, v, i, j, d = acc; w, cur = v, o
        print(f"[{tag}] {step+1}: {i}->{j} {d*100:.0f}% | U1ce3 {np.exp(cur):.3f} | {S.show(w)} | {time.time()-t0:.0f}s", flush=True)
    return w
if __name__ == "__main__":
    BK = json.load(open(os.environ["SEEDS_FILE"])); tag = sys.argv[1]
    w = climb({t: v for t, v in BK[tag].items() if v > 0}, tag)
    json.dump({tag: {t: round(float(w[IDX[t]]), 2) for t in S.held(w)}}, open(os.environ["OUTJSON"], "w"), indent=1)
    print(f"[{tag}] FINAL {S.show(w)}", flush=True)
