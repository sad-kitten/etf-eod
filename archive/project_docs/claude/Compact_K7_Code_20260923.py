# ---- search_k7.py ----
"""SEARCH K7 — compact-book search for small custodial accounts (owner request 2026-09-23): at most KMAX holdings,
each at least MINW, per-holding cap MAXW, under the STANDING rules (v3.3, deck R8k) otherwise, with ONE stated relaxation:
the single-holding risk-share cap is RS_NAME_K (0.33) instead of 0.24, because a 7-holding book's equal risk share is 14%
and the volatile sleeves (SHLD, NUKZ) at 15% weight carry 25-29% of variance -- the 24% cap was calibrated on 12-14 holdings.
Objective = search_v3 objective (mean log of the 4-convention fast evaluator, OBJ=GEO|CE3), gates at FAST_P5.
Moves: 1/2/3/5-point shifts between held names (min-weight rule), close-into, replace (i -> j at the same weight),
open j at MINW funded from i (only when below KMAX). Endpoints confirmed engine-exact with report_k7.py.
Run: R7=1 R8=1 R9=1 R7B=1 DECK_FILE=Deck_Recal_R8k_20260910.csv SCEN_FILE=scenarios_25_R8i_20260909.csv
     GRADES_FILE=out/Row_Grades_R8j_20260909.csv REGCOV=1 NPATHS=6000 OBJ=GEO python3 search_k7.py [seeds]
"""
import sys, os, json, time, numpy as np, warnings; warnings.filterwarnings('ignore')
import standing as ST
import search_B as S
from core import UNIV, IDX, DECKM, ERV, wvec, CONVS, SINGLES, PMEQ, SEMIS, DEF, NUC, BAL
KMAX = int(os.environ.get("KMAX", 7)); MINW = float(os.environ.get("MINW", 0.05)); MAXW = float(os.environ.get("MAXW", 0.20))
RS_NAME_K = float(os.environ.get("RS_NAME_K", 0.33))
RS_BLOCK_K = float(os.environ.get("RS_BLOCK_K", ST.RS_BLOCK))   # micro books (<= 5 holdings): block cap relaxed with the single-holding cap
OBJ = os.environ.get("OBJ", "GEO"); KEY = {"CE3": "CE3", "GEO": "PathCE", "CE5": "CE5"}[OBJ]

def feasible_compact(w):
    if w.min() < -1e-12 or abs(w.sum() - 1) > 1e-9: return False
    held = w > 1e-9
    if held.sum() > KMAX: return False
    if np.any(w[held] < MINW - 1e-9): return False
    if w.max() > MAXW + 1e-9: return False
    for t in SINGLES:
        if t in IDX and w[IDX[t]] > 0.05 + 1e-9: return False
    ssum = lambda ts: sum(w[IDX[t]] for t in ts if t in IDX)
    if ssum(SINGLES) > 0.20 + 1e-9: return False
    if ssum(PMEQ) > 0.10 + 1e-9: return False
    if ssum(SEMIS) > 0.12 + 1e-9: return False
    if ssum(DEF) > 0.35 + 1e-9: return False
    if ssum(NUC) > 0.35 + 1e-9: return False
    if ssum(BAL) < 0.20 - 1e-9: return False
    return True
def apply_compact_rules():
    ST.feasible = feasible_compact; ST.RS_NAME = RS_NAME_K; ST.RS_BLOCK = RS_BLOCK_K
    if os.environ.get("P5_FLOOR"): ST.FAST_P5 = float(os.environ["P5_FLOOR"])   # 'similar tail' climbs: fast P5 floor above the gate (1.06)
apply_compact_rules()

def objective(w):
    r = DECKM.T @ w - float(ERV @ w); vals = []
    for pin in ("robust", "canonical"):
        sigv, _ = ST.sigma_vec(w, pin)
        for conv in ("G", "L"): vals.append(np.log(ST.run_fast(r, sigv, pin, conv)[KEY]))
    return float(np.mean(vals))
def admissible(v): return not ST.violations(v)
def gates_ok(v): return not ST.evaluate_fast(v)["fails"]

FAMILY = ["GLDM", "SHLD", "CIBR", "DBMF", "NUKZ", "URNM", "XLE", "BRKB", "CME", "MSCI", "FNV", "XAR", "SMH", "HACK", "KMLM", "CTA"]
CORE = ["GLDM", "SHLD", "CIBR", "DBMF", "NUKZ", "URNM", "XLE", "KMLM", "CTA", "IVV", "BRKB", "CME", "MSCI", "FNV", "XAR", "SMH", "HACK", "GRID", "NLR", "PPA", "QUAL", "COWZ"]
def neighbours(w, pool):
    out = {}; H = S.held(w); n = len(H)
    def add(v, i, j, d):
        v = v.copy(); v[np.abs(v) < 1e-12] = 0.0
        key = tuple(np.round(v * 1000).astype(int))
        if key not in out: out[key] = (v, i, j, d)
    for i in H:
        wi = w[IDX[i]]
        for j in pool:
            if j == i: continue
            wj = w[IDX[j]]
            if wj > 1e-9:
                for d in (0.01, 0.02, 0.03, 0.05):
                    if d > wi + 1e-9: continue
                    rem = wi - d
                    if 0 < rem < MINW - 1e-9: continue
                    v = w.copy(); v[IDX[i]] -= d; v[IDX[j]] += d; add(v, i, j, d)
                v = w.copy(); v[IDX[j]] += wi; v[IDX[i]] = 0.0; add(v, i, j, wi)          # close i into j
            else:
                v = w.copy(); v[IDX[j]] = wi; v[IDX[i]] = 0.0; add(v, i, j, wi)           # replace i by j
                if n < KMAX and wi - MINW >= MINW - 1e-9:                                 # open j at MINW from i
                    v = w.copy(); v[IDX[i]] -= MINW; v[IDX[j]] += MINW; add(v, i, j, MINW)
    return list(out.values())

def climb(seed, tag, max_steps=40, shortlist=16):
    w = wvec(seed); assert abs(w.sum() - 1) < 1e-9
    viol = ST.violations(w)
    if viol: print(f"[{tag}] seed violates constraints: {viol}; skipped", flush=True); return w, float("-inf")
    if not gates_ok(w): print(f"[{tag}] seed fails absolute gates: {ST.evaluate_fast(w)['fails']}", flush=True); return w, objective(w)
    cur = objective(w); t0 = time.time()
    pool_all = sorted(set(S.POOL) | {"BRKB"} | set(S.S1))
    sc = {}
    for j in pool_all:
        v = w * 0.95; v[IDX[j]] += 0.05; v /= v.sum(); sc[j] = objective(v)
    pool = sorted(set(sorted(sc, key=lambda j: -sc[j])[:shortlist]) | set(S.held(w)) | set(t for t in CORE if t in IDX))
    if os.environ.get("FAMILY") == "1": pool = sorted(set(FAMILY) | set(S.held(w)))   # condensation of the owner's book: family tickers only
    e = ST.evaluate_fast(w)
    print(f"[{tag}] seed {S.show(w)} | {OBJ} {np.exp(cur):.3f} geo {e['geo']:.3f} ce3 {e['ce3']:.3f} P5 {e['P5']:.2f} | pool {len(pool)}", flush=True)
    for step in range(max_steps):
        cands = []
        for v, i, j, d in neighbours(w, pool):
            if not admissible(v): continue
            o = objective(v)
            if o > cur + 1e-6: cands.append((o, v, i, j, d))
        if not cands: break
        cands.sort(key=lambda x: -x[0]); accepted = None
        for o, v, i, j, d in cands[:20]:
            if gates_ok(v): accepted = (o, v, i, j, d); break
        if accepted is None: print(f"[{tag}] no improving move clears the gates (top-20 tried); stopping", flush=True); break
        o, v, i, j, d = accepted
        w, cur = v, o; e = ST.evaluate_fast(w)
        print(f"[{tag}] {step+1}: {i}->{j} {d*100:.0f}% | {OBJ} {np.exp(cur):.3f} geo {e['geo']:.3f} ce3 {e['ce3']:.3f} ce5 {e['ce5']:.3f} P5 {e['P5']:.2f} worst {e['worst']*100:.2f} | {S.show(w)} | {time.time()-t0:.0f}s", flush=True)
    return w, cur

SEEDS = {
 "K_a": {'GLDM':.20,'SHLD':.15,'CIBR':.15,'DBMF':.15,'NUKZ':.15,'XLE':.15,'BRKB':.05},
 "K_b": {'GLDM':.20,'SHLD':.15,'CIBR':.15,'DBMF':.20,'NUKZ':.10,'URNM':.05,'XLE':.15},
 "K_c": {'GLDM':.20,'SHLD':.15,'CIBR':.20,'DBMF':.20,'NUKZ':.15,'CME':.05,'MSCI':.05},
 "K_d": {'GLDM':.20,'DBMF':.15,'KMLM':.10,'SHLD':.15,'CIBR':.15,'NUKZ':.15,'XLE':.10},
 "K_e": {'GLDM':.20,'SHLD':.15,'CIBR':.20,'DBMF':.15,'NUKZ':.15,'XLE':.15},
 "K_f": {'GLDM':.15,'SHLD':.15,'CIBR':.15,'DBMF':.20,'NUKZ':.10,'URNM':.05,'IVV':.20},
 "K_g": {'GLDM':.20,'SHLD':.15,'CIBR':.20,'DBMF':.15,'NUKZ':.15,'FNV':.05,'XLE':.10},
 "K6_a": {'GLDM':.20,'SHLD':.20,'CIBR':.20,'DBMF':.15,'URNM':.10,'XLE':.15},
 "K6_b": {'GLDM':.20,'SHLD':.15,'CIBR':.20,'DBMF':.15,'NUKZ':.15,'XLE':.15},
 "K5_a": {'GLDM':.20,'SHLD':.20,'CIBR':.20,'DBMF':.20,'XLE':.20},
 "K6_c": {'GLDM':.20,'SHLD':.20,'DBMF':.20,'CIBR':.20,'URNM':.10,'XLE':.10},
 "K6_d": {'GLDM':.20,'SHLD':.20,'DBMF':.20,'CIBR':.20,'KMLM':.15,'BRKB':.05},
 "K6_e": {'GLDM':.20,'SHLD':.20,'DBMF':.20,'CIBR':.20,'KMLM':.10,'URNM':.10},
 "K_j": {'GLDM':.20,'CIBR':.20,'SHLD':.20,'DBMF':.18,'URNM':.12,'XLE':.05,'BRKB':.05},
 "K_k": {'GLDM':.20,'SHLD':.20,'CIBR':.20,'DBMF':.20,'URNM':.10,'FNV':.05,'CME':.05},
 "K_l": {'GLDM':.20,'SHLD':.20,'CIBR':.15,'DBMF':.20,'URNM':.10,'CTA':.10,'FNV':.05},
 "K_m": {'GLDM':.20,'DBMF':.20,'SHLD':.18,'JEPI':.18,'CIBR':.14,'MSCI':.05,'BRKB':.05},
 "K6_f": {'GLDM':.20,'CIBR':.20,'SHLD':.20,'DBMF':.20,'URNM':.15,'FNV':.05},
 "K6_g": {'GLDM':.20,'CIBR':.20,'SHLD':.20,'DBMF':.20,'URNM':.12,'XLE':.08},
 "K6_h": {'GLDM':.20,'CIBR':.20,'SHLD':.20,'DBMF':.20,'URNM':.12,'BRKB':.05,'FNV':.03},
 "M5_a": {'GLDM':.25,'SHLD':.20,'CIBR':.20,'DBMF':.20,'URNM':.15},
 "M5_b": {'GLDM':.30,'SHLD':.20,'CIBR':.20,'DBMF':.20,'URNM':.10},
 "M5_c": {'GLDM':.20,'SHLD':.20,'CIBR':.20,'DBMF':.20,'XLE':.20},
 "M5_d": {'GLDM':.25,'SHLD':.20,'CIBR':.25,'DBMF':.20,'KMLM':.10},
 "M5_e": {'GLDM':.20,'SHLD':.20,'CIBR':.20,'DBMF':.25,'NUKZ':.15},
 "M4_a": {'GLDM':.30,'SHLD':.25,'CIBR':.25,'DBMF':.20},
 "M4_b": {'GLDM':.25,'SHLD':.25,'CIBR':.25,'DBMF':.25},
 "M4_c": {'GLDM':.30,'CIBR':.30,'DBMF':.20,'URNM':.20},
 "M4_d": {'GLDM':.30,'SHLD':.20,'CIBR':.30,'DBMF':.20},
 "M5_f": {'GLDM':.25,'SHLD':.25,'CIBR':.20,'DBMF':.20,'URNM':.10},
 "M4_f": {'GLDM':.30,'SHLD':.20,'CIBR':.20,'DBMF':.30},
 "M4_g": {'GLDM':.30,'SHLD':.20,'CIBR':.25,'DBMF':.25},
 "K_h":  {'GLDM':.20,'DBMF':.20,'CIBR':.20,'SHLD':.18,'KMLM':.12,'BRKB':.05,'MSCI':.05},
 "K_i":  {'GLDM':.20,'SHLD':.15,'DBMF':.20,'CIBR':.15,'KMLM':.10,'URNM':.10,'XLE':.10},
 "K5_b": {'GLDM':.20,'SHLD':.20,'CIBR':.20,'DBMF':.20,'URNM':.20},
}
if __name__ == "__main__":
    if sys.argv[1:2] == ["screen"]:
        for tag, d in SEEDS.items():
            w = wvec(d); e = ST.evaluate_fast(w); ns, mx = ST.risk_shares(w)
            print(f"{tag}: geo {e['geo']:.3f} ce3 {e['ce3']:.3f} P5 {e['P5']:.3f} worst {e['worst']*100:.2f} fails {e['fails']} | viol {ST.violations(w)} | nucRS {ns*100:.0f} maxRS {mx*100:.0f}")
        sys.exit()
    which = sys.argv[1:] or list(SEEDS); out = {}
    for tag in which:
        w, cur = climb(SEEDS[tag], tag)
        out[tag] = {t: round(float(w[IDX[t]]), 2) for t in S.held(w)}
        e = ST.evaluate_fast(w)
        print(f"[{tag}] FINAL {OBJ} {np.exp(cur):.3f} | geo {e['geo']:.3f} ce3 {e['ce3']:.3f} ce5 {e['ce5']:.3f} P5 {e['P5']:.2f} worst {e['worst']*100:.2f} | {S.show(w)}", flush=True)
        json.dump(out, open(os.environ.get("OUTJSON", f"out/searchK7_{OBJ}{'_fam' if os.environ.get('FAMILY')=='1' else ''}.json"), "w"), indent=1)

# ---- report_k7.py ----
"""REPORT K7 — engine-exact reporting (report_v3.profile) under the COMPACT rule set of search_k7 (<= KMAX holdings, MINW, MAXW,
RS_NAME_K); everything else is the standing v3.3 reporting set. Run with the standard env (see search_k7.py):
  ... python3 report_k7.py books.json out.csv
"""
import sys, json, pandas as pd, warnings; warnings.filterwarnings('ignore')
import search_k7 as K            # applies the compact rules to standing on import
import report_v3 as R
import standing as ST
if __name__ == "__main__":
    BK = json.load(open(sys.argv[1])); rows = []
    for nm, d in BK.items():
        d = {t: v for t, v in d.items() if v > 0}; rows.append(R.profile(nm, d))
        print(nm, rows[-1]["geo x"], rows[-1]["g3 CE-CAGR %"], rows[-1]["P5 terminal x"], rows[-1]["gates"], rows[-1]["constraints"], flush=True)
    df = pd.DataFrame(rows).set_index("Port"); df.to_csv(sys.argv[2])
    ok = df[(df.gates == "PASS") & (df.constraints == "ok")]
    nd = [p for p in ok.index if not any((ok.loc[q, "geo x"] >= ok.loc[p, "geo x"]) and (ok.loc[q, "g3 CE-CAGR %"] >= ok.loc[p, "g3 CE-CAGR %"])
                                        and ((ok.loc[q, "geo x"] > ok.loc[p, "geo x"]) or (ok.loc[q, "g3 CE-CAGR %"] > ok.loc[p, "g3 CE-CAGR %"])) for q in ok.index if q != p)]
    print("\nnon-dominated on (geo, g3):", sorted(nd, key=lambda p: -ok.loc[p, "geo x"]))
