# ===== standing.py (v3.1, 2026-09-10) =====
"""STANDING RULES v3.1 (v3 = Addendum 31 items applied 2026-09-09; v3.1 = Addendum 33 gate set ratified 2026-09-10).

One module that states, in code, what a book must satisfy and how it is reported:
  - constraints (items 6, 7, 8): risk-share caps, grade-C cap, singles, duplicates, short-history cap,
    block caps, ballast floor
  - absolute gates (items 3, 5): thresholds on basis (B), not relative to P1; DRB40 reported, not gated;
    smooth pi-weighted shortfall below 6% reported
  - regime-conditional volatility (item 11): sigma per scenario class, calibrated so the pi-weighted
    variance equals the standing 48-month sigma; only the ALLOCATION of variance across states changes
  - reporting set (items 4, 12): the dial, -100bp level, ladder inputs, blocks, short-history weight, tie bands
"""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L
from core import UNIV, IDX, DECKM, ERV, sigma_of, wvec, SINGLES, feasible, CONVS
import search_B as S

# ---------------------------------------------------------------- constraints (6, 7, 8)
RS_NUC, RS_NAME, RS_BLOCK = 0.38, 0.24, 0.35
CW_CAP, SINGLES_MAX, DUP, SHORT_CAP, SHORT_MONTHS, BALLAST_FLOOR = 0.20, 3, 0.03, 0.30, 60, 0.09
BLOCKS = {"nuclear": ["NUKZ", "URNM", "CEG", "SETM", "NLR", "URA"],
          "defence": ["SHLD", "XAR", "PPA", "ITA", "EUAD", "ROKT", "JEDI"],
          "gold":    ["GLDM", "FNV", "PSLV", "GDX", "WPM", "SIL"],
          "cyber":   ["CIBR", "HACK"],
          "semis":   ["SMH", "SOXX", "XSD", "XLK", "FTEC"]}
BALLAST = ["DBMF", "CTA", "KMLM"]
_mr = L.monthly_returns()
HIST = {t: int(_mr[t].notna().sum()) for t in _mr.columns}
CW = np.array([1.0 if S.grades.get(t, "B") == "C" else 0.0 for t in UNIV])
SHORT = np.array([1.0 if HIST.get(t, 0) < SHORT_MONTHS else 0.0 for t in UNIV])
ETF_LIKE = {"BRKB"}

def risk_share(w, names):
    Cv = S.COV @ w; var = float(w @ Cv); mrc = Cv * w / var
    return float(sum(mrc[IDX[t]] for t in names if t in IDX))

def violations(w):
    """Every standing constraint the book breaks; empty list = admissible."""
    v = []
    if not feasible(w): v.append("feasibility")
    ns, mx = S.risk_shares(w)
    if ns > RS_NUC + 1e-9: v.append(f"nuclear risk share {ns*100:.0f}%>{RS_NUC*100:.0f}")
    if mx > RS_NAME + 1e-9: v.append(f"single-holding risk share {mx*100:.0f}%>{RS_NAME*100:.0f}")
    for b, names in BLOCKS.items():
        if b == "nuclear": continue
        r = risk_share(w, names)
        if r > RS_BLOCK + 1e-9: v.append(f"{b} block risk share {r*100:.0f}%>{RS_BLOCK*100:.0f}")
    if float(CW @ w) > CW_CAP + 1e-9: v.append(f"grade-C weight {float(CW@w)*100:.0f}%>{CW_CAP*100:.0f}")
    if float(SHORT @ w) > SHORT_CAP + 1e-9: v.append(f"<{SHORT_MONTHS}mo weight {float(SHORT@w)*100:.0f}%>{SHORT_CAP*100:.0f}")
    nsing = sum(1 for t in SINGLES if t in IDX and w[IDX[t]] > 1e-9 and t not in ETF_LIKE)
    if nsing > SINGLES_MAX: v.append(f"{nsing} singles>{SINGLES_MAX}")
    bal = sum(w[IDX[t]] for t in BALLAST if t in IDX)
    if bal < BALLAST_FLOOR - 1e-9: v.append(f"ballast {bal*100:.0f}%<{BALLAST_FLOOR*100:.0f}")
    return v

def duplicate_of(w, others):
    """Name of an existing book within DUP (share of portfolio that must move), else None."""
    for k, u in others.items():
        if 0.5 * np.abs(w - u).sum() <= DUP + 1e-9: return k
    return None

# ---------------------------------------------------------------- regime-conditional sigma (11)
STRESS = np.array([1.0 if s >= 4 else 0.0 for s in L.SEV])         # scenario class by severity
_ivv = _mr["IVV"]; _cut = _ivv.quantile(1 / 3)
_mS = _mr[_ivv <= _cut]; _mN = _mr[_ivv > _cut]                      # bottom-tercile equity months vs the rest
# short-history names are represented by their long-history sector anchor in the CONDITIONAL panel only
# (same anchor logic as the rows); this gives 100-250 stress months instead of ~10.
ANCHOR = {"NUKZ": "NLR", "SHLD": "PPA", "SETM": "COPX", "JEDI": "PPA", "EUAD": "ITA", "CEG": "NLR", "IBIT": "IBIT", "CTA": "DBMF"}
_cols = [t for t in UNIV if t in _mr.columns]
def _cov(m):
    src = m[[ANCHOR.get(t, t) if HIST.get(t, 0) < SHORT_MONTHS and ANCHOR.get(t, t) in m.columns else t for t in _cols]].copy()
    src.columns = _cols
    c = src.cov(min_periods=18) * 12.0
    full = pd.DataFrame(0.0, index=UNIV, columns=UNIV); full.loc[_cols, _cols] = c.values
    return np.nan_to_num(full.values)
COV_S, COV_N = _cov(_mS), _cov(_mN)
SHRINK = 0.5   # ratio shrunk halfway toward 1 (conditional covariances are noisy)
P_STRESS = {pin: float(pi @ STRESS) for pin, pi in (("robust", L.PI_ROBUST), ("canonical", L.PI_CANON))}

def sigma_vec(w, pin="robust"):
    """Per-scenario sigma vector: sigma0 x (ratio in stress states, 1 in normal), rescaled so that the
    pi-weighted variance equals sigma0^2. Level unchanged; allocation across states from the conditional panel."""
    s0 = sigma_of(w)
    vs, vn = float(w @ COV_S @ w), float(w @ COV_N @ w)
    ratio = np.sqrt(max(vs, 1e-12) / max(vn, 1e-12)); ratio = float(np.clip(1.0 + SHRINK * (ratio - 1.0), 0.6, 2.0))
    p = P_STRESS[pin]
    k = 1.0 / np.sqrt(p * ratio**2 + (1 - p))
    return s0 * k * np.where(STRESS > 0, ratio, 1.0), ratio

# ---------------------------------------------------------------- absolute gates (3, 5)
# v3.1 (ratified 2026-09-10, Addendum 33 s3): four gates. -200bp P5, HCC P5, HCC ruin, DRB40, RCEW40, shortfall6, P10 are REPORTED, not gated.
GATES = dict(ruin=0, P5=1.00, worst=0.0525, R3AF=-0.002, LS_ruin=0)
# The fast evaluator (search basis, no taxes) overstates P5 by +0.048 (sd 0.012) against the exact engine: the search-time
# threshold is 1.06 so that books admitted by the search also clear the exact 1.00. Exact reporting uses GATES["P5"].
FAST_P5 = 1.06
def p5_threshold(mode="exact"): return FAST_P5 if mode == "fast" else GATES["P5"]
def shortfall6(r, pi=None):
    """pi-weighted expected shortfall of scenario net CAGR below 6% (smooth replacement for DRB40 as a gate)."""
    pi = (np.array(L.PI_ROBUST) + np.array(L.PI_CANON)) / 2 if pi is None else pi
    return float(pi @ np.maximum(0.0, 0.06 - r))

def gate_fails_abs(res_base, res_ls, res_hcc=None, r=None, mode="exact"):
    """res_*: dicts from the engine for the robust-G convention; mode='fast' applies FAST_P5 (search), 'exact' GATES['P5'].
    res_hcc is accepted for signature compatibility and reported elsewhere; it is not gated (v3.1). Returns list of failed gates."""
    f = []; p5 = p5_threshold(mode)
    if res_base["ruin"] > GATES["ruin"]: f.append("ruin")
    if res_base["P5"] < p5 - 1e-9: f.append(f"P5 {res_base['P5']:.2f}<{p5:.2f}")
    if float(r.min()) < GATES["worst"] - 1e-12: f.append(f"worst {r.min()*100:.2f}<{GATES['worst']*100:.2f}")
    ra = L.r3af(r, L.prow(L.EVG11) - L.wer_evg11())
    if ra < GATES["R3AF"]: f.append(f"R3AF {ra:.4f}<{GATES['R3AF']}")
    if res_ls["ruin"] > GATES["LS_ruin"]: f.append("LS ruin")
    return f

# ---------------------------------------------------------------- fast evaluation with regime sigma
def run_fast(r, sigv, pin, conv):
    """search_B.run_B with a per-scenario sigma vector (regime-conditional)."""
    FE = S.FE; s = FE.S[pin]; sig = sigv[s]
    mu = np.log1p(r)[s]
    if conv == "G": mu = mu - sig * sig / 2
    X = np.exp(mu + sig * FE.Z) / (1.0 + S.INFL)
    W = np.ones(FE.npaths); red = np.zeros(FE.npaths, bool); ru = np.zeros(FE.npaths, bool)
    for t in range(1, S.T + 1):
        W = W * X[:, t - 1]
        red = np.where(W < 0.80, True, np.where(W >= 1.0, False, red))
        draw = np.maximum(0.03 * np.where(red, 0.75, 1.0) - (S.SS if t >= 11 else 0.0), 0.0)
        W = W - draw; ru |= W <= 0; W = np.maximum(W, 1e-9)
    lg = np.log(W)
    return dict(PathCE=float(np.exp(lg.mean())), CE3=float(np.mean(W ** -2.0) ** -0.5), CE5=float(np.mean(W ** -4.0) ** -0.25),
                P5=float(np.quantile(W, .05)), P10=float(np.quantile(W, .10)), P25=float(np.quantile(W, .25)), ruin=int(ru.sum()))

def evaluate_fast(w, deck=None):
    """Objectives and gates on the fast evaluator, regime sigma. Returns dict."""
    M = DECKM if deck is None else deck
    r = M.T @ w - float(ERV @ w)
    out = {}
    for pin in ("robust", "canonical"):
        sigv, ratio = sigma_vec(w, pin)
        for conv in ("G", "L"):
            out[(pin, conv)] = run_fast(r, sigv, pin, conv)
    g1 = float(np.mean([np.log(out[c]["PathCE"]) for c in CONVS]))
    g3 = float(np.mean([np.log(out[c]["CE3"]) for c in CONVS]))
    g5 = float(np.mean([np.log(out[c]["CE5"]) for c in CONVS]))
    sigv, ratio = sigma_vec(w, "robust")
    ls = run_fast(r - 0.02, sigv, "robust", "G")
    hcc = run_fast(S.HCC.T @ w - float(ERV @ w), sigv, "robust", "G")
    fails = gate_fails_abs(out[("robust", "G")], ls, hcc, r, mode="fast")   # search-time threshold FAST_P5
    return dict(geo=np.exp(g1), ce3=np.exp(g3), ce5=np.exp(g5), P5=out[("robust", "G")]["P5"], ruin=out[("robust", "G")]["ruin"],
                ls_geo=np.exp(np.mean([np.log(ls["PathCE"])])), ls_P5=ls["P5"], hcc_P5=hcc["P5"], fails=fails, stress_ratio=ratio,
                shortfall6=shortfall6(r), worst=float(r.min()))

# ---------------------------------------------------------------- reporting set (4, 12)
TIE = dict(geo=0.03, ce3_cagr=0.02, sace=0.015)
def cagr(x): return (x ** (1 / 40.0) - 1) * 100
def report_structure(w, d):
    ns, mx = S.risk_shares(w)
    return {"nuclear RS %": round(ns * 100), "defence RS %": round(risk_share(w, BLOCKS["defence"]) * 100),
            "gold RS %": round(risk_share(w, BLOCKS["gold"]) * 100), "cyber RS %": round(risk_share(w, BLOCKS["cyber"]) * 100),
            "largest RS %": round(mx * 100), "grade-C wt %": round(float(CW @ w) * 100), "<60mo wt %": round(float(SHORT @ w) * 100),
            "ballast wt %": round(sum(d.get(t, 0) for t in BALLAST) * 100), "singles": sum(1 for t in SINGLES if d.get(t, 0) > 0 and t not in ETF_LIKE),
            "holdings": len(d), "ER %": round(float(ERV @ w) * 100, 3), "stress σ ratio": round(sigma_vec(w)[1], 2)}

# ===== search_v3.py =====
"""SEARCH v3 — Addendum 31 items 1-2: climb on the reported criterion (gamma=3 CE by default; GEO / CE5 selectable)
under the STANDING constraints (standing.violations) and the ABSOLUTE gates (standing.gate_fails_abs), with
regime-conditional sigma, from multiple seeds. Deck = DECK_FILE (R8j). Objective evaluated on the base deck; the
C-haircut deck enters through the HCC P5 gate.
Run: R7=1 R8=1 R9=1 R7B=1 DECK_FILE=Deck_Recal_R8j_20260909.csv SCEN_FILE=scenarios_25_R8i_20260909.csv GRADES_FILE=out/Row_Grades_R8j_20260909.csv NPATHS=6000 OBJ=CE3 python3 search_v3.py [seeds]
v3.1 (2026-09-10): gates use standing.FAST_P5 (1.06) at search time; NPATHS must be the FAST_P5 calibration count (6000) —
fast P5 has a seed sd of ~0.017, so every endpoint is confirmed engine-exact (report_v3.py) before it is called admissible.
Run from ALL seeds for both OBJ=GEO and OBJ=CE3 (Addendum 34 s3).
"""
import sys, os, json, time, numpy as np, warnings; warnings.filterwarnings('ignore')
import standing as ST
import search_B as S
from core import UNIV, IDX, DECKM, ERV, wvec, CONVS
OBJ = os.environ.get("OBJ", "CE3"); KEY = {"CE3": "CE3", "GEO": "PathCE", "CE5": "CE5"}[OBJ]

def objective(w):
    r = DECKM.T @ w - float(ERV @ w); vals = []
    for pin in ("robust", "canonical"):
        sigv, _ = ST.sigma_vec(w, pin)
        for conv in ("G", "L"): vals.append(np.log(ST.run_fast(r, sigv, pin, conv)[KEY]))
    return float(np.mean(vals))

def admissible(v): return not ST.violations(v)
def gates_ok(v): return not ST.evaluate_fast(v)["fails"]

def climb(seed, tag, max_steps=40, shortlist=30):
    w = wvec(seed); assert abs(w.sum() - 1) < 1e-9
    viol = ST.violations(w); assert not viol, (tag, viol)
    if not gates_ok(w): print(f"[{tag}] seed fails absolute gates: {ST.evaluate_fast(w)['fails']}", flush=True); return w, objective(w)
    cur = objective(w); t0 = time.time()
    pool_all = sorted(set(S.POOL) | {"BRKB"} | set(S.S1))
    sc = {}
    for j in pool_all:
        v = w * 0.97; v[IDX[j]] += 0.03; v /= v.sum(); sc[j] = objective(v)
    pool = sorted(set(sorted(sc, key=lambda j: -sc[j])[:shortlist]) | set(S.held(w)))
    e = ST.evaluate_fast(w)
    print(f"[{tag}] seed {S.show(w)} | {OBJ} {np.exp(cur):.3f} geo {e['geo']:.3f} ce3 {e['ce3']:.3f} P5 {e['P5']:.2f}", flush=True)
    for step in range(max_steps):
        cands = []
        for i in S.held(w):
            wi = w[IDX[i]]
            for j in pool:
                if j == i: continue
                wj = w[IDX[j]]
                sizes = ([0.01, 0.02, 0.03] if wj > 1e-9 else [0.02, 0.03])
                for d in sizes:
                    if d > wi + 1e-9 or (0 < wi - d < 0.02 - 1e-9): continue
                    v = w.copy(); v[IDX[i]] -= d; v[IDX[j]] += d; v[np.abs(v) < 1e-12] = 0.0
                    if not admissible(v): continue
                    o = objective(v)
                    if o > cur + 1e-6: cands.append((o, v, i, j, d))
        if not cands: break
        cands.sort(key=lambda x: -x[0]); accepted = None
        for o, v, i, j, d in cands[:20]:                    # best improving move that also clears the absolute gates
            if gates_ok(v): accepted = (o, v, i, j, d); break
        if accepted is None: print(f"[{tag}] no improving move clears the gates (top-20 tried); stopping", flush=True); break
        o, v, i, j, d = accepted
        w, cur = v, o; e = ST.evaluate_fast(w)
        print(f"[{tag}] {step+1}: {i}->{j} {d*100:.0f}% | {OBJ} {np.exp(cur):.3f} geo {e['geo']:.3f} ce3 {e['ce3']:.3f} ce5 {e['ce5']:.3f} P5 {e['P5']:.2f} | {S.show(w)} | {time.time()-t0:.0f}s", flush=True)
    return w, cur

SEEDS = {
 "V_F2":   {'GLDM':.15,'SHLD':.15,'CIBR':.13,'NUKZ':.10,'DBMF':.09,'URNM':.08,'XLE':.08,'BRKB':.05,'CME':.05,'FNV':.05,'MSCI':.05,'XAR':.02},
 "V_F3r":  {'CIBR':.15,'GLDM':.15,'SHLD':.15,'NUKZ':.08,'URNM':.06,'XLE':.05,'BRKB':.05,'CME':.05,'DBMF':.09,'FNV':.05,'MSCI':.05,'XAR':.04,'SETM':.03},
 "V_X3c":  {'GLDM':.15,'SHLD':.15,'CIBR':.15,'DBMF':.11,'NUKZ':.10,'URNM':.05,'XLE':.05,'BRKB':.05,'CME':.05,'FNV':.05,'MSCI':.05,'XAR':.04},
 "V_X2":   {'GLDM':.15,'SHLD':.15,'CIBR':.13,'NUKZ':.10,'DBMF':.15,'URNM':.05,'XLE':.05,'BRKB':.05,'CME':.05,'FNV':.05,'MSCI':.05,'XAR':.02},
 "V_C5":   {'CIBR':.15,'DBMF':.15,'GLDM':.15,'SHLD':.15,'NUKZ':.09,'BRKB':.05,'CME':.05,'FNV':.05,'MSCI':.05,'XLE':.05,'URNM':.04,'HACK':.02},
 "V_F11":  {'CIBR':.15,'DBMF':.15,'GLDM':.15,'SHLD':.14,'NUKZ':.07,'BRKB':.05,'CME':.05,'IVV':.05,'MSCI':.05,'XLE':.05,'FNV':.04,'URNM':.03,'PDBC':.02},
 "V_G1":   {'DBMF':.15,'GLDM':.15,'CIBR':.14,'SHLD':.13,'KMLM':.08,'NUKZ':.06,'BRKB':.05,'CME':.05,'FNV':.05,'MSCI':.05,'JEPI':.03,'CTA':.02,'HACK':.02,'XLU':.02},
 "V_bal":  {'GLDM':.15,'DBMF':.15,'SHLD':.14,'CIBR':.13,'NUKZ':.08,'CTA':.05,'BRKB':.05,'CME':.05,'FNV':.05,'MSCI':.05,'XLE':.04,'URNM':.03,'XAR':.03},
}
if __name__ == "__main__":
    which = sys.argv[1:] or list(SEEDS); out = {}
    for tag in which:
        w, cur = climb(SEEDS[tag], tag)
        out[tag] = {t: round(float(w[IDX[t]]), 2) for t in S.held(w)}
        e = ST.evaluate_fast(w)
        print(f"[{tag}] FINAL {OBJ} {np.exp(cur):.3f} | geo {e['geo']:.3f} ce3 {e['ce3']:.3f} ce5 {e['ce5']:.3f} P5 {e['P5']:.2f} | {S.show(w)}", flush=True)
        json.dump(out, open(os.environ.get("OUTJSON", f"out/searchV3_{OBJ}.json"), "w"), indent=1)

# ===== report_v3.py =====
"""REPORT v3 — Addendum 31 items 4, 12: the standing reporting set, engine-exact (60k x 4 conventions), regime sigma.
Per book: the dial (gamma 1/3/5 CE-CAGR), P5 terminal and path CAGR, -100bp headline and -200bp stress, C-haircut P5,
symmetric-deck sensitivity, smooth shortfall, DRB40 (reported), absolute gates, standing constraints, structure, ER, tax.
Then the non-dominated set on (gamma=1, gamma=3).
Run: DECK_FILE=Deck_Recal_R8j_20260909.csv SCEN_FILE=scenarios_25_R8i_20260909.csv GRADES_FILE=out/Row_Grades_R8j_20260909.csv REGCOV=1 python3 report_v3.py books.json out.csv
"""
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, policy_engine2 as E, standing as ST, search_B as S
from core import DECKM, ERV, wvec, UNIV, IDX
CONV = [("robust","G"),("robust","L"),("canonical","G"),("canonical","L")]
SYM = pd.read_csv("Deck_Recal_R8j_sym_20260909.csv"); SYM["Ticker"] = SYM["Ticker"].str.strip(); SYM = SYM.set_index("Ticker")
SYMM = np.array([[SYM.loc[t, s] for s in L.SC] for t in UNIV]) / 100.0
def ce_cagr(W, gam):
    ce = float(np.exp(np.log(W).mean())) if gam == 1 else float(np.mean(W ** (1 - gam)) ** (1 / (1 - gam)))
    return ce, (ce ** (1 / 40.) - 1) * 100
def pooled(mc, key="W"): return np.concatenate([np.maximum(mc[c][key], 1e-9) for c in CONV])
def profile(nm, d):
    w = wvec(d); r = DECKM.T @ w - float(ERV @ w)
    sig = {p: ST.sigma_vec(w, p)[0] for p in ("robust", "canonical")}
    mc = E.suite(r, sig, E.POL); W = pooled(mc); rg = mc[("robust", "G")]
    m1 = E.suite(r - 0.01, sig, E.POL); m2 = E.suite(r - 0.02, sig, E.POL)
    mh = E.suite(S.HCC.T @ w - float(ERV @ w), sig, E.POL)
    ms = E.suite(SYMM.T @ w - float(ERV @ w), sig, E.POL)
    g1x, g1 = ce_cagr(W, 1); g3x, g3 = ce_cagr(W, 3); g5x, g5 = ce_cagr(W, 5)
    pc = W ** (1 / 40.) - 1
    fails = ST.gate_fails_abs(rg, m2[("robust", "G")], mh[("robust", "G")], r)
    st = ST.report_structure(w, d)
    row = {"Port": nm,
        "geo x": round(g1x, 3), "real CAGR %": round(g1, 2), "nominal x": round(g1x * 1.025 ** 40, 2),
        "g3 CE-CAGR %": round(g3, 2), "g5 CE-CAGR %": round(g5, 2),
        "P5 terminal x": round(rg["P5"], 2), "P5 path CAGR %": round(float(np.quantile(pc, .05)) * 100, 2),
        "P25 path CAGR %": round(float(np.quantile(pc, .25)) * 100, 2),
        "-100bp geo x": round(E.geo(m1), 3), "-100bp CAGR %": round(ce_cagr(pooled(m1), 1)[1], 2),
        "-200bp geo / P5": f"{E.geo(m2):.2f} / {m2[('robust','G')]['P5']:.2f}",
        "HCC geo / P5": f"{E.geo(mh):.2f} / {mh[('robust','G')]['P5']:.2f}",
        "sym-deck geo x": round(E.geo(ms), 3), "sym-deck CAGR %": round(ce_cagr(pooled(ms), 1)[1], 2),
        "worst scen %": round(float(r.min()) * 100, 2), "shortfall6 (bp)": round(ST.shortfall6(r) * 1e4, 1),
        "DRB40@6 (reported)": round(L.drb40(r, h=0.06), 4), "RCEW40 (reported)": round(L.rcew40(r)["robust"], 2),
        "ruin": int(rg["ruin"]), "gates": "PASS" if not fails else "; ".join(fails),
        "constraints": "ok" if not ST.violations(w) else "; ".join(ST.violations(w)),
        "lifetime tax x": round(rg["tax_paid"], 3), "vol48 %": round(S.sigma_of(w) * 100, 2)}
    row.update(st); row["weights"] = " ".join(f"{t}{int(round(x*100))}" for t, x in sorted(d.items(), key=lambda kv: -kv[1]))
    return row
if __name__ == "__main__":
    BK = json.load(open(sys.argv[1])); rows = []
    for nm, d in BK.items():
        d = {t: v for t, v in d.items() if v > 0}; rows.append(profile(nm, d)); print(nm, rows[-1]["geo x"], rows[-1]["g3 CE-CAGR %"], rows[-1]["gates"], flush=True)
    df = pd.DataFrame(rows).set_index("Port"); df.to_csv(sys.argv[2])
    # non-dominated on (gamma=1, gamma=3) among gate-passing, constraint-satisfying books
    ok = df[(df.gates == "PASS") & (df.constraints == "ok")]
    nd = [p for p in ok.index if not any((ok.loc[q, "geo x"] >= ok.loc[p, "geo x"]) and (ok.loc[q, "g3 CE-CAGR %"] >= ok.loc[p, "g3 CE-CAGR %"])
                                        and ((ok.loc[q, "geo x"] > ok.loc[p, "geo x"]) or (ok.loc[q, "g3 CE-CAGR %"] > ok.loc[p, "g3 CE-CAGR %"])) for q in ok.index if q != p)]
    print("\nnon-dominated on (geo, g3):", sorted(nd, key=lambda p: -ok.loc[p, "geo x"]))
    pd.set_option("display.width", 300); print(df.drop(columns=["weights"]).T.to_string())

# ===== pool_v31.py =====
"""v3.1 gate set on the 70-book pool, fast evaluator (search-time threshold FAST_P5) — admissibility count and who changes status vs v3."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import standing as ST
from core import wvec, feasible
pool = json.load(open("out/gate_pool.json")); old = pd.read_csv("out/Gate_Pool_Stats_20260910.csv").set_index("Port")
rows = []
for nm, d in pool.items():
    w = wvec(d)
    if not feasible(w): continue
    e = ST.evaluate_fast(w)
    v3_f = []  # v3 status recomputed from stored stats (exact-threshold 1.00 on fast P5, worst 5.5, LS_P5, HCC_P5)
    o = old.loc[nm]
    if o.P5 < 1.0: v3_f.append("P5")
    if o.worst < 0.055: v3_f.append("worst")
    if o.R3AF < -0.002: v3_f.append("R3AF")
    if o.LS_P5 < 0.45: v3_f.append("LS_P5")
    if o.HCC_P5 < 0.85: v3_f.append("HCC_P5")
    rows.append(dict(Port=nm, constraints="ok" if not ST.violations(w) else "viol", fastP5=round(e["P5"],3), worst=round(e["worst"]*100,2),
                     v3=";".join(v3_f) or "PASS", v31=";".join(x.split(" ")[0] for x in e["fails"]) or "PASS", geo=round(e["geo"],3), ce3=round(e["ce3"],3)))
df = pd.DataFrame(rows).set_index("Port"); df.to_csv("out/Pool_v31_20260910.csv")
ok = df[df.constraints == "ok"]
print(f"pool {len(df)}; constraint-ok {len(ok)}; v3 gate-pass {int((ok.v3=='PASS').sum())}; v3.1 gate-pass (fast 1.06) {int((ok.v31=='PASS').sum())}")
ch = ok[ok.v3 != ok.v31]; print("\nstatus changes among constraint-ok books:"); print(ch.to_string())
print("\nfast P5 in [1.00, 1.06) among constraint-ok (admitted by v3 search threshold, excluded by 1.06):")
print(ok[(ok.fastP5 >= 1.0) & (ok.fastP5 < 1.06)][["fastP5","worst","v3","v31","geo","ce3"]].to_string())

# ===== p5noise.py =====
"""Fast-evaluator P5 noise vs path count: the search-time gate must be read at the calibration path count."""
import json, numpy as np, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, standing as ST, search_B as S
from core import DECKM, ERV, wvec
B = {**json.load(open("out/v31_books.json")), **json.load(open("out/v31_new.json"))}
for nm in ["V3_wealth","F2_bal2","S_X2end","V3_geoX3c","V31_balG"]:
    w = wvec(B[nm]); r = DECKM.T @ w - float(ERV @ w); sigv, _ = ST.sigma_vec(w, "robust")
    line = f"{nm:11s}"
    for n in (4000, 6000, 14000):
        vals = []
        for sd in (777, 1, 2, 3, 4):
            S.FE = L.FastEval(npaths=n, seed=sd); vals.append(ST.run_fast(r, sigv, "robust", "G")["P5"])
        line += f" | n={n:5d}: seed777 {vals[0]:.3f} mean {np.mean(vals):.3f} sd {np.std(vals, ddof=1):.3f}"
    print(line, flush=True)
