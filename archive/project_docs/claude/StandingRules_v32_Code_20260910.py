# ===== standing.py (v3.2, 2026-09-10) =====
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
GATES = dict(ruin=0, P5=1.00, worst=0.0525, R3AF=0.05, LS_ruin=0)
# R3AF threshold 0.05 belongs to the RE-BASED statistic (v3.2, below): it is what a book scores that beats the equity market by
# 2 pp in severity-5 states with a beta of 0.8 to it there (0.55*0.02 + 0.20*0.2). The legacy -0.002 belonged to the EVG11
# formulation, whose scale was set by a beta on six scenarios (ortho = 1-|beta| turns negative past beta 1).
# v3.2 (Addendum 34 s4, accepted 2026-09-10): a book whose exact P5 lies within P5_BAND of the line is decided by the mean of
# P5_SEEDS engine-exact runs (robust-G), not by the single seed-42 run. Exact 6-seed sd is 0.009; the band is ~2 sd.
P5_BAND, P5_SEEDS = 0.02, [42, 7, 101, 2024, 555, 31337]
# v3.2 (Addendum 34 s5.1, accepted 2026-09-10): R3AF re-based off the legacy EVG11 book. Partition is the DECK's severity
# (bad = severity 5, 23% of canonical mass; good = severity 2, 25%; mid = 3-4) and the reference for tail alpha and
# bad-state beta is the equity market (IVV net of its ER). Same 0.55/0.25/0.20 formula and canonical weights as lib_pathce.r3af.
R3AF_BENCH = "IVV"; R3AF_BAD, R3AF_GOOD = L.SEV >= 5, L.SEV <= 2
def r3af_std(r_net):
    b = DECKM[IDX[R3AF_BENCH]] - float(ERV[IDX[R3AF_BENCH]]); r = np.asarray(r_net); w = L.W_CAN
    bad, good = R3AF_BAD, R3AF_GOOD; mid = ~(bad | good)
    wm = lambda x, m: float(np.sum(w[m] * x[m]) / np.sum(w[m]))
    tail_alpha = wm(r, bad) - wm(b, bad)
    convexity = (wm(r, good) - wm(r, mid)) - (wm(r, mid) - wm(r, bad))
    bw, rw = wm(b, bad), wm(r, bad); vb = np.sum(w[bad] * (b[bad] - bw) ** 2)
    beta = np.sum(w[bad] * (b[bad] - bw) * (r[bad] - rw)) / vb if vb > 1e-18 else np.nan
    ortho = 1.0 - min(2.0, abs(beta)) if np.isfinite(beta) else np.nan
    return 0.55 * tail_alpha + 0.25 * convexity + 0.20 * ortho
def r3af_legacy(r_net): return L.r3af(r_net, L.prow(L.EVG11) - L.wer_evg11())
# The fast evaluator (search basis, no taxes) overstates P5 by +0.048 (sd 0.012) against the exact engine: the search-time
# threshold is 1.06 so that books admitted by the search also clear the exact 1.00. Exact reporting uses GATES["P5"].
FAST_P5 = 1.06
def p5_threshold(mode="exact"): return FAST_P5 if mode == "fast" else GATES["P5"]
def shortfall6(r, pi=None):
    """pi-weighted expected shortfall of scenario net CAGR below 6% (smooth replacement for DRB40 as a gate)."""
    pi = (np.array(L.PI_ROBUST) + np.array(L.PI_CANON)) / 2 if pi is None else pi
    return float(pi @ np.maximum(0.0, 0.06 - r))

def gate_fails_abs(res_base, res_ls, res_hcc=None, r=None, mode="exact", p5_mean=None):
    """res_*: dicts from the engine for the robust-G convention; mode='fast' applies FAST_P5 (search), 'exact' GATES['P5'].
    p5_mean: the P5_SEEDS-mean exact P5, supplied by report_v3 when the single run is within P5_BAND of the line (v3.2);
    it then decides the gate. res_hcc is accepted for signature compatibility; it is not gated (v3.1). Returns failed gates."""
    f = []; p5 = p5_threshold(mode)
    if res_base["ruin"] > GATES["ruin"]: f.append("ruin")
    p5v = res_base["P5"] if (mode == "fast" or p5_mean is None) else p5_mean
    if p5v < p5 - 1e-9: f.append(f"P5 {p5v:.3f}<{p5:.2f}" + (" (6-seed mean)" if p5_mean is not None and mode != "fast" else ""))
    if float(r.min()) < GATES["worst"] - 1e-12: f.append(f"worst {r.min()*100:.2f}<{GATES['worst']*100:.2f}")
    ra = r3af_std(r)
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
    # v3.2: a P5 within P5_BAND of the line is decided by the P5_SEEDS-mean of exact robust-G runs, not one seed
    p5_mean = None
    if abs(rg["P5"] - ST.GATES["P5"]) < ST.P5_BAND:
        p5_mean = float(np.mean([E.run(np.random.default_rng(s), r, L.PI_ROBUST, "G", sig["robust"], E.POL)["P5"] for s in ST.P5_SEEDS]))
    fails = ST.gate_fails_abs(rg, m2[("robust", "G")], mh[("robust", "G")], r, p5_mean=p5_mean)
    st = ST.report_structure(w, d)
    row = {"Port": nm,
        "geo x": round(g1x, 3), "real CAGR %": round(g1, 2), "nominal x": round(g1x * 1.025 ** 40, 2),
        "g3 CE-CAGR %": round(g3, 2), "g5 CE-CAGR %": round(g5, 2),
        "P5 terminal x": round(rg["P5"], 3), "P5 6-seed mean": (round(p5_mean, 3) if p5_mean is not None else ""),
        "R3AF (sev5 vs IVV)": round(ST.r3af_std(r), 4), "R3AF legacy (EVG11)": round(ST.r3af_legacy(r), 4),
        "P5 path CAGR %": round(float(np.quantile(pc, .05)) * 100, 2),
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

# ===== build_r8k.py =====
"""R8k — the AI Buildout Disappointment de-rating made MULTIPLICATIVE (Addendum 31 caveat; Addendum 34 s5.7, accepted 2026-09-10).

R8i/R8j applied the overlay ADDITIVELY: power -3.0 pp, semis -2.5, speculative tech -1.5, energy -1.0, ballast +0.5, so a
high-row ticker stayed high (URNM 8.60 in the scenario meant to hurt it, above IVV's 7.30). The scenario's thesis is that
the thematic PREMIUM evaporates, so R8k de-rates the premium: for a ticker in negative set S,
    hit = max( additive_S ,  premium )     with premium = max(blend - IVV_blend, 0),
    blend = 0.50 Credit + 0.35 Base + 0.15 Stagflation on the R8j rows.
Additive floor, premium cap: nothing gets softer than R8j; a name whose premium exceeds the additive hit loses the whole
premium and lands at the market. (Addendum 31's "e.g. x0.7 on the excess" would have changed nothing material: URNM's
premium is 4.30, 0.7 x 4.30 = 3.0 = the additive hit; a multiplier is shown for reference only.)
Ballast +0.5 pp stays additive (a benefit, not a premium). Scenario weight and severity unchanged; only this column changes.
"""
import numpy as np, pandas as pd
DECK_IN, DECK_OUT = "Deck_Recal_R8j_20260909.csv", "Deck_Recal_R8k_20260910.csv"
NAME = "AI Buildout Disappointment – Power/Capex De-rating"
d = pd.read_csv(DECK_IN); d["Ticker"] = d["Ticker"].str.strip(); d = d.set_index("Ticker")
C = lambda k: d[[c for c in d.columns if c.startswith(k)][0]]
POWER = {'NUKZ','URNM','CEG','NLR','GRID','SETM','URA','AQWA'}; SEMI = {'SMH','SOXX','XSD','XLK','FTEC'}
SPEC = {'QTUM','BOTZ','ROBO','KOMP','XT','PRNT','ROKT','JEDI'}; ENER = {'XLE','GUNR','PDBC'}
BALL = {'GLDM','DBMF','CTA','KMLM','PSLV','TLT','IEF'}
blend = 0.50 * C("Credit Stress") + 0.35 * C("Base Case") + 0.15 * C("Stagflation")
ivv = float(blend["IVV"]); prem = (blend - ivv).clip(lower=0)
old = d[NAME].copy(); new = blend.copy()
print(f"IVV blend {ivv:.2f}")
for tag, S, a in (("power", POWER, 3.0), ("semis", SEMI, 2.5), ("spec", SPEC, 1.5), ("energy", ENER, 1.0)):
    ts = [t for t in d.index if t in S]
    hit = np.maximum(a, prem[ts]); new[ts] = blend[ts] - hit
    print(f"{tag:6s} additive {a:.1f}: " + ", ".join(f"{t} prem {prem[t]:.2f} hit {hit[t]:.2f}" for t in ts))
bt = [t for t in d.index if t in BALL]; new[bt] = blend[bt] + 0.5
d[NAME] = np.round(new, 4); d.to_csv(DECK_OUT)
chg = d[NAME] - old; aff = chg[chg.abs() > 1e-9]
print(f"\n{DECK_OUT}: column changed for {len(aff)} tickers: " + ", ".join(f"{t} {v:+.2f}" for t, v in aff.sort_values().items()))
print("AIBUST R8j -> R8k, held names:")
for t in ['URNM','URA','CEG','NUKZ','SETM','NLR','GRID','SMH','SOXX','XLE','JEDI','QTUM','GLDM','DBMF','IVV']:
    if t in d.index: print(f"  {t:5s} {old[t]:6.2f} -> {d.loc[t, NAME]:6.2f}   (blend {blend[t]:.2f}; row min now {d.loc[t].min():.2f})")

# ===== constraint_assess.py =====
"""Constraint retention assessment (Addendum 34 s5.2, accepted): the gate_assess pattern applied to the eight standing
constraint values — fails, UNIQUE fails, margins, correlation of the constraint statistics, and knife-edge sensitivity
to 1% trades on the frontier books. Constraints are closed-form (historical covariance, weights, grades, history), so
this is deck-independent apart from feasibility."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import standing as ST, search_B as S
from core import wvec, IDX, UNIV, feasible, SINGLES
pool = json.load(open("out/gate_pool.json")); pool.update({k: v for k, v in json.load(open("out/v31_books_all.json")).items() if k not in pool})
TH = dict(nuclear=ST.RS_NUC, name=ST.RS_NAME, defence=ST.RS_BLOCK, gold=ST.RS_BLOCK, cyber=ST.RS_BLOCK, semis=ST.RS_BLOCK,
          gradeC=ST.CW_CAP, short=ST.SHORT_CAP, singles=ST.SINGLES_MAX, ballast=ST.BALLAST_FLOOR)
def stats(w, d):
    ns, mx = S.risk_shares(w)
    return dict(nuclear=ns, name=mx, **{b: ST.risk_share(w, ST.BLOCKS[b]) for b in ("defence", "gold", "cyber", "semis")},
                gradeC=float(ST.CW @ w), short=float(ST.SHORT @ w), singles=sum(1 for t in SINGLES if d.get(t, 0) > 0 and t not in ST.ETF_LIKE),
                ballast=sum(d.get(t, 0) for t in ST.BALLAST), feasible=feasible(w))
rows = {}
for nm, d in pool.items():
    d = {t: v for t, v in d.items() if v > 0}; w = wvec(d); rows[nm] = stats(w, d)
df = pd.DataFrame(rows).T; df.to_csv("out/Constraint_Pool_Stats_20260910.csv")
# margin: positive = inside the constraint
M = pd.DataFrame({k: (TH[k] - df[k]) if k != "ballast" else (df[k] - TH[k]) for k in TH}); M.index = df.index
F = M < -1e-9; F["feasibility"] = ~df.feasible.astype(bool)
n = len(df); print(f"pool: {n} books ({int(F.any(axis=1).sum())} violate at least one constraint)\n")
print(f"{'constraint':12s} {'thr':>5s} {'fails':>5s} {'unique':>6s}  margin p10 / median / p90 (pp or count; + = inside)   fails within 2pp")
for k in list(TH) + ["feasibility"]:
    f = F[k]; others = F[[x for x in F.columns if x != k]].any(axis=1); uniq = int((f & ~others).sum())
    if k in M:
        m = M[k] * (1 if k == "singles" else 100); near = int((f & (M[k] > -0.02)).sum()) if k != "singles" else ""
        print(f"{k:12s} {TH[k]*(1 if k=='singles' else 100):5.0f} {int(f.sum()):5d} {uniq:6d}  {m.quantile(.1):+6.1f} / {m.median():+6.1f} / {m.quantile(.9):+6.1f}   {near}")
    else: print(f"{k:12s} {'':5s} {int(f.sum()):5d} {uniq:6d}")
print("\ncorrelation of constraint statistics across the pool:")
print(df[list(TH)].astype(float).corr().round(2).to_string())
print("\nbooks failing exactly one constraint, by margin (pp):")
one = F[F.sum(axis=1) == 1]
for nm in one.index:
    k = one.columns[one.loc[nm].values.argmax()]
    mv = M.loc[nm, k] * (1 if k == "singles" else 100) if k in M else float("nan")
    print(f"  {nm:14s} {k:10s} {mv:+.1f}")
# knife-edge on the frontier books: share of admissible-by-feasibility 1% trades that break each constraint
print("\nknife-edge: share of feasible 1%-trade neighbours of each frontier book that break a constraint the book satisfies")
FR = ["V3_geoX3c", "V31_C5g", "X2_F2_DBMF15", "C5_bal", "V3_reliab"]
pool_t = sorted(set(S.POOL) | {"BRKB"} | set(S.S1))
for nm in FR:
    d = {t: v for t, v in pool[nm].items() if v > 0}; w0 = wvec(d); base = set(x.split(" ")[0] for x in ST.violations(w0)); flips = {}; cnt = 0
    for i in S.held(w0):
        for j in pool_t:
            if j == i: continue
            v = w0.copy(); v[IDX[i]] -= 0.01; v[IDX[j]] += 0.01; v[np.abs(v) < 1e-12] = 0
            if v[IDX[i]] > 0 and v[IDX[i]] < 0.02 - 1e-9: continue
            if v[IDX[j]] < 0.02 - 1e-9 or not feasible(v): continue
            cnt += 1
            for g in set(x.split(" ")[0] for x in ST.violations(v)) - base: flips[g] = flips.get(g, 0) + 1
    st = rows[nm]
    print(f"  {nm:13s} {cnt:4d} neighbours: " + (", ".join(f"{g} {c/cnt*100:.0f}%" for g, c in sorted(flips.items(), key=lambda kv: -kv[1])) or "none")
          + f"   | margins: nuclear {(TH['nuclear']-st['nuclear'])*100:+.1f} name {(TH['name']-st['name'])*100:+.1f} short {(TH['short']-st['short'])*100:+.1f} ballast {(st['ballast']-TH['ballast'])*100:+.1f} gradeC {(TH['gradeC']-st['gradeC'])*100:+.1f}")

# ===== rs_boot.py =====
"""Estimation noise of the risk-share constraints: month-resampling bootstrap of the 48-month covariance."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import search_B as S, standing as ST
from core import wvec, IDX, UNIV
pool = json.load(open("out/gate_pool.json")); pool.update(json.load(open("out/v31_books_all.json")))
W48 = S._W48[[t for t in UNIV if t in S._W48.columns]]; cols = list(W48.columns); ci = [IDX[t] for t in cols]
rng = np.random.default_rng(7); B = 300
def rs_from(C, w, names):
    Cv = C @ w; var = float(w @ Cv); mrc = Cv * w / var; return float(sum(mrc[IDX[t]] for t in names if t in IDX))
books = ["R7a", "F2_bal2", "S_X2end", "V3_geoX3c", "X2_F2_DBMF15", "V3_reliab", "X2p3_both"]
res = {b: {"nuc": [], "name": []} for b in books}
for _ in range(B):
    idx = rng.integers(0, len(W48), len(W48)); C = np.zeros((len(UNIV), len(UNIV)))
    c = W48.iloc[idx].cov(min_periods=18).values * 12.0; c = np.nan_to_num(c)
    for a, ia in enumerate(ci):
        C[ia, ci] = c[a]
    for b in books:
        w = wvec({t: v for t, v in pool[b].items() if v > 0})
        res[b]["nuc"].append(rs_from(C, w, S.NUC_SET))
        Cv = C @ w; var = float(w @ Cv); res[b]["name"].append(float((Cv * w / var).max()))
print(f"{'book':13s} {'nuclear RS':>10s} {'boot sd':>7s} {'name RS':>8s} {'boot sd':>7s}   (point estimates from the standing 48-month covariance; sd from 300 month-resamples)")
for b in books:
    w = wvec({t: v for t, v in pool[b].items() if v > 0}); ns, mx = S.risk_shares(w)
    print(f"{b:13s} {ns*100:10.1f} {np.std(res[b]['nuc'])*100:7.1f} {mx*100:8.1f} {np.std(res[b]['name'])*100:7.1f}")

# ===== rs_long.py =====
"""Nuclear risk share on a longer, anchored covariance panel (120 months, short-history names via sector anchors) vs the standing 48-month estimate."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import search_B as S, standing as ST
from core import wvec, IDX, UNIV
pool = json.load(open("out/gate_pool.json")); pool.update(json.load(open("out/v31_books_all.json")))
m120 = ST._mr.loc[ST._mr.index[-120]:]
def rs(C, w, names):
    Cv = C @ w; var = float(w @ Cv); mrc = Cv * w / var; return float(sum(mrc[IDX[t]] for t in names if t in IDX)), float(mrc.max())
C120 = ST._cov(m120); rng = np.random.default_rng(11)
books = ["R7a", "F2_bal2", "S_X2end", "V3_geoX3c", "X2_F2_DBMF15", "V31_C5g", "C5_bal", "V3_reliab", "X2p3_both"]
boot = {b: [] for b in books}
for _ in range(200):
    idx = rng.integers(0, len(m120), len(m120)); C = ST._cov(m120.iloc[idx])
    for b in books: boot[b].append(rs(C, wvec({t: v for t, v in pool[b].items() if v > 0}), S.NUC_SET)[0])
print(f"{'book':13s} {'48mo nuc RS':>11s} {'120mo anchored':>14s} {'boot sd':>7s} {'48mo name RS':>12s} {'120mo':>6s}")
for b in books:
    w = wvec({t: v for t, v in pool[b].items() if v > 0}); ns, mx = S.risk_shares(w); n2, m2 = rs(C120, w, S.NUC_SET)
    print(f"{b:13s} {ns*100:11.1f} {n2*100:14.1f} {np.std(boot[b])*100:7.1f} {mx*100:12.1f} {m2*100:6.1f}")

# ===== rs_blend.py =====
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import search_B as S, standing as ST
from core import wvec, IDX, UNIV, feasible
pool = json.load(open("out/gate_pool.json")); pool.update({k: v for k, v in json.load(open("out/v31_books_all.json")).items() if k not in pool})
C120 = ST._cov(ST._mr.loc[ST._mr.index[-120]:])
def rs(C, w, names):
    Cv = C @ w; var = float(w @ Cv); mrc = Cv * w / var; return float(sum(mrc[IDX[t]] for t in names if t in IDX)), float(mrc.max())
rows = []
for nm, d in pool.items():
    d = {t: v for t, v in d.items() if v > 0}; w = wvec(d)
    if not feasible(w): continue
    n48, m48 = S.risk_shares(w); n120, m120 = rs(C120, w, S.NUC_SET)
    other = [x for x in ST.violations(w) if "risk share" not in x]
    rows.append(dict(Port=nm, n48=n48*100, n120=n120*100, navg=(n48+n120)*50, m48=m48*100, m120=m120*100, mavg=(m48+m120)*50, other_ok=not other))
df = pd.DataFrame(rows).set_index("Port")
for lab, nc, mc in (("48-month (standing)", "n48", "m48"), ("120-month anchored", "n120", "m120"), ("average", "navg", "mavg")):
    ok = (df[nc] <= 38 + 1e-9) & (df[mc] <= 24 + 1e-9)
    print(f"{lab:20s}: pass nuclear&name caps {int(ok.sum()):3d}/{len(df)}; fully admissible {int((ok & df.other_ok).sum()):3d}; nuclear fails {int((df[nc] > 38).sum()):3d}, name fails {int((df[mc] > 24).sum()):3d}")
chg = df[((df.n48 > 38) != (df.navg > 38)) | ((df.m48 > 24) != (df.mavg > 24))]
print("\nbooks whose risk-share status changes between the 48-month and the averaged estimate:")
print(chg.round(1).to_string())

# ===== readmit_check.py =====
"""Books the averaged risk-share estimator would re-admit: fast gates (FAST_P5) on R8k and fast-space dominance vs the frontier."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import standing as ST, search_B as S
from core import wvec, IDX, feasible
pool = json.load(open("out/gate_pool.json")); pool.update({k: v for k, v in json.load(open("out/v31_books_all.json")).items() if k not in pool})
C120 = ST._cov(ST._mr.loc[ST._mr.index[-120]:])
def rs(C, w, names):
    Cv = C @ w; var = float(w @ Cv); mrc = Cv * w / var; return float(sum(mrc[IDX[t]] for t in names if t in IDX)), float(mrc.max())
FR = ["V3_geoX3c", "V31_balG", "V31_C5g", "F9_semi", "C5_bal", "V3_reliab"]; fr = {}
for nm in FR:
    e = ST.evaluate_fast(wvec({t: v for t, v in pool[nm].items() if v > 0})); fr[nm] = (e["geo"], e["ce3"])
rows = []
for nm, d in pool.items():
    d = {t: v for t, v in d.items() if v > 0}; w = wvec(d)
    if not feasible(w): continue
    n48, m48 = S.risk_shares(w); n120, m120 = rs(C120, w, S.NUC_SET); na, ma = (n48 + n120) / 2, (m48 + m120) / 2
    other = [x for x in ST.violations(w) if "risk share" not in x]
    now_ok = (n48 <= .38) and (m48 <= .24) and not other; avg_ok = (na <= .38) and (ma <= .24) and not other
    if avg_ok and not now_ok:
        e = ST.evaluate_fast(w); dom = [f for f, (g, c) in fr.items() if g >= e["geo"] and c >= e["ce3"]]
        rows.append(dict(Port=nm, nuc48=round(n48*100,1), nucavg=round(na*100,1), fast_geo=round(e["geo"],3), fast_ce3=round(e["ce3"],3), fastP5=round(e["P5"],3), worst=round(e["worst"]*100,2), gates=";".join(x.split(" ")[0] for x in e["fails"]) or "PASS", dominated_by=",".join(dom) or "NONE"))
df = pd.DataFrame(rows).set_index("Port"); print(df.to_string()); print(f"\n{len(df)} re-admitted; gate-pass {int((df.gates=='PASS').sum())}; dominated {int((df.dominated_by!='NONE').sum())}")
print("frontier fast (geo, ce3):", {k: (round(a,3), round(b,3)) for k, (a, b) in fr.items()})

# ===== r3af_check.py =====
"""R3AF re-based (severity-5 partition, IVV reference) vs legacy (EVG11 quintiles): distribution over the pool, who fails at -0.002, knife-edge."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import standing as ST, search_B as S, lib_pathce as L
from core import wvec, IDX, DECKM, ERV, feasible
pool = json.load(open("out/gate_pool.json")); pool.update({k: v for k, v in json.load(open("out/v31_books_all.json")).items() if k not in pool})
rows = []
for nm, d in pool.items():
    d = {t: v for t, v in d.items() if v > 0}; w = wvec(d)
    if not feasible(w): continue
    r = DECKM.T @ w - float(ERV @ w)
    rows.append(dict(Port=nm, new=ST.r3af_std(r), legacy=ST.r3af_legacy(r), worst=r.min()*100, ok=not ST.violations(w)))
df = pd.DataFrame(rows).set_index("Port"); df.to_csv("out/R3AF_Rebased_Pool_20260910.csv")
print(f"pool {len(df)}: corr(new, legacy) {df.new.corr(df.legacy):.2f}")
print(f"new    : min {df.new.min():+.4f} p10 {df.new.quantile(.1):+.4f} median {df.new.median():+.4f} p90 {df.new.quantile(.9):+.4f}; fails at -0.002: {int((df.new < -0.002).sum())}")
print(f"legacy : min {df.legacy.min():+.4f} p10 {df.legacy.quantile(.1):+.4f} median {df.legacy.median():+.4f} p90 {df.legacy.quantile(.9):+.4f}; fails at -0.002: {int((df.legacy < -0.002).sum())}")
print("\nlowest 10 on the re-based R3AF:"); print(df.sort_values("new").head(10).round(4).to_string())
print("\nfrontier / named books:"); print(df.loc[[b for b in ["V3_geoX3c","V31_C5g","X2_F2_DBMF15","C5_bal","V3_reliab","F9_semi","V3_wealth","F2_bal2","S_X2end","X3d","R7a"] if b in df.index]].round(4).to_string())
# components for F9_semi and V3_geoX3c
def comps(r):
    b = DECKM[IDX["IVV"]] - float(ERV[IDX["IVV"]]); w = L.W_CAN; bad, good = ST.R3AF_BAD, ST.R3AF_GOOD; mid = ~(bad | good)
    wm = lambda x, m: float(np.sum(w[m] * x[m]) / np.sum(w[m]))
    ta = wm(r, bad) - wm(b, bad); cv = (wm(r, good) - wm(r, mid)) - (wm(r, mid) - wm(r, bad))
    bw, rw = wm(b, bad), wm(r, bad); vb = np.sum(w[bad] * (b[bad] - bw) ** 2); beta = np.sum(w[bad] * (b[bad] - bw) * (r[bad] - rw)) / vb
    return ta, cv, beta, wm(r, bad), wm(b, bad)
print("\ncomponents (tail alpha vs IVV, convexity, beta in sev-5 states, book sev-5 mean, IVV sev-5 mean):")
for nm in ["F9_semi", "V3_geoX3c", "X2_F2_DBMF15", "V3_reliab", "SMH-heavy test"]:
    if nm == "SMH-heavy test": d = {"SMH": .12, "GLDM": .15, "SHLD": .15, "DBMF": .15, "CIBR": .13, "CME": .05, "MSCI": .05, "BRKB": .05, "XLE": .05, "NUKZ": .05, "FNV": .05}
    else: d = {t: v for t, v in pool[nm].items() if v > 0}
    w = wvec(d); r = DECKM.T @ w - float(ERV @ w); ta, cv, beta, rb, bb = comps(r)
    print(f"  {nm:14s} R3AF {ST.r3af_std(r):+.4f} | alpha {ta*100:+.2f} pp, convexity {cv*100:+.2f} pp, beta {beta:+.2f} (ortho {1-min(2,abs(beta)):+.2f}) | sev-5 mean {rb*100:.2f} vs IVV {bb*100:.2f}")
print("\nknife-edge (share of admissible 1%-trade neighbours whose R3AF gate status differs):")
pool_t = sorted(set(S.POOL) | {"BRKB"} | set(S.S1))
for nm in ["V3_geoX3c", "V31_C5g", "X2_F2_DBMF15", "C5_bal", "V3_reliab"]:
    d = {t: v for t, v in pool[nm].items() if v > 0}; w0 = wvec(d); r0 = DECKM.T @ w0 - float(ERV @ w0); s0 = ST.r3af_std(r0) < ST.GATES["R3AF"]; cnt = fl = 0; mn = 9
    for i in S.held(w0):
        for j in pool_t:
            if j == i: continue
            v = w0.copy(); v[IDX[i]] -= 0.01; v[IDX[j]] += 0.01; v[np.abs(v) < 1e-12] = 0
            if v[IDX[i]] > 0 and v[IDX[i]] < 0.02 - 1e-9: continue
            if v[IDX[j]] < 0.02 - 1e-9 or ST.violations(v): continue
            cnt += 1; x = ST.r3af_std(DECKM.T @ v - float(ERV @ v)); mn = min(mn, x); fl += int((x < ST.GATES["R3AF"]) != s0)
    print(f"  {nm:13s} R3AF {ST.r3af_std(r0):+.4f}; {cnt} neighbours, flips {fl} ({fl/cnt*100:.0f}%), lowest neighbour {mn:+.4f}")

# ===== legacy_beta.py =====
import json, numpy as np, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, standing as ST
from core import wvec, DECKM, ERV
pool = json.load(open("out/gate_pool.json")); pool.update(json.load(open("out/v31_books_all.json")))
b = L.prow(L.EVG11) - L.wer_evg11(); w = L.W_CAN
order = np.argsort(b); xs = b[order]; ws = w[order]; cw = np.cumsum(ws); cw = cw / cw[-1]
ql = float(np.interp(0.20, cw, xs)); qh = float(np.interp(0.80, cw, xs)); bad = b <= ql; good = b >= qh
print("EVG11 bottom-quintile states:", [L.SC[i][:28] for i in np.where(bad)[0]], "mass", round(float(w[bad].sum()),3))
for nm in ["F9_semi", "V31_C5g", "V3_geoX3c"]:
    v = wvec({t: x for t, x in pool[nm].items() if x > 0}); r = DECKM.T @ v - float(ERV @ v)
    wm = lambda x, m: float(np.sum(w[m] * x[m]) / np.sum(w[m]))
    bw, rw = wm(b, bad), wm(r, bad); vb = np.sum(w[bad] * (b[bad] - bw) ** 2); beta = np.sum(w[bad] * (b[bad] - bw) * (r[bad] - rw)) / vb
    print(f"{nm:10s} legacy beta in EVG11-bad states {beta:+.2f} -> ortho {1-min(2,abs(beta)):+.2f}; tail alpha {(rw-bw)*100:+.2f} pp; R3AF {ST.r3af_legacy(r):+.4f}")

# ===== noise_v32.py =====
"""Noise floor re-measured under regime sigma on deck R8k (Addendum 34 s5.6): 6 seeds x 2 books, engine-exact (60k x 4 conventions)."""
import json, numpy as np, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, policy_engine2 as E, standing as ST
from core import DECKM, ERV, wvec
CONV = [("robust","G"),("robust","L"),("canonical","G"),("canonical","L")]
def suite_seed(r, sig, seed):
    out = {}
    for pin, pi in (("robust", L.PI_ROBUST), ("canonical", L.PI_CANON)):
        rng = np.random.default_rng(seed)
        for conv in ("G", "L"): out[(pin, conv)] = E.run(rng, r, pi, conv, sig[pin], E.POL)
    return out
def ce(W, g): return float(np.exp(np.log(W).mean())) if g == 1 else float(np.mean(W ** (1 - g)) ** (1 / (1 - g)))
B = json.load(open("out/v31_books_all.json")); SEEDS = [42, 7, 101, 2024, 555, 31337]
for nm in ["X2_F2_DBMF15", "V3_geoX3c"]:
    w = wvec({t: v for t, v in B[nm].items() if v > 0}); r = DECKM.T @ w - float(ERV @ w); sig = {p: ST.sigma_vec(w, p)[0] for p in ("robust", "canonical")}
    G, C3, P = [], [], []
    for s in SEEDS:
        mc = suite_seed(r, sig, s); W = np.concatenate([np.maximum(mc[c]["W"], 1e-9) for c in CONV])
        G.append(ce(W, 1)); C3.append((ce(W, 3) ** (1/40.) - 1) * 100); P.append(mc[("robust", "G")]["P5"])
    G, C3, P = map(np.array, (G, C3, P))
    print(f"{nm:13s} geo mean {G.mean():.4f} sd {G.std(ddof=1):.4f} | g3 CE-CAGR mean {C3.mean():.3f} sd {C3.std(ddof=1):.3f} pp | P5 mean {P.mean():.4f} sd {P.std(ddof=1):.4f}", flush=True)

# ===== fastcal.py =====
"""Re-derive FAST_P5 on the current deck: fast-evaluator P5 (6000 paths, seed 777, regime sigma) minus engine-exact P5 across the standing books."""
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import standing as ST
from core import wvec, DECKM, ERV
rep = pd.read_csv(sys.argv[1]).set_index("Port"); B = json.load(open("out/v31_books_all.json")); rows = []
for nm in rep.index:
    if nm not in B: continue
    w = wvec({t: v for t, v in B[nm].items() if v > 0}); r = DECKM.T @ w - float(ERV @ w); sigv, _ = ST.sigma_vec(w, "robust")
    f = ST.run_fast(r, sigv, "robust", "G"); ex = float(rep.loc[nm, "P5 terminal x"])
    rows.append(dict(Port=nm, fast=f["P5"], exact=ex, off=f["P5"] - ex, geo_ratio=ST.evaluate_fast(w)["geo"] / float(rep.loc[nm, "geo x"])))
df = pd.DataFrame(rows).set_index("Port"); print(df.round(3).to_string())
print(f"\nP5 offset: mean {df.off.mean():+.3f} sd {df.off.std(ddof=1):.3f} -> FAST_P5 = 1.00 + mean + 1 sd = {1 + df.off.mean() + df.off.std(ddof=1):.3f}; geo ratio mean {df.geo_ratio.mean():.3f} sd {df.geo_ratio.std(ddof=1):.3f}")
