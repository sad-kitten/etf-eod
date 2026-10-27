# ===== standing.py (v3.3, 2026-09-10) =====
"""STANDING RULES v3.3 (v3 = Addendum 31 items applied 2026-09-09; v3.1 = Addendum 33 gate set ratified 2026-09-10;
v3.2 = Addendum 34 items applied 2026-09-10; v3.3 = averaged risk-share estimator and 3-sd tie bands, Addendum 35 s3/s4, 2026-09-10).

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

def _mrc(C, w):
    Cv = C @ w; var = float(w @ Cv)
    return Cv * w / var if var > 1e-12 else np.zeros_like(w)
# v3.3 (Addendum 35 s3, accepted 2026-09-10): risk shares are the AVERAGE of the marginal risk contributions from the standing
# 48-month covariance (search_B.COV) and from a 120-month panel in which short-history names are represented by their sector
# anchor (the sigma_vec panel logic). Month-resampling sd falls from 3.4-4.4 pp to ~2.5 pp and one regime no longer sets the
# number. COV120 is defined after the anchored panel machinery below; risk_mrc() is what every constraint reads.
def risk_mrc(w): return 0.5 * (_mrc(S.COV, w) + _mrc(COV120, w))
def risk_share(w, names):
    mrc = risk_mrc(w); return float(sum(mrc[IDX[t]] for t in names if t in IDX))
def risk_shares(w):
    """(nuclear-complex share, largest single-holding share) on the averaged estimator."""
    mrc = risk_mrc(w); return float(sum(mrc[IDX[t]] for t in S.NUC_SET)), float(mrc.max())
def risk_shares_48(w): return S.risk_shares(w)   # the pre-v3.3 statistic, reported for one revision

def violations(w):
    """Every standing constraint the book breaks; empty list = admissible."""
    v = []
    if not feasible(w): v.append("feasibility")
    ns, mx = risk_shares(w)
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
COV120 = _cov(_mr.loc[_mr.index[-120]:])   # v3.3: 120-month anchored panel for the averaged risk-share estimator
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
# v3.3 (Addendum 35 s4, accepted 2026-09-10): 3-sd bands from the regime-sigma noise floor (geo sd 0.006-0.007, g3 sd 0.011-0.014 pp,
# P5 sd ~0.005; re-measured at every deck revision). Inside a band, choose on cost and structure, never on the number.
TIE = dict(geo=0.02, ce3_cagr=0.04, p5=0.015)
def cagr(x): return (x ** (1 / 40.0) - 1) * 100
def report_structure(w, d):
    ns, mx = risk_shares(w); n48, m48 = risk_shares_48(w)
    return {"nuclear RS %": round(ns * 100), "nuclear RS 48mo %": round(n48 * 100), "defence RS %": round(risk_share(w, BLOCKS["defence"]) * 100),
            "gold RS %": round(risk_share(w, BLOCKS["gold"]) * 100), "cyber RS %": round(risk_share(w, BLOCKS["cyber"]) * 100),
            "largest RS %": round(mx * 100), "largest RS 48mo %": round(m48 * 100), "grade-C wt %": round(float(CW @ w) * 100), "<60mo wt %": round(float(SHORT @ w) * 100),
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

# ===== report_v33.py =====
"""Report v3.3 = the v3.2 engine-exact results (unchanged: gates and metrics do not depend on the constraint estimator or the
tie bands) with the constraint and structure columns recomputed under v3.3, then the non-dominated set and the tie-band reading."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import standing as ST
from core import wvec
df = pd.read_csv("out/Report_v32_R8k_20260910.csv").set_index("Port"); B = json.load(open("out/v31_books_all.json"))
for nm in df.index:
    d = {t: v for t, v in B[nm].items() if v > 0}; w = wvec(d); st = ST.report_structure(w, d)
    for k, v in st.items(): df.loc[nm, k] = v
    df.loc[nm, "constraints"] = "ok" if not ST.violations(w) else "; ".join(ST.violations(w))
df.to_csv("out/Report_v33_R8k_20260910.csv")
ok = df[(df.gates == "PASS") & (df.constraints == "ok")]
nd = [p for p in ok.index if not any((ok.loc[q, "geo x"] >= ok.loc[p, "geo x"]) and (ok.loc[q, "g3 CE-CAGR %"] >= ok.loc[p, "g3 CE-CAGR %"])
      and ((ok.loc[q, "geo x"] > ok.loc[p, "geo x"]) or (ok.loc[q, "g3 CE-CAGR %"] > ok.loc[p, "g3 CE-CAGR %"])) for q in ok.index if q != p)]
nd = sorted(nd, key=lambda p: -ok.loc[p, "geo x"]); print("admissible:", len(ok), "of", len(df)); print("non-dominated (geo, g3):", nd)
# tie-band reading (v3.3 bands): a frontier point that is inside the g3 band of a higher-wealth frontier point is its neighbour
T = ST.TIE; sub = []
for p in nd:
    better = [q for q in nd if ok.loc[q, "geo x"] > ok.loc[p, "geo x"] + T["geo"] and ok.loc[q, "g3 CE-CAGR %"] >= ok.loc[p, "g3 CE-CAGR %"] - T["ce3_cagr"]]
    sub.append((p, better[0] if better else None))
print("substantive frontier (after tie bands):", [p for p, b in sub if b is None]); print("tie-band neighbours:", {p: b for p, b in sub if b})
print(df.loc[[p for p in nd], ["geo x", "g3 CE-CAGR %", "P5 terminal x", "worst scen %", "nuclear RS %", "nuclear RS 48mo %", "largest RS %", "constraints"]].to_string())
print("\nstatus changes vs v3.2:", [nm for nm in df.index if (df.loc[nm, "constraints"] == "ok") != (pd.read_csv("out/Report_v32_R8k_20260910.csv").set_index("Port").loc[nm, "constraints"] == "ok")])

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
