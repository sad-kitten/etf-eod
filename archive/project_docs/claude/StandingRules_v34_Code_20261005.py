# ===== StandingRules_v34_Code_20261005 — code bundle for EngineV2_Addendum42 (2026-10-05) =====
# Stack: enginev2/lib_pathce.py (investment-lab 4bad722) + core.py (package + ENGINEV2_EXTRA hook) + search_B.py + policy_engine2.py
# (Bench_2FFL_Code_20260919 reconstructions, verbatim) + the modules below. Env for every run in Addendum 42:
#   ENGINEV2_DATA=<dir with decks/> DECK_FILE=Deck_Recal_R8l_20261005.csv SCEN_FILE=scenarios_25_R8i_20260909.csv
#   GRADES_FILE=<decks>/Row_Grades_R8j_20260909.csv ENGINEV2_PANEL=<etf-eod>/monthly_returns.csv (pull of 2026-10-04) ENGINEV2_END=2026-08
#   STRESS_GATE=1 [SIGMA_LEVEL=cov120 for the M1 column] [ETF_CAP=0.20 for the cap test] NPATHS=6000 (search) / 2000 (report; FastEval unused)

# ===== standing.py (v3.3 + 2026-10-05 additions) =====
"""STANDING RULES v3.3 (verbatim from StandingRules_v33_Code_20260910.py, standing.py section).
v3 = Addendum 31 items applied 2026-09-09; v3.1 = Addendum 33 gate set ratified 2026-09-10; v3.2 = Addendum 34 items applied
2026-09-10; v3.3 = averaged risk-share estimator and 3-sd tie bands, Addendum 35 s3/s4, 2026-09-10.
2026-10-05 additions (Addendum 42 session), OFF unless the environment variable is set:
  SIGMA_LEVEL=cov120  -> M1: sigma0 from the 120-month anchored panel (COV120) instead of the 48-month window (level only;
                         the per-scenario allocation of variance by severity is unchanged)
  ETF_CAP=0.20        -> cap test: uniform per-holding cap for the feasibility check (default 0.15 from core.feasible)
"""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L
from core import UNIV, IDX, DECKM, ERV, sigma_of, wvec, SINGLES, feasible as _feasible_core, CONVS
import search_B as S

ETF_CAP = float(os.environ.get("ETF_CAP", "0.15"))
def feasible(w):
    """core.feasible with a configurable per-holding cap (cap test, 2026-10-05); identical to core.feasible at 0.15."""
    if ETF_CAP == 0.15: return _feasible_core(w)
    if w.min() < -1e-12 or abs(w.sum()-1) > 1e-9: return False
    held = w > 1e-9
    if held.sum() > 14: return False
    if np.any(w[held] < 0.02 - 1e-9): return False
    if w.max() > ETF_CAP + 1e-9: return False
    from core import PMEQ, SEMIS, DEF, NUC, BAL
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
def risk_mrc(w): return 0.5 * (_mrc(S.COV, w) + _mrc(COV120, w))
def risk_share(w, names):
    mrc = risk_mrc(w); return float(sum(mrc[IDX[t]] for t in names if t in IDX))
def risk_shares(w):
    """(nuclear-complex share, largest single-holding share) on the averaged estimator."""
    mrc = risk_mrc(w); return float(sum(mrc[IDX[t]] for t in S.NUC_SET)), float(mrc.max())
def risk_shares_48(w): return S.risk_shares(w)

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
    for k, u in others.items():
        if 0.5 * np.abs(w - u).sum() <= DUP + 1e-9: return k
    return None

# ---------------------------------------------------------------- regime-conditional sigma (11)
STRESS = np.array([1.0 if s >= 4 else 0.0 for s in L.SEV])
_ivv = _mr["IVV"]; _cut = _ivv.quantile(1 / 3)
_mS = _mr[_ivv <= _cut]; _mN = _mr[_ivv > _cut]
ANCHOR = {"NUKZ": "NLR", "SHLD": "PPA", "SETM": "COPX", "JEDI": "PPA", "EUAD": "ITA", "CEG": "NLR", "IBIT": "IBIT", "CTA": "DBMF"}
_cols = [t for t in UNIV if t in _mr.columns]
def _cov(m):
    src = m[[ANCHOR.get(t, t) if HIST.get(t, 0) < SHORT_MONTHS and ANCHOR.get(t, t) in m.columns else t for t in _cols]].copy()
    src.columns = _cols
    c = src.cov(min_periods=18) * 12.0
    full = pd.DataFrame(0.0, index=UNIV, columns=UNIV); full.loc[_cols, _cols] = c.values
    return np.nan_to_num(full.values)
COV_S, COV_N = _cov(_mS), _cov(_mN)
COV120 = _cov(_mr.loc[_mr.index[-120]:])
SHRINK = 0.5
P_STRESS = {pin: float(pi @ STRESS) for pin, pi in (("robust", L.PI_ROBUST), ("canonical", L.PI_CANON))}

SIGMA_LEVEL = os.environ.get("SIGMA_LEVEL", "vol48")
def sigma0_of(w):
    """Book sigma level: standing = 48-month realized (core.sigma_of); M1 (2026-10-05) = 120-month anchored panel."""
    if SIGMA_LEVEL == "cov120":
        return float(np.sqrt(max(float(w @ COV120 @ w), 1e-12)))
    return sigma_of(w)

def sigma_vec(w, pin="robust"):
    """Per-scenario sigma vector: sigma0 x (ratio in stress states, 1 in normal), rescaled so that the
    pi-weighted variance equals sigma0^2. Level unchanged; allocation across states from the conditional panel."""
    s0 = sigma0_of(w)
    vs, vn = float(w @ COV_S @ w), float(w @ COV_N @ w)
    ratio = np.sqrt(max(vs, 1e-12) / max(vn, 1e-12)); ratio = float(np.clip(1.0 + SHRINK * (ratio - 1.0), 0.6, 2.0))
    p = P_STRESS[pin]
    k = 1.0 / np.sqrt(p * ratio**2 + (1 - p))
    return s0 * k * np.where(STRESS > 0, ratio, 1.0), ratio


# ---------------------------------------------------------------- v3.4 candidate (2026-10-05): stressed-floor gate
# Stressed deck = current deck with every premium the floor can lean on haircut at once, so the optimizer cannot migrate to the
# un-stressed one: gold family -1.5 pp all scenarios (base -> realized 40-y median), trend rows -3.0 pp in severity-4/5 scenarios
# (Addendum 39 item 3), thematic equity rows -2.0 pp all scenarios (Addendum 39 item 1). Broad-market, bond and single-name rows
# unchanged. Gate: worst stressed net CAGR >= STRESS_FLOOR (default 3.5% = corpus survives the 40-year fixed 3% draw in the
# worst scenario at ~1% real). Active only when STRESS_GATE=1.
import json as _json
GOLD_FAM = ["GLD", "GLDM", "FNV", "WPM", "GDX", "SIL"]
TREND = ["DBMF", "KMLM", "CTA", "CTA_OVERLAY"]
THEMATIC = sorted(set(BLOCKS["nuclear"] + BLOCKS["defence"] + BLOCKS["cyber"] + BLOCKS["semis"] +
                      ["XLE", "GRID", "PAVE", "AIRR", "QTUM", "COPX", "REMX", "GUNR", "AMLP", "XBI", "IBB", "IGV", "XLK", "FTEC",
                       "BOTZ", "ROBO", "KOMP", "XT", "PRNT", "PSCI", "AQWA", "IHI", "IHAK", "SETM", "CEG", "URA", "NLR"]))
STRESS_SPEC = dict(gold_all=-1.5, trend_sev45=-3.0, thematic_all=-2.0)
STRESS_FLOOR = float(os.environ.get("STRESS_FLOOR", "0.035"))
STRESS_GATE = os.environ.get("STRESS_GATE", "0") == "1"
def stressed_deck(M=None):
    M = (DECKM if M is None else M).copy(); sev45 = L.SEV >= 4
    for t in GOLD_FAM:
        if t in IDX: M[IDX[t]] += STRESS_SPEC["gold_all"] / 100.0
    for t in TREND:
        if t in IDX: M[IDX[t], sev45] += STRESS_SPEC["trend_sev45"] / 100.0
    for t in THEMATIC:
        if t in IDX: M[IDX[t]] += STRESS_SPEC["thematic_all"] / 100.0
    return M
DECK_STRESSED = stressed_deck()
def stressed_worst(w):
    r = DECK_STRESSED.T @ w - float(ERV @ w); i = int(np.argmin(r)); return float(r[i]), L.SC[i]
def stressed_fail(w):
    v, sc = stressed_worst(w); return None if v >= STRESS_FLOOR - 1e-12 else f"stressed worst {v*100:.2f}<{STRESS_FLOOR*100:.2f} ({sc[:20]})"

# ---------------------------------------------------------------- absolute gates (3, 5)
GATES = dict(ruin=0, P5=1.00, worst=0.0525, R3AF=0.05, LS_ruin=0)
P5_BAND, P5_SEEDS = 0.02, [42, 7, 101, 2024, 555, 31337]
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
FAST_P5 = 1.06
def p5_threshold(mode="exact"): return FAST_P5 if mode == "fast" else GATES["P5"]
def shortfall6(r, pi=None):
    pi = (np.array(L.PI_ROBUST) + np.array(L.PI_CANON)) / 2 if pi is None else pi
    return float(pi @ np.maximum(0.0, 0.06 - r))

def gate_fails_abs(res_base, res_ls, res_hcc=None, r=None, mode="exact", p5_mean=None):
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
    fails = gate_fails_abs(out[("robust", "G")], ls, hcc, r, mode="fast")
    if STRESS_GATE:
        sf = stressed_fail(w)
        if sf: fails.append(sf)
    return dict(geo=np.exp(g1), ce3=np.exp(g3), ce5=np.exp(g5), P5=out[("robust", "G")]["P5"], ruin=out[("robust", "G")]["ruin"],
                ls_geo=np.exp(np.mean([np.log(ls["PathCE"])])), ls_P5=ls["P5"], hcc_P5=hcc["P5"], fails=fails, stress_ratio=ratio,
                shortfall6=shortfall6(r), worst=float(r.min()))

# ---------------------------------------------------------------- reporting set (4, 12)
TIE = dict(geo=0.02, ce3_cagr=0.04, p5=0.015)
def cagr(x): return (x ** (1 / 40.0) - 1) * 100
def report_structure(w, d):
    ns, mx = risk_shares(w); n48, m48 = risk_shares_48(w)
    return {"nuclear RS %": round(ns * 100), "nuclear RS 48mo %": round(n48 * 100), "defence RS %": round(risk_share(w, BLOCKS["defence"]) * 100),
            "gold RS %": round(risk_share(w, BLOCKS["gold"]) * 100), "cyber RS %": round(risk_share(w, BLOCKS["cyber"]) * 100),
            "largest RS %": round(mx * 100), "largest RS 48mo %": round(m48 * 100), "grade-C wt %": round(float(CW @ w) * 100), "<60mo wt %": round(float(SHORT @ w) * 100),
            "ballast wt %": round(sum(d.get(t, 0) for t in BALLAST) * 100), "singles": sum(1 for t in SINGLES if d.get(t, 0) > 0 and t not in ETF_LIKE),
            "holdings": len(d), "ER %": round(float(ERV @ w) * 100, 3), "stress σ ratio": round(sigma_vec(w)[1], 2),
            "sigma0 %": round(sigma0_of(w) * 100, 2)}

# ===== report_v3.py =====
"""REPORT v3 (StandingRules_v33_Code_20260910.py report_v3.py) — engine-exact (60k x 4 conventions), regime sigma.
2026-10-05: the symmetric-deck columns are omitted (Deck_Recal_R8j_sym not loaded); an optional DECK_SHIFT json
{ticker: pp} applies a parallel row shift to the deck AND the HCC deck (deck-revision candidates, e.g. D1); optional
STRESS_DECK json applies a second shift set used only for the stressed-floor column (v3.4 candidate).
Run: DECK_FILE=... SCEN_FILE=... GRADES_FILE=... ENGINEV2_PANEL=... ENGINEV2_END=2026-08 python3 report_v3.py books.json out.csv
"""
import sys, os, json, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, policy_engine2 as E, standing as ST, search_B as S
from core import DECKM, ERV, wvec, UNIV, IDX
CONV = [("robust","G"),("robust","L"),("canonical","G"),("canonical","L")]
SEVv = L.SEV

def shifted(M, spec):
    """spec: {ticker: pp} or {ticker: {"all": pp} / {"sev45": pp}} — parallel shift of deck rows in percent."""
    M = M.copy()
    for t, v in (spec or {}).items():
        if t not in IDX: continue
        if isinstance(v, dict):
            if "all" in v: M[IDX[t]] += v["all"] / 100.0
            if "sev45" in v: M[IDX[t], SEVv >= 4] += v["sev45"] / 100.0
        else:
            M[IDX[t]] += v / 100.0
    return M
DSHIFT = json.loads(os.environ["DECK_SHIFT"]) if os.environ.get("DECK_SHIFT") else {}
M0 = shifted(DECKM, DSHIFT); H0 = shifted(S.HCC, DSHIFT); MS = ST.stressed_deck(M0); SSHIFT = True
STRESS_FLOOR = ST.STRESS_FLOOR

def ce_cagr(W, gam):
    ce = float(np.exp(np.log(W).mean())) if gam == 1 else float(np.mean(W ** (1 - gam)) ** (1 / (1 - gam)))
    return ce, (ce ** (1 / 40.) - 1) * 100
def pooled(mc, key="W"): return np.concatenate([np.maximum(mc[c][key], 1e-9) for c in CONV])
def profile(nm, d):
    t0 = time.time()
    w = wvec(d); wer = float(ERV @ w); r = M0.T @ w - wer; rs = MS.T @ w - wer
    sig = {p: ST.sigma_vec(w, p)[0] for p in ("robust", "canonical")}
    mc = E.suite(r, sig, E.POL); W = pooled(mc); rg = mc[("robust", "G")]
    m1 = E.suite(r - 0.01, sig, E.POL); m2 = E.suite(r - 0.02, sig, E.POL)
    mh = E.suite(H0.T @ w - wer, sig, E.POL)
    g1x, g1 = ce_cagr(W, 1); g3x, g3 = ce_cagr(W, 3); g5x, g5 = ce_cagr(W, 5)
    pc = W ** (1 / 40.) - 1
    p5_mean = None
    if abs(rg["P5"] - ST.GATES["P5"]) < ST.P5_BAND:
        p5_mean = float(np.mean([E.run(np.random.default_rng(s), r, L.PI_ROBUST, "G", sig["robust"], E.POL)["P5"] for s in ST.P5_SEEDS]))
    fails = ST.gate_fails_abs(rg, m2[("robust", "G")], mh[("robust", "G")], r, p5_mean=p5_mean)
    sfail = float(rs.min()) < STRESS_FLOOR - 1e-12
    if ST.STRESS_GATE and sfail: fails.append(f"stressed worst {rs.min()*100:.2f}<{STRESS_FLOOR*100:.2f}")
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
        "worst scen %": round(float(r.min()) * 100, 2), "worst scen": L.SC[int(np.argmin(r))][:28],
        "stressed worst %": round(float(rs.min()) * 100, 2), "stressed worst scen": L.SC[int(np.argmin(rs))][:28],
        "stressed floor": ("PASS" if not sfail else f"FAIL <{STRESS_FLOOR*100:.2f}") if SSHIFT else "",
        "shortfall6 (bp)": round(ST.shortfall6(r) * 1e4, 1),
        "DRB40@6 (reported)": round(L.drb40(r, h=0.06), 4), "RCEW40 (reported)": round(L.rcew40(r)["robust"], 2),
        "ruin": int(rg["ruin"]), "gates": "PASS" if not fails else "; ".join(fails),
        "constraints": "ok" if not ST.violations(w) else "; ".join(ST.violations(w)),
        "lifetime tax x": round(rg["tax_paid"], 3), "vol48 %": round(S.sigma_of(w) * 100, 2),
        "P10 terminal x": round(float(np.quantile(W, .10)), 3), "P50 terminal x": round(float(np.median(W)), 3),
        "P(W40<1) %": round(float((W < 1.0).mean()) * 100, 1), "yrs reduced draw %": round(rg["yrs_red"] * 100, 1),
        "secs": round(time.time() - t0)}
    row.update(st); row["weights"] = " ".join(f"{t}{int(round(x*100))}" for t, x in sorted(d.items(), key=lambda kv: -kv[1]))
    return row

if __name__ == "__main__":
    BK = json.load(open(sys.argv[1])); rows = []
    for nm, d in BK.items():
        d = {t: v for t, v in d.items() if v > 0}; rows.append(profile(nm, d))
        print(nm, rows[-1]["geo x"], rows[-1]["g3 CE-CAGR %"], rows[-1]["P5 terminal x"], rows[-1]["worst scen %"], rows[-1]["gates"], rows[-1]["constraints"], rows[-1]["secs"], "s", flush=True)
    df = pd.DataFrame(rows).set_index("Port"); df.to_csv(sys.argv[2])
    ok = df[(df.gates == "PASS") & (df.constraints == "ok")]
    nd = [p for p in ok.index if not any((ok.loc[q, "geo x"] >= ok.loc[p, "geo x"]) and (ok.loc[q, "g3 CE-CAGR %"] >= ok.loc[p, "g3 CE-CAGR %"])
                                        and ((ok.loc[q, "geo x"] > ok.loc[p, "geo x"]) or (ok.loc[q, "g3 CE-CAGR %"] > ok.loc[p, "g3 CE-CAGR %"])) for q in ok.index if q != p)]
    print("\nnon-dominated on (geo, g3):", sorted(nd, key=lambda p: -ok.loc[p, "geo x"]))

# ===== search_v3.py (with repair phase) =====
"""SEARCH v3 (StandingRules_v33_Code_20260910.py search_v3.py) with a 2026-10-05 repair phase: a seed that fails the absolute
gates is first moved, one admissible 1–3-point trade at a time, along the direction that reduces the total gate deficit
(sum of shortfalls on central worst, stressed worst, fast P5), then climbed on the objective as before.
Objective: mean log of the 4-convention fast evaluator (OBJ=CE3 default, GEO, CE5), regime sigma; constraints standing.violations
(ETF_CAP env for the cap test); gates standing.evaluate_fast (FAST_P5 1.06 at search time; stressed floor when STRESS_GATE=1).
Run: DECK_FILE=Deck_Recal_R8l_20261005.csv ... NPATHS=6000 OBJ=CE3 ETF_CAP=0.20 STRESS_GATE=1 python3 search_v3.py C5_SMH3b
"""
import sys, os, json, time, numpy as np, warnings; warnings.filterwarnings('ignore')
import standing as ST
import search_B as S
from core import UNIV, IDX, DECKM, ERV, wvec, CONVS
OBJ = os.environ.get("OBJ", "CE3"); KEY = {"CE3": "CE3", "GEO": "PathCE", "CE5": "CE5"}[OBJ]
SIZES = [float(x) for x in os.environ.get("SIZES", "0.01,0.02,0.03").split(",")]
MAXSTEPS = int(os.environ.get("MAXSTEPS", "40")); SHORTLIST = int(os.environ.get("SHORTLIST", "30"))

def objective(w):
    r = DECKM.T @ w - float(ERV @ w); vals = []
    for pin in ("robust", "canonical"):
        sigv, _ = ST.sigma_vec(w, pin)
        for conv in ("G", "L"): vals.append(np.log(ST.run_fast(r, sigv, pin, conv)[KEY]))
    return float(np.mean(vals))

def admissible(v): return not ST.violations(v)
def gates_ok(v): return not ST.evaluate_fast(v)["fails"]
def deficit(v):
    """Total gate shortfall (percentage points + P5 units), 0 when all gates pass."""
    e = ST.evaluate_fast(v); r = DECKM.T @ v - float(ERV @ v); d = 0.0
    d += max(0.0, ST.GATES["worst"] - float(r.min())) * 100
    if ST.STRESS_GATE: d += max(0.0, ST.STRESS_FLOOR - ST.stressed_worst(v)[0]) * 100
    d += max(0.0, ST.FAST_P5 - e["P5"])
    d += max(0.0, ST.GATES["R3AF"] - ST.r3af_std(r)) * 10
    d += 1.0 * (e["ruin"] > 0) + 1.0 * (ST.run_fast(r - 0.02, ST.sigma_vec(v, "robust")[0], "robust", "G")["ruin"] > 0)
    return d

def neighbours(w, pool):
    out = []
    for i in S.held(w):
        wi = w[IDX[i]]
        for j in pool:
            if j == i: continue
            wj = w[IDX[j]]
            sizes = SIZES if wj > 1e-9 else [s for s in SIZES if s >= 0.02]
            for d in sizes:
                if d > wi + 1e-9 or (0 < wi - d < 0.02 - 1e-9): continue
                v = w.copy(); v[IDX[i]] -= d; v[IDX[j]] += d; v[np.abs(v) < 1e-12] = 0.0
                if admissible(v): out.append((v, i, j, d))
    return out

def climb(seed, tag):
    w = wvec(seed); assert abs(w.sum() - 1) < 1e-9
    viol = ST.violations(w); assert not viol, (tag, viol)
    t0 = time.time()
    pool_all = sorted(set(S.POOL) | {"BRKB"} | set(S.S1))
    sc = {}
    for j in pool_all:
        v = w * 0.97; v[IDX[j]] += 0.03; v /= v.sum(); sc[j] = objective(v)
    pool = sorted(set(sorted(sc, key=lambda j: -sc[j])[:SHORTLIST]) | set(S.held(w)))
    # ---- repair phase ----
    dcur = deficit(w); steps = 0
    while dcur > 1e-9 and steps < 12:
        best = None
        for v, i, j, d in neighbours(w, pool):
            dv = deficit(v)
            if best is None or dv < best[0] - 1e-12 or (abs(dv - best[0]) < 1e-12 and objective(v) > best[5]):
                best = (dv, v, i, j, d, objective(v) if best is None or abs(dv - best[0]) < 1e-12 else -np.inf)
        if best is None or best[0] >= dcur - 1e-12:
            print(f"[{tag}] repair stalled at deficit {dcur:.3f}: {ST.evaluate_fast(w)['fails']}", flush=True); break
        dcur, w = best[0], best[1]; steps += 1
        print(f"[{tag}] repair {steps}: {best[2]}->{best[3]} {best[4]*100:.0f}% | deficit {dcur:.3f} | {S.show(w)}", flush=True)
    cur = objective(w); e = ST.evaluate_fast(w)
    print(f"[{tag}] start {S.show(w)} | {OBJ} {np.exp(cur):.3f} geo {e['geo']:.3f} ce3 {e['ce3']:.3f} P5 {e['P5']:.2f} worst {e['worst']*100:.2f} fails {e['fails']}", flush=True)
    for step in range(MAXSTEPS):
        cands = []
        for v, i, j, d in neighbours(w, pool):
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

if __name__ == "__main__":
    BK = json.load(open(os.environ.get("SEEDS_FILE", "books_all.json")))
    which = sys.argv[1:] or ["C5_SMH3b"]; out = {}
    for tag in which:
        w, cur = climb({t: v for t, v in BK[tag].items() if v > 0}, tag)
        out[tag] = {t: round(float(w[IDX[t]]), 2) for t in S.held(w)}
        e = ST.evaluate_fast(w)
        print(f"[{tag}] FINAL {OBJ} {np.exp(cur):.3f} | geo {e['geo']:.3f} ce3 {e['ce3']:.3f} ce5 {e['ce5']:.3f} P5 {e['P5']:.2f} worst {e['worst']*100:.2f} | {S.show(w)}", flush=True)
        json.dump(out, open(os.environ.get("OUTJSON", f"out/searchV3_{OBJ}.json"), "w"), indent=1)

# ===== run_batch.sh =====
#!/bin/bash
cd /home/claude/engine
export ENGINEV2_DATA=/home/claude/engine/data SCEN_FILE=scenarios_25_R8i_20260909.csv GRADES_FILE=/home/claude/engine/data/decks/Row_Grades_R8j_20260909.csv ENGINEV2_PANEL=/home/claude/etf-eod/monthly_returns.csv ENGINEV2_END=2026-08 NPATHS=2000 STRESS_GATE=1
# A: R8l, v3.3 sigma level (vol48), stressed-floor gate reported+gated (v3.4 candidate)
DECK_FILE=Deck_Recal_R8l_20261005.csv python3 report_v3.py books_all.json out/Report_v34_R8l_ALL_20261005.csv > out/logA.txt 2>&1 &
# B: R8l, M1 sigma level (120-month anchored panel)
DECK_FILE=Deck_Recal_R8l_20261005.csv SIGMA_LEVEL=cov120 python3 report_v3.py books_all.json out/Report_v34_R8l_M1_ALL_20261005.csv > out/logB.txt 2>&1 &
# C: R8k reference re-run on the current panel (attribution baseline)
DECK_FILE=Deck_Recal_R8k_20260910.csv python3 report_v3.py books_all.json out/Report_v33_R8k_ALL_rerun_20261005.csv > out/logC.txt 2>&1 &
wait
echo BATCH_DONE >> out/logA.txt

# ===== run_search.sh =====
#!/bin/bash
cd /home/claude/engine
export ENGINEV2_DATA=/home/claude/engine/data DECK_FILE=Deck_Recal_R8l_20261005.csv SCEN_FILE=scenarios_25_R8i_20260909.csv GRADES_FILE=/home/claude/engine/data/decks/Row_Grades_R8j_20260909.csv ENGINEV2_PANEL=/home/claude/etf-eod/monthly_returns.csv ENGINEV2_END=2026-08 NPATHS=6000 STRESS_GATE=1
OBJ=CE3 ETF_CAP=0.15 OUTJSON=out/search_R8l_cap15_CE3.json nice -n 5 python3 search_v3.py C5_SMH3b > out/search_cap15_CE3.log 2>&1 &
OBJ=CE3 ETF_CAP=0.20 OUTJSON=out/search_R8l_cap20_CE3.json nice -n 5 python3 search_v3.py C5_SMH3b > out/search_cap20_CE3.log 2>&1 &
wait
OBJ=GEO ETF_CAP=0.15 OUTJSON=out/search_R8l_cap15_GEO.json nice -n 5 python3 search_v3.py C5_SMH3b > out/search_cap15_GEO.log 2>&1 &
OBJ=GEO ETF_CAP=0.20 OUTJSON=out/search_R8l_cap20_GEO.json nice -n 5 python3 search_v3.py C5_SMH3b > out/search_cap20_GEO.log 2>&1 &
wait
echo SEARCH_DONE > out/search_done.txt

# ===== books_all.json =====
{
 "C5_SMH3b": {
  "GLDM": 0.15,
  "SHLD": 0.15,
  "CIBR": 0.15,
  "DBMF": 0.15,
  "NUKZ": 0.08,
  "BRKB": 0.05,
  "CME": 0.05,
  "FNV": 0.05,
  "MSCI": 0.05,
  "URNM": 0.05,
  "XLE": 0.03,
  "XAR": 0.02,
  "SMH": 0.02
 },
 "K7": {
  "GLDM": 0.2,
  "SHLD": 0.2,
  "CIBR": 0.2,
  "DBMF": 0.2,
  "URNM": 0.1,
  "MSCI": 0.05,
  "FNV": 0.05
 },
 "M5": {
  "GLDM": 0.3,
  "SHLD": 0.23,
  "CIBR": 0.2,
  "DBMF": 0.17,
  "URNM": 0.1
 },
 "C5_smh3_NUKZ": {
  "CIBR": 0.15,
  "GLDM": 0.15,
  "SHLD": 0.15,
  "DBMF": 0.15,
  "NUKZ": 0.06,
  "CME": 0.05,
  "MSCI": 0.05,
  "XLE": 0.05,
  "FNV": 0.05,
  "BRKB": 0.05,
  "URNM": 0.04,
  "SMH": 0.03,
  "HACK": 0.02
 },
 "C5_bal": {
  "CIBR": 0.15,
  "GLDM": 0.15,
  "SHLD": 0.15,
  "DBMF": 0.15,
  "NUKZ": 0.09,
  "CME": 0.05,
  "MSCI": 0.05,
  "XLE": 0.05,
  "FNV": 0.05,
  "BRKB": 0.05,
  "URNM": 0.04,
  "HACK": 0.02
 },
 "C5_bal2": {
  "CIBR": 0.15,
  "GLDM": 0.15,
  "SHLD": 0.15,
  "DBMF": 0.15,
  "NUKZ": 0.09,
  "CME": 0.05,
  "MSCI": 0.05,
  "XLE": 0.05,
  "FNV": 0.05,
  "BRKB": 0.05,
  "URNM": 0.04,
  "XAR": 0.02
 },
 "C5_SMH3c": {
  "CIBR": 0.15,
  "GLDM": 0.15,
  "SHLD": 0.15,
  "DBMF": 0.15,
  "NUKZ": 0.07,
  "URNM": 0.06,
  "CME": 0.05,
  "MSCI": 0.05,
  "FNV": 0.05,
  "BRKB": 0.05,
  "XLE": 0.03,
  "XAR": 0.02,
  "SMH": 0.02
 },
 "X2_F2_DBMF15": {
  "GLDM": 0.15,
  "SHLD": 0.15,
  "DBMF": 0.15,
  "CIBR": 0.13,
  "NUKZ": 0.1,
  "URNM": 0.05,
  "XLE": 0.05,
  "BRKB": 0.05,
  "CME": 0.05,
  "FNV": 0.05,
  "MSCI": 0.05,
  "XAR": 0.02
 },
 "X3c": {
  "GLDM": 0.15,
  "SHLD": 0.15,
  "CIBR": 0.15,
  "DBMF": 0.11,
  "NUKZ": 0.1,
  "CME": 0.05,
  "MSCI": 0.05,
  "URNM": 0.05,
  "XLE": 0.05,
  "FNV": 0.05,
  "BRKB": 0.05,
  "XAR": 0.04
 },
 "V3_geoX3c": {
  "GLDM": 0.15,
  "SHLD": 0.15,
  "CIBR": 0.15,
  "DBMF": 0.11,
  "URNM": 0.08,
  "NUKZ": 0.07,
  "CME": 0.05,
  "MSCI": 0.05,
  "XLE": 0.05,
  "FNV": 0.05,
  "BRKB": 0.05,
  "XAR": 0.04
 },
 "V33_geoX3c": {
  "GLDM": 0.15,
  "SHLD": 0.15,
  "CIBR": 0.13,
  "DBMF": 0.11,
  "URNM": 0.08,
  "CME": 0.05,
  "MSCI": 0.05,
  "SOXX": 0.05,
  "XLE": 0.05,
  "FNV": 0.05,
  "BRKB": 0.05,
  "NUKZ": 0.04,
  "XAR": 0.04
 },
 "V33_geoC5": {
  "GLDM": 0.15,
  "SHLD": 0.15,
  "CIBR": 0.15,
  "DBMF": 0.09,
  "URNM": 0.09,
  "NUKZ": 0.07,
  "CME": 0.05,
  "MSCI": 0.05,
  "XLE": 0.05,
  "FNV": 0.05,
  "BRKB": 0.05,
  "SOXX": 0.03,
  "HACK": 0.02
 },
 "V31_C5g": {
  "GLDM": 0.15,
  "SHLD": 0.15,
  "CIBR": 0.15,
  "DBMF": 0.12,
  "NUKZ": 0.09,
  "URNM": 0.07,
  "CME": 0.05,
  "MSCI": 0.05,
  "XLE": 0.05,
  "FNV": 0.05,
  "BRKB": 0.05,
  "HACK": 0.02
 },
 "V3_reliab": {
  "GLDM": 0.15,
  "DBMF": 0.15,
  "CIBR": 0.15,
  "KMLM": 0.12,
  "SHLD": 0.12,
  "CME": 0.05,
  "MSCI": 0.05,
  "URNM": 0.05,
  "FNV": 0.05,
  "BRKB": 0.05,
  "XLE": 0.02,
  "HACK": 0.02,
  "XLU": 0.02
 },
 "V31_reliab": {
  "GLDM": 0.15,
  "DBMF": 0.15,
  "CIBR": 0.15,
  "KMLM": 0.13,
  "SHLD": 0.12,
  "CME": 0.05,
  "MSCI": 0.05,
  "FNV": 0.05,
  "BRKB": 0.05,
  "URNM": 0.04,
  "NUKZ": 0.02,
  "XLE": 0.02,
  "XBI": 0.02
 },
 "G1_g3bal": {
  "GLDM": 0.15,
  "DBMF": 0.15,
  "CIBR": 0.14,
  "SHLD": 0.13,
  "KMLM": 0.08,
  "NUKZ": 0.06,
  "CME": 0.05,
  "MSCI": 0.05,
  "FNV": 0.05,
  "BRKB": 0.05,
  "JEPI": 0.03,
  "HACK": 0.02,
  "XLU": 0.02,
  "CTA": 0.02
 },
 "F9_semi": {
  "GLDM": 0.15,
  "DBMF": 0.15,
  "SHLD": 0.14,
  "CIBR": 0.13,
  "NUKZ": 0.09,
  "CME": 0.05,
  "MSCI": 0.05,
  "FNV": 0.05,
  "BRKB": 0.05,
  "SMH": 0.04,
  "URNM": 0.04,
  "XLE": 0.04,
  "AGG": 0.02
 },
 "F11_ivv": {
  "GLDM": 0.15,
  "DBMF": 0.15,
  "CIBR": 0.15,
  "SHLD": 0.14,
  "NUKZ": 0.07,
  "CME": 0.05,
  "MSCI": 0.05,
  "XLE": 0.05,
  "IVV": 0.05,
  "BRKB": 0.05,
  "FNV": 0.04,
  "URNM": 0.03,
  "PDBC": 0.02
 },
 "P1": {
  "GLDM": 0.15,
  "SHLD": 0.15,
  "DBMF": 0.15,
  "NUKZ": 0.15,
  "CIBR": 0.1,
  "GRID": 0.1,
  "CTA": 0.05,
  "CME": 0.05,
  "JEDI": 0.05,
  "MSCI": 0.05
 }
}
# ===== books_cand.json =====
{
 "C5_SMH3b": {"GLDM":0.15,"SHLD":0.15,"CIBR":0.15,"DBMF":0.15,"NUKZ":0.08,"BRKB":0.05,"CME":0.05,"FNV":0.05,"MSCI":0.05,"URNM":0.05,"XLE":0.03,"XAR":0.02,"SMH":0.02},
 "C5r15_KMLM3": {"GLDM":0.15,"SHLD":0.15,"DBMF":0.15,"CIBR":0.12,"NUKZ":0.08,"CME":0.05,"MSCI":0.05,"URNM":0.05,"FNV":0.05,"BRKB":0.05,"XLE":0.03,"KMLM":0.03,"SMH":0.02,"XAR":0.02},
 "C5r20_GLDM18": {"GLDM":0.18,"SHLD":0.15,"DBMF":0.15,"CIBR":0.15,"NUKZ":0.05,"CME":0.05,"MSCI":0.05,"URNM":0.05,"FNV":0.05,"BRKB":0.05,"XLE":0.03,"SMH":0.02,"XAR":0.02},
 "C5g_XAR5": {"GLDM":0.15,"SHLD":0.15,"DBMF":0.15,"CIBR":0.15,"NUKZ":0.05,"CME":0.05,"MSCI":0.05,"URNM":0.05,"FNV":0.05,"BRKB":0.05,"XAR":0.05,"XLE":0.03,"SMH":0.02},
 "CE3_15_KMLM12": {"GLDM":0.15,"SHLD":0.15,"DBMF":0.15,"CIBR":0.12,"KMLM":0.12,"CME":0.05,"MSCI":0.05,"URNM":0.05,"BRKB":0.05,"FNV":0.03,"NUKZ":0.02,"SMH":0.02,"XAR":0.02,"XLE":0.02},
 "CE3_20_DBMF20": {"DBMF":0.20,"GLDM":0.18,"SHLD":0.15,"CIBR":0.11,"KMLM":0.06,"CME":0.05,"MSCI":0.05,"BRKB":0.05,"URNM":0.04,"SMH":0.03,"NUKZ":0.02,"FNV":0.02,"XAR":0.02,"XLE":0.02}
}

# ===== search logs =====
[C5_SMH3b] repair 1: CIBR->KMLM 3% | deficit 0.000 | GLDM15 SHLD15 DBMF15 CIBR12 NUKZ8 CME5 MSCI5 URNM5 FNV5 BRKB5 XLE3 KMLM3 SMH2 XAR2
[C5_SMH3b] start GLDM15 SHLD15 DBMF15 CIBR12 NUKZ8 CME5 MSCI5 URNM5 FNV5 BRKB5 XLE3 KMLM3 SMH2 XAR2 | CE3 2.716 geo 4.925 ce3 2.716 P5 1.11 worst 5.33 fails []
[C5_SMH3b] 1: NUKZ->KMLM 3% | CE3 2.813 geo 4.815 ce3 2.813 ce5 1.670 P5 1.15 worst 5.35 | GLDM15 SHLD15 DBMF15 CIBR12 KMLM6 NUKZ5 CME5 MSCI5 URNM5 FNV5 BRKB5 XLE3 SMH2 XAR2 | 88s
[C5_SMH3b] 2: NUKZ->KMLM 3% | CE3 2.883 geo 4.699 ce3 2.883 ce5 1.807 P5 1.21 worst 5.37 | GLDM15 SHLD15 DBMF15 CIBR12 KMLM9 CME5 MSCI5 URNM5 FNV5 BRKB5 XLE3 NUKZ2 SMH2 XAR2 | 103s
[C5_SMH3b] 3: FNV->KMLM 1% | CE3 2.902 geo 4.648 ce3 2.902 ce5 1.853 P5 1.23 worst 5.37 | GLDM15 SHLD15 DBMF15 CIBR12 KMLM10 CME5 MSCI5 URNM5 BRKB5 FNV4 XLE3 NUKZ2 SMH2 XAR2 | 117s
[C5_SMH3b] 4: URNM->KMLM 1% | CE3 2.914 geo 4.555 ce3 2.914 ce5 1.908 P5 1.25 worst 5.37 | GLDM15 SHLD15 DBMF15 CIBR12 KMLM11 CME5 MSCI5 BRKB5 URNM4 FNV4 XLE3 NUKZ2 SMH2 XAR2 | 131s
[C5_SMH3b] 5: XLE->CIBR 1% | CE3 2.923 geo 4.552 ce3 2.923 ce5 1.920 P5 1.26 worst 5.38 | GLDM15 SHLD15 DBMF15 CIBR13 KMLM11 CME5 MSCI5 BRKB5 URNM4 FNV4 NUKZ2 SMH2 XAR2 XLE2 | 142s
[C5_SMH3b] 6: FNV->URNM 1% | CE3 2.926 geo 4.593 ce3 2.926 ce5 1.906 P5 1.25 worst 5.37 | GLDM15 SHLD15 DBMF15 CIBR13 KMLM11 CME5 MSCI5 URNM5 BRKB5 FNV3 NUKZ2 SMH2 XAR2 XLE2 | 152s
[C5_SMH3b] 7: CIBR->KMLM 1% | CE3 2.928 geo 4.553 ce3 2.928 ce5 1.927 P5 1.26 worst 5.40 | GLDM15 SHLD15 DBMF15 CIBR12 KMLM12 CME5 MSCI5 URNM5 BRKB5 FNV3 NUKZ2 SMH2 XAR2 XLE2 | 164s
[C5_SMH3b] no improving move clears the gates (top-20 tried); stopping
[C5_SMH3b] FINAL CE3 2.928 | geo 4.553 ce3 2.928 ce5 1.927 P5 1.26 worst 5.40 | GLDM15 SHLD15 DBMF15 CIBR12 KMLM12 CME5 MSCI5 URNM5 BRKB5 FNV3 NUKZ2 SMH2 XAR2 XLE2
[C5_SMH3b] repair 1: NUKZ->GLDM 3% | deficit 0.000 | GLDM18 SHLD15 DBMF15 CIBR15 NUKZ5 CME5 MSCI5 URNM5 FNV5 BRKB5 XLE3 SMH2 XAR2
[C5_SMH3b] start GLDM18 SHLD15 DBMF15 CIBR15 NUKZ5 CME5 MSCI5 URNM5 FNV5 BRKB5 XLE3 SMH2 XAR2 | CE3 2.733 geo 5.070 ce3 2.733 P5 1.10 worst 5.31 fails []
[C5_SMH3b] 1: NUKZ->DBMF 3% | CE3 2.833 geo 5.019 ce3 2.833 ce5 1.617 P5 1.16 worst 5.34 | GLDM18 DBMF18 SHLD15 CIBR15 CME5 MSCI5 URNM5 FNV5 BRKB5 XLE3 NUKZ2 SMH2 XAR2 | 124s
[C5_SMH3b] 2: FNV->KMLM 3% | CE3 2.906 geo 4.864 ce3 2.906 ce5 1.766 P5 1.21 worst 5.33 | GLDM18 DBMF18 SHLD15 CIBR15 CME5 MSCI5 URNM5 BRKB5 XLE3 KMLM3 NUKZ2 FNV2 SMH2 XAR2 | 158s
[C5_SMH3b] 3: CIBR->KMLM 3% | CE3 2.933 geo 4.744 ce3 2.933 ce5 1.851 P5 1.24 worst 5.42 | GLDM18 DBMF18 SHLD15 CIBR12 KMLM6 CME5 MSCI5 URNM5 BRKB5 XLE3 NUKZ2 FNV2 SMH2 XAR2 | 175s
[C5_SMH3b] 4: URNM->DBMF 1% | CE3 2.951 geo 4.670 ce3 2.951 ce5 1.905 P5 1.26 worst 5.43 | DBMF19 GLDM18 SHLD15 CIBR12 KMLM6 CME5 MSCI5 BRKB5 URNM4 XLE3 NUKZ2 FNV2 SMH2 XAR2 | 184s
[C5_SMH3b] 5: XLE->DBMF 1% | CE3 2.965 geo 4.646 ce3 2.965 ce5 1.933 P5 1.28 worst 5.47 | DBMF20 GLDM18 SHLD15 CIBR12 KMLM6 CME5 MSCI5 BRKB5 URNM4 NUKZ2 FNV2 SMH2 XAR2 XLE2 | 190s
[C5_SMH3b] 6: CIBR->SMH 1% | CE3 2.971 geo 4.690 ce3 2.971 ce5 1.921 P5 1.27 worst 5.38 | DBMF20 GLDM18 SHLD15 CIBR11 KMLM6 CME5 MSCI5 BRKB5 URNM4 SMH3 NUKZ2 FNV2 XAR2 XLE2 | 196s
[C5_SMH3b] no improving move clears the gates (top-20 tried); stopping
[C5_SMH3b] FINAL CE3 2.971 | geo 4.690 ce3 2.971 ce5 1.921 P5 1.27 worst 5.38 | DBMF20 GLDM18 SHLD15 CIBR11 KMLM6 CME5 MSCI5 BRKB5 URNM4 SMH3 NUKZ2 FNV2 XAR2 XLE2
[C5_SMH3b] repair 1: NUKZ->XAR 3% | deficit 0.000 | GLDM15 SHLD15 DBMF15 CIBR15 NUKZ5 CME5 MSCI5 URNM5 FNV5 BRKB5 XAR5 XLE3 SMH2
[C5_SMH3b] start GLDM15 SHLD15 DBMF15 CIBR15 NUKZ5 CME5 MSCI5 URNM5 FNV5 BRKB5 XAR5 XLE3 SMH2 | GEO 5.096 geo 5.096 ce3 2.685 P5 1.07 worst 5.25 fails []
[C5_SMH3b] no improving move clears the gates (top-20 tried); stopping
[C5_SMH3b] FINAL GEO 5.096 | geo 5.096 ce3 2.685 ce5 1.412 P5 1.07 worst 5.25 | GLDM15 SHLD15 DBMF15 CIBR15 NUKZ5 CME5 MSCI5 URNM5 FNV5 BRKB5 XAR5 XLE3 SMH2
[C5_SMH3b] repair 1: NUKZ->XAR 3% | deficit 0.000 | GLDM15 SHLD15 DBMF15 CIBR15 NUKZ5 CME5 MSCI5 URNM5 FNV5 BRKB5 XAR5 XLE3 SMH2
[C5_SMH3b] start GLDM15 SHLD15 DBMF15 CIBR15 NUKZ5 CME5 MSCI5 URNM5 FNV5 BRKB5 XAR5 XLE3 SMH2 | GEO 5.096 geo 5.096 ce3 2.685 P5 1.07 worst 5.25 fails []
[C5_SMH3b] no improving move clears the gates (top-20 tried); stopping
[C5_SMH3b] FINAL GEO 5.096 | geo 5.096 ce3 2.685 ce5 1.412 P5 1.07 worst 5.25 | GLDM15 SHLD15 DBMF15 CIBR15 NUKZ5 CME5 MSCI5 URNM5 FNV5 BRKB5 XAR5 XLE3 SMH2
