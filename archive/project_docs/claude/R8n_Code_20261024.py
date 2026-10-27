# ===== build_r8n.py =====
"""Deck R8n (2026-10-24) = R8m + (D3) mean-preserving trend-row reshape + (D6) stacked-fund rows with structure haircut.
D3: realized SG Trend 2000-2025 CAGR 5.23% vs ~1.9% T-bills -> ~3.3 pp net excess (BTOP50 ~2.2, TTU ~5.4); the deck's DBMF
net excess (2.29 pi-mean) sits inside that range, so LEVELS are kept. SHAPE is not: trend lagged in steady equity up-markets
(2010s, ~a decade near flat), yet the deck pays >=2.1 pp excess in every state. Rule: in severity-2 (steady-growth) scenarios
the excess over BIL is capped at 0.5 pp; the removed mass is added back pro rata to severity>=3 excesses so the canonical
pi-mean excess is unchanged.  D6: stacked rows carry BIL+0.30 financing plus a structure haircut (0.50 pp RSST/RSSB/GDE,
0.25 pp NTSX)."""
import sys, pandas as pd, numpy as np
deck_in, scen, panel_in, deck_out, panel_out = sys.argv[1:6]
D = pd.read_csv(deck_in).set_index("Ticker"); S = pd.read_csv(scen)
w = S.Weight_Normalized.values; sev = S.Severity_1to5.values; cols = D.columns
for t in ["DBMF", "KMLM", "CTA", "CTA_OVERLAY"]:
    x = (D.loc[t] - D.loc["BIL"]).values.copy(); m0 = w @ x
    lo = sev <= 2; cut = np.where(lo, np.maximum(x - 0.5, 0.0), 0.0); x = x - cut
    hi = sev >= 3; x[hi] += x[hi] * (w @ cut) / (w[hi] @ x[hi])
    assert abs(w @ x - m0) < 1e-9
    D.loc[t] = D.loc["BIL"].values + x
    print(f"{t}: pi-mean excess {m0:.2f} kept; sev2 excess now <=0.5; sev>=3 x{1 + (w @ cut)/(w[hi] @ (x[hi]/(1+0)) ):.3f}")
L = lambda t: D.loc[t]; SP = 0.30
tsy = 0.25*L("SHY") + 0.5*L("IEF") + 0.25*L("TLT")
rows = {"RSST": L("IVV") + (L("DBMF") - L("BIL")) - SP - 0.50,
        "RSSB": L("VT") + (tsy - L("BIL")) - SP - 0.50,
        "NTSX": 0.9*L("IVV") + 0.6*L("IEF") - 0.5*L("BIL") - 0.6*SP - 0.25,
        "GDE":  0.9*L("IVV") + 0.9*L("GLDM") - 0.8*L("BIL") - 0.9*SP - 0.50,
        "DBMFh": L("BIL") + 0.5*(L("DBMF") - L("BIL")),
        "RSSTh": L("IVV") + 0.5*(L("DBMF") - L("BIL")) - SP - 0.50}
for k, v in rows.items(): D.loc[k] = v
D.reset_index().to_csv(deck_out, index=False)
print("DBMF-BIL by severity:", {s: round(float((D.loc['DBMF']-D.loc['BIL'])[sev == s].mean()), 2) for s in (2, 3, 4, 5)})
M = pd.read_csv(panel_in, index_col=0); sp = SP/100/12
tsym = 0.25*M["SHY"] + 0.5*M["IEF"] + 0.25*M["TLT"]
new = pd.DataFrame({"RSST": M["IVV"] + M["DBMF"] - M["BIL"] - sp, "RSSB": M["VT"] + tsym - M["BIL"] - sp,
                    "NTSX": 0.9*M["IVV"] + 0.6*M["IEF"] - 0.5*M["BIL"] - 0.6*sp, "GDE": 0.9*M["IVV"] + 0.9*M["GLDM"] - 0.8*M["BIL"] - 0.9*sp})
new["DBMFh"] = M["DBMF"]; new["RSSTh"] = new["RSST"]
pd.concat([M.drop(columns=[c for c in new if c in M]), new], axis=1).to_csv(panel_out)

# ===== standing.py =====
"""standing (v3.4, verbatim from StandingRules_v34_Code_20261005.py standing.py section; T-generic CAGR)."""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L
from core import UNIV, IDX, DECKM, ERV, sigma_of, wvec, SINGLES, feasible as _feasible_core, CONVS
import search_B as S

ETF_CAP = float(os.environ.get("ETF_CAP", "0.15"))
def feasible(w):
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

RS_NUC, RS_NAME, RS_BLOCK = 0.38, 0.24, 0.35
CW_CAP, SINGLES_MAX, DUP, SHORT_CAP, SHORT_MONTHS, BALLAST_FLOOR = 0.20, 3, 0.03, 0.30, 60, 0.09
BLOCKS = {"nuclear": ["NUKZ", "URNM", "CEG", "SETM", "NLR", "URA"],
          "defence": ["SHLD", "XAR", "PPA", "ITA", "EUAD", "ROKT", "JEDI"],
          "gold":    ["GLDM", "FNV", "PSLV", "GDX", "WPM", "SIL", "GDE"],
          "cyber":   ["CIBR", "HACK"],
          "semis":   ["SMH", "SOXX", "XSD", "XLK", "FTEC"]}
BALLAST = ["DBMF", "CTA", "KMLM", "RSST"]   # RSST carries 1.0x trend exposure (2026-10-24)
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
    mrc = risk_mrc(w); return float(sum(mrc[IDX[t]] for t in S.NUC_SET)), float(mrc.max())
def risk_shares_48(w): return S.risk_shares(w)

def violations(w):
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
    if PREF:   # owner preferences (Addendum 43): gold-linked <= 15% (look-through GDE 0.9), trend >= gold-linked
        g = sum(w[IDX[t]] * k for t, k in GOLDLINK.items() if t in IDX); tr = sum(w[IDX[t]] for t in TRENDW if t in IDX)
        if g > 0.15 + 1e-9: v.append(f"gold-linked {g*100:.1f}%>15")
        if tr < g - 1e-9: v.append(f"trend {tr*100:.0f}%<gold {g*100:.0f}%")
    return v
PREF = os.environ.get("PREF", "0") == "1"
GOLDLINK = {"GLDM": 1, "GLD": 1, "FNV": 1, "WPM": 1, "GDX": 1, "SIL": 1, "PSLV": 1, "GDE": 0.9}
TRENDW = ["DBMF", "KMLM", "CTA", "RSST"]

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
    if SIGMA_LEVEL == "cov120": return float(np.sqrt(max(float(w @ COV120 @ w), 1e-12)))
    return sigma_of(w)
def sigma_vec(w, pin="robust"):
    s0 = sigma0_of(w)
    vs, vn = float(w @ COV_S @ w), float(w @ COV_N @ w)
    ratio = np.sqrt(max(vs, 1e-12) / max(vn, 1e-12)); ratio = float(np.clip(1.0 + SHRINK * (ratio - 1.0), 0.6, 2.0))
    p = P_STRESS[pin]; k = 1.0 / np.sqrt(p * ratio**2 + (1 - p))
    return s0 * k * np.where(STRESS > 0, ratio, 1.0), ratio

GOLD_FAM = ["GLD", "GLDM", "FNV", "WPM", "GDX", "SIL"]
TREND = ["DBMF", "KMLM", "CTA", "CTA_OVERLAY"]
THEMATIC = sorted(set(BLOCKS["nuclear"] + BLOCKS["defence"] + BLOCKS["cyber"] + BLOCKS["semis"] +
                      ["XLE", "GRID", "PAVE", "AIRR", "QTUM", "COPX", "REMX", "GUNR", "AMLP", "XBI", "IBB", "IGV", "XLK", "FTEC",
                       "BOTZ", "ROBO", "KOMP", "XT", "PRNT", "PSCI", "AQWA", "IHI", "IHAK", "SETM", "CEG", "URA", "NLR"]))
STRESS_SPEC = dict(gold_all=-1.5, trend_sev45=-3.0, thematic_all=-2.0)
# Stacked funds carry their sleeve's stressed haircut in proportion to the stacked exposure (added 2026-10-24).
STACK_STRESS = {"RSST": dict(trend_sev45=1.0), "GDE": dict(gold_all=0.9)}
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
    for t, spec in STACK_STRESS.items():
        if t not in IDX: continue
        if "trend_sev45" in spec: M[IDX[t], sev45] += spec["trend_sev45"] * STRESS_SPEC["trend_sev45"] / 100.0
        if "gold_all" in spec: M[IDX[t]] += spec["gold_all"] * STRESS_SPEC["gold_all"] / 100.0
    return M
DECK_STRESSED = stressed_deck()
def stressed_worst(w):
    r = DECK_STRESSED.T @ w - float(ERV @ w); i = int(np.argmin(r)); return float(r[i]), L.SC[i]
def stressed_fail(w):
    v, sc = stressed_worst(w); return None if v >= STRESS_FLOOR - 1e-12 else f"stressed worst {v*100:.2f}<{STRESS_FLOOR*100:.2f} ({sc[:20]})"

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
    if p5v < p5 - 1e-9: f.append(f"P5 {p5v:.3f}<{p5:.2f}")
    if float(r.min()) < GATES["worst"] - 1e-12: f.append(f"worst {r.min()*100:.2f}<{GATES['worst']*100:.2f}")
    ra = r3af_std(r)
    if ra < GATES["R3AF"]: f.append(f"R3AF {ra:.4f}<{GATES['R3AF']}")
    if res_ls["ruin"] > GATES["LS_ruin"]: f.append("LS ruin")
    return f

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
    r = M.T @ w - float(ERV @ w); out = {}
    for pin in ("robust", "canonical"):
        sigv, ratio = sigma_vec(w, pin)
        for conv in ("G", "L"): out[(pin, conv)] = run_fast(r, sigv, pin, conv)
    g1 = float(np.mean([np.log(out[c]["PathCE"]) for c in CONVS]))
    g3 = float(np.mean([np.log(out[c]["CE3"]) for c in CONVS]))
    g5 = float(np.mean([np.log(out[c]["CE5"]) for c in CONVS]))
    sigv, ratio = sigma_vec(w, "robust")
    ls = run_fast(r - 0.02, sigv, "robust", "G")
    fails = gate_fails_abs(out[("robust", "G")], ls, None, r, mode="fast")
    if STRESS_GATE:
        sf = stressed_fail(w)
        if sf: fails.append(sf)
    return dict(geo=np.exp(g1), ce3=np.exp(g3), ce5=np.exp(g5), P5=out[("robust", "G")]["P5"], ruin=out[("robust", "G")]["ruin"],
                fails=fails, stress_ratio=ratio, worst=float(r.min()))

def report_structure(w, d):
    ns, mx = risk_shares(w)
    return {"nuclear RS %": round(ns * 100), "defence RS %": round(risk_share(w, BLOCKS["defence"]) * 100),
            "gold RS %": round(risk_share(w, BLOCKS["gold"]) * 100), "largest RS %": round(mx * 100),
            "grade-C wt %": round(float(CW @ w) * 100), "ballast wt %": round(sum(d.get(t, 0) for t in BALLAST) * 100),
            "holdings": len(d), "ER %": round(float(ERV @ w) * 100, 3), "stress σ ratio": round(sigma_vec(w)[1], 2),
            "sigma0 %": round(sigma0_of(w) * 100, 2)}

# ===== search_v3.py =====
"""search_v3 (v3.4 bundle, verbatim logic): repair phase then greedy 1-3 pt climbs on OBJ (CE3/GEO), standing constraints, fast gates."""
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
def climb(seed, tag):
    w = wvec(seed); t0 = time.time()
    pool_all = sorted((set(S.POOL) | {"BRKB"} | set(S.S1)) - EXCL)
    sc = {}
    for j in pool_all:
        v = w * 0.97; v[IDX[j]] += 0.03; v /= v.sum(); sc[j] = objective(v)
    pool = sorted(set(sorted(sc, key=lambda j: -sc[j])[:SHORTLIST]) | set(S.held(w)) | {"RSST"})
    print(f"[{tag}] pool {pool}", flush=True)
    dcur = deficit(w); steps = 0
    while dcur > 1e-9 and steps < 12:
        best = None
        for v, i, j, d in neighbours(w, pool):
            dv = deficit(v)
            if best is None or dv < best[0] - 1e-12: best = (dv, v, i, j, d)
        if best is None or best[0] >= dcur - 1e-12: print(f"[{tag}] repair stalled {dcur:.3f}", flush=True); break
        dcur, w = best[0], best[1]; steps += 1
        print(f"[{tag}] repair {steps}: {best[2]}->{best[3]} {best[4]*100:.0f}% deficit {dcur:.3f} | {S.show(w)}", flush=True)
    cur = objective(w); e = ST.evaluate_fast(w)
    print(f"[{tag}] start {S.show(w)} | {OBJ} {np.exp(cur):.3f} geo {e['geo']:.3f} ce3 {e['ce3']:.3f} P5 {e['P5']:.2f} fails {e['fails']}", flush=True)
    for step in range(MAXSTEPS):
        cands = [(o, v, i, j, d) for v, i, j, d in neighbours(w, pool) for o in [objective(v)] if o > cur + 1e-6]
        if not cands: break
        cands.sort(key=lambda x: -x[0]); acc = None
        for c in cands[:20]:
            if gates_ok(c[1]): acc = c; break
        if acc is None: print(f"[{tag}] no improving move clears gates", flush=True); break
        o, v, i, j, d = acc; w, cur = v, o; e = ST.evaluate_fast(w)
        print(f"[{tag}] {step+1}: {i}->{j} {d*100:.0f}% | {OBJ} {np.exp(cur):.3f} geo {e['geo']:.3f} ce3 {e['ce3']:.3f} P5 {e['P5']:.2f} | {S.show(w)} | {time.time()-t0:.0f}s", flush=True)
    return w
if __name__ == "__main__":
    BK = json.load(open(sys.argv[1])); tag = sys.argv[2]
    w = climb({t: v for t, v in BK[tag].items() if v > 0}, tag)
    json.dump({tag + "_" + OBJ: {t: round(float(w[IDX[t]]), 2) for t in S.held(w)}}, open(sys.argv[3], "w"), indent=1)
    print(f"[{tag}] FINAL {S.show(w)}", flush=True)

# ===== report_v3.py =====
"""report_v3 (v3.4 bundle, engine-exact 60k x 4 conventions, regime sigma), T-generic (50-y basis via ENGINEV2_T).
Usage: python report_v3.py books.json out.csv"""
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

# ===== env.sh =====
export ENGINEV2_DATA=/home/claude/engine/data DECK_FILE=Deck_R8n_20261024.csv SCEN_FILE=scenarios_25_R8i_20260909.csv GRADES_FILE=/home/claude/engine/data/decks/Row_Grades_R8j_20260909.csv ENGINEV2_PANEL=/home/claude/engine/data/panel_stk.csv ENGINEV2_END=2026-08 ENGINEV2_T=50 NPATHS=2000 STRESS_GATE=1 ETF_CAP=0.20 ENGINEV2_EXTRA=RSST,RSSB,NTSX,GDE,DBMFh,RSSTh PYTHONWARNINGS=ignore

# ===== seeds.json =====
{"STK_CE3": {"GLDM":0.19,"RSST":0.18,"SHLD":0.14,"DBMF":0.12,"CIBR":0.08,"KMLM":0.08,"CME":0.05,"MSCI":0.05,"BRKB":0.05,"NUKZ":0.02,"URNM":0.02,"FNV":0.02},
 "C5_SMH3b": {"GLDM":0.15,"SHLD":0.15,"CIBR":0.15,"DBMF":0.15,"NUKZ":0.08,"BRKB":0.05,"CME":0.05,"FNV":0.05,"MSCI":0.05,"URNM":0.05,"XLE":0.03,"XAR":0.02,"SMH":0.02},
 "T15_rel": {"DBMF":0.15,"SHLD":0.14,"CIBR":0.14,"GLDM":0.12,"KMLM":0.12,"URNM":0.06,"CME":0.05,"MSCI":0.05,"BRKB":0.05,"CTA":0.03,"FNV":0.03,"JEPI":0.03,"XLI":0.03}}

# ===== books_r8n.json =====
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
 "STK_CE3": {
  "GLDM": 0.19,
  "RSST": 0.18,
  "SHLD": 0.14,
  "DBMF": 0.12,
  "CIBR": 0.08,
  "KMLM": 0.08,
  "CME": 0.05,
  "MSCI": 0.05,
  "BRKB": 0.05,
  "NUKZ": 0.02,
  "URNM": 0.02,
  "FNV": 0.02
 },
 "R8n_A_cap20": {
  "GLDM": 0.2,
  "SHLD": 0.15,
  "DBMF": 0.2,
  "CIBR": 0.09,
  "CME": 0.05,
  "MSCI": 0.05,
  "SMH": 0.02,
  "KMLM": 0.03,
  "BRKB": 0.05,
  "RSST": 0.16
 },
 "R8n_B_cap20_M1": {
  "GLDM": 0.19,
  "SHLD": 0.14,
  "DBMF": 0.2,
  "NUKZ": 0.02,
  "CIBR": 0.08,
  "CME": 0.05,
  "MSCI": 0.05,
  "SMH": 0.02,
  "KMLM": 0.08,
  "FNV": 0.02,
  "BRKB": 0.05,
  "RSST": 0.1
 },
 "R8n_C_pref15": {
  "GLDM": 0.15,
  "SHLD": 0.14,
  "DBMF": 0.15,
  "CIBR": 0.11,
  "CME": 0.05,
  "MSCI": 0.05,
  "URNM": 0.03,
  "KMLM": 0.12,
  "BRKB": 0.05,
  "JEPI": 0.03,
  "RSST": 0.12
 },
 "R8n_D_pref15_M1": {
  "GLDM": 0.15,
  "SHLD": 0.14,
  "DBMF": 0.15,
  "CIBR": 0.14,
  "CTA": 0.03,
  "CME": 0.05,
  "MSCI": 0.05,
  "URNM": 0.04,
  "KMLM": 0.12,
  "BRKB": 0.05,
  "JEPI": 0.03,
  "RSST": 0.05
 }
}
