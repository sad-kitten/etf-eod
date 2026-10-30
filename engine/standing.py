"""standing (v3.4, verbatim from R8n_Code_20261024.py standing.py section; T-generic CAGR).
2026-10-26 (Addendum 57): SINGLES_MAX env override (owner dropped the 3-singles count cap 2026-10-26)."""
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
    if np.any(w[held] < float(os.environ.get('MIN_HOLD', '0.02')) - 1e-9): return False
    if w.max() > ETF_CAP + 1e-9: return False
    from core import PMEQ, SEMIS, DEF, NUC, BAL
    _bc = float(os.environ.get('BRKB_CAP', '0.05'))   # 2026-10-26 owner: BRKB up to 15%, treated like a fund
    for t in SINGLES:
        if t in IDX and w[IDX[t]] > (_bc if t == 'BRKB' else 0.05) + 1e-9: return False
    ssum = lambda ts: sum(w[IDX[t]] for t in ts if t in IDX)
    if ssum([t for t in SINGLES if not (t == 'BRKB' and _bc > 0.05)]) > float(os.environ.get('SINGLES_TOT', '0.20')) + 1e-9: return False
    if ssum(PMEQ) > 0.10 + 1e-9: return False
    if ssum(SEMIS) > 0.12 + 1e-9: return False
    if ssum(DEF) > 0.35 + 1e-9: return False
    if ssum(NUC) > 0.35 + 1e-9: return False
    if ssum(BAL) < 0.20 - 1e-9: return False
    return True

RS_NUC, RS_NAME, RS_BLOCK = 0.38, 0.24, 0.35
CW_CAP, SINGLES_MAX, DUP, SHORT_CAP, SHORT_MONTHS, BALLAST_FLOOR = float(os.environ.get("CW_CAP", "0.20")), int(os.environ.get("SINGLES_MAX", "3")), 0.03, 0.30, 60, 0.09
BLOCKS = {"nuclear": ["NUKZ", "URNM", "CEG", "SETM", "NLR", "URA"],
          "defence": ["SHLD", "XAR", "PPA", "ITA", "EUAD", "ROKT", "JEDI"],
          "gold":    ["GLDM", "FNV", "PSLV", "GDX", "WPM", "SIL", "GDE"],
          "cyber":   ["CIBR", "HACK"],
          "semis":   ["SMH", "SOXX", "XSD", "XLK", "FTEC"]}
BALLAST = ["DBMF", "CTA", "KMLM", "RSST"]
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
    if PREF:
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
STACK_STRESS = {"RSST": dict(trend_sev45=1.0), "GDE": dict(gold_all=0.9)}
STRESS_FLOOR = float(os.environ.get("STRESS_FLOOR", "0.035"))
STRESS_GATE = os.environ.get("STRESS_GATE", "0") == "1"
def stressed_deck(M=None):
    M = (DECKM if M is None else M).copy(); sev45 = L.SEV >= 4
    if os.environ.get("STRESS_SHIFT_FILE"):   # A65 gates v4: G2 stress deck = explicit shift file (all / sev45 pp)
        import json as _j
        for t, v in _j.load(open(os.environ["STRESS_SHIFT_FILE"])).items():
            if t not in IDX: continue
            if "all" in v: M[IDX[t]] += v["all"] / 100.0
            if "sev45" in v: M[IDX[t], sev45] += v["sev45"] / 100.0
        return M
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

GATES = dict(ruin=0, P5=1.00, worst=float(os.environ.get("WORST_GATE", "0.0525")), R3AF=float(os.environ.get("R3AF_GATE", "0.05")), LS_ruin=0)   # A65: env (gates v4: worst 4.27)
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
