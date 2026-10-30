"""Shared library: EngineV2 reproduction + scenario metrics + vol pipeline.

Conventions locked to EngineV2_Rebuild_20260829.py (assumption register AR1-AR7)
and rcew40.py / KEEP_DRB40_Definition_2026-06-01.md / metrics_r3af.py.
"""
import numpy as np, pandas as pd, os

WORK = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ENGINEV2_DATA", os.path.join(WORK, "data"))
DECKS = os.path.join(DATA, "decks")
DECK = pd.read_csv(os.path.join(DECKS, os.environ.get("DECK_FILE", "Deck_Recal_R8cand_20260903.csv")))
DECK["Ticker"] = DECK["Ticker"].str.strip()
DECK = DECK.set_index("Ticker")
SC = list(DECK.columns)                      # 20 scenario names, canonical order
SCEN = pd.read_csv(os.path.join(DECKS, os.environ.get("SCEN_FILE", "scenarios_24_R6_20260831.csv")))
W_CAN = SCEN["Weight_Normalized"].values
SEV = SCEN["Severity_1to5"].values.astype(float)
NSC = len(SC)
assert NSC == len(W_CAN), (NSC, len(W_CAN))
_lamf = os.environ.get("ENGINEV2_LAM_FILE")
_lam0 = float(os.environ.get('ENGINEV2_LAM', '0.125'))
LAMV = np.full(NSC, _lam0)
if _lamf:
    for k, v in __import__("json").load(open(_lamf)).items():
        i = int(k) - 1 if str(k).isdigit() else SC.index(k)
        LAMV[i] = float(v)

# AR4 benchmark-only IAI proxy
DECK.loc["IAI"] = DECK.loc[["CME", "CBOE", "NDAQ"]].mean()

P1 = {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.15,'CIBR':.10,'GRID':.10,
      'CTA':.05,'CME':.05,'JEDI':.05,'MSCI':.05}
EVG11 = {'CIBR':.15,'SHLD':.15,'CTA':.10,'DBMF':.10,'GLDM':.10,'IAI':.10,
         'NUKZ':.10,'QTUM':.10,'XAR':.10}

# Expense ratios, percent/yr. Baseline block = engine AR1 verbatim (continuity).
# Additions: prospectus values to be re-verified for any final holding.
ER = {
 'GLDM':0.10,'SHLD':0.50,'DBMF':0.85,'NUKZ':0.85,'CIBR':0.59,'GRID':0.65,
 'CTA':0.76,'CME':0.00,'JEDI':0.50,'MSCI':0.00,
 'SMH':0.35,'ROKT':0.45,'XAR':0.35,'URNM':0.75,'CEG':0.00,'SETM':0.65,
 'PPA':0.64,'ITA':0.40,'EUAD':0.75,'PAVE':0.47,'QTUM':0.40,'KMLM':0.90,
 'IEF':0.15,'CBOE':0.00,'ICE':0.00,'SPGI':0.00,'NDAQ':0.00,'NLR':0.61,
 'XSD':0.35,'SOXX':0.35,'QUAL':0.15,'COWZ':0.49,'AVUV':0.25,'XLE':0.09,
 'HACK':0.60,'IVV':0.03,'MOAT':0.46,'XLV':0.09,'XBI':0.35,'TLT':0.15,'GOVZ':0.10,'IAI':0.42,
 'LTPZ':0.20,'SCHP':0.03,'VTIP':0.04,'GDX':0.51,'WPM':0.00,'FNV':0.00,'PSLV':0.60,'COPX':0.65,
 'BRKB':0.00,'WM':0.00,'INDA':0.62,'JEPI':0.35,'SPMO':0.13,
 'V':0.00,'IGV':0.41,'FTEC':0.084,'COIN':0.00,'EMXC':0.25,'EWT':0.59,'SIL':0.65,'VTV':0.04,'SCHG':0.04,'IBB':0.45,
}

NP_, T, LAM, DRAW = 60000, int(os.environ.get('ENGINEV2_T', '40')), float(os.environ.get('ENGINEV2_LAM', '0.125')), 0.03   # LAM: annual regime-redraw probability (A60 override)

import json as _json

# A65: scenario-specific regime persistence. ENGINEV2_LAM_FILE = JSON {scenario_id(1-based) or name: lambda}; others = LAM.
# Exit probability per scenario LAMV[s]; on exit the next scenario is drawn from q ∝ pi·LAMV, which keeps pi as the
# long-run occupancy (time share). Uniform LAMV reproduces the original engine exactly (q is pi itself).
def entry(pi):
    if LAMV is None or np.all(LAMV == LAMV[0]): return pi
    q = np.asarray(pi) * LAMV; return q / q.sum()
_er8 = os.path.join(DECKS, "ER_R8.json")
if os.path.exists(_er8):
    ER.update({k: v for k, v in _json.load(open(_er8)).items() if k not in ER})

def prow(b):
    """Portfolio per-scenario gross CAGR vector (decimal) from deck rows."""
    return np.array([sum(b[t] * DECK.loc[t, s] for t in b) for s in SC]) / 100.0

def wer(b):
    return sum(b[t] * ER[t] for t in b) / 100.0

def make_pi_robust(bench=EVG11):
    evg = prow(bench)
    dsl = np.maximum(0.0, evg[0] - evg)
    st = np.sqrt(W_CAN) * SEV * dsl
    st = st / st.sum()
    pi = 0.5 * W_CAN + 0.25 / NSC + 0.25 * st
    return pi / pi.sum()

def stress_weights(bench=EVG11):
    """Stress leg alone (for RCEW/DRB robust blends)."""
    evg = prow(bench)
    dsl = np.maximum(0.0, evg[0] - evg)
    st = np.sqrt(W_CAN) * SEV * dsl
    return st / st.sum()

PI_ROBUST = make_pi_robust()
if os.environ.get('PI_ROBUST_FILE'):
    PI_ROBUST = np.loadtxt(os.environ['PI_ROBUST_FILE'], delimiter=','); PI_ROBUST = PI_ROBUST / PI_ROBUST.sum()
PI_CANON = W_CAN / W_CAN.sum()
ST_W = stress_weights()

# ---------------- engine-exact Monte Carlo (AR3/AR7) ----------------
def run_engine(rng, r_g, pi, conv, sigma, wer_):
    """Verbatim port of EngineV2 run(); shared rng => exact stream replication."""
    rn = r_g - wer_
    s = rng.choice(len(pi), size=NP_, p=pi)
    W = np.ones(NP_); ru = np.zeros(NP_, bool)
    for _ in range(T):
        j = rng.random(NP_) < LAMV[s]
        if j.any():
            s[j] = rng.choice(len(pi), size=int(j.sum()), p=entry(pi))
        mu = np.log1p(rn[s]) - (sigma * sigma / 2 if conv == "G" else 0.0)
        W = W * np.exp(mu + sigma * rng.standard_normal(NP_)) - DRAW
        ru |= (W <= 0)
        W = np.maximum(W, 1e-9)
    lg = np.log(W)
    return dict(PathCE=float(np.exp(lg.mean())), P50=float(np.median(W)),
                P10=float(np.quantile(W, .10)), P5=float(np.quantile(W, .05)),
                ruin=int(ru.sum()))

# ---------------- fast CRN evaluator for search ----------------
class FastEval:
    """Precompute regime paths per pi and shocks once; evaluate many portfolios
    with common random numbers. Same distributions as engine, different stream."""
    def __init__(self, npaths=24000, seed=777):
        rng = np.random.default_rng(seed)
        self.npaths = npaths
        self.Z = rng.standard_normal((npaths, T))
        self.S = {}
        for name, pi in (("robust", PI_ROBUST), ("canonical", PI_CANON)):
            s = np.empty((npaths, T), dtype=np.int8)
            cur = rng.choice(len(pi), size=npaths, p=pi)
            for t in range(T):
                j = rng.random(npaths) < LAMV[cur]
                nj = int(j.sum())
                if nj:
                    cur = cur.copy()
                    cur[j] = rng.choice(len(pi), size=nj, p=entry(pi))
                s[:, t] = cur
            self.S[name] = s

    def run(self, r_net, pi_name, conv, sigma):
        s = self.S[pi_name]
        lr = np.log1p(r_net)          # 20-vector
        mu = lr[s]                    # [npaths,T]
        if conv == "G":
            mu = mu - sigma * sigma / 2
        W = np.ones(self.npaths); ru = np.zeros(self.npaths, bool)
        X = np.exp(mu + sigma * self.Z)
        for t in range(T):
            W = W * X[:, t] - DRAW
            ru |= (W <= 0)
            W = np.maximum(W, 1e-9)
        lg = np.log(W)
        return dict(PathCE=float(np.exp(lg.mean())), P50=float(np.median(W)),
                    P10=float(np.quantile(W, .10)), P5=float(np.quantile(W, .05)),
                    ruin=int(ru.sum()))

# ---------------- closed-form scenario metrics ----------------
def tw_after_draw(g, T_=None, d=0.03):
    """Terminal multiple after a fixed draw on initial corpus; horizon defaults to the engine T (A58/A60: was hard-coded 40)."""
    T_ = T if T_ is None else T_
    g = np.asarray(g, dtype=float)
    growth = (1.0 + g) ** T_
    out = np.where(np.abs(g) < 1e-15, 1.0 - d * T_, growth - d * (growth - 1.0) / np.where(np.abs(g) < 1e-15, 1.0, g))
    return out

def log_cew(tm, w, eps=1e-12):
    w = np.asarray(w) / np.sum(w)
    return float(np.exp(np.sum(w * np.log(np.maximum(tm, eps)))))

def rcew40(r_net):
    """Robust CEW on the engine horizon (RCEW50 at T=50). Diagnostic only since 2026-10-29 (retired as a ranking input)."""
    tm = tw_after_draw(r_net)
    c = log_cew(tm, W_CAN); e = log_cew(tm, np.ones(NSC) / NSC); s = log_cew(tm, ST_W)
    robust = float(np.exp(0.50 * np.log(max(c, 1e-12)) + 0.25 * np.log(max(e, 1e-12)) + 0.25 * np.log(max(s, 1e-12))))
    return dict(canonical=c, equal=e, stress=s, robust=robust)

def drb40(r_net, h=0.06, eps=1e-9):   # ratified hurdle 6% (2026-08-31); horizon = engine T
    tm = np.maximum(tw_after_draw(r_net), eps)
    twh = tw_after_draw(np.array([h]))[0]
    gap = np.maximum(0.0, np.log(twh / tm))
    def leg(w):
        w = np.asarray(w) / np.sum(w)
        return float(np.sqrt(np.sum(w * gap ** 2)))
    return 0.50 * leg(W_CAN) + 0.25 * leg(np.ones(NSC) / NSC) + 0.25 * leg(ST_W)

def r3af(r_net, bench_net):
    """metrics_r3af.py logic, canonical weights, q 0.20/0.80."""
    w = W_CAN.copy(); r = np.asarray(r_net); b = np.asarray(bench_net)
    order = np.argsort(b); xs = b[order]; ws = w[order]
    cw = np.cumsum(ws); cw = cw / cw[-1]
    ql = float(np.interp(0.20, cw, xs)); qh = float(np.interp(0.80, cw, xs))
    bad = b <= ql; good = b >= qh; mid = ~(bad | good)
    wm = lambda x, m: float(np.sum(w[m] * x[m]) / np.sum(w[m])) if np.sum(w[m]) > 0 else np.nan
    mu_bad_r, mu_mid_r, mu_good_r = wm(r, bad), wm(r, mid), wm(r, good)
    mu_bad_b = wm(b, bad)
    tail_alpha = mu_bad_r - mu_bad_b
    convexity = (mu_good_r - mu_mid_r) - (mu_mid_r - mu_bad_r)
    bw = np.sum(w[bad] * b[bad]) / np.sum(w[bad]); rw = np.sum(w[bad] * r[bad]) / np.sum(w[bad])
    vb = np.sum(w[bad] * (b[bad] - bw) ** 2)
    beta = np.sum(w[bad] * (b[bad] - bw) * (r[bad] - rw)) / vb if vb > 1e-18 else np.nan
    ortho = 1.0 - min(2.0, abs(beta)) if np.isfinite(beta) else np.nan
    return 0.55 * tail_alpha + 0.25 * convexity + 0.20 * ortho

# ---------------- historical vol / Sortino pipeline ----------------
_MRET = None
def monthly_returns():
    """Monthly total-return panel (period index, one column per ticker).

    Source, in order: ENGINEV2_PANEL (a monthly panel CSV such as etf-eod's
    monthly_returns.csv or the packaged canonical panel) -> ENGINEV2_PRICES (a
    directory of daily files with Date/Adj_Close, e.g. etf-eod's prices/; month-end
    last observation, optionally truncated at ENGINEV2_ASOF=YYYY-MM-DD) -> the
    packaged canonical panel data/panel/monthly_returns_canonical_20260821.csv,
    which reproduces every reference metric to the digit.
    """
    global _MRET
    if _MRET is not None:
        return _MRET
    panel = os.environ.get("ENGINEV2_PANEL")
    pdir = os.environ.get("ENGINEV2_PRICES")
    if panel is None and pdir is None:
        panel = os.path.join(DATA, "panel", "monthly_returns_canonical_20260821.csv")
    if panel is not None:
        df = pd.read_csv(panel, index_col=0)
        df.columns = [c.replace("-", "") for c in df.columns]      # etf-eod uses BRK-B; house key is BRKB
        df.index = pd.PeriodIndex(df.index.astype(str), freq="M")
        _MRET = df.sort_index()
        return _MRET
    asof = os.environ.get("ENGINEV2_ASOF")
    out = {}
    for f in sorted(os.listdir(pdir)):
        if not f.endswith(".csv") or f == "manifest.csv":
            continue
        t = f[:-4]
        df = pd.read_csv(os.path.join(pdir, f), usecols=["Date", "Adj_Close"])
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        if asof:
            df = df[df["Date"] <= pd.Timestamp(asof + " 23:59", tz="UTC")]
        df = df.set_index("Date").sort_index()
        m = df["Adj_Close"].resample("ME").last()
        out[t] = m.pct_change()
    _MRET = pd.DataFrame(out)
    _MRET.index = _MRET.index.to_period("M")
    return _MRET

def port_vol48(weights, end=None, nmonths=48, ddof=1):
    """Annualized vol of portfolio monthly returns, live-weight renormalized (AR2).
    end defaults to ENGINEV2_END (2026-08), the last month of the 48-month window."""
    mr = monthly_returns()
    end_p = pd.Period(end or os.environ.get("ENGINEV2_END", "2026-08"), "M")
    idx = pd.period_range(end_p - (nmonths - 1), end_p, freq="M")
    sub = mr.reindex(idx)[list(weights.keys())]
    wv = np.array([weights[t] for t in sub.columns])
    live = sub.notna().values
    Wm = live * wv
    Wm = Wm / Wm.sum(axis=1, keepdims=True)
    pr = np.nansum(sub.values * Wm, axis=1)
    return float(np.std(pr, ddof=ddof) * np.sqrt(12)), pr, idx

def hist_diagnostics(weights, end=None, nmonths=48):
    v, pr, idx = port_vol48(weights, end, nmonths)
    mean_m = pr.mean()
    downside = pr[pr < 0]
    dd = np.sqrt(np.mean(downside ** 2)) if len(downside) else np.nan
    sortino = (mean_m * 12) / (dd * np.sqrt(12)) if dd and dd > 0 else np.nan
    cum = np.cumprod(1 + pr)
    peak = np.maximum.accumulate(cum)
    maxdd = float((cum / peak - 1).min())
    cagr = float(cum[-1] ** (12 / len(pr)) - 1)
    return dict(vol=v, sortino=float(sortino), maxdd=maxdd, cagr48=cagr)

def evaluate_closed_form(weights, bench=EVG11):
    r_net = prow(weights) - wer(weights)
    b_net = prow(bench) - wer_evg11()
    rc = rcew40(r_net)
    return dict(
        WER=wer(weights) * 100,
        SWCAGR=float(np.sum(W_CAN * r_net)) * 100,
        worst=float(r_net.min()) * 100,
        worst_scen=SC[int(np.argmin(r_net))],
        RCEW40_robust=rc["robust"], RCEW40_canonical=rc["canonical"],
        RCEW40_stress=rc["stress"],
        DRB40=drb40(r_net), R3AF=r3af(r_net, b_net),
    )

def wer_evg11():
    er_evg = dict(ER); er_evg.setdefault("IAI", 0.33)
    return sum(EVG11[t] * er_evg.get(t, 0.4) for t in EVG11) / 100.0

rcew, drb = rcew40, drb40   # horizon-neutral names
