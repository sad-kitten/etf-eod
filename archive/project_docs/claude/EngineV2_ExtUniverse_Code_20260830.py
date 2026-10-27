"""EngineV2 Extended-Universe Optimization — consolidated code archive (2026-08-30).
Ran as three modules: lib_pathce.py, core.py, final_verify.py (sections below).
Data: Deck_Recal_R1R4_20260822.csv, scenarios_20_canonical.csv (project docs),
per-ticker daily price CSVs (project files) in ./prices/.
"""

# ======== SECTION 1: lib_pathce.py ========
"""Shared library: EngineV2 reproduction + scenario metrics + vol pipeline.

Conventions locked to EngineV2_Rebuild_20260829.py (assumption register AR1-AR7)
and rcew40.py / KEEP_DRB40_Definition_2026-06-01.md / metrics_r3af.py.
"""
import numpy as np, pandas as pd, os

WORK = os.path.dirname(os.path.abspath(__file__))
DECK = pd.read_csv(os.path.join(WORK, "Deck_Recal_R1R4_20260822.csv"))
DECK["Ticker"] = DECK["Ticker"].str.strip()
DECK = DECK.set_index("Ticker")
SC = list(DECK.columns)                      # 20 scenario names, canonical order
SCEN = pd.read_csv(os.path.join(WORK, "scenarios_20_canonical.csv"))
W_CAN = SCEN["Weight_Normalized"].values
SEV = SCEN["Severity_1to5"].values.astype(float)

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
 'HACK':0.60,'IVV':0.03,'MOAT':0.46,
}

NP_, T, LAM, DRAW = 60000, 40, 0.125, 0.03

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
    pi = 0.5 * W_CAN + 0.25 / 20 + 0.25 * st
    return pi / pi.sum()

def stress_weights(bench=EVG11):
    """Stress leg alone (for RCEW/DRB robust blends)."""
    evg = prow(bench)
    dsl = np.maximum(0.0, evg[0] - evg)
    st = np.sqrt(W_CAN) * SEV * dsl
    return st / st.sum()

PI_ROBUST = make_pi_robust()
PI_CANON = W_CAN / W_CAN.sum()
ST_W = stress_weights()

# ---------------- engine-exact Monte Carlo (AR3/AR7) ----------------
def run_engine(rng, r_g, pi, conv, sigma, wer_):
    """Verbatim port of EngineV2 run(); shared rng => exact stream replication."""
    rn = r_g - wer_
    s = rng.choice(20, size=NP_, p=pi)
    W = np.ones(NP_); ru = np.zeros(NP_, bool)
    for _ in range(T):
        j = rng.random(NP_) < LAM
        if j.any():
            s[j] = rng.choice(20, size=int(j.sum()), p=pi)
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
            cur = rng.choice(20, size=npaths, p=pi)
            for t in range(T):
                j = rng.random(npaths) < LAM
                nj = int(j.sum())
                if nj:
                    cur = cur.copy()
                    cur[j] = rng.choice(20, size=nj, p=pi)
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
def tw_after_draw(g, T_=40, d=0.03):
    g = np.asarray(g, dtype=float)
    growth = (1.0 + g) ** T_
    out = np.where(np.abs(g) < 1e-15, 1.0 - d * T_, growth - d * (growth - 1.0) / np.where(np.abs(g) < 1e-15, 1.0, g))
    return out

def log_cew(tm, w, eps=1e-12):
    w = np.asarray(w) / np.sum(w)
    return float(np.exp(np.sum(w * np.log(np.maximum(tm, eps)))))

def rcew40(r_net):
    tm = tw_after_draw(r_net)
    c = log_cew(tm, W_CAN); e = log_cew(tm, np.ones(20) / 20); s = log_cew(tm, ST_W)
    robust = float(np.exp(0.50 * np.log(max(c, 1e-12)) + 0.25 * np.log(max(e, 1e-12)) + 0.25 * np.log(max(s, 1e-12))))
    return dict(canonical=c, equal=e, stress=s, robust=robust)

def drb40(r_net, h=0.05, eps=1e-9):
    tm = np.maximum(tw_after_draw(r_net), eps)
    twh = tw_after_draw(np.array([h]))[0]
    gap = np.maximum(0.0, np.log(twh / tm))
    def leg(w):
        w = np.asarray(w) / np.sum(w)
        return float(np.sqrt(np.sum(w * gap ** 2)))
    return 0.50 * leg(W_CAN) + 0.25 * leg(np.ones(20) / 20) + 0.25 * leg(ST_W)

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
    """Monthly total-return series per ticker from Adj_Close (month-end last obs)."""
    global _MRET
    if _MRET is not None:
        return _MRET
    out = {}
    pdir = os.path.join(WORK, "prices")
    for f in sorted(os.listdir(pdir)):
        if not f.endswith(".csv") or f == "manifest.csv":
            continue
        t = f[:-4]
        df = pd.read_csv(os.path.join(pdir, f), usecols=["Date", "Adj_Close"])
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.set_index("Date").sort_index()
        m = df["Adj_Close"].resample("ME").last()
        out[t] = m.pct_change()
    _MRET = pd.DataFrame(out)
    _MRET.index = _MRET.index.to_period("M")
    return _MRET

def port_vol48(weights, end="2026-08", nmonths=48, ddof=1):
    """Annualized vol of portfolio monthly returns, live-weight renormalized (AR2)."""
    mr = monthly_returns()
    end_p = pd.Period(end, "M")
    idx = pd.period_range(end_p - (nmonths - 1), end_p, freq="M")
    sub = mr.reindex(idx)[list(weights.keys())]
    wv = np.array([weights[t] for t in sub.columns])
    live = sub.notna().values
    Wm = live * wv
    Wm = Wm / Wm.sum(axis=1, keepdims=True)
    pr = np.nansum(sub.values * Wm, axis=1)
    return float(np.std(pr, ddof=ddof) * np.sqrt(12)), pr, idx

def hist_diagnostics(weights, end="2026-08", nmonths=48):
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

# ======== SECTION 2: core.py ========
"""Shared optimization core (no search executed at import)."""
import numpy as np, pandas as pd
import lib_pathce as L

UNIV = ['GLDM','SHLD','DBMF','NUKZ','CIBR','GRID','CTA','CME','JEDI','MSCI',
        'SMH','ROKT','XAR','URNM','CEG','SETM','PPA','ITA','EUAD','PAVE',
        'QTUM','KMLM','IEF','CBOE','ICE','SPGI','NDAQ','NLR','XSD','SOXX',
        'QUAL','COWZ','AVUV','XLE','HACK','MOAT']
N = len(UNIV)
IDX = {t:i for i,t in enumerate(UNIV)}
SINGLES = ['CME','MSCI','CEG','CBOE','ICE','SPGI','NDAQ']
SEMIS = ['SMH','SOXX','XSD']
DEF = ['SHLD','XAR','PPA','ITA','EUAD','ROKT','JEDI']
NUC = ['NUKZ','NLR','URNM','CEG','SETM','GRID']
BAL = ['GLDM','DBMF','CTA','KMLM','IEF']

DECKM = np.array([[L.DECK.loc[t,s] for s in L.SC] for t in UNIV])/100.0
ERV = np.array([L.ER[t] for t in UNIV])/100.0

mr = L.monthly_returns()
idx48 = pd.period_range(pd.Period("2026-08","M")-47, pd.Period("2026-08","M"), freq="M")
R48 = mr.reindex(idx48)[UNIV].values
LIVE = ~np.isnan(R48)
R48z = np.nan_to_num(R48)

def sigma_of(w):
    Wm = LIVE * w
    s = Wm.sum(axis=1, keepdims=True)
    pr = (R48z * Wm).sum(axis=1) / s[:,0]
    return float(np.std(pr, ddof=1) * np.sqrt(12))

import os as _os
FE = L.FastEval(npaths=int(_os.environ.get("NPATHS", "14000")), seed=777)
CONVS = [("robust","G"),("robust","L"),("canonical","G"),("canonical","L")]

def eval_fast(w, full=False):
    r_net = DECKM.T @ w - float(ERV @ w)
    sig = sigma_of(w)
    res = {(pin,conv): FE.run(r_net, pin, conv, sig) for pin,conv in CONVS}
    score = float(np.mean([np.log(res[c]["PathCE"]) for c in CONVS]))
    out = dict(score=score, sigma=sig, res=res, r_net=r_net)
    if full:
        rc = L.rcew40(r_net)
        out.update(RCEW=rc["robust"], DRB=L.drb40(r_net),
                   R3AF=L.r3af(r_net, L.prow(L.EVG11)-L.wer_evg11()),
                   worst=float(r_net.min()))
    return out

def wvec(d):
    w = np.zeros(N)
    for t,x in d.items(): w[IDX[t]] = x
    return w/w.sum()

P1v = wvec(L.P1)
base = eval_fast(P1v, full=True)
FLOOR = dict(P10={c: base["res"][c]["P10"] for c in CONVS},
             P5={c: base["res"][c]["P5"] for c in CONVS},
             ruin={c: base["res"][c]["ruin"] for c in CONVS},
             DRB=base["DRB"], worst=base["worst"], RCEW=base["RCEW"], R3AF=base["R3AF"])

def feasible(w):
    if w.min() < -1e-12 or abs(w.sum()-1) > 1e-9: return False
    held = w > 1e-9
    if held.sum() > 14: return False
    if np.any(w[held] < 0.02 - 1e-9): return False
    if w.max() > 0.15 + 1e-9: return False
    for t in SINGLES:
        if w[IDX[t]] > 0.05 + 1e-9: return False
    ssum = lambda ts: sum(w[IDX[t]] for t in ts)
    if ssum(SINGLES) > 0.20 + 1e-9: return False
    if ssum(SEMIS) > 0.12 + 1e-9: return False
    if ssum(DEF) > 0.35 + 1e-9: return False
    if ssum(NUC) > 0.35 + 1e-9: return False
    if ssum(BAL) < 0.20 - 1e-9: return False
    return True

def gates_ok(ev):
    for c in CONVS:
        if ev["res"][c]["P10"] < FLOOR["P10"][c] - 1e-9: return False
        if ev["res"][c]["P5"]  < FLOOR["P5"][c]  - 1e-9: return False
        if ev["res"][c]["ruin"] > FLOOR["ruin"][c]: return False
    return True

def gates_full_ok(ev):
    return (ev["DRB"] <= FLOOR["DRB"] + 1e-12 and ev["worst"] >= FLOOR["worst"] - 1e-12
            and ev["RCEW"] >= FLOOR["RCEW"] - 1e-9 and ev["R3AF"] >= FLOOR["R3AF"] - 0.002)

# ======== SECTION 3: final_verify.py ========
"""Final verification at engine fidelity (60k paths, seed 42, engine-exact
streams mirroring the store: robust G,L on one shared stream; canonical G,L on
a fresh stream — the convention that reproduces V1-V4 to the digit).
Plus closed-form suite, haircut-deck check, historical diagnostics, and
WER/sigma sensitivities. Writes deliverable CSVs to out/.
"""
import numpy as np, pandas as pd, json, sys
import lib_pathce as L
from core import (UNIV, N, IDX, DECKM, ERV, sigma_of, wvec, P1v, CONVS)

# haircut deck (25% shrink of non-incumbent rows toward per-scenario column median)
colmed = np.median(DECKM, axis=0)
HC = DECKM.copy()
P1set = {IDX[t] for t in L.P1}
for i in range(N):
    if i not in P1set:
        HC[i] = 0.75 * DECKM[i] + 0.25 * colmed

def engine_exact(r_g, sigma, wer_):
    out = {}
    rng = np.random.default_rng(42)
    for conv in ("G", "L"):
        out[("robust", conv)] = L.run_engine(rng, r_g, L.PI_ROBUST, conv, sigma, wer_)
    rng = np.random.default_rng(42)
    for conv in ("G", "L"):
        out[("canonical", conv)] = L.run_engine(rng, r_g, L.PI_CANON, conv, sigma, wer_)
    return out

def full_eval(w, label):
    r_g = DECKM.T @ w
    wer_ = float(ERV @ w)
    sig = sigma_of(w)
    mc = engine_exact(r_g, sig, wer_)
    r_net = r_g - wer_
    rc = L.rcew40(r_net)
    weights = {UNIV[i]: float(w[i]) for i in np.where(w > 1e-9)[0]}
    hist = L.hist_diagnostics(weights)
    # haircut-deck MC
    r_g_hc = HC.T @ w
    mc_hc = engine_exact(r_g_hc, sig, wer_)
    # sensitivities on the two declared engine variants (robust G/L), like S3-S8
    sens = {}
    for dw, tag in ((0.001, "WER+0.10"), (-0.001, "WER-0.10")):
        m = engine_exact(r_g, sig, wer_ + dw)
        sens[tag] = {c: m[c]["PathCE"] for c in (("robust","G"),("robust","L"))}
    m = engine_exact(r_g, 0.1236, wer_)
    sens["sigma=12.36"] = {c: m[c]["PathCE"] for c in (("robust","G"),("robust","L"))}
    geo = float(np.exp(np.mean([np.log(mc[c]["PathCE"]) for c in CONVS])))
    geo_hc = float(np.exp(np.mean([np.log(mc_hc[c]["PathCE"]) for c in CONVS])))
    return dict(label=label, weights=weights, WER=wer_*100, sigma=sig*100, mc=mc,
                mc_hc=mc_hc, geo=geo, geo_hc=geo_hc,
                RCEW40=rc["robust"], DRB40=L.drb40(r_net),
                R3AF=L.r3af(r_net, L.prow(L.EVG11) - L.wer_evg11()),
                worst=float(r_net.min())*100, worst_scen=L.SC[int(np.argmin(r_net))],
                SWCAGR=float(np.sum(L.W_CAN * r_net))*100, hist=hist, sens=sens)

def report(fe):
    print(f"\n===== {fe['label']} =====")
    print("  " + "  ".join(f"{k} {v*100:.0f}" for k, v in sorted(fe["weights"].items(), key=lambda kv: -kv[1])))
    print(f"  WER={fe['WER']:.3f}%  vol48={fe['sigma']:.2f}%  spec-robust geo-PathCE={fe['geo']:.2f} (haircut {fe['geo_hc']:.2f})")
    for c in CONVS:
        r = fe["mc"][c]; rh = fe["mc_hc"][c]
        print(f"  {c[0]:9s}-{c[1]}: PathCE={r['PathCE']:6.2f} P50={r['P50']:6.1f} P10={r['P10']:5.2f} P5={r['P5']:5.2f} ruin={r['ruin']:4d} | haircut PathCE={rh['PathCE']:6.2f} P10={rh['P10']:5.2f} ruin={rh['ruin']:4d}")
    print(f"  RCEW40={fe['RCEW40']:.2f}  DRB40={fe['DRB40']:.4f}  R3AF={fe['R3AF']:.4f}  worst={fe['worst']:.2f}% ({fe['worst_scen']})  SWCAGR={fe['SWCAGR']:.2f}%")
    h = fe["hist"]
    print(f"  48m diagnostics: Sortino={h['sortino']:.2f}  vol={h['vol']*100:.2f}%  maxDD={h['maxdd']*100:.1f}%  CAGR48={h['cagr48']*100:.1f}%")
    for tag, d in fe["sens"].items():
        print(f"  sens {tag}: " + ", ".join(f"{c[0][:3]}-{c[1]} {v:.2f}" for c, v in d.items()))

R2_FINAL = {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.13,'CIBR':.09,'GRID':.06,
            'CME':.05,'MSCI':.05,'CTA':.04,'URNM':.04,'XAR':.03,'CEG':.03,'XLE':.03}

if __name__ == "__main__":
    cands = [("P1_baseline", P1v),
             ("R2_evolution", wvec(R2_FINAL)),
             ("R1_maximin", np.load("/home/claude/work/out/final_R1_maximin_H_CBOE_ICE.npy"))]
    results = []
    for tag, w in cands:
        fe = full_eval(w, tag)
        report(fe)
        results.append(fe)
    # deliverable CSVs
    rows = []
    for fe in results:
        row = dict(Portfolio=fe["label"], WER_pct=round(fe["WER"], 3), vol48_pct=round(fe["sigma"], 2),
                   geo_PathCE=round(fe["geo"], 2), geo_PathCE_haircut=round(fe["geo_hc"], 2),
                   RCEW40_robust=round(fe["RCEW40"], 2), DRB40=round(fe["DRB40"], 4),
                   R3AF=round(fe["R3AF"], 4), worst_net_CAGR_pct=round(fe["worst"], 2),
                   worst_scenario=fe["worst_scen"], SWCAGR_pct=round(fe["SWCAGR"], 2),
                   Sortino48=round(fe["hist"]["sortino"], 2), maxDD48_pct=round(fe["hist"]["maxdd"]*100, 1))
        for c in CONVS:
            k = f"{c[0][:3]}{c[1]}"
            row[f"PathCE_{k}"] = round(fe["mc"][c]["PathCE"], 2)
            row[f"P10_{k}"] = round(fe["mc"][c]["P10"], 2)
            row[f"P5_{k}"] = round(fe["mc"][c]["P5"], 2)
            row[f"ruin_{k}"] = fe["mc"][c]["ruin"]
        rows.append(row)
    pd.DataFrame(rows).to_csv("/home/claude/work/out/EngineV2_ExtUniverse_Metrics_20260830.csv", index=False)
    wr = []
    for fe in results:
        for t, v in fe["weights"].items():
            wr.append(dict(Portfolio=fe["label"], Ticker=t, Weight=round(v, 4)))
    pd.DataFrame(wr).to_csv("/home/claude/work/out/EngineV2_ExtUniverse_Weights_20260830.csv", index=False)
    print("\nwrote out/EngineV2_ExtUniverse_Metrics_20260830.csv and _Weights_")

# ======== SECTION 4: qtum_stress.py (2026-08-31) ========
"""QTUM row correction stress + re-optimization + DRB40 h=6%.

Synthetic composition-based QTUM row = 0.50*SMH + 0.35*QQEW + 0.15*ROBO
(semis / established software-tech / speculative sleeve), motivated by
holdings mix (~51% electronic tech, ~29% tech services, small pure-play
quantum sleeve) and 0.91 monthly corr to SMH.
Corrected deck C: QTUM row -> 0.5*deck + 0.5*synthetic. Stress deck S: full synthetic.
"""
import numpy as np, pandas as pd
import lib_pathce as L
from core import (UNIV, N, IDX, DECKM, ERV, sigma_of, wvec, P1v, CONVS, FE,
                  feasible)

syn = (0.50*L.DECK.loc['SMH'] + 0.35*L.DECK.loc['QQEW'] + 0.15*L.DECK.loc['ROBO'])/100.0
qi = IDX['QTUM']
DECK_C = DECKM.copy(); DECK_C[qi] = 0.5*DECKM[qi] + 0.5*syn.values
DECK_S = DECKM.copy(); DECK_S[qi] = syn.values
print("QTUM rows (deck / corrected / synthetic): Taiwan %.2f / %.2f / %.2f ; worst %.2f / %.2f / %.2f"
      % (DECKM[qi][11]*100, DECK_C[qi][11]*100, DECK_S[qi][11]*100,
         DECKM[qi].min()*100, DECK_C[qi].min()*100, DECK_S[qi].min()*100))

colmed_C = np.median(DECK_C, axis=0)
HC_C = DECK_C.copy()
P1set = {IDX[t] for t in L.P1}
for i in range(N):
    if i not in P1set:
        HC_C[i] = 0.75*DECK_C[i] + 0.25*colmed_C

def ev_deck(w, M, full=False):
    r_net = M.T @ w - float(ERV @ w)
    sig = sigma_of(w)
    res = {(p,c): FE.run(r_net, p, c, sig) for p,c in CONVS}
    score = float(np.mean([np.log(res[k]["PathCE"]) for k in CONVS]))
    out = dict(score=score, sigma=sig, res=res, r_net=r_net, worst=float(r_net.min()))
    if full:
        out.update(RCEW=L.rcew40(r_net)["robust"], DRB6=L.drb40(r_net, h=0.06),
                   R3AF=L.r3af(r_net, L.prow(L.EVG11)-L.wer_evg11()))
    return out

base = ev_deck(P1v, DECKM, full=True)
FLOOR = dict(P10={c: base["res"][c]["P10"] for c in CONVS},
             P5={c: base["res"][c]["P5"] for c in CONVS},
             ruin={c: base["res"][c]["ruin"] for c in CONVS},
             worst=base["worst"], RCEW=base["RCEW"], R3AF=base["R3AF"], DRB6=base["DRB6"])

def gates(w, M):
    ev = ev_deck(w, M, full=True)
    fails=[]
    for c in CONVS:
        r=ev["res"][c]
        if r["P10"]<FLOOR["P10"][c]-1e-9: fails.append(f"P10@{c[0][:3]}{c[1]}")
        if r["P5"]<FLOOR["P5"][c]-1e-9: fails.append(f"P5@{c[0][:3]}{c[1]}")
        if r["ruin"]>FLOOR["ruin"][c]: fails.append(f"ruin@{c[0][:3]}{c[1]}")
    if ev["worst"]<FLOOR["worst"]-1e-12: fails.append(f"worst({ev['worst']*100:.2f}<{FLOOR['worst']*100:.2f})")
    if ev["RCEW"]<FLOOR["RCEW"]-1e-9: fails.append("RCEW")
    if ev["R3AF"]<FLOOR["R3AF"]-0.002: fails.append("R3AF")
    if ev["DRB6"]>FLOOR["DRB6"]+1e-12: fails.append(f"DRB6({ev['DRB6']:.4f}>{FLOOR['DRB6']:.4f})")
    return ev, fails

R1 = np.load("out/final_R1_maximin_H_CBOE_ICE.npy")
R2 = wvec({'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.13,'CIBR':.09,'GRID':.06,
           'CME':.05,'MSCI':.05,'CTA':.04,'URNM':.04,'XAR':.03,'CEG':.03,'XLE':.03})

print("\nDRB40@6%% baselines: P1=%.4f" % base["DRB6"])
for name,w in (("R2",R2),("R1",R1)):
    for tag,M in (("deck",DECKM),("corrected",DECK_C),("synthetic",DECK_S)):
        ev,fails = gates(w,M)
        print(f"{name}@{tag:9s}: geoCE={np.exp(ev['score']):.2f} worst={ev['worst']*100:.2f}% DRB6={ev['DRB6']:.4f} -> {'PASS' if not fails else 'FAIL: '+', '.join(fails)}")

# --- re-polish an aggressive variant under the CORRECTED deck (maximin with its haircut) ---
rng = np.random.default_rng(11)
def gates_pair(w):
    ev, f1 = gates(w, DECK_C)
    if f1: return None
    eh, f2 = gates(w, HC_C)
    if f2: return None
    return ev, eh

def polish(w0, iters=50, sample=32):
    g = gates_pair(w0)
    if g is None: return None
    ev, eh = g
    best = (min(ev["score"], eh["score"]), w0.copy(), ev, eh)
    for it in range(iters):
        cur_s, cur_w, _, _ = best
        src = np.where(cur_w > 1e-9)[0]
        improved=False
        for _ in range(sample):
            i=int(rng.choice(src)); j=int(rng.integers(N))
            if i==j: continue
            wn=cur_w.copy(); take=min(0.01 if rng.random()<0.8 else 0.02, wn[i])
            if wn[i]-take<0.02-1e-9 and wn[i]-take>1e-9: take=wn[i]-0.02
            if take<=1e-9: continue
            wn[i]-=take
            if wn[i]<1e-9: wn[i]=0
            add=take
            if wn[j]<1e-9 and add<0.02-1e-9:
                extra=0.02-add
                if wn[i]>=extra+0.02 or abs(wn[i]-extra)<1e-9:
                    wn[i]-=extra; add=0.02
                    if wn[i]<1e-9: wn[i]=0
                else: continue
            wn[j]+=add
            if not feasible(wn): continue
            q = ev_deck(wn, DECK_C)
            if q["score"] <= best[0]+1e-5: continue
            g2 = gates_pair(wn)
            if g2 is None: continue
            s2 = min(g2[0]["score"], g2[1]["score"])
            if s2 > best[0]+1e-5:
                best=(s2, wn.copy(), g2[0], g2[1]); improved=True
        if not improved and it>10: break
    return best

out=[]
for name,w0 in (("R1seed",R1),("R2seed",R2)):
    r = polish(w0)
    if r: out.append((r[0],name)+r[1:])
    print(f"polish from {name}: {np.exp(r[0]) if r else 'seed fails corrected-deck gates'}")
out.sort(key=lambda x:-x[0])
if out:
    sc,name,w,ev,eh = out[0]
    hold={UNIV[i]: round(float(w[i])*100,1) for i in np.where(w>1e-9)[0]}
    print(f"\n[R1b from {name}] maximin-geoCE={np.exp(sc):.2f} (corrected {np.exp(ev['score']):.2f} / haircut {np.exp(eh['score']):.2f}) sigma={ev['sigma']*100:.2f}%")
    print("  ", dict(sorted(hold.items(), key=lambda kv:-kv[1])))
    # cross-check under original deck and full-synthetic deck
    for tag,M in (("orig-deck",DECKM),("synthetic",DECK_S)):
        e2,f2 = gates(w,M)
        print(f"   under {tag}: geoCE={np.exp(e2['score']):.2f} worst={e2['worst']*100:.2f}% -> {'PASS' if not f2 else 'FAIL: '+', '.join(f2)}")
    np.save("out/final_R1b_corrected.npy", w)

# ======== SECTION 5: final_r5.py (2026-08-31) ========
"""Final R5-deck optimization: floors under R5, haircut(R5), S-A (URNM/CEG 50%->median,
like-for-like), S-B (power complex 25%->median, like-for-like). XLV/XBI available.
Objective: maximin(score@R5, score@haircut). Run with DECK_FILE=Deck_Recal_R5_20260831.csv.
"""
import numpy as np
import lib_pathce as L
from core import (UNIV, N, IDX, DECKM, ERV, sigma_of, wvec, P1v, CONVS, FE, feasible)

colmed = np.median(DECKM, axis=0)
HC = DECKM.copy()
P1set = {IDX[t] for t in L.P1}
for i in range(N):
    if i not in P1set:
        HC[i] = 0.75*DECKM[i] + 0.25*colmed
SA = np.load("out/SA.npy"); SB = np.load("out/SB.npy")

def ev(w, M, full=False):
    r_net = M.T @ w - float(ERV @ w)
    sig = sigma_of(w)
    res = {(p,c): FE.run(r_net, p, c, sig) for p,c in CONVS}
    score = float(np.mean([np.log(res[k]["PathCE"]) for k in CONVS]))
    out = dict(score=score, sigma=sig, res=res, r_net=r_net, worst=float(r_net.min()))
    if full:
        out.update(RCEW=L.rcew40(r_net)["robust"], DRB6=L.drb40(r_net, h=0.06),
                   R3AF=L.r3af(r_net, L.prow(L.EVG11)-L.wer_evg11()))
    return out

BASES = {"R5": ev(P1v, DECKM, True), "SA": ev(P1v, SA, True), "SB": ev(P1v, SB, True)}

def fails_vs(e, b, tol_r3af=0.002):
    fails=[]
    for c in CONVS:
        if e["res"][c]["P10"] < b["res"][c]["P10"]-1e-9: fails.append(f"P10{c[1]}")
        if e["res"][c]["P5"]  < b["res"][c]["P5"] -1e-9: fails.append(f"P5{c[1]}")
        if e["res"][c]["ruin"] > b["res"][c]["ruin"]: fails.append(f"ruin@{c[0][:3]}{c[1]}")
    if e["worst"]<b["worst"]-1e-12: fails.append("worst")
    if e["RCEW"]<b["RCEW"]-1e-9: fails.append("RCEW")
    if e["R3AF"]<b["R3AF"]-tol_r3af: fails.append("R3AF")
    if e["DRB6"]>b["DRB6"]+1e-12: fails.append("DRB6")
    return fails

def gates_all(w):
    e1 = ev(w, DECKM, True)
    if fails_vs(e1, BASES["R5"]): return None
    e2 = ev(w, HC, True)
    if fails_vs(e2, BASES["R5"]): return None
    e3 = ev(w, SA, True)
    if fails_vs(e3, BASES["SA"]): return None
    e4 = ev(w, SB, True)
    if fails_vs(e4, BASES["SB"]): return None
    return e1, e2

rng = np.random.default_rng(31)
def polish(w0, iters=45, sample=30):
    g = gates_all(w0)
    if g is None:
        print("  seed fails gates"); return None
    best = (min(g[0]["score"], g[1]["score"]), w0.copy(), g[0], g[1])
    for it in range(iters):
        cur_s, cur_w, _, _ = best
        src = np.where(cur_w>1e-9)[0]; improved=False
        for _ in range(sample):
            i=int(rng.choice(src)); j=int(rng.integers(N))
            if i==j: continue
            wn=cur_w.copy(); take=min(0.01 if rng.random()<0.8 else 0.02, wn[i])
            if wn[i]-take<0.02-1e-9 and wn[i]-take>1e-9: take=wn[i]-0.02
            if take<=1e-9: continue
            wn[i]-=take
            if wn[i]<1e-9: wn[i]=0
            add=take
            if wn[j]<1e-9 and add<0.02-1e-9:
                extra=0.02-add
                if wn[i]>=extra+0.02 or abs(wn[i]-extra)<1e-9:
                    wn[i]-=extra; add=0.02
                    if wn[i]<1e-9: wn[i]=0
                else: continue
            wn[j]+=add
            if not feasible(wn): continue
            q = ev(wn, DECKM)
            if q["score"] <= best[0]+1e-5: continue
            g2 = gates_all(wn)
            if g2 is None: continue
            s2 = min(g2[0]["score"], g2[1]["score"])
            if s2 > best[0]+1e-5:
                best = (s2, wn.copy(), g2[0], g2[1]); improved=True
        if not improved and it>10: break
    return best

R2 = np.load("out/R2v.npy"); R1b = np.load("out/R1b38.npy")
def tweak(w, changes):
    wn = w.copy()
    for t, dv in changes.items(): wn[IDX[t]] += dv
    return wn
seeds = [
  ("R2_trim", tweak(R2, {'URNM':-0.02,'CEG':-0.01,'XAR':+0.01,'CTA':+0.01,'CIBR':+0.01})),
  ("R1b_trim", tweak(R1b, {'URNM':-0.02,'CEG':-0.02,'DBMF':+0.01,'XAR':+0.02,'CTA':+0.01})),
  ("R2_raw", R2), ("R1b_raw", R1b),
]
out=[]
for name, w0 in seeds:
    print(f"seed {name}:")
    r = polish(w0)
    if r:
        out.append((r[0], name)+r[1:])
        print(f"  -> maximin {np.exp(r[0]):.3f}")
out.sort(key=lambda x:-x[0])
for sc,name,w,e1,e2 in out[:3]:
    hold={UNIV[i]: round(float(w[i])*100,1) for i in np.where(w>1e-9)[0]}
    to=50*np.abs(w-P1v).sum()
    print(f"\n[{name}] maximin={np.exp(sc):.2f} (R5 {np.exp(e1['score']):.2f} / haircut {np.exp(e2['score']):.2f}) turnover={to:.0f}%")
    print("  ", dict(sorted(hold.items(), key=lambda kv:-kv[1])))
    np.save(f"out/r5_{name}.npy", w)

# ======== SECTION 6: build_r6.py (2026-08-31) ========
"""Build Deck R6 (24 engine scenarios, +TLT/GOVZ) + scenarios_24 + row grades.

New scenarios (documented blends of R5 columns + explicit overlays, CM24-style):
 S21 Deflationary Bust – Balance-Sheet Recession = .50*CreditStress + .30*China + .20*CreditAccidentUS
 S22 USD / Reserve-Institution Crisis            = .45*JGB + .35*Stagflation + .20*Political
 S23 Broad Kinetic Conflict – Multi-Theater      = .50*Taiwan + .30*Cyber + .20*EnergyShock
 S24 AI Abundance – Disinflationary Boom         = .60*AIProd + .40*Base
Duration rows: TLT/GOVZ legacy columns = m + beta*(IEF_s - m), m = IEF scenario mean,
beta empirical vs IEF (TLT 1.85, GOVZ 2.55); duration overlays scale by beta.
"""
import numpy as np, pandas as pd

deck = pd.read_csv("Deck_Recal_R5_20260831.csv"); deck["Ticker"]=deck["Ticker"].str.strip()
D = deck.set_index("Ticker")
SC = list(D.columns)
IX = dict(Base=0,Refl=1,Stag=2,Debt=3,Credit=4,AIProd=5,AIPower=6,TradeTech=7,Cyber=8,Crony=9,
          China=10,Taiwan=11,USCredit=12,Space=13,Pandemic=14,ClimPol=15,ClimPhys=16,Energy=17,Political=18,JGB=19)

# ---- duration rows on legacy 20 columns ----
BT, BG = 1.85, 2.55
ief = D.loc["IEF"].astype(float); m = float(ief.mean())
D.loc["TLT"]  = (m + BT*(ief - m)).round(4)
D.loc["GOVZ"] = (m + BG*(ief - m)).round(4)
print(f"IEF scenario mean m={m:.3f}; TLT row range {D.loc['TLT'].min():.2f}..{D.loc['TLT'].max():.2f}; GOVZ {D.loc['GOVZ'].min():.2f}..{D.loc['GOVZ'].max():.2f}")

def c(t,k): return float(D.loc[t].iloc[IX[k]])
def dur(x): return {"IEF":x,"IEF/aggregate-bond proxy":x,"TLT":round(BT*x,2),"GOVZ":round(BG*x,2)}
MF=['DBMF','CTA','CTA_OVERLAY','KMLM']; GOLD=['GLDM','GLD']
DEFP=['SHLD','XAR','PPA','ITA','EUAD','JEDI','LMT','NOC','GD','RTX']
CYB=['CIBR','HACK','IHAK']; EXCH4=['CME','CBOE','NDAQ','ICE']
EXD=['CME','CBOE','ICE','SPGI','NDAQ','MSCI','MOAT','QUAL']
INTL=['AVDE','AVDV','AVES','IXUS','VT']; BROAD=['IVV','VOO','VT','QQEW','AVUS','VTI','VTSAX']
NEW = {
 "Deflationary Bust – Balance-Sheet Recession": dict(
    blend={'Credit':.50,'China':.30,'USCredit':.20}, w=4.0, sev=4,
    defn="Decade-scale deleveraging; nominal growth 0-2%; ZIRP returns; duration rewarded, real assets soft.",
    ov={**dur(+2.0), **{t:+1.0 for t in MF}, **{t:+0.5 for t in GOLD},
        'XLE':-1.0,'URNM':-1.0,'SETM':-1.0,'CEG':-0.5,'NLR':-0.5,'NUKZ':-0.5}),
 "USD / Reserve-Institution Crisis": dict(
    blend={'JGB':.45,'Stag':.35,'Political':.20}, w=4.0, sev=5,
    defn="Term-premium blowout + reserve-status erosion; repression with negative real returns on USD financial assets; hard assets rewarded, nominal duration the epicenter.",
    ov={**dur(-2.0), **{t:+2.5 for t in GOLD}, **{t:+0.75 for t in INTL},
        'URNM':+0.75,'XLE':+0.75,'SETM':+0.75, **{t:+0.25 for t in EXCH4}}),
 "Broad Kinetic Conflict – Multi-Theater": dict(
    blend={'Taiwan':.50,'Cyber':.30,'Energy':.20}, w=3.0, sev=5,
    defn="Major-power war beyond Taiwan; homeland cyber-physical strikes; mobilization procurement; domestic infrastructure targeted.",
    ov={**dur(+0.5), **{t:+2.0 for t in DEFP}, 'ROKT':+1.0, **{t:+1.5 for t in CYB},
        **{t:+1.5 for t in GOLD}, **{t:+1.0 for t in MF},
        'GRID':-1.5,'CEG':-1.5,'NLR':-1.5,'NUKZ':-0.5,'SMH':-1.0,'SOXX':-1.0,'XSD':-1.0}),
 "AI Abundance – Disinflationary Boom": dict(
    blend={'AIProd':.60,'Base':.40}, w=3.0, sev=2,
    defn="Productivity boom compresses prices and margins; disinflationary breadth; pricing-power moats and gold underperform; duration fine.",
    ov={**dur(+1.0), **{t:-0.75 for t in EXD}, **{t:-1.0 for t in GOLD},
        **{t:+0.5 for t in BROAD}, 'CIBR':+0.5}),
}
for name, spec in NEW.items():
    col = sum(wt*D.iloc[:, IX[k]] for k,wt in spec['blend'].items())
    for t,adj in spec['ov'].items():
        if t in D.index: col.loc[t] = col.loc[t] + adj
    D[name] = col.round(4)

D.reset_index().to_csv("Deck_Recal_R6_20260831.csv", index=False)
print("R6 deck:", D.shape)

# ---- scenarios_24 file ----
s20 = pd.read_csv("scenarios_20_canonical.csv")
rows = s20[["Scenario_ID","Scenario","Weight_raw","Severity_1to5","Canonical_Parent","Definition"]].to_dict("records")
for i,(name,spec) in enumerate(NEW.items(), start=21):
    rows.append(dict(Scenario_ID=i, Scenario=name, Weight_raw=spec['w'], Severity_1to5=spec['sev'],
                     Canonical_Parent=name, Definition=spec['defn']))
s24 = pd.DataFrame(rows)
s24["Weight_Normalized"] = s24["Weight_raw"]/s24["Weight_raw"].sum()
s24.to_csv("scenarios_24_R6_20260831.csv", index=False)
print("scenario weights sum:", s24["Weight_raw"].sum(), "| base-case mass:", round(float(s24["Weight_Normalized"].iloc[0])*100,1), "%")

# ---- row confidence grades ----
A = set(('GLDM GLD DBMF CTA CTA_OVERLAY KMLM IEF IVV VOO VT VTI VTSAX AVUS AVUV AVDE AVDV AVES IXUS '
         'QUAL USMV SPLV SCHD MOAT COWZ JQUA VFQY VALQ DFAS XLV CME CBOE ICE SPGI NDAQ MSCI IBKR LMT NOC GD RTX').split())
A.add('IEF/aggregate-bond proxy')
Cg = set('QTUM URNM CEG NUKZ JEDI EUAD ZTS_proxy TLT GOVZ'.split())
grades = []
for t in D.index:
    g = 'A' if t in A else ('C' if t in Cg else 'B')
    grades.append(dict(Ticker=t, Grade=g))
gdf = pd.DataFrame(grades)
note = {"A":"empirically corroborated / low model risk","B":"thesis-consistent thematic with history","C":"model-derived, short-history, or audit-flagged (25% haircut toward column median by default in optimization)"}
gdf["Meaning"] = gdf["Grade"].map(note)
gdf.to_csv("Row_Grades_R6_20260831.csv", index=False)
print("grades:", gdf.Grade.value_counts().to_dict())
print("\nSpot checks (new columns):")
for t in ['GLDM','SHLD','GOVZ','TLT','IEF','DBMF','SMH','CEG','CME','IVV','XAR','URNM']:
    r=D.loc[t]
    print(f"  {t:5s} Defl={r.iloc[20]:6.2f} USD={r.iloc[21]:6.2f} War={r.iloc[22]:6.2f} Abund={r.iloc[23]:6.2f}")

# ======== SECTION 7: r6_opt.py ========
"""R6 optimization: gates = {R6, C-haircut deck, -200bp level stress} x 4 conventions,
like-for-like vs P1 under the same deck. Objective: maximin(score@R6, score@C-haircut).
Run with DECK_FILE=Deck_Recal_R6_20260831.csv SCEN_FILE=scenarios_24_R6_20260831.csv.
"""
import numpy as np, pandas as pd
import lib_pathce as L
from core import (UNIV, N, IDX, DECKM, ERV, sigma_of, wvec, P1v, CONVS, FE, feasible)

grades = pd.read_csv("Row_Grades_R6_20260831.csv").set_index("Ticker")["Grade"]
colmed = np.median(DECKM, axis=0)
HCC = DECKM.copy()
for i, t in enumerate(UNIV):
    if grades.get(t, 'B') == 'C':
        HCC[i] = 0.75 * DECKM[i] + 0.25 * colmed

def ev(w, M, shift=0.0, full=True):
    r_net = M.T @ w - float(ERV @ w) + shift
    sig = sigma_of(w)
    res = {(p, c): FE.run(r_net, p, c, sig) for p, c in CONVS}
    score = float(np.mean([np.log(res[k]["PathCE"]) for k in CONVS]))
    out = dict(score=score, sigma=sig, res=res, worst=float(r_net.min()))
    if full:
        out.update(RCEW=L.rcew40(r_net)["robust"], DRB6=L.drb40(r_net, h=0.06),
                   R3AF=L.r3af(r_net, (DECKM.T @ P1EVG - L.wer_evg11()) if False else L.prow(L.EVG11) - L.wer_evg11()))
    return out
P1EVG = None

DECKS = {"R6": (DECKM, 0.0), "HCC": (HCC, 0.0), "LS": (DECKM, -0.02)}
BASES = {k: ev(P1v, M, s) for k, (M, s) in DECKS.items()}

def fails_vs(e, b):
    f = []
    for c in CONVS:
        if e["res"][c]["P10"] < b["res"][c]["P10"] - 1e-9: f.append(f"P10{c[1]}")
        if e["res"][c]["P5"] < b["res"][c]["P5"] - 1e-9: f.append(f"P5{c[1]}")
        if e["res"][c]["ruin"] > b["res"][c]["ruin"]: f.append(f"ruin{c[0][:3]}{c[1]}")
    if e["worst"] < b["worst"] - 1e-12: f.append("worst")
    if e["RCEW"] < b["RCEW"] - 1e-9: f.append("RCEW")
    if e["R3AF"] < b["R3AF"] - 0.002: f.append("R3AF")
    if e["DRB6"] > b["DRB6"] + 1e-12: f.append("DRB6")
    return f

def gates_all(w):
    es = {}
    for k, (M, s) in DECKS.items():
        e = ev(w, M, s)
        if fails_vs(e, BASES[k]):
            return None, {k: fails_vs(e, BASES[k])}
        es[k] = e
    return es, None

def show_gate(name, w):
    es, why = gates_all(w)
    if why:
        k, f = list(why.items())[0]
        print(f"{name:22s}: FAIL@{k}: {', '.join(f[:5])}")
        return None
    mm = min(es["R6"]["score"], es["HCC"]["score"])
    print(f"{name:22s}: maximin={np.exp(mm):6.2f}  R6={np.exp(es['R6']['score']):6.2f} HCC={np.exp(es['HCC']['score']):6.2f} LSgeo={np.exp(es['LS']['score']):5.2f} LS-P5robG={es['LS']['res'][('robust','G')]['P5']:.2f}")
    return mm

R2f = wvec({'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.13,'CIBR':.09,'GRID':.05,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.04,'URNM':.03,'XAR':.03,'CEG':.03})
R1bf = wvec({'GLDM':.15,'SHLD':.14,'DBMF':.15,'NUKZ':.12,'CIBR':.10,'CME':.05,'MSCI':.05,'XLE':.05,'XAR':.04,'CEG':.03,'URNM':.03,'CBOE':.03,'GRID':.03,'CTA':.03})
P2 = wvec({'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.15,'CIBR':.10,'GRID':.10,'CTA':.05,'CME':.05,'XAR':.05,'MSCI':.05})

print("=== existing books under R6 gate battery ===")
show_gate("P2", P2); show_gate("R2_final", R2f); show_gate("R1b_final", R1bf)

print("\n=== duration marginals on R2_final ===")
V = {
 "R2+GOVZ3 (NUKZ-1,CIBR-1,XLE-1)": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.12,'CIBR':.08,'GRID':.05,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.03,'URNM':.03,'XAR':.03,'CEG':.03,'GOVZ':.03},
 "R2+GOVZ4 (GRID-2,NUKZ-2)": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.11,'CIBR':.09,'GRID':.03,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.04,'URNM':.03,'XAR':.03,'CEG':.03,'GOVZ':.04},
 "R2+TLT4 (GRID-2,NUKZ-2)": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.11,'CIBR':.09,'GRID':.03,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.04,'URNM':.03,'XAR':.03,'CEG':.03,'TLT':.04},
 "R2+GOVZ3 (CTA-2,DBMF-1)": {'GLDM':.15,'SHLD':.15,'DBMF':.14,'NUKZ':.13,'CIBR':.09,'GRID':.05,'CME':.05,'MSCI':.05,'CTA':.03,'XLE':.04,'URNM':.03,'XAR':.03,'CEG':.03,'GOVZ':.03},
}
seeds = [("R2_final", R2f), ("R1b_final", R1bf)]
for nm, d in V.items():
    w = wvec(d)
    mm = show_gate(nm, w)
    if mm is not None:
        seeds.append((nm, w))

rng = np.random.default_rng(63)
def polish(w0, iters=40, sample=28):
    es, why = gates_all(w0)
    if why: return None
    best = (min(es["R6"]["score"], es["HCC"]["score"]), w0.copy(), es)
    for it in range(iters):
        _, cur_w, _ = best
        src = np.where(cur_w > 1e-9)[0]; improved = False
        for _ in range(sample):
            i = int(rng.choice(src)); j = int(rng.integers(N))
            if i == j: continue
            wn = cur_w.copy(); take = min(0.01 if rng.random() < 0.8 else 0.02, wn[i])
            if wn[i] - take < 0.02 - 1e-9 and wn[i] - take > 1e-9: take = wn[i] - 0.02
            if take <= 1e-9: continue
            wn[i] -= take
            if wn[i] < 1e-9: wn[i] = 0
            add = take
            if wn[j] < 1e-9 and add < 0.02 - 1e-9:
                extra = 0.02 - add
                if wn[i] >= extra + 0.02 or abs(wn[i] - extra) < 1e-9:
                    wn[i] -= extra; add = 0.02
                    if wn[i] < 1e-9: wn[i] = 0
                else: continue
            wn[j] += add
            if not feasible(wn): continue
            q = ev(wn, DECKM, 0.0, full=False)
            if q["score"] <= best[0] + 1e-5: continue
            es2, why2 = gates_all(wn)
            if why2: continue
            s2 = min(es2["R6"]["score"], es2["HCC"]["score"])
            if s2 > best[0] + 1e-5:
                best = (s2, wn.copy(), es2); improved = True
        if not improved and it > 10: break
    return best

print("\n=== polish ===")
out = []
seeds.sort(key=lambda x: 0)  # keep insertion order
for nm, w0 in seeds[:5]:
    r = polish(w0)
    if r:
        out.append((r[0], nm, r[1], r[2]))
        print(f"from {nm}: maximin {np.exp(r[0]):.3f}")
out.sort(key=lambda x: -x[0])
for sc, nm, w, es in out[:3]:
    hold = {UNIV[i]: round(float(w[i]) * 100, 1) for i in np.where(w > 1e-9)[0]}
    to = 50 * np.abs(w - P1v).sum()
    print(f"\n[{nm}] maximin={np.exp(sc):.2f} R6={np.exp(es['R6']['score']):.2f} HCC={np.exp(es['HCC']['score']):.2f} LS-P5robG={es['LS']['res'][('robust','G')]['P5']:.2f} turnover={to:.0f}%")
    print("  ", dict(sorted(hold.items(), key=lambda kv: -kv[1])))
    np.save(f"out/r6_{nm.split()[0].replace('+','_')}.npy", w)

# ======== SECTION 8: r6_verify.py ========
"""R6 final verification at 60k: P1, P2, R2_final, R1b_final, R2d (+TLT4).
Decks: R6, C-grade haircut (HCC), level stress LS = R6 - 200bp. Engine-exact streams.
Run with DECK_FILE=Deck_Recal_R6_20260831.csv SCEN_FILE=scenarios_24_R6_20260831.csv.
"""
import numpy as np, pandas as pd
import lib_pathce as L
from core import UNIV, IDX, N, DECKM, ERV, sigma_of, wvec, P1v, CONVS

grades = pd.read_csv("Row_Grades_R6_20260831.csv").set_index("Ticker")["Grade"]
colmed = np.median(DECKM, axis=0)
HCC = DECKM.copy()
for i, t in enumerate(UNIV):
    if grades.get(t, 'B') == 'C':
        HCC[i] = 0.75 * DECKM[i] + 0.25 * colmed

def engine_exact(r_net, sigma):
    out = {}
    rng = np.random.default_rng(42)
    for conv in ("G", "L"):
        out[("robust", conv)] = L.run_engine(rng, r_net + 0.0, L.PI_ROBUST, conv, sigma, 0.0)
    rng = np.random.default_rng(42)
    for conv in ("G", "L"):
        out[("canonical", conv)] = L.run_engine(rng, r_net + 0.0, L.PI_CANON, conv, sigma, 0.0)
    return out

BOOKS = {
 "P1_baseline": L.P1,
 "P2_JEDItoXAR": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.15,'CIBR':.10,'GRID':.10,'CTA':.05,'CME':.05,'XAR':.05,'MSCI':.05},
 "R2_final": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.13,'CIBR':.09,'GRID':.05,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.04,'URNM':.03,'XAR':.03,'CEG':.03},
 "R1b_final": {'GLDM':.15,'SHLD':.14,'DBMF':.15,'NUKZ':.12,'CIBR':.10,'CME':.05,'MSCI':.05,'XLE':.05,'XAR':.04,'CEG':.03,'URNM':.03,'CBOE':.03,'GRID':.03,'CTA':.03},
 "R2d_duration": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.11,'CIBR':.08,'GRID':.05,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.03,'URNM':.03,'XAR':.03,'CEG':.03,'TLT':.04},
}
rows = []; wrows = []
for tag, d in BOOKS.items():
    w = wvec(d)
    wer = float(ERV @ w); sig = sigma_of(w)
    r6 = DECKM.T @ w - wer
    mc = engine_exact(r6, sig)
    mch = engine_exact(HCC.T @ w - wer, sig)
    mcl = engine_exact(r6 - 0.02, sig)
    geo = float(np.exp(np.mean([np.log(mc[c]["PathCE"]) for c in CONVS])))
    geoh = float(np.exp(np.mean([np.log(mch[c]["PathCE"]) for c in CONVS])))
    geol = float(np.exp(np.mean([np.log(mcl[c]["PathCE"]) for c in CONVS])))
    rc = L.rcew40(r6)["robust"]; drb6 = L.drb40(r6, h=0.06)
    r3 = L.r3af(r6, L.prow(L.EVG11) - L.wer_evg11())
    weights = {t: v for t, v in d.items()}
    hist = L.hist_diagnostics(weights)
    to = 50 * np.abs(w - P1v).sum()
    lsr = mcl[("robust", "G")]
    print(f"{tag}: geo={geo:.2f} HCC={geoh:.2f} | LS geo={geol:.2f} P5robG={lsr['P5']:.2f} ruinrobG={lsr['ruin']} | worst={r6.min()*100:.2f}% DRB6={drb6:.4f} R3AF={r3:.4f} WER={wer*100:.3f} vol={sig*100:.2f} Sortino={hist['sortino']:.2f} turn={to:.0f}%")
    for c in CONVS:
        r = mc[c]
        print(f"   {c[0][:3]}-{c[1]}: PathCE={r['PathCE']:.2f} P10={r['P10']:.2f} P5={r['P5']:.2f} ruin={r['ruin']}")
    row = dict(Portfolio=tag, Deck="R6_20260831", WER_pct=round(wer*100,3), vol48_pct=round(sig*100,2),
               geo_PathCE=round(geo,2), geo_PathCE_Chaircut=round(geoh,2), geo_PathCE_LS200=round(geol,2),
               LS200_P5_robG=round(lsr["P5"],2), LS200_ruin_robG=lsr["ruin"],
               RCEW40_robust=round(rc,2), DRB40_h6=round(drb6,4), R3AF=round(r3,4),
               worst_net_CAGR_pct=round(float(r6.min())*100,2),
               worst_scenario=L.SC[int(np.argmin(r6))],
               SWCAGR_pct=round(float(np.sum(L.W_CAN*r6))*100,2),
               Sortino48=round(hist["sortino"],2), maxDD48_pct=round(hist["maxdd"]*100,1),
               turnover_vs_P1_pct=round(to,0))
    for c in CONVS:
        k = f"{c[0][:3]}{c[1]}"
        row[f"PathCE_{k}"] = round(mc[c]["PathCE"],2); row[f"P10_{k}"] = round(mc[c]["P10"],2)
        row[f"P5_{k}"] = round(mc[c]["P5"],2); row[f"ruin_{k}"] = mc[c]["ruin"]
    rows.append(row)
    for t, v in sorted(d.items(), key=lambda kv: -kv[1]):
        wrows.append(dict(Portfolio=tag, Ticker=t, Weight=v))
pd.DataFrame(rows).to_csv("out/EngineV2_Metrics_R6_20260831.csv", index=False)
pd.DataFrame(wrows).to_csv("out/EngineV2_Weights_R6_20260831.csv", index=False)

# formal gate check vs P1 like-for-like at 60k
print("\n=== formal 60k gates vs P1 (R6 / HCC / LS, all conventions) ===")
P1w = wvec(L.P1); P1wer = float(ERV @ P1w); P1sig = sigma_of(P1w)
P1mc = {"R6": engine_exact(DECKM.T @ P1w - P1wer, P1sig),
        "HCC": engine_exact(HCC.T @ P1w - P1wer, P1sig),
        "LS": engine_exact(DECKM.T @ P1w - P1wer - 0.02, P1sig)}
for tag, d in BOOKS.items():
    if tag == "P1_baseline": continue
    w = wvec(d); wer = float(ERV @ w); sig = sigma_of(w)
    fails = []
    for dk, M, sh in (("R6", DECKM, 0.0), ("HCC", HCC, 0.0), ("LS", DECKM, -0.02)):
        mc = engine_exact(M.T @ w - wer + sh, sig)
        for c in CONVS:
            a, b = mc[c], P1mc[dk][c]
            if a["P10"] < b["P10"] - 1e-9: fails.append(f"P10@{dk}")
            if a["P5"] < b["P5"] - 1e-9: fails.append(f"P5@{dk}")
            if a["ruin"] > b["ruin"]: fails.append(f"ruin@{dk}{c[0][:3]}{c[1]}")
    print(f"{tag}: {'ALL GATES PASS' if not fails else 'FAIL: ' + ', '.join(sorted(set(fails)))}")
print("\nwrote out/EngineV2_Metrics_R6_20260831.csv and Weights_R6")

# ======== SECTION 9: build_r7.py (2026-09-01) ========
"""R7 candidate deck: R6 + 13 previously-unmodeled tickers via the house 'added_proxy'
method (anchor-row blends scaled by empirical betas + documented deltas), R6 frame.
All new rows graded C. Run with R6 env; writes Deck_Recal_R7cand_20260901.csv.
"""
import numpy as np, pandas as pd
import lib_pathce as L

D = L.DECK.copy().drop(index=["IAI"], errors="ignore")   # R6 rows (percent), 24 cols
SC = list(D.columns); LEG = SC[:20]
IX = {k:i for i,k in enumerate(['Base','Refl','Stag','Debt','Credit','AIProd','AIPower','TradeTech','Cyber','Crony',
      'China','Taiwan','USCredit','Space','Pandemic','ClimPol','ClimPhys','Energy','Political','JGB'])}
def row(t): return D.loc[t, LEG].astype(float).values
def dev(t): r=row(t); return r - r.mean()
def mean(t): return float(row(t).mean())
def ov(**kw):
    v = np.zeros(20)
    for k,x in kw.items(): v[IX[k]] = x
    return v
MOM = np.array([2,2,0,1,-1,4,2,0.5,-1.5,1,-0.5,-2,-1,2.5,3,1.5,0.5,-0.5,0.5,0.0])  # house Mom premia

INFL = ov(Stag=2.0,Debt=1.5,Energy=1.5,JGB=0.5,Refl=0.5,Crony=0.5,Political=0.5,Credit=-1.0,China=-1.0,
          USCredit=-0.5,Cyber=-0.5,AIProd=-0.5,AIPower=0.5,ClimPol=0.5,ClimPhys=0.5,TradeTech=0.5)
NEWROWS = {
 "LTPZ": mean('IEF') + 1.46*dev('IEF') + 1.0*INFL,
 "SCHP": mean('IEF') + 0.61*dev('IEF') + 0.5*INFL,
 "VTIP": mean('IEF') - 0.7 + 0.18*dev('IEF') + 0.3*INFL,
 "GDX":  mean('GLDM') - 1.5 + 1.98*dev('GLDM') + 0.5*dev('IVV') + ov(Stag=-1.0,Energy=-1.0,Refl=-0.5,Credit=-1.0),
 "WPM":  mean('GLDM') + 1.0 + 1.88*dev('GLDM') + 0.3*dev('IVV'),
 "FNV":  mean('GLDM') + 1.0 + 1.40*dev('GLDM') + 0.3*dev('IVV'),
 "PSLV": mean('GLDM') - 0.5 + 1.52*dev('GLDM') + ov(Refl=1.5,AIProd=1.0,AIPower=1.0,ClimPol=1.5,China=-1.5,Credit=-1.5,USCredit=-1.0,Taiwan=-0.5),
 "COPX": 0.5*row('SETM') + 0.3*row('XLE') + 0.2*row('AVUV') + ov(AIPower=2.0,Refl=2.0,ClimPol=2.0,Base=0.5,China=-2.5,Credit=-1.5,USCredit=-1.0,Taiwan=-1.0,Cyber=-0.5,Stag=0.5,Debt=0.5),
 "BRKB": 0.5*row('QUAL') + 0.3*row('MOAT') + 0.2*row('IVV') + ov(Credit=1.5,USCredit=1.5,Cyber=0.5,Taiwan=0.5,AIProd=-1.0,Space=-0.5,Stag=0.5),
 "WM":   0.5*row('QUAL') + 0.3*row('USMV') + 0.2*row('PAVE') + ov(Stag=1.0,Debt=0.5,Credit=0.5,USCredit=0.5,AIProd=-0.5,ClimPol=0.5,Energy=0.5),
 "INDA": 0.5*row('AVES') + 0.5*row('IXUS') + ov(Base=1.5,TradeTech=2.0,China=1.5,Refl=1.0,AIProd=0.5,Energy=-2.0,Stag=-0.5,Taiwan=-1.0,Cyber=-1.0,Credit=-1.0,JGB=-1.0,USCredit=-0.5),
 "JEPI": 0.75*row('IVV') + 0.25*row('USMV') + ov(AIProd=-2.5,Refl=-1.5,Base=-1.0,Space=-1.0,AIPower=-0.5,Stag=0.5,Credit=0.5,USCredit=0.5,Cyber=0.5,China=0.5,Political=0.5,JGB=0.5,Taiwan=0.5),
 "SPMO": row('IVV') + 0.5*(MOM - MOM.mean()) + ov(Credit=-1.0,Taiwan=-1.0),
}
# new-scenario columns via R6 blend formulas + class overlays
NEWOV = {  # (Defl, USDcrisis, War, Abund)
 "LTPZ":(0.9,-0.9,1.2,0.5),"SCHP":(0.2,-0.2,0.8,-0.4),"VTIP":(-0.1,0.3,0.4,-0.8),
 "GDX":(0.0,5.0,3.0,-2.0),"WPM":(0.2,4.7,2.8,-1.9),"FNV":(0.2,3.5,2.1,-1.4),"PSLV":(-0.5,3.8,2.3,-0.5),
 "COPX":(-2.0,0.75,-1.0,1.0),"BRKB":(1.0,0.0,0.0,-0.5),"WM":(0.5,0.0,0.5,-0.75),
 "INDA":(-1.0,0.75,-1.5,0.5),"JEPI":(0.5,-0.5,0.0,-1.0),"SPMO":(-1.0,0.0,-1.0,1.0),
}
def c(r,k): return r[IX[k]]
out = {}
for t, r in NEWROWS.items():
    defl = .50*c(r,'Credit')+.30*c(r,'China')+.20*c(r,'USCredit') + NEWOV[t][0]
    usd  = .45*c(r,'JGB')+.35*c(r,'Stag')+.20*c(r,'Political') + NEWOV[t][1]
    war  = .50*c(r,'Taiwan')+.30*c(r,'Cyber')+.20*c(r,'Energy') + NEWOV[t][2]
    abu  = .60*c(r,'AIProd')+.40*c(r,'Base') + NEWOV[t][3]
    out[t] = np.concatenate([r, [defl, usd, war, abu]])
R7 = pd.concat([D, pd.DataFrame(out, index=SC).T])
R7.round(4).reset_index().rename(columns={"index":"Ticker"}).to_csv("Deck_Recal_R7cand_20260901.csv", index=False)
g = pd.read_csv("Row_Grades_R6_20260831.csv")
add = pd.DataFrame([dict(Ticker=t, Grade="C", Meaning="model-derived (R7 proxy-blend), 25% haircut by default") for t in NEWROWS if t not in set(g.Ticker)])
pd.concat([g, add]).to_csv("Row_Grades_R7cand_20260901.csv", index=False)
print("R7 candidate deck:", R7.shape)
print(f"{'tkr':5s} {'mean':>6s} {'worst':>6s} {'Stag':>6s} {'Credit':>7s} {'Taiwan':>7s} {'Defl':>6s} {'USD':>6s} {'War':>6s} {'Abund':>6s}")
for t in NEWROWS:
    r = R7.loc[t]
    print(f"{t:5s} {r.iloc[:20].mean():6.2f} {r.min():6.2f} {r.iloc[2]:6.2f} {r.iloc[4]:7.2f} {r.iloc[11]:7.2f} {r.iloc[20]:6.2f} {r.iloc[21]:6.2f} {r.iloc[22]:6.2f} {r.iloc[23]:6.2f}")
print("reference: GLDM mean %.2f worst %.2f | IEF mean %.2f | TLT mean %.2f | IVV mean %.2f worst %.2f" % (
    D.loc['GLDM'].iloc[:20].mean(), D.loc['GLDM'].min(), D.loc['IEF'].iloc[:20].mean(), D.loc['TLT'].iloc[:20].mean(), D.loc['IVV'].iloc[:20].mean(), D.loc['IVV'].min()))

# ======== SECTION 10: r7_opt.py ========
"""R7 widened-universe test: do any of the 13 new rows earn weight under the full
standing battery (R7 deck, C-haircut, -200bp, like-for-like vs P1) + concentration
stress on entrants?  Run: R7=1 DECK_FILE=Deck_Recal_R7cand_20260901.csv SCEN_FILE=scenarios_24_R6_20260831.csv NPATHS=10000
"""
import numpy as np, pandas as pd
import lib_pathce as L
from core import (UNIV, N, IDX, DECKM, ERV, sigma_of, wvec, P1v, CONVS, FE, feasible)

grades = pd.read_csv("Row_Grades_R7cand_20260901.csv").set_index("Ticker")["Grade"]
colmed = np.median(DECKM, axis=0)
HCC = DECKM.copy()
for i, t in enumerate(UNIV):
    if grades.get(t, 'B') == 'C':
        HCC[i] = 0.75*DECKM[i] + 0.25*colmed

def ev(w, M, shift=0.0, full=True):
    r_net = M.T @ w - float(ERV @ w) + shift
    sig = sigma_of(w)
    res = {(p,c): FE.run(r_net, p, c, sig) for p,c in CONVS}
    score = float(np.mean([np.log(res[k]["PathCE"]) for k in CONVS]))
    out = dict(score=score, sigma=sig, res=res, worst=float(r_net.min()))
    if full:
        out.update(RCEW=L.rcew40(r_net)["robust"], DRB6=L.drb40(r_net,h=0.06),
                   R3AF=L.r3af(r_net, L.prow(L.EVG11)-L.wer_evg11()))
    return out
DECKS = {"R7":(DECKM,0.0),"HCC":(HCC,0.0),"LS":(DECKM,-0.02)}
BASES = {k: ev(P1v,M,s) for k,(M,s) in DECKS.items()}
def fails_vs(e,b):
    f=[]
    for c in CONVS:
        if e["res"][c]["P10"]<b["res"][c]["P10"]-1e-9: f.append("P10")
        if e["res"][c]["P5"]<b["res"][c]["P5"]-1e-9: f.append("P5")
        if e["res"][c]["ruin"]>b["res"][c]["ruin"]: f.append("ruin")
    if e["worst"]<b["worst"]-1e-12: f.append("worst")
    if e["RCEW"]<b["RCEW"]-1e-9: f.append("RCEW")
    if e["R3AF"]<b["R3AF"]-0.002: f.append("R3AF")
    if e["DRB6"]>b["DRB6"]+1e-12: f.append("DRB6")
    return f
def gates_all(w):
    es={}
    for k,(M,s) in DECKS.items():
        e=ev(w,M,s); f=fails_vs(e,BASES[k])
        if f: return None,(k,f)
        es[k]=e
    return es,None
def mm(es): return min(es["R7"]["score"], es["HCC"]["score"])

R2f = {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.13,'CIBR':.09,'GRID':.05,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.04,'URNM':.03,'XAR':.03,'CEG':.03}
R1bf = {'GLDM':.15,'SHLD':.14,'DBMF':.15,'NUKZ':.12,'CIBR':.10,'CME':.05,'MSCI':.05,'XLE':.05,'XAR':.04,'CEG':.03,'URNM':.03,'CBOE':.03,'GRID':.03,'CTA':.03}
R2d = {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.11,'CIBR':.08,'GRID':.05,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.03,'URNM':.03,'XAR':.03,'CEG':.03,'TLT':.04}
base_es,_ = gates_all(wvec(R2f)); base_mm = mm(base_es)
print(f"R2_final on R7 deck: maximin={np.exp(base_mm):.2f} (R7 {np.exp(base_es['R7']['score']):.2f} / HCC {np.exp(base_es['HCC']['score']):.2f})")

NEW = ['LTPZ','SCHP','VTIP','GDX','WPM','FNV','PSLV','COPX','BRKB','WM','INDA','JEPI','SPMO']
print("\n=== marginal adds to R2_final at 4% (funded NUKZ-2, CIBR-1, XLE-1) ===")
funded = dict(R2f); funded['NUKZ']-=.02; funded['CIBR']-=.01; funded['XLE']-=.01
marg=[]
for t in NEW:
    d = dict(funded); d[t]=0.04
    w = wvec(d)
    if not feasible(w): print(f"{t:5s}: infeasible"); continue
    es,why = gates_all(w)
    if why: print(f"{t:5s}: FAIL@{why[0]}: {sorted(set(why[1]))}"); continue
    s = mm(es); marg.append((s,t,w))
    print(f"{t:5s}: maximin={np.exp(s):.2f} ({np.exp(s-base_mm)-1:+.1%} vs R2_final) R7={np.exp(es['R7']['score']):.2f} LS-P5={es['LS']['res'][('robust','G')]['P5']:.2f}")

# concentration stress for passing entrants: shrink entrant row 25%->median, like-for-like
print("\n=== concentration stress on passing entrants (entrant row 25%->median) ===")
for s,t,w in sorted(marg, key=lambda x:-x[0])[:6]:
    CS = DECKM.copy(); CS[IDX[t]] = 0.75*DECKM[IDX[t]] + 0.25*colmed
    e = ev(w, CS); b = ev(P1v, CS); f = fails_vs(e,b)
    print(f"{t:5s}: {'PASS' if not f else 'FAIL: '+', '.join(sorted(set(f)))}  geoCE@CS={np.exp(e['score']):.2f}")

# bounded polish from R2_final with new names available
rng = np.random.default_rng(5)
def polish(w0, iters=40, sample=30):
    es,why = gates_all(w0)
    if why: return None
    best=(mm(es), w0.copy(), es)
    for it in range(iters):
        _,cur_w,_=best; src=np.where(cur_w>1e-9)[0]; improved=False
        for _ in range(sample):
            i=int(rng.choice(src)); j=int(rng.integers(N))
            if i==j: continue
            wn=cur_w.copy(); take=min(0.01 if rng.random()<0.8 else 0.02, wn[i])
            if wn[i]-take<0.02-1e-9 and wn[i]-take>1e-9: take=wn[i]-0.02
            if take<=1e-9: continue
            wn[i]-=take
            if wn[i]<1e-9: wn[i]=0
            add=take
            if wn[j]<1e-9 and add<0.02-1e-9:
                extra=0.02-add
                if wn[i]>=extra+0.02 or abs(wn[i]-extra)<1e-9:
                    wn[i]-=extra; add=0.02
                    if wn[i]<1e-9: wn[i]=0
                else: continue
            wn[j]+=add
            if not feasible(wn): continue
            q=ev(wn,DECKM,full=False)
            if q["score"]<=best[0]+1e-5: continue
            es2,why2=gates_all(wn)
            if why2: continue
            s2=mm(es2)
            if s2>best[0]+1e-5: best=(s2,wn.copy(),es2); improved=True
        if not improved and it>10: break
    return best
print("\n=== polish with widened universe ===")
for nm,d in (("R2_final",R2f),("R1b_final",R1bf),("R2d",R2d)):
    r=polish(wvec(d))
    if r:
        sc,w,es=r
        hold={UNIV[i]:round(float(w[i])*100,1) for i in np.where(w>1e-9)[0]}
        new_in={k:v for k,v in hold.items() if k in NEW}
        print(f"from {nm}: maximin {np.exp(sc):.2f} | new entrants: {new_in if new_in else 'none'}")
        print("   ", dict(sorted(hold.items(), key=lambda kv:-kv[1])))
        np.save(f"out/r7_{nm}.npy", w)

# ======== SECTION 11: r7_verify.py ========
"""60k engine-exact verification of R3 candidates on the R7 candidate deck + flexible-draw
experiment. Run: R7=1 DECK_FILE=Deck_Recal_R7cand_20260901.csv SCEN_FILE=scenarios_24_R6_20260831.csv
"""
import numpy as np, pandas as pd
import lib_pathce as L
from core import UNIV, IDX, DECKM, ERV, sigma_of, wvec, P1v, CONVS
from r6_verify import engine_exact

grades = pd.read_csv("Row_Grades_R7cand_20260901.csv").set_index("Ticker")["Grade"]
colmed = np.median(DECKM, axis=0)
HCC = DECKM.copy()
for i, t in enumerate(UNIV):
    if grades.get(t, 'B') == 'C': HCC[i] = 0.75*DECKM[i] + 0.25*colmed

BOOKS = {
 "P1": dict(L.P1),
 "R2_final": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.13,'CIBR':.09,'GRID':.05,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.04,'URNM':.03,'XAR':.03,'CEG':.03},
 "R2d": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.11,'CIBR':.08,'GRID':.05,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.03,'URNM':.03,'XAR':.03,'CEG':.03,'TLT':.04},
 "R1b_final": {'GLDM':.15,'SHLD':.14,'DBMF':.15,'NUKZ':.12,'CIBR':.10,'CME':.05,'MSCI':.05,'XLE':.05,'XAR':.04,'CEG':.03,'URNM':.03,'CBOE':.03,'GRID':.03,'CTA':.03},
 "R3b": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.12,'CIBR':.08,'GRID':.05,'CME':.05,'BRKB':.04,'CTA':.05,'URNM':.03,'XAR':.03,'CEG':.03,'FNV':.04,'MSCI':.03},
 "R3d": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.10,'CIBR':.08,'GRID':.05,'CME':.05,'MSCI':.04,'CTA':.05,'URNM':.03,'XAR':.03,'TLT':.04,'FNV':.04,'BRKB':.04},
}

def suite(w, draw_rule=None):
    wer = float(ERV @ w); sig = sigma_of(w); r = DECKM.T @ w - wer
    mc = engine_exact(r, sig); mch = engine_exact(HCC.T @ w - wer, sig); mcl = engine_exact(r - 0.02, sig)
    geo = lambda m: float(np.exp(np.mean([np.log(m[c]["PathCE"]) for c in CONVS])))
    return dict(geo=geo(mc), geoh=geo(mch), geol=geo(mcl), mc=mc, mcl=mcl, r=r, sig=sig, wer=wer)

P1s = suite(P1v)
print("=== 60k, R7 deck: books vs P1 gates (R7 / HCC / LS) ===")
rows=[]
for tag, d in BOOKS.items():
    w = wvec(d); s = suite(w)
    fails=[]
    for dk, a_mc, b_mc in (("R7", s["mc"], P1s["mc"]), ("LS", s["mcl"], P1s["mcl"])):
        for c in CONVS:
            if a_mc[c]["P10"] < b_mc[c]["P10"]-1e-9: fails.append(f"P10@{dk}")
            if a_mc[c]["P5"]  < b_mc[c]["P5"]-1e-9: fails.append(f"P5@{dk}")
            if a_mc[c]["ruin"] > b_mc[c]["ruin"]: fails.append(f"ruin@{dk}")
    rg = s["mc"][("robust","G")]; lg = s["mcl"][("robust","G")]
    to = 50*np.abs(w-P1v).sum()
    drb6 = L.drb40(s["r"], h=0.06)
    print(f"{tag:10s} geo={s['geo']:.2f} HCC={s['geoh']:.2f} LS={s['geol']:.2f} | robG PathCE={rg['PathCE']:.2f} P5={rg['P5']:.2f} ruin={rg['ruin']} | LS P5={lg['P5']:.2f} ruin={lg['ruin']} | worst={s['r'].min()*100:.2f}% DRB6={drb6:.4f} WER={s['wer']*100:.3f} vol={s['sig']*100:.2f} turn={to:.0f}% | {'PASS' if not fails else 'FAIL '+','.join(sorted(set(fails)))}")
    rows.append(dict(Portfolio=tag, geo_PathCE=round(s['geo'],2), geo_Chaircut=round(s['geoh'],2), geo_LS200=round(s['geol'],2),
                     PathCE_robG=round(rg['PathCE'],2), P5_robG=round(rg['P5'],2), ruin_robG=rg['ruin'],
                     LS_P5_robG=round(lg['P5'],2), LS_ruin_robG=lg['ruin'], worst_pct=round(float(s['r'].min())*100,2),
                     DRB40_h6=round(drb6,4), WER_pct=round(s['wer']*100,3), vol48_pct=round(s['sig']*100,2), turnover_pct=round(to,0)))
pd.DataFrame(rows).to_csv("out/R7_Books_Metrics_20260901.csv", index=False)

# ======== SECTION 12: flexdraw.py ========
"""Flexible-draw (guardrail) rule: draw 3% of initial normally; when wealth < 0.80x initial,
draw 2.25% (-25%); restore at >= 1.0x. Engine-exact streams; reports ruin/P5/PathCE and the
spending cost (share of years at reduced draw, mean total draws vs fixed = 1.20 units)."""
import numpy as np, pandas as pd
import lib_pathce as L
from core import UNIV, DECKM, ERV, sigma_of, wvec, P1v, CONVS

def run_flex(rng, r_net, pi, conv, sigma, lo=0.80, cut=0.75, hi=1.00):
    NP_, T, LAM = L.NP_, L.T, L.LAM
    s = rng.choice(len(pi), size=NP_, p=pi)
    W = np.ones(NP_); ru = np.zeros(NP_, bool); reduced = np.zeros(NP_, bool)
    yrs_red = np.zeros(NP_); total_draw = np.zeros(NP_)
    for _ in range(T):
        j = rng.random(NP_) < LAM
        if j.any(): s[j] = rng.choice(len(pi), size=int(j.sum()), p=pi)
        mu = np.log1p(r_net[s]) - (sigma*sigma/2 if conv == "G" else 0.0)
        W = W * np.exp(mu + sigma * rng.standard_normal(NP_))
        reduced = np.where(W < lo, True, np.where(W >= hi, False, reduced))
        d = np.where(reduced, 0.03*cut, 0.03)
        yrs_red += reduced; total_draw += np.where(W > 0, d, 0.0)
        W = W - d
        ru |= (W <= 0); W = np.maximum(W, 1e-9)
    lg = np.log(W)
    return dict(PathCE=float(np.exp(lg.mean())), P5=float(np.quantile(W,.05)), P10=float(np.quantile(W,.10)),
                ruin=int(ru.sum()), yrs_red=float(yrs_red.mean()/T), draw=float(total_draw.mean()))

def both(r_net, sigma):
    out = {}
    for mode in ("fixed","flex"):
        rng = np.random.default_rng(42); res = {}
        for conv in ("G","L"):
            res[("robust",conv)] = (L.run_engine(rng, r_net, L.PI_ROBUST, conv, sigma, 0.0) if mode=="fixed"
                                    else run_flex(rng, r_net, L.PI_ROBUST, conv, sigma))
        rng = np.random.default_rng(42)
        for conv in ("G","L"):
            res[("canonical",conv)] = (L.run_engine(rng, r_net, L.PI_CANON, conv, sigma, 0.0) if mode=="fixed"
                                       else run_flex(rng, r_net, L.PI_CANON, conv, sigma))
        out[mode] = res
    return out

BOOKS = {
 "P1": dict(L.P1),
 "R2_final": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.13,'CIBR':.09,'GRID':.05,'CME':.05,'MSCI':.05,'CTA':.05,'XLE':.04,'URNM':.03,'XAR':.03,'CEG':.03},
 "R3b": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.12,'CIBR':.08,'GRID':.05,'CME':.05,'BRKB':.04,'CTA':.05,'URNM':.03,'XAR':.03,'CEG':.03,'FNV':.04,'MSCI':.03},
 "R3d": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.10,'CIBR':.08,'GRID':.05,'CME':.05,'MSCI':.04,'CTA':.05,'URNM':.03,'XAR':.03,'TLT':.04,'FNV':.04,'BRKB':.04},
 "SP500": {'IVV':1.0},
}
rows=[]
for tag, d in BOOKS.items():
    w = wvec(d); wer = float(ERV@w); sig = sigma_of(w); r = DECKM.T@w - wer
    for shift, lab in ((0.0,"base"), (-0.02,"LS200")):
        o = both(r + shift, sig)
        geo = {m: float(np.exp(np.mean([np.log(o[m][c]["PathCE"]) for c in CONVS]))) for m in o}
        fg = o["fixed"][("robust","G")]; xg = o["flex"][("robust","G")]
        rows.append(dict(book=tag, deck=lab, geo_fixed=round(geo["fixed"],2), geo_flex=round(geo["flex"],2),
                         ruin_fixed=fg["ruin"], ruin_flex=xg["ruin"], P5_fixed=round(fg["P5"],2), P5_flex=round(xg["P5"],2),
                         yrs_reduced_pct=round(xg["yrs_red"]*100,1), draws_flex_vs_fixed=f"{xg['draw']/1.20*100:.0f}%"))
    print("done", tag, flush=True)
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("out/FlexDraw_Experiment_20260901.csv", index=False)

# ======== SECTION 13: core.py R8 delta (2026-09-03) ========
# R8 (2026-09-03) delta in core.py — inserted after the R7B block:
R8_NEW = ['BIL','SHY','AGG','TIP','HYG','LQD','EMB','VNQ','XLU','XLI','XLF','XLP','XLK','KRE','AMLP','PDBC',
          'GUNR','REMX','URA','IEMG','EWJ','IBIT','TAIL','BTAL','MTUM','STIP','AQWA']
R8_HIST = ['AIRR','BOTZ','KOMP','PRNT','ROBO','SCHD','XT','PSCI','IBKR','VTI']   # deck rows that lacked history until etf-eod run 9
if _os0.environ.get("R8", "0") == "1":
    import json as _json
    UNIV += [t for t in R8_NEW + R8_HIST if t not in UNIV]
    L.ER.update({k: v for k, v in _json.load(open("ER_R8.json")).items() if k not in L.ER})
# class lists gained: SINGLES += IBKR; NUC += URA; BAL += BIL SHY AGG TIP STIP TAIL BTAL
# prices/: the 37 new histories were added from etf-eod branch commit a6102f0, truncated at 2026-08-21.

# ======== SECTION 14: build_r8.py ========
"""R8 candidate deck: R7cand + 27 asset-class-gap tickers (histories now available from
etf-eod) via the house 'added_proxy' method, plus two frame corrections (VTI, SCHD).
Rows are built directly on the 24-column R6/R7 frame from anchor rows (blends or
mean + beta*deviation) with documented overlays; empirical anchors/betas were read
from the 2016-09..2026-08 monthly panel (see Addendum 6). All derived rows graded C
(VTI graded B: identical-exposure transfer from IVV, beta 1.02 / corr 1.00).
Run with: R7=1 DECK_FILE=Deck_Recal_R7cand_20260901.csv SCEN_FILE=scenarios_24_R6_20260831.csv
Writes Deck_Recal_R8cand_20260903.csv, Row_Grades_R8cand_20260903.csv, ER_R8.json
"""
import json
import numpy as np, pandas as pd
import lib_pathce as L

D = L.DECK.copy()                      # R7cand rows (percent), 24 cols
SC = list(D.columns)
IX = {k: i for i, k in enumerate(['Base','Refl','Stag','Debt','Credit','AIProd','AIPower','TradeTech',
      'Cyber','Crony','China','Taiwan','USCredit','Space','Pandemic','ClimPol','ClimPhys','Energy',
      'Political','JGB','Defl','USD','War','Abund'])}
assert len(IX) == len(SC) == 24

R = {}                                  # working rows (24-vector, percent)
def row(t): return (R[t] if t in R else D.loc[t].astype(float).values).copy()
def mean(t): return float(row(t).mean())
def dev(t): r = row(t); return r - r.mean()
def ov(**kw):
    v = np.zeros(24)
    for k, x in kw.items(): v[IX[k]] = x
    return v

# ---- cash / rates ---------------------------------------------------------------
R['BIL'] = (mean('IEF') - 1.0 + 0.10*dev('IEF')
            + ov(Stag=1.5, Refl=0.5, Energy=1.0, AIPower=0.5, Debt=-1.0, Credit=-1.5, USCredit=-1.0,
                 Pandemic=-1.0, AIProd=-0.5, Taiwan=-0.5, Cyber=-0.5, China=-0.5, Political=-0.3,
                 Crony=-0.3, Defl=-3.0, USD=2.5, War=-0.5, Abund=-1.5))
R['SHY'] = 0.6*row('BIL') + 0.4*row('IEF')
R['LQD'] = (row('IEF') + 0.8 + 0.15*dev('IVV')
            + ov(Credit=-2.0, USCredit=-1.2, Defl=-1.0, Stag=-0.5, Cyber=-0.5, War=-0.5, Taiwan=-0.5,
                 Energy=-0.3, Refl=0.3, Abund=0.3))
R['AGG'] = 0.75*row('IEF') + 0.25*row('LQD')
R['HYG'] = (mean('IEF') + 1.5 + 0.35*dev('IVV') + 0.20*dev('IEF')
            + ov(Credit=-2.5, USCredit=-1.5, Defl=-2.0, Stag=-0.5, War=-1.0, Cyber=-0.5, Taiwan=-0.5,
                 China=-0.5, Pandemic=-0.5, Refl=0.5, Energy=0.5, Abund=0.5))
R['EMB'] = (mean('IEF') + 1.5 + 0.30*dev('AVES') + 0.40*dev('IEF')
            + ov(China=-1.5, Credit=-1.5, USCredit=-0.5, Defl=-1.0, USD=-1.0, War=-1.5, Taiwan=-1.0,
                 TradeTech=-0.8, Cyber=-0.5, Refl=0.5, Energy=0.5))
R['TIP']  = row('SCHP')                 # same index (Bloomberg US TIPS): beta 1.01 / corr 1.00
R['STIP'] = row('VTIP')                 # 0-5y TIPS: beta 0.98 / corr 1.00
# ---- hedges ----------------------------------------------------------------------
R['TAIL'] = (0.6*row('IEF') - 2.5 - 0.35*dev('IVV')
             + ov(Credit=1.0, Cyber=1.0, Taiwan=1.5, War=1.5, Defl=1.0, Pandemic=1.0, USCredit=0.5,
                  China=0.5, Refl=-0.5, AIProd=-0.5, Abund=-0.5))
R['BTAL'] = (2.0 - 0.45*dev('IVV')
             + ov(Credit=1.0, Defl=1.0, Cyber=0.5, Taiwan=0.5, War=0.5, Stag=1.0, AIProd=-2.0,
                  Abund=-2.0, Refl=-1.0, Space=-1.0))
# ---- US sectors ------------------------------------------------------------------
R['XLU'] = (mean('IVV') - 1.0 + 0.45*dev('IVV') + 0.5*dev('IEF')
            + ov(AIPower=3.0, Credit=1.0, Defl=1.0, Taiwan=1.0, China=0.5, Energy=0.5, ClimPol=0.5,
                 Abund=0.5, Pandemic=0.5, Cyber=-1.0, ClimPhys=-1.5, Stag=-0.5))
R['VNQ'] = (mean('IVV') - 1.5 + 0.7*dev('IVV') + 0.5*dev('IEF')
            + ov(Stag=-2.0, Debt=0.5, Credit=-1.5, USCredit=-2.0, Defl=-0.5, ClimPhys=-2.0, Pandemic=-1.5,
                 AIPower=1.0, AIProd=-0.5, Energy=-0.5, War=-0.5, Political=-0.3, Refl=0.3))
R['XLI'] = 0.6*row('IVV') + 0.4*row('PAVE') + ov(Space=1.0, Taiwan=1.5, War=1.5, Refl=0.3, Cyber=0.3, Credit=-0.5)
R['XLF'] = (0.6*row('IVV') + 0.4*row('BRKB')
            + ov(Credit=-2.5, USCredit=-3.0, Defl=-2.0, Debt=-1.0, Cyber=-1.5, Crony=-0.5, Political=-0.5,
                 War=-0.5, Refl=1.0, Stag=0.5, AIProd=0.5, Abund=0.3))
R['KRE'] = (row('XLF') - 1.0 + 0.3*dev('AVUV')
            + ov(USCredit=-4.0, Credit=-2.0, Defl=-2.5, Debt=-1.0, ClimPhys=-1.0, AIProd=-0.5, Cyber=-0.5,
                 Refl=1.5, Stag=0.5))
R['XLP'] = (0.5*row('USMV') + 0.5*row('IVV') - 1.0
            + ov(Credit=0.8, Cyber=0.5, Taiwan=1.0, War=0.5, Defl=1.0, Pandemic=1.0, Stag=0.5,
                 Refl=-0.5, AIProd=-1.0, Abund=-0.5, Energy=-0.5))
R['XLK'] = row('FTEC')                  # same sector exposure: beta 1.00 / corr 0.99
# ---- real assets / commodities ---------------------------------------------------
R['AMLP'] = (0.7*row('XLE') + 0.3*row('XLU') - 0.5
             + ov(AIPower=1.5, Energy=0.5, Stag=0.5, ClimPol=-2.0, Credit=-1.0, Defl=-1.0, Abund=-0.5))
R['PDBC'] = (0.4*row('XLE') + 0.3*row('GLDM') + 0.3*row('COPX') - 5.0
             + ov(Taiwan=2.5, War=2.0, USD=2.0, Stag=1.0, Energy=1.0, ClimPhys=1.0, Refl=0.5,
                  Credit=-0.5, Defl=-2.0, Abund=-1.5, AIProd=-1.5, China=-1.0, Pandemic=-1.0, USCredit=-1.0))
R['GUNR'] = (0.45*row('XLE') + 0.25*row('COPX') + 0.15*row('AVES') + 0.15*row('IXUS')
             + ov(ClimPhys=1.0, China=-0.5, Abund=-0.5))
R['REMX'] = (mean('SETM') - 2.0 + 1.3*dev('SETM')
             + ov(ClimPol=2.0, AIPower=1.0, TradeTech=0.5, USD=0.5, China=-2.0, Taiwan=-3.0, War=-2.0,
                  Cyber=-0.5, Abund=-0.5, Credit=-1.0))
R['URA']  = 0.7*row('URNM') + 0.3*row('NLR') - 0.3
R['AQWA'] = 0.5*row('PAVE') + 0.3*row('XLU') + 0.2*row('IXUS') - 0.5 + ov(ClimPhys=2.0, ClimPol=1.0)
# ---- international ---------------------------------------------------------------
R['IEMG'] = (0.5*row('AVES') + 0.3*row('EMXC') + 0.2*row('IXUS')
             + ov(China=-1.5, Taiwan=-2.0, TradeTech=-1.0, War=-1.0, Credit=-0.5, AIProd=0.5, Abund=0.5))
R['EWJ']  = (0.7*row('AVDE') + 0.3*row('IVV') - 0.5
             + ov(JGB=-4.0, Taiwan=-4.0, War=-3.0, China=-2.0, TradeTech=-0.5, Defl=-1.0,
                  Refl=1.0, USD=2.0, Debt=0.5, AIProd=1.0, AIPower=0.5, Space=0.5))
# ---- crypto / factor -------------------------------------------------------------
R['IBIT'] = (mean('COIN') - 3.0 + 0.9*dev('COIN')
             + ov(USD=6.0, Debt=3.0, Stag=1.0, Political=1.0, Cyber=-3.0, Crony=-1.0, Credit=-2.0,
                  Defl=-3.0, AIProd=-1.0, Abund=-1.0, Taiwan=-1.0, War=-1.0))
R['MTUM'] = row('SPMO')                 # beta 0.96 / corr 0.96
# ---- frame corrections (old rows sat in a stale, lower frame: VTI base 6.5 vs IVV 9.8) --
R['VTI']  = row('IVV') - 0.1
R['SCHD'] = 0.6*row('VTV') + 0.4*row('QUAL') - 0.3 + ov(Credit=0.5, Cyber=0.3, Taiwan=0.3, AIProd=-1.0, Abund=-0.5)

NEW = [t for t in R if t not in D.index]
REPLACED = [t for t in R if t in D.index]
out = D.copy()
for t, v in R.items(): out.loc[t] = v
out.round(4).reset_index().rename(columns={"index": "Ticker"}).to_csv("Deck_Recal_R8cand_20260903.csv", index=False)

g = pd.read_csv("Row_Grades_R7cand_20260901.csv")
g = g[~g.Ticker.isin(REPLACED)]
add = [dict(Ticker=t, Grade="C", Meaning="model-derived (R8 proxy-blend), 25% haircut by default") for t in NEW]
add.append(dict(Ticker="VTI", Grade="B", Meaning="R8 re-anchor: IVV row -0.1 (beta 1.02, corr 1.00); old row was in a stale frame"))
add.append(dict(Ticker="SCHD", Grade="C", Meaning="R8 re-anchor: 0.6 VTV + 0.4 QUAL blend; old row was in a stale frame"))
pd.concat([g, pd.DataFrame(add)]).to_csv("Row_Grades_R8cand_20260903.csv", index=False)

ER_R8 = {"BIL":0.1356,"SHY":0.15,"AGG":0.03,"TIP":0.18,"HYG":0.49,"LQD":0.14,"EMB":0.39,"VNQ":0.13,"XLU":0.08,
         "XLI":0.08,"XLF":0.08,"XLP":0.08,"XLK":0.08,"KRE":0.35,"AMLP":0.85,"PDBC":0.59,"GUNR":0.46,"REMX":0.53,
         "URA":0.69,"IEMG":0.09,"EWJ":0.50,"IBIT":0.25,"TAIL":0.59,"BTAL":0.45,"MTUM":0.15,"STIP":0.03,"AQWA":0.40,
         "AIRR":0.70,"BOTZ":0.68,"KOMP":0.20,"PRNT":0.66,"ROBO":0.95,"SCHD":0.06,"XT":0.46,"PSCI":0.29,"IBKR":0.0,"VTI":0.03}
json.dump(ER_R8, open("ER_R8.json", "w"), indent=1)

print("R8 candidate deck:", out.shape, "| new rows:", len(NEW), "| replaced:", REPLACED)
print(f"{'tkr':5s} {'mean':>6s} {'worst':>6s} {'wcol':>9s} {'Stag':>6s} {'Credit':>7s} {'Taiwan':>7s} {'Defl':>6s} {'USD':>6s} {'War':>6s} {'Abund':>6s}")
for t in R:
    r = R[t]; wi = int(np.argmin(r))
    print(f"{t:5s} {r.mean():6.2f} {r.min():6.2f} {list(IX)[wi]:>9s} {r[IX['Stag']]:6.1f} {r[IX['Credit']]:7.1f} {r[IX['Taiwan']]:7.1f} {r[IX['Defl']]:6.1f} {r[IX['USD']]:6.1f} {r[IX['War']]:6.1f} {r[IX['Abund']]:6.1f}")

# ======== SECTION 15: r8_test.py (audit, marginals, polish) ========
"""R8: row audit for the 37 newly-historied tickers, marginal adds into R3b, concentration
stress, and bounded polish of R3b / R3d over the widened universe (full standing battery:
R8 deck, C-haircut deck, -200bp, like-for-like vs P1).
Run: R7=1 R8=1 DECK_FILE=Deck_Recal_R8cand_20260903.csv SCEN_FILE=scenarios_24_R6_20260831.csv NPATHS=10000
"""
import sys, numpy as np, pandas as pd
import lib_pathce as L
from core import (UNIV, N, IDX, DECKM, ERV, sigma_of, wvec, P1v, CONVS, FE, feasible, R8_NEW, R8_HIST)

grades = pd.read_csv("Row_Grades_R8cand_20260903.csv").set_index("Ticker")["Grade"]
colmed = np.median(DECKM, axis=0)
HCC = DECKM.copy()
for i, t in enumerate(UNIV):
    if grades.get(t, 'B') == 'C':
        HCC[i] = 0.75*DECKM[i] + 0.25*colmed

# ---------------- 1. row-consistency audit ----------------
mr = L.monthly_returns()
win = mr.loc[pd.Period("2018-11","M"):pd.Period("2026-08","M")]
ivv = win["IVV"]; thr = ivv.quantile(0.10); bad = ivv <= thr
deck_worst = pd.Series({t: DECKM[IDX[t]].min()*100 for t in UNIV})
pctl = deck_worst.rank(pct=True)*100
rows = []
for t in R8_NEW + R8_HIST:
    s = win[t]; n = int(s.notna().sum())
    wd = float(s[bad].mean()*100); c = float(s.corr(ivv))
    v48 = float(mr.loc[pd.Period("2022-09","M"):pd.Period("2026-08","M")][t].std(ddof=1)*np.sqrt(12)*100)
    flag = ""
    if pctl[t] >= 80 and wd < ivv[bad].mean()*100: flag = "optimistic-tail"
    if pctl[t] <= 20 and wd > 0: flag = "pessimistic-tail"
    rows.append(dict(T=t, n=n, corrIVV=round(c,2), worstDec=round(wd,1), vol48=round(v48,1),
                     deck_worst=round(deck_worst[t],2), deck_pctl=round(pctl[t],0), grade=grades.get(t,'?'), flag=flag))
aud = pd.DataFrame(rows); aud.to_csv("out/Row_Audit_R8_20260903.csv", index=False)
print(f"=== row audit (IVV worst-decile mean {ivv[bad].mean()*100:.1f}%, {int(bad.sum())} months) ===")
print(aud.to_string(index=False))

# ---------------- 2. evaluation / gates ----------------
def ev(w, M, shift=0.0, full=True):
    r_net = M.T @ w - float(ERV @ w) + shift
    sig = sigma_of(w)
    res = {(p,c): FE.run(r_net, p, c, sig) for p,c in CONVS}
    score = float(np.mean([np.log(res[k]["PathCE"]) for k in CONVS]))
    out = dict(score=score, sigma=sig, res=res, worst=float(r_net.min()))
    if full:
        out.update(RCEW=L.rcew40(r_net)["robust"], DRB6=L.drb40(r_net,h=0.06),
                   R3AF=L.r3af(r_net, L.prow(L.EVG11)-L.wer_evg11()))
    return out
DECKS = {"R8":(DECKM,0.0),"HCC":(HCC,0.0),"LS":(DECKM,-0.02)}
BASES = {k: ev(P1v,M,s) for k,(M,s) in DECKS.items()}
def fails_vs(e,b):
    f=[]
    for c in CONVS:
        if e["res"][c]["P10"]<b["res"][c]["P10"]-1e-9: f.append("P10")
        if e["res"][c]["P5"]<b["res"][c]["P5"]-1e-9: f.append("P5")
        if e["res"][c]["ruin"]>b["res"][c]["ruin"]: f.append("ruin")
    if e["worst"]<b["worst"]-1e-12: f.append("worst")
    if e["RCEW"]<b["RCEW"]-1e-9: f.append("RCEW")
    if e["R3AF"]<b["R3AF"]-0.002: f.append("R3AF")
    if e["DRB6"]>b["DRB6"]+1e-12: f.append("DRB6")
    return f
def gates_all(w):
    es={}
    for k,(M,s) in DECKS.items():
        e=ev(w,M,s); f=fails_vs(e,BASES[k])
        if f: return None,(k,f)
        es[k]=e
    return es,None
def mm(es): return min(es["R8"]["score"], es["HCC"]["score"])

R3b = {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.12,'CIBR':.08,'GRID':.05,'CME':.05,'BRKB':.04,'CTA':.05,'URNM':.03,'XAR':.03,'CEG':.03,'FNV':.04,'MSCI':.03}
R3d = {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.10,'CIBR':.08,'GRID':.05,'CME':.05,'MSCI':.04,'CTA':.05,'URNM':.03,'XAR':.03,'TLT':.04,'FNV':.04,'BRKB':.04}
base_es,why = gates_all(wvec(R3b)); assert why is None, why
base_mm = mm(base_es)
print(f"\nR3b on R8 deck: maximin={np.exp(base_mm):.2f} (R8 {np.exp(base_es['R8']['score']):.2f} / HCC {np.exp(base_es['HCC']['score']):.2f} / LS {np.exp(base_es['LS']['score']):.2f})")

# ---------------- 3. marginal adds: 4% into R3b, position-neutral (drop CEG 3, NUKZ -1) ----------------
funded = dict(R3b); funded.pop('CEG'); funded['NUKZ'] -= .01
print("\n=== marginal adds to R3b at 4% (CEG out, NUKZ -1; maximin vs R3b) ===")
marg=[]; mrows=[]
for t in R8_NEW + R8_HIST:
    d = dict(funded); d[t]=0.04
    w = wvec(d)
    if not feasible(w): print(f"{t:5s}: infeasible"); mrows.append(dict(T=t, result="infeasible")); continue
    es,why = gates_all(w)
    if why:
        print(f"{t:5s}: FAIL@{why[0]}: {sorted(set(why[1]))}"); mrows.append(dict(T=t, result=f"FAIL@{why[0]}:{'/'.join(sorted(set(why[1])))}")); continue
    s = mm(es); marg.append((s,t,w))
    lg = es['LS']['res'][('robust','G')]
    print(f"{t:5s}: maximin={np.exp(s):.2f} ({np.exp(s-base_mm)-1:+.1%} vs R3b) R8={np.exp(es['R8']['score']):.2f} HCC={np.exp(es['HCC']['score']):.2f} LS-P5={lg['P5']:.2f} ruin={lg['ruin']} vol={es['R8']['sigma']*100:.2f}")
    mrows.append(dict(T=t, result="PASS", maximin=round(float(np.exp(s)),2), d_vs_R3b_pct=round((np.exp(s-base_mm)-1)*100,1),
                      geo_R8=round(float(np.exp(es['R8']['score'])),2), geo_HCC=round(float(np.exp(es['HCC']['score'])),2),
                      LS_P5=round(lg['P5'],2), LS_ruin=lg['ruin'], vol48=round(es['R8']['sigma']*100,2)))
pd.DataFrame(mrows).to_csv("out/R8_Marginal_20260903.csv", index=False)

print("\n=== concentration stress on positive entrants (entrant row 25%->median, vs P1 like-for-like) ===")
for s,t,w in sorted(marg, key=lambda x:-x[0]):
    if s <= base_mm: break
    CS = DECKM.copy(); CS[IDX[t]] = 0.75*DECKM[IDX[t]] + 0.25*colmed
    e = ev(w, CS); b = ev(P1v, CS); f = fails_vs(e,b)
    print(f"{t:5s}: {'PASS' if not f else 'FAIL: '+', '.join(sorted(set(f)))}  geoCE@CS={np.exp(e['score']):.2f}")
sys.stdout.flush()

# ---------------- 4. bounded polish over the widened universe ----------------
rng = np.random.default_rng(8)
def polish(w0, iters=40, sample=30):
    es,why = gates_all(w0)
    if why: return None
    best=(mm(es), w0.copy(), es)
    for it in range(iters):
        _,cur_w,_=best; src=np.where(cur_w>1e-9)[0]; improved=False
        for _ in range(sample):
            i=int(rng.choice(src)); j=int(rng.integers(N))
            if i==j: continue
            wn=cur_w.copy(); take=min(0.01 if rng.random()<0.8 else 0.02, wn[i])
            if wn[i]-take<0.02-1e-9 and wn[i]-take>1e-9: take=wn[i]-0.02
            if take<=1e-9: continue
            wn[i]-=take
            if wn[i]<1e-9: wn[i]=0
            add=take
            if wn[j]<1e-9 and add<0.02-1e-9:
                extra=0.02-add
                if wn[i]>=extra+0.02 or abs(wn[i]-extra)<1e-9:
                    wn[i]-=extra; add=0.02
                    if wn[i]<1e-9: wn[i]=0
                else: continue
            wn[j]+=add
            if not feasible(wn): continue
            q=ev(wn,DECKM,full=False)
            if q["score"]<=best[0]+1e-5: continue
            es2,why2=gates_all(wn)
            if why2: continue
            s2=mm(es2)
            if s2>best[0]+1e-5: best=(s2,wn.copy(),es2); improved=True
        if not improved and it>10: break
    return best
print("\n=== polish over widened universe ===")
for nm,d in (("R3b",R3b),("R3d",R3d)):
    r=polish(wvec(d))
    if r:
        sc,w,es=r
        hold={UNIV[i]:round(float(w[i])*100,1) for i in np.where(w>1e-9)[0]}
        new_in={k:v for k,v in hold.items() if k in R8_NEW+R8_HIST}
        e=es['R8']['res'][('robust','G')]; lg=es['LS']['res'][('robust','G')]
        print(f"from {nm}: maximin {np.exp(sc):.2f} (start {np.exp(mm(gates_all(wvec(d))[0])):.2f}) | robG PathCE {e['PathCE']:.2f} P5 {e['P5']:.2f} ruin {e['ruin']} | LS P5 {lg['P5']:.2f} ruin {lg['ruin']} | new entrants: {new_in if new_in else 'none'}")
        print("   ", dict(sorted(hold.items(), key=lambda kv:-kv[1])))
        np.save(f"out/r8_{nm}.npy", w)
    sys.stdout.flush()
print("DONE")

# ======== SECTION 16: r8_ballast.py (ballast ladder; r8_test_lib = sections 1-2 of r8_test.py) ========
"""Ballast ladder: 4% sleeves into R3b (CEG out, NUKZ -1) for existing vs new ballast/hedge rows."""
import numpy as np, pandas as pd
from r8_test_lib import *  # noqa
funded = dict(R3b); funded.pop('CEG'); funded['NUKZ'] -= .01
b8 = base_es['R8']['res'][('robust','G')]; bl = base_es['LS']['res'][('robust','G')]
print(f"R3b      : maximin={np.exp(base_mm):.2f} | robG P5={b8['P5']:.2f} ruin={b8['ruin']} | LS P5={bl['P5']:.2f} ruin={bl['ruin']} | vol={base_es['R8']['sigma']*100:.2f}")
rows=[]
for t in ['TLT','IEF','GOVZ','KMLM','SCHP','VTIP','BIL','SHY','AGG','STIP','TIP','LQD','HYG','BTAL','TAIL','XLU','XLP']:
    d=dict(funded); d[t]=d.get(t,0)+0.04; w=wvec(d)
    if not feasible(w): print(t,"infeasible"); continue
    es,why=gates_all(w)
    if why: print(f"{t:5s}: FAIL@{why[0]}: {sorted(set(why[1]))}"); continue
    s=mm(es); e=es['R8']['res'][('robust','G')]; lg=es['LS']['res'][('robust','G')]
    dce=(np.exp(s-base_mm)-1)*100; druin=(lg['ruin']/bl['ruin']-1)*100
    print(f"{t:5s}: maximin={np.exp(s):.2f} ({dce:+.1f}%) | robG P5={e['P5']:.2f} ruin={e['ruin']} | LS P5={lg['P5']:.2f} ruin={lg['ruin']} ({druin:+.0f}%) | vol={es['R8']['sigma']*100:.2f} | CE cost per 10% LS-ruin cut={(-dce/max(-druin,1e-9))*10:.2f}%")
    rows.append(dict(sleeve=t, maximin=round(float(np.exp(s)),2), dCE_pct=round(dce,1), P5_robG=round(e['P5'],2), ruin_robG=e['ruin'], LS_P5=round(lg['P5'],2), LS_ruin=lg['ruin'], dLSruin_pct=round(druin,0), vol48=round(es['R8']['sigma']*100,2)))
pd.DataFrame(rows).to_csv("out/R8_Ballast_Ladder_20260903.csv", index=False)

# ======== SECTION 17: r8_verify.py (60k engine-exact, R8 deck) + r8_trendstress2.py ========
"""60k engine-exact verification of books on the R8 candidate deck + flexible-draw
experiment. Run: R7=1 DECK_FILE=Deck_Recal_R7cand_20260901.csv SCEN_FILE=scenarios_24_R6_20260831.csv
"""
import numpy as np, pandas as pd
import lib_pathce as L
from core import UNIV, IDX, DECKM, ERV, sigma_of, wvec, P1v, CONVS
from r6_verify import engine_exact

grades = pd.read_csv("Row_Grades_R8cand_20260903.csv").set_index("Ticker")["Grade"]
colmed = np.median(DECKM, axis=0)
HCC = DECKM.copy()
for i, t in enumerate(UNIV):
    if grades.get(t, 'B') == 'C': HCC[i] = 0.75*DECKM[i] + 0.25*colmed

BOOKS = {
 "P1": dict(L.P1),
 "R3b": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.12,'CIBR':.08,'GRID':.05,'CME':.05,'BRKB':.04,'CTA':.05,'URNM':.03,'XAR':.03,'CEG':.03,'FNV':.04,'MSCI':.03},
 "R3d": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.10,'CIBR':.08,'GRID':.05,'CME':.05,'MSCI':.04,'CTA':.05,'URNM':.03,'XAR':.03,'TLT':.04,'FNV':.04,'BRKB':.04},
 "R3e_KMLM": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.11,'CIBR':.08,'GRID':.05,'CME':.05,'BRKB':.04,'CTA':.05,'URNM':.03,'XAR':.03,'FNV':.04,'MSCI':.03,'KMLM':.04},
 "R3dK": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.10,'CIBR':.08,'GRID':.05,'CME':.05,'MSCI':.04,'CTA':.05,'URNM':.03,'XAR':.03,'KMLM':.04,'FNV':.04,'BRKB':.04},
 "R3b_AGG4": {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.11,'CIBR':.08,'GRID':.05,'CME':.05,'BRKB':.04,'CTA':.05,'URNM':.03,'XAR':.03,'FNV':.04,'MSCI':.03,'AGG':.04},
}

def suite(w, draw_rule=None):
    wer = float(ERV @ w); sig = sigma_of(w); r = DECKM.T @ w - wer
    mc = engine_exact(r, sig); mch = engine_exact(HCC.T @ w - wer, sig); mcl = engine_exact(r - 0.02, sig)
    geo = lambda m: float(np.exp(np.mean([np.log(m[c]["PathCE"]) for c in CONVS])))
    return dict(geo=geo(mc), geoh=geo(mch), geol=geo(mcl), mc=mc, mcl=mcl, r=r, sig=sig, wer=wer)

P1s = suite(P1v)
print("=== 60k, R8 deck: books vs P1 gates (R7 / HCC / LS) ===")
rows=[]
for tag, d in BOOKS.items():
    w = wvec(d); s = suite(w)
    fails=[]
    for dk, a_mc, b_mc in (("R8", s["mc"], P1s["mc"]), ("LS", s["mcl"], P1s["mcl"])):
        for c in CONVS:
            if a_mc[c]["P10"] < b_mc[c]["P10"]-1e-9: fails.append(f"P10@{dk}")
            if a_mc[c]["P5"]  < b_mc[c]["P5"]-1e-9: fails.append(f"P5@{dk}")
            if a_mc[c]["ruin"] > b_mc[c]["ruin"]: fails.append(f"ruin@{dk}")
    rg = s["mc"][("robust","G")]; lg = s["mcl"][("robust","G")]
    to = 50*np.abs(w-P1v).sum()
    drb6 = L.drb40(s["r"], h=0.06)
    print(f"{tag:10s} geo={s['geo']:.2f} HCC={s['geoh']:.2f} LS={s['geol']:.2f} | robG PathCE={rg['PathCE']:.2f} P5={rg['P5']:.2f} ruin={rg['ruin']} | LS P5={lg['P5']:.2f} ruin={lg['ruin']} | worst={s['r'].min()*100:.2f}% DRB6={drb6:.4f} WER={s['wer']*100:.3f} vol={s['sig']*100:.2f} turn={to:.0f}% | {'PASS' if not fails else 'FAIL '+','.join(sorted(set(fails)))}")
    rows.append(dict(Portfolio=tag, geo_PathCE=round(s['geo'],2), geo_Chaircut=round(s['geoh'],2), geo_LS200=round(s['geol'],2),
                     PathCE_robG=round(rg['PathCE'],2), P5_robG=round(rg['P5'],2), ruin_robG=rg['ruin'],
                     LS_P5_robG=round(lg['P5'],2), LS_ruin_robG=lg['ruin'], worst_pct=round(float(s['r'].min())*100,2),
                     DRB40_h6=round(drb6,4), WER_pct=round(s['wer']*100,3), vol48_pct=round(s['sig']*100,2), turnover_pct=round(to,0)))
pd.DataFrame(rows).to_csv("out/R8_Books_Metrics_20260903.csv", index=False)


# ---- r8_trendstress2.py ----
"""Trend 'no crisis alpha' stress: flatten DBMF/CTA/KMLM rows toward their own row mean
(50% and 100%) — removes scenario convexity but keeps the mean. Gates like-for-like vs P1."""
import numpy as np
from r8_test_lib import *  # noqa
TREND=['DBMF','CTA','KMLM']
R3dK = {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.10,'CIBR':.08,'GRID':.05,'CME':.05,'MSCI':.04,'CTA':.05,'URNM':.03,'XAR':.03,'KMLM':.04,'FNV':.04,'BRKB':.04}
for lam in (0.5, 1.0):
    CS = DECKM.copy()
    for t in TREND:
        r = DECKM[IDX[t]]; CS[IDX[t]] = (1-lam)*r + lam*r.mean()
    b = ev(P1v, CS); bg=b['res'][('robust','G')]
    print(f"\n--- flatten trend rows {lam:.0%} toward own mean --- P1: geo={np.exp(b['score']):.2f} P5={bg['P5']:.2f} ruin={bg['ruin']} worst={b['worst']*100:.2f}%")
    for nm,d in (("R3b",R3b),("R3d",R3d),("R3dK",R3dK)):
        w=wvec(d); e=ev(w,CS); f=fails_vs(e,b); rg=e['res'][('robust','G')]; e0=ev(w,DECKM)
        print(f"{nm:5s}: geo={np.exp(e['score']):.2f} ({np.exp(e['score']-e0['score'])-1:+.1%} vs unstressed) P5={rg['P5']:.2f} ruin={rg['ruin']} worst={e['worst']*100:.2f}% | {'PASS' if not f else 'FAIL: '+', '.join(sorted(set(f)))}")

# ======== SECTION 18: app_parity.py + app_parity2.py (2026-09-03) — the app's canonical scorer (canonical_critmin_pathce.py, user's Drive copy, sha256 8be415a7…) run on campaign decks ========
"""Run the app's canonical scorer (canonical_critmin_pathce.py, unchanged) on the campaign's decks
and price panel, to (1) test whether it reproduces the 2026-08-27 store target PathCE_v2 = 12.35
for P1 on the R1R4 deck, and (2) put its numbers beside enginev2 on the R8 deck.
Gross = app convention (rows as they are); net = rows minus each holding's expense ratio."""
import os, io, zipfile, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/work")
import canonical_critmin_pathce as A

# price zip from the canonical snapshot files (same files behind enginev2's panel)
ZIP = "/tmp/app_prices.zip"
with zipfile.ZipFile(ZIP, "w") as z:
    for f in os.listdir("prices"):
        if f.endswith(".csv") and f != "manifest.csv":
            z.write(os.path.join("prices", f), f)

ER = json.load(open("ER_R8.json"))
import lib_pathce as L   # for the baseline ER block (needs env DECK/SCEN, set by caller)
ER.update({k: v for k, v in L.ER.items()})

BOOKS = json.load(open("pkg/enginev2/books.json")); BOOKS = {k: v for k, v in BOOKS.items() if not k.startswith("_")}

def net_deck(mat, weights):
    m = mat.set_index("Ticker").copy()
    for t in weights:
        m.loc[t] = m.loc[t] - ER[t]          # percent units
    return m.reset_index()

def run(deck_file, scen_file, names, label, npaths=60000):
    mat = pd.read_csv(deck_file); mat["Ticker"] = mat["Ticker"].str.strip()
    scen = pd.read_csv(scen_file)
    print(f"\n=== {label}: {deck_file} × {scen_file} ===")
    print(f"{'book':10s} {'gross PathCE':>12s} {'net PathCE':>10s} | {'net P10':>7s} {'net P5':>6s} {'deplete%':>8s} {'RCEW40':>7s} {'DRB40@5':>7s} {'R3AF':>6s} {'vol':>6s} | canon/equal/stress (net)")
    rows = []
    for nm in names:
        w = BOOKS[nm]
        try:
            g, _ = A.score_portfolio(nm, dict(w), mat, scen, ZIP, n_paths=npaths, seed=20260711)
            n, _ = A.score_portfolio(nm, dict(w), net_deck(mat, w), scen, ZIP, n_paths=npaths, seed=20260711)
        except Exception as e:
            print(f"{nm:10s} ERROR {e}"); continue
        print(f"{nm:10s} {g['PathCE']:12.2f} {n['PathCE']:10.2f} | {n['P10_TW']:7.2f} {n['P5_TW']:6.2f} {n['P_deplete']*100:8.2f} {n['RCEW40']:7.2f} {n['DRB40']:7.4f} {n['R3AF']:6.3f} {n['Vol']*100:6.2f} | {n['PathCE_canonical']:.2f}/{n['PathCE_equal']:.2f}/{n['PathCE_stress']:.2f}")
        rows.append(dict(deck=label, book=nm, gross_PathCE=round(g['PathCE'],2), net_PathCE=round(n['PathCE'],2), net_P10=round(n['P10_TW'],2), net_P5=round(n['P5_TW'],2),
                         net_deplete_pct=round(n['P_deplete']*100,2), net_RCEW40=round(n['RCEW40'],2), net_DRB40_h5=round(n['DRB40'],4), net_R3AF=round(n['R3AF'],3), vol48=round(n['Vol']*100,2),
                         net_PathCE_canonical=round(n['PathCE_canonical'],2), net_PathCE_equal=round(n['PathCE_equal'],2), net_PathCE_stress=round(n['PathCE_stress'],2)))
    return rows

out = []
out += run("Deck_Recal_R1R4_20260822.csv", "scenarios_20_canonical.csv", ["P1", "P2", "SP500", "60_40"], "R1R4 / 20 scenarios (store frame of 2026-08-27)")
out += run("Deck_Recal_R8cand_20260903.csv", "scenarios_24_R6_20260831.csv", ["P1", "P2", "R2_final", "R1b_final", "R3b", "R3d", "R3dK", "SP500", "60_40"], "R8cand / 24 scenarios")
pd.DataFrame(out).to_csv("out/App_Parity_20260903.csv", index=False)

# ---- app_parity2.py ----
"""Variants of the app scorer for P1 on the R1R4 deck, hunting the 2026-08-27 store target 12.35."""
import os, zipfile, json, sys, math
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/work")
import canonical_critmin_pathce as A, lib_pathce as L
ZIP="/tmp/app_prices.zip"
mat=pd.read_csv("Deck_Recal_R1R4_20260822.csv"); mat["Ticker"]=mat["Ticker"].str.strip(); scen=pd.read_csv("scenarios_20_canonical.csv")
P1=dict(L.P1); ER={t:L.ER[t] for t in P1}
def net(m,w): m=m.set_index("Ticker").copy(); [m.__setitem__(slice(None),None) for _ in ()]; 
def netdeck(m,w,scale=1.0):
    m=m.set_index("Ticker").copy()
    for t in w: m.loc[t]=m.loc[t]-scale*ER[t]
    return m.reset_index()
# (a) baseline net, seeds 20260711 and 42 and 20260702 (seed sensitivity)
for seed in (20260711, 20260702, 42):
    n,_=A.score_portfolio("P1",dict(P1),netdeck(mat,P1),scen,ZIP,n_paths=60000,seed=seed)
    print(f"net, seed {seed}: PathCE={n['PathCE']:.2f} canon={n['PathCE_canonical']:.2f} equal={n['PathCE_equal']:.2f} stress={n['PathCE_stress']:.2f} vol={n['Vol']*100:.2f}")
# (b) vol window ending 2026-07 (archive truncated before August)
import io
with zipfile.ZipFile("/tmp/app_prices_jul.zip","w") as z:
    for f in os.listdir("prices"):
        if f.endswith(".csv") and f!="manifest.csv":
            df=pd.read_csv(os.path.join("prices",f)); df=df[pd.to_datetime(df.Date,utc=True)<=pd.Timestamp("2026-07-31",tz="UTC")]
            z.writestr(f, df.to_csv(index=False))
n,_=A.score_portfolio("P1",dict(P1),netdeck(mat,P1),scen,"/tmp/app_prices_jul.zip",n_paths=60000,seed=20260711)
print(f"net, archive to 2026-07-31: PathCE={n['PathCE']:.2f} vol={n['Vol']*100:.2f}")
# (c) ER sensitivity: +-0.10pp WER
for sc in (0.8, 1.2):
    n,_=A.score_portfolio("P1",dict(P1),netdeck(mat,P1,sc),scen,ZIP,n_paths=60000,seed=20260711)
    print(f"net with ER x{sc}: PathCE={n['PathCE']:.2f}")
# (d) canonical weights only (no 3-vector blend), net
n,_=A.score_portfolio("P1",dict(P1),netdeck(mat,P1),scen,ZIP,n_paths=60000,seed=20260711)
print(f"canonical-only net component: {n['PathCE_canonical']:.2f}; equal-only {n['PathCE_equal']:.2f}")

# ======== SECTION 19: regime_metrics.py (2026-09-03) — persistence sensitivity, bad-start, dispersion, empirical 48m ========
"""Regime-robust metric set for the six recent books (R8 deck, 60k engine-exact)."""
import numpy as np, pandas as pd, json
import lib_pathce as L
from core import DECKM, ERV, sigma_of, wvec, CONVS, UNIV, IDX
books=json.load(open("pkg/enginev2/books.json"))
NAMES=["R3b","R1c","R1b_final","R2_final","R3dK","R3d"]
bench=L.prow(L.EVG11)-L.wer_evg11()

def engine(r_net, sigma, lam=None, start=None):
    """Engine-exact streams; optional Markov rate override and forced starting scenario."""
    lam0=L.LAM
    if lam is not None: L.LAM=lam
    out={}
    for pin,pi in (("robust",L.PI_ROBUST),("canonical",L.PI_CANON)):
        rng=np.random.default_rng(42)
        for conv in ("G","L"):
            if start is None:
                out[(pin,conv)]=L.run_engine(rng, r_net+0.0, pi, conv, sigma, 0.0)
            else:
                NP_,T=L.NP_,L.T
                rn=r_net; s=np.full(NP_, start); W=np.ones(NP_); ru=np.zeros(NP_,bool)
                rng.choice(len(pi), size=NP_, p=pi)   # consume the same draw as run_engine for stream parity
                for _ in range(T):
                    j=rng.random(NP_)<L.LAM
                    if j.any(): s[j]=rng.choice(len(pi), size=int(j.sum()), p=pi)
                    mu=np.log1p(rn[s])-(sigma*sigma/2 if conv=="G" else 0.0)
                    W=W*np.exp(mu+sigma*rng.standard_normal(NP_))-L.DRAW
                    ru|=(W<=0); W=np.maximum(W,1e-9)
                lg=np.log(W)
                out[(pin,conv)]=dict(PathCE=float(np.exp(lg.mean())),P50=float(np.median(W)),P10=float(np.quantile(W,.10)),P5=float(np.quantile(W,.05)),ruin=int(ru.sum()))
    L.LAM=lam0
    return out
geo=lambda mc: float(np.exp(np.mean([np.log(mc[c]["PathCE"]) for c in CONVS])))

mr=L.monthly_returns(); win=mr.loc[pd.Period("2018-11","M"):pd.Period("2026-08","M")]
ivv=win["IVV"]; bad=ivv<=ivv.quantile(0.10)
def port_series(d, frame):
    sub=frame[list(d)]; wv=np.array([d[t] for t in sub.columns]); live=sub.notna().values
    Wm=live*wv; Wm=Wm/Wm.sum(1,keepdims=True); return pd.Series(np.nansum(sub.values*Wm,1), index=frame.index)

rows=[]
for nm in NAMES:
    d=books[nm]; w=wvec(d); wer=float(ERV@w); sig=sigma_of(w); r=DECKM.T@w-wer
    base=engine(r,sig); ls=engine(r-0.02,sig)
    fast=engine(r,sig,lam=0.25); slow=engine(r,sig,lam=1/16)
    worst_i=int(np.argmin(r)); badstart=engine(r,sig,start=worst_i)
    rg=base[("robust","G")]; bg=badstart[("robust","G")]
    hd=L.hist_diagnostics(d)
    ps=port_series(d, win)
    rows.append({
      "Port":nm,
      "geo-PathCE (λ=1/8)":round(geo(base),2),
      "geo-PathCE λ=1/4 / λ=1/16":f"{geo(fast):.2f} / {geo(slow):.2f}",
      "ruin rob-G λ=1/8 / 1/4 / 1/16":f"{rg['ruin']} / {fast[('robust','G')]['ruin']} / {slow[('robust','G')]['ruin']}",
      "P5 rob-G":round(rg["P5"],2),
      "−200bp ruin rob-G":ls[("robust","G")]["ruin"],
      "bad-start (worst scen. first): PathCE / P5 / ruin":f"{bg['PathCE']:.2f} / {bg['P5']:.2f} / {bg['ruin']}",
      "worst net CAGR %":round(float(r.min())*100,2),
      "scenario CAGR spread: std / range (pp)":f"{np.std(r)*100:.2f} / {(r.max()-r.min())*100:.2f}",
      "scenarios < 6% net (of 24)":int((r<0.06).sum()),
      "RCEW40 robust":round(L.rcew40(r)["robust"],2),
      "R3AF vs EVG11":round(L.r3af(r,bench),3),
      "vol48 %":round(sig*100,2),
      "Sortino 48m":round(hd["sortino"],2) if "sortino" in hd else np.nan,
      "maxDD 48m %":round(hd["maxdd"]*100,1) if "maxdd" in hd else np.nan,
      "CAGR 48m %":round(hd["cagr"]*100,1) if "cagr" in hd else np.nan,
      "IVV worst-decile months: book mean %":round(float(ps[bad].mean()*100),1),
      "corr to IVV (2018-11..)":round(float(ps.corr(ivv)),2),
    })
df=pd.DataFrame(rows).set_index("Port").T
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",40)
print(df.to_string())
df.to_csv("out/Regime_Metrics_20260903.csv")

# ======== SECTION 20: R8b deck revision (FNV premium +1.0 -> +0.5; BRKB credit deltas +1.5 -> +1.0; grades C -> B) — see Addendum 8; r8b_verify.py = r8_verify.py with the R8b grades file and the BOOKS below ========
# BOOKS added: R3b_B5 = R3b with BRKB 5 / NUKZ 11; R3b_F5B5 = R3b with FNV 5 / BRKB 5 / NUKZ 11 / CEG 2

# ======== SECTION 21: policy_engine.py (2026-09-04) — real draw, guardrail on real wealth, two-account taxes (10-year rule), stress vol ========
"""Policy-basis engine (2026-09-03): real (inflation-indexed) draw, flexible-draw guardrail on real
wealth, two-account draw sourcing with taxes (taxable brokerage + inherited IRA under the 10-year rule),
and a stress-volatility variant. Same scenario Markov process, seeds and conventions as lib_pathce.run_engine.

Run: R7=1 R8=1 DECK_FILE=Deck_Recal_R8b_20260903.csv SCEN_FILE=scenarios_24_R6_20260831.csv python3 policy_engine.py
"""
import json, sys
import numpy as np, pandas as pd
import lib_pathce as L
from core import DECKM, ERV, sigma_of, wvec, CONVS, UNIV

INFL = 0.025          # steady inflation; deck rows are nominal, draw is indexed
DRAW0 = 0.03          # first-year draw, share of initial corpus (real spending need)
FLEX = dict(lo=0.80, cut=0.75, hi=1.00)   # guardrail on REAL wealth vs initial
# Illustrative tax / account parameters — replace with actual split and brackets
TAX = dict(ira_share=0.50,   # inherited IRA as share of initial corpus
           t_ord=0.27,       # ordinary rate on IRA distributions (federal 22% bracket + Missouri)
           t_cg=0.20,        # long-term gains rate on brokerage sales (15% federal + state)
           gain0=0.50,       # unrealized-gain share of the brokerage account at start
           ira_years=10)     # 10-year rule: IRA emptied straight-line over years 1..10

def run(rng, r_net, pi, conv, sigma, *, indexed=True, flex=False, tax=None):
    """One convention. Returns real terminal-wealth stats (units of initial corpus, today's prices)."""
    NP_, T, LAM = L.NP_, L.T, L.LAM
    s = rng.choice(len(pi), size=NP_, p=pi)
    W = np.ones(NP_)                                   # total nominal wealth
    if tax:
        ira = np.full(NP_, tax["ira_share"]); brk = 1.0 - ira
        basis = brk * (1.0 - tax["gain0"])             # brokerage cost basis
    ru = np.zeros(NP_, bool); red = np.zeros(NP_, bool); lvl = 1.0
    yrs_red = np.zeros(NP_); taxes_paid = np.zeros(NP_)
    for t in range(1, T + 1):
        j = rng.random(NP_) < LAM
        if j.any(): s[j] = rng.choice(len(pi), size=int(j.sum()), p=pi)
        mu = np.log1p(r_net[s]) - (sigma * sigma / 2 if conv == "G" else 0.0)
        g = np.exp(mu + sigma * rng.standard_normal(NP_))
        if tax:
            ira = ira * g; brk = brk * g               # both accounts earn the portfolio return
            W = ira + brk
        else:
            W = W * g
        if indexed: lvl *= (1 + INFL)
        need = DRAW0 * lvl                              # spending need (nominal), real 3% of initial
        if flex:
            real = W / lvl
            red = np.where(real < FLEX["lo"], True, np.where(real >= FLEX["hi"], False, red))
            need = np.where(red, need * FLEX["cut"], need)
            yrs_red += red
        if tax:
            # forced IRA distribution under the 10-year rule (straight-line), taxed as ordinary income
            forced = ira / max(tax["ira_years"] - t + 1, 1) if t <= tax["ira_years"] else np.zeros(NP_)
            # spending first from IRA money (forced, then extra if needed), net of ordinary tax
            take_ira = np.minimum(np.maximum(forced, need / (1 - tax["t_ord"])), ira)
            net_ira = take_ira * (1 - tax["t_ord"]); taxes_paid += take_ira * tax["t_ord"]
            ira = ira - take_ira
            surplus = np.maximum(net_ira - need, 0.0); shortfall = np.maximum(need - net_ira, 0.0)
            # surplus after-tax IRA money is reinvested in the brokerage (adds to basis)
            brk = brk + surplus; basis = basis + surplus
            # shortfall funded by brokerage sales, gains taxed
            gf = np.where(brk > 1e-12, np.clip(1.0 - basis / np.maximum(brk, 1e-12), 0.0, 1.0), 0.0)
            gross = shortfall / np.maximum(1 - tax["t_cg"] * gf, 1e-9)
            gross = np.minimum(gross, brk)
            taxes_paid += gross * gf * tax["t_cg"]
            basis = basis * np.where(brk > 1e-12, 1 - gross / np.maximum(brk, 1e-12), 1.0)
            brk = brk - gross
            W = ira + brk
            ru |= (W <= 1e-9) | ((shortfall > 0) & (gross < shortfall - 1e-12))
            ira = np.maximum(ira, 0.0); brk = np.maximum(brk, 1e-9); basis = np.minimum(basis, brk)
        else:
            W = W - need
            ru |= (W <= 0)
        W = np.maximum(W, 1e-9)
    real = W / lvl if indexed else W
    lg = np.log(np.maximum(real, 1e-9))
    return dict(PathCE=float(np.exp(lg.mean())), P10=float(np.quantile(real, .10)), P5=float(np.quantile(real, .05)),
                ruin=int(ru.sum()), yrs_red=float(yrs_red.mean() / T), tax_paid=float(taxes_paid.mean() / lvl))

def suite(r_net, sigma, **kw):
    out = {}
    for pin, pi in (("robust", L.PI_ROBUST), ("canonical", L.PI_CANON)):
        rng = np.random.default_rng(42)
        for conv in ("G", "L"):
            out[(pin, conv)] = run(rng, r_net, pi, conv, sigma, **kw)
    return out

def geo(mc): return float(np.exp(np.mean([np.log(mc[c]["PathCE"]) for c in CONVS])))

def stress_vol(d, start="2010-01", end="2026-08", n=48, min_live=0.70):
    """Highest live-weight-renormalized 48-month vol over rolling windows in [start, end], using only
    windows in which at least min_live of the book's weight has history in every month (so the
    window measures this book, not a handful of its oldest holdings)."""
    mr = L.monthly_returns().loc[pd.Period(start, "M"):pd.Period(end, "M")]
    sub = mr[list(d)]; wv = np.array([d[t] for t in sub.columns]); live = sub.notna().values
    livew = (live * wv).sum(1)
    Wm = live * wv; Wm = Wm / np.maximum(Wm.sum(1, keepdims=True), 1e-12)
    pr = pd.Series(np.nansum(sub.values * Wm, 1), index=sub.index)
    roll = pr.rolling(n).std(ddof=1) * np.sqrt(12)
    ok = pd.Series(livew, index=sub.index).rolling(n).min() >= min_live
    roll = roll.where(ok)
    i = roll.idxmax(); return float(roll.max()), str(i - (n - 1)), str(i)

if __name__ == "__main__":
    books = json.load(open("pkg/enginev2/books.json"))
    books["R3b_B5"] = {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.11,'CIBR':.08,'GRID':.05,'CME':.05,'BRKB':.05,'CTA':.05,'URNM':.03,'XAR':.03,'CEG':.03,'FNV':.04,'MSCI':.03}
    NAMES = ["P2", "R3b", "R3b_B5", "R3dK", "R1b_final", "R2_final"]
    THEM = ['SHLD','NUKZ','CIBR','GRID','XAR','URNM','CEG','JEDI']; SING = ['CME','MSCI','CEG','CBOE','BRKB','FNV']
    S4 = {"P2":0.06,"R3b":0.10,"R3b_B5":0.10,"R3dK":0.12,"R1b_final":0.10,"R2_final":0.08}
    rows = []
    for nm in NAMES:
        d = books[nm]; w = wvec(d); wer = float(ERV @ w); sig = sigma_of(w); r = DECKM.T @ w - wer
        A = suite(r, sig, indexed=True, flex=False)                   # real draw, fixed
        B = suite(r, sig, indexed=True, flex=True)                    # real draw + guardrail (policy basis)
        BL = suite(r - 0.02, sig, indexed=True, flex=True)            # policy basis under -200bp
        C = suite(r, sig, indexed=True, flex=True, tax=TAX)           # + taxes / 10-year rule (illustrative)
        sv, w0, w1 = stress_vol(d)
        D = suite(r, sv, indexed=True, flex=True)                     # + stress volatility
        wt = sum(d.get(t, 0) for t in THEM); ws = sum(d.get(t, 0) for t in SING)
        slope = (np.log(geo(B)) - np.log(geo(BL))) / 2
        pen = 0.6 * slope + 0.8 * wt * slope + S4[nm] + 0.30 * slope + 0.25 * ws
        rg = lambda X: X[("robust", "G")]
        rows.append(dict(Port=nm,
            real_fixed_geo=round(geo(A), 2), real_fixed_P5=round(rg(A)["P5"], 2), real_fixed_ruin=rg(A)["ruin"],
            policy_geo=round(geo(B), 2), policy_P5=round(rg(B)["P5"], 2), policy_ruin=rg(B)["ruin"], policy_yrs_reduced_pct=round(rg(B)["yrs_red"] * 100, 1),
            policy_LS_geo=round(geo(BL), 2), policy_LS_ruin=rg(BL)["ruin"], slope=round(slope, 2), surprise_adj_CE=round(geo(B) * np.exp(-pen), 2),
            tax_geo=round(geo(C), 2), tax_ruin=rg(C)["ruin"], tax_paid_real=round(rg(C)["tax_paid"], 3),
            stress_vol_pct=round(sv * 100, 2), stress_window=f"{w0}..{w1}", stressvol_geo=round(geo(D), 2), stressvol_ruin=rg(D)["ruin"]))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows).set_index("Port")
    df.to_csv("out/Policy_Basis_20260903.csv")
    pd.set_option("display.width", 250); print(df.T.to_string())
