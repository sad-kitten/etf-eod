"""Bench_2FFL_Code_20260919 — code bundle for EngineV2_Bench_2FFL_vs_C5smh3_20260919.md (v2: 50/60/70% AVUV + Target Retirement 2070).
Stack = patch-0001 lib_pathce.py + core.py (unchanged apart from the ENGINEV2_EXTRA hook) + the three modules below + standing.py v3.3
(verbatim from StandingRules_v33_Code_20260910.py). Env: DECK_FILE=Deck_Recal_R8k_20260910.csv SCEN_FILE=scenarios_25_R8i_20260909.csv
GRADES_FILE=Row_Grades_R8j_20260909.csv ENGINEV2_PRICES=<etf-eod@008a5a5>/prices ENGINEV2_END=2026-08 NPATHS=6000 ENGINEV2_EXTRA=IXUS
Run: python3 report_cmp.py books_cmp2.json out.csv (books JSON at the end; "shift" = parallel pp shift of a deck row, sensitivity only)."""

# ===== core.py hook (added lines) =====
"""Shared optimization core (patch 0001, enginev2/core.py) with an ENGINEV2_EXTRA hook: extra tickers appended AFTER the
standard universe so that every base-universe index, DECKM row order and the C-haircut column median (computed over the
first N_BASE rows in search_B) are unchanged."""
import numpy as np, pandas as pd
import lib_pathce as L

UNIV = ['GLDM','SHLD','DBMF','NUKZ','CIBR','GRID','CTA','CME','JEDI','MSCI',
        'SMH','ROKT','XAR','URNM','CEG','SETM','PPA','ITA','EUAD','PAVE',
        'QTUM','KMLM','IEF','CBOE','ICE','SPGI','NDAQ','NLR','XSD','SOXX',
        'QUAL','COWZ','AVUV','XLE','HACK','MOAT','XLV','XBI','TLT','GOVZ','IAI','IVV']
UNIV += ['LTPZ','SCHP','VTIP','GDX','WPM','FNV','PSLV','COPX','BRKB','WM','INDA','JEPI','SPMO']      # R7
UNIV += ['V','IGV','FTEC','COIN','EMXC','EWT','SIL','VTV','SCHG','IBB']                              # R7B
R8_NEW = ['BIL','SHY','AGG','TIP','HYG','LQD','EMB','VNQ','XLU','XLI','XLF','XLP','XLK','KRE','AMLP','PDBC',
          'GUNR','REMX','URA','IEMG','EWJ','IBIT','TAIL','BTAL','MTUM','STIP','AQWA']
R8_HIST = ['AIRR','BOTZ','KOMP','PRNT','ROBO','SCHD','XT','PSCI','IBKR','VTI']
UNIV += [t for t in R8_NEW + R8_HIST if t not in UNIV]
import os as _os
if _os.environ.get("ENGINEV2_RPRX", "0") == "1": UNIV += ["RPRX"]
N_BASE = len(UNIV)
EXTRA = [t for t in _os.environ.get("ENGINEV2_EXTRA", "").split(",") if t and t not in UNIV]
UNIV += EXTRA
N = len(UNIV)
IDX = {t:i for i,t in enumerate(UNIV)}
SINGLES = ['CME','MSCI','CEG','CBOE','ICE','SPGI','NDAQ','BRKB','WM','WPM','FNV','V','COIN','IBKR']
PMEQ = ['GDX','WPM','FNV','PSLV','SIL']
SEMIS = ['SMH','SOXX','XSD']
DEF = ['SHLD','XAR','PPA','ITA','EUAD','ROKT','JEDI']
NUC = ['NUKZ','NLR','URNM','CEG','SETM','GRID','URA']
BAL = ['GLDM','DBMF','CTA','KMLM','IEF','TLT','GOVZ','LTPZ','SCHP','VTIP','BIL','SHY','AGG','TIP','STIP','TAIL','BTAL']


# ===== search_B.py (reconstruction) =====
"""search_B — the parts standing.py / report_v3.py read (reconstructed from search_B_20260904.py + Addendum 15 risk-share
machinery + the GRADES_FILE convention of the v3 bundles). C-haircut column median is taken over the BASE universe rows so
that ENGINEV2_EXTRA tickers do not move any standing book's HCC deck."""
import os, numpy as np, pandas as pd
import lib_pathce as L
from core import UNIV, N, N_BASE, IDX, DECKM, ERV, sigma_of, wvec, P1v, CONVS, FE, feasible, SINGLES, idx48, mr

grades = pd.read_csv(os.environ.get("GRADES_FILE", os.path.join(L.DECKS, "Row_Grades_R8cand_20260903.csv"))).set_index("Ticker")["Grade"]
colmed = np.median(DECKM[:N_BASE], axis=0); HCC = DECKM.copy()
for i, t in enumerate(UNIV):
    if grades.get(t, "B") == "C": HCC[i] = 0.75 * DECKM[i] + 0.25 * colmed
INFL = 0.025; SS = 45000.0 / 1999798.78; T = L.T

# ---- 48-month covariance (Addendum 15): share of book variance = w_i (Sigma w)_i / (w' Sigma w) ----
_W48 = mr.reindex(idx48)
_cols = [t for t in UNIV if t in _W48.columns]
_c = _W48[_cols].cov(min_periods=18).values * 12.0
COV = np.zeros((N, N)); _ci = [IDX[t] for t in _cols]
for a, ia in enumerate(_ci): COV[ia, _ci] = _c[a]
COV = np.nan_to_num(COV)
NUC_SET = ["NUKZ", "URNM", "CEG", "SETM", "NLR", "URA"]
def risk_shares(w):
    Cv = COV @ w; var = float(w @ Cv)
    mrc = Cv * w / var if var > 1e-12 else np.zeros_like(w)
    return float(sum(mrc[IDX[t]] for t in NUC_SET if t in IDX)), float(mrc.max())

def run_B(r_net, pin, conv, sigma):
    s = FE.S[pin]; mu = np.log1p(r_net)[s]
    if conv == "G": mu = mu - sigma * sigma / 2
    X = np.exp(mu + sigma * FE.Z) / (1.0 + INFL)
    W = np.ones(FE.npaths); red = np.zeros(FE.npaths, bool); ru = np.zeros(FE.npaths, bool)
    for t in range(1, T + 1):
        W = W * X[:, t - 1]
        red = np.where(W < 0.80, True, np.where(W >= 1.0, False, red))
        draw = np.maximum(0.03 * np.where(red, 0.75, 1.0) - (SS if t >= 11 else 0.0), 0.0)
        W = W - draw; ru |= W <= 0; W = np.maximum(W, 1e-9)
    lg = np.log(W)
    return dict(PathCE=float(np.exp(lg.mean())), P5=float(np.quantile(W, .05)), P10=float(np.quantile(W, .10)), ruin=int(ru.sum()))

def n_singles(w): return int(sum(w[IDX[t]] > 1e-9 for t in SINGLES if t in IDX))
def held(w): return [t for t in UNIV if w[IDX[t]] > 1e-9]
def show(w): return " ".join(f"{t}{int(round(w[IDX[t]]*100))}" for t in sorted(held(w), key=lambda t: -w[IDX[t]]))
POOL = [t for t in UNIV if t not in SINGLES and t != "QTUM"]
S1 = [t for t in SINGLES if t in IDX]

# ===== policy_engine2.py (per-scenario sigma patch) =====
"""Account-aware policy engine (policy_engine2_20260904.py) with the Addendum 31 item-11 patch: sigma may be a per-scenario
vector (indexed by the path's current scenario); a scalar sigma reproduces the 2026-09-04 behaviour exactly."""
import json, sys
import numpy as np, pandas as pd
import lib_pathce as L
from core import DECKM, ERV, sigma_of, wvec, CONVS

INFL = 0.025
ACCT = dict(ira=384120.14 + 67216.37, brk=199677.93 + 596544.64 + 129629.35, roth=500318.19, hsa=122292.16)
W0 = float(sum(ACCT.values()))
DRAW0 = 0.03 * W0
FLEX = dict(lo=0.80, cut=0.75, hi=1.00)
FED = {
    "single": dict(std=16100, std65=2050, cg0=49450, cg15=545500,
                   br=[(12400, .10), (50400, .12), (105700, .22), (201775, .24), (256225, .32), (640600, .35), (np.inf, .37)]),
    "mfj":    dict(std=32200, std65=3300, cg0=98900, cg15=613700,
                   br=[(24800, .10), (100800, .12), (211400, .22), (403550, .24), (512450, .32), (768700, .35), (np.inf, .37)])}
MO_RATE = 0.047
POL = dict(status="mfj", fill=0.10, ira_years=10, gain0=0.50, y_ord=0.010, y_qual=0.005, age65_year=7, amort_r=0.05,
           rmd_f0=27.1, ss_start=11, ss=45000.0, ss_taxable=0.85, ss_offsets=True)

def bracket_tax(ti, br):
    tax = np.zeros_like(ti); lo = 0.0
    for top, rate in br:
        tax += np.clip(ti - lo, 0.0, top - lo) * rate; lo = top
    return tax

def fed_tax(ordinc, pref, P, std):
    ti_ord = np.maximum(ordinc - std, 0.0)
    ti_pref = np.maximum(pref - np.maximum(std - ordinc, 0.0), 0.0)
    lo = ti_ord; hi = ti_ord + ti_pref
    a15 = np.maximum(0.0, np.minimum(hi, P["cg15"]) - np.maximum(lo, P["cg0"]))
    a20 = np.maximum(0.0, hi - np.maximum(lo, P["cg15"]))
    return bracket_tax(ti_ord, P["br"]) + 0.15 * a15 + 0.20 * a20

def mo_tax(ordinc, qual, std):
    return MO_RATE * np.maximum(ordinc + qual - std, 0.0)

def run(rng, r_net, pi, conv, sigma, pol=POL):
    NP_, T, LAM = L.NP_, L.T, L.LAM
    P = FED[pol["status"]]; fill_top = [top for top, rate in P["br"] if rate == pol["fill"]][0]
    N = pol["ira_years"]; r = pol["amort_r"]
    sigma = np.asarray(sigma, dtype=float)
    vec = sigma.ndim == 1
    s = rng.choice(len(pi), size=NP_, p=pi)
    ira = np.full(NP_, ACCT["ira"]); brk = np.full(NP_, ACCT["brk"]); roth = np.full(NP_, ACCT["roth"] + ACCT["hsa"])
    basis = brk * (1.0 - pol["gain0"])
    ru = np.zeros(NP_, bool); red = np.zeros(NP_, bool); brk_out = np.zeros(NP_, bool); roth_tap = np.zeros(NP_, bool)
    yrs_red = np.zeros(NP_); tax_paid = np.zeros(NP_); tax_yr = np.zeros(T); ira_yr = np.zeros(T)
    Wts = []
    for t in range(1, T + 1):
        j = rng.random(NP_) < LAM
        if j.any(): s[j] = rng.choice(len(pi), size=int(j.sum()), p=pi)
        sg = sigma[s] if vec else sigma
        mu = np.log1p(r_net[s]) - (sg * sg / 2 if conv == "G" else 0.0)
        g = np.exp(mu + sg * rng.standard_normal(NP_)) / (1.0 + INFL)
        d_ord = pol["y_ord"] * brk; d_qual = pol["y_qual"] * brk
        rmd = ira / max(pol["rmd_f0"] - (t - 1), 1.0)
        ira = ira * g; roth = roth * g; brk = np.maximum(brk * g - d_ord - d_qual, 0.0)
        basis = basis / (1.0 + INFL)
        W = ira + brk + roth
        red = np.where(W < FLEX["lo"] * W0, True, np.where(W >= FLEX["hi"] * W0, False, red))
        need = DRAW0 * np.where(red, FLEX["cut"], 1.0); yrs_red += red
        std = P["std"] + (P["std65"] if t >= pol["age65_year"] else 0.0)
        ss = pol["ss"] if t >= pol["ss_start"] else 0.0
        ss_ord = pol["ss_taxable"] * ss
        if t < N:
            n_left = N - t + 1
            room = np.maximum(0.0, std + fill_top - d_ord - ss_ord)
            amort = ira * r / (1.0 - (1.0 + r) ** (-n_left))
            need_gross = np.maximum(0.0, need - d_ord - d_qual - (ss if pol["ss_offsets"] else 0.0)) / (1.0 - pol["fill"] - MO_RATE)
            take = np.minimum(ira, np.maximum(np.maximum(np.maximum(room, amort), need_gross), rmd))
        elif t == N:
            take = ira.copy()
        else:
            take = np.zeros(NP_)
        ira = ira - take; ira_yr[t - 1] = take.mean()
        ordinc = d_ord + take
        gf = np.where(brk > 1e-9, np.clip(1.0 - basis / np.maximum(brk, 1e-9), 0.0, 1.0), 0.0)
        base_cash = d_ord + d_qual + take + (ss if pol["ss_offsets"] else 0.0)
        def total_tax(S):
            t_all = fed_tax(ordinc + ss_ord, d_qual + S * gf, P, std) + mo_tax(ordinc, d_qual, std)
            if pol["ss_offsets"] or ss == 0.0: return t_all
            return fed_tax(ordinc, d_qual + S * gf, P, std) + mo_tax(ordinc, d_qual, std)
        S = np.zeros(NP_)
        for _ in range(20):
            S = np.maximum(0.0, need + total_tax(S) - base_cash)
        S = np.minimum(S, brk)
        tax = total_tax(S)
        cash = base_cash + S - tax
        surplus = np.maximum(cash - need, 0.0); short = np.maximum(need - cash, 0.0)
        basis = basis * np.where(brk > 1e-9, 1.0 - S / np.maximum(brk, 1e-9), 1.0); brk = brk - S
        brk = brk + surplus; basis = basis + surplus
        take_r = np.minimum(short, roth); roth = roth - take_r; roth_tap |= take_r > 1.0
        ru |= (short - take_r) > 1.0
        brk_out |= brk <= 1.0
        tax_paid += tax; tax_yr[t - 1] = tax.mean()
        ira = np.maximum(ira, 0.0); brk = np.maximum(brk, 0.0); roth = np.maximum(roth, 0.0); basis = np.minimum(basis, brk)
        if t in (10, 20, 30): Wts.append((ira + brk + roth) / W0)
    W = (ira + brk + roth) / W0
    lg = np.log(np.maximum(W, 1e-9))
    return dict(PathCE=float(np.exp(lg.mean())), P10=float(np.quantile(W, .10)), P5=float(np.quantile(W, .05)),
                ruin=int(ru.sum()), yrs_red=float(yrs_red.mean() / T), tax_paid=float(tax_paid.mean() / W0),
                tax_yr=tax_yr, ira_yr=ira_yr, brk_out=float(brk_out.mean()), roth_tap=float(roth_tap.mean()),
                roth_share=float((roth / W0 / np.maximum(W, 1e-9)).mean()), W=W, W10=Wts[0], W20=Wts[1], W30=Wts[2])

def suite(r_net, sigma, pol=POL):
    out = {}
    for pin, pi in (("robust", L.PI_ROBUST), ("canonical", L.PI_CANON)):
        rng = np.random.default_rng(42)
        sg = sigma[pin] if isinstance(sigma, dict) else sigma
        for conv in ("G", "L"):
            out[(pin, conv)] = run(rng, r_net, pi, conv, sg, pol)
    return out

def geo(mc): return float(np.exp(np.mean([np.log(mc[c]["PathCE"]) for c in CONVS])))

# ===== report_cmp.py =====
"""report_v3.profile() (v3.2/v3.3 reporting set, engine-exact 60k x 4 conventions, regime sigma) with (a) an optional WER
override so a book can carry a fund-level expense ratio, (b) extra path/tail columns, (c) no symmetric-deck column
(R8j_sym not needed for the comparison). Usage: python report_cmp.py books.json out.csv"""
import sys, json, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, policy_engine2 as E, standing as ST, search_B as S
from core import DECKM, ERV, wvec, UNIV, IDX
CONV = [("robust","G"),("robust","L"),("canonical","G"),("canonical","L")]
def ce_cagr(W, gam):
    ce = float(np.exp(np.log(W).mean())) if gam == 1 else float(np.mean(W ** (1 - gam)) ** (1 / (1 - gam)))
    return ce, (ce ** (1 / 40.) - 1) * 100
def pooled(mc, key="W"): return np.concatenate([np.maximum(mc[c][key], 1e-9) for c in CONV])
def profile(nm, d, wer=None, full=True, shift=None):
    t0 = time.time()
    M = DECKM.copy(); H = S.HCC.copy()
    for t, pp in (shift or {}).items(): M[IDX[t]] += pp / 100.0; H[IDX[t]] += pp / 100.0   # parallel row shift (sensitivity only)
    w = wvec(d); wer = float(ERV @ w) if wer is None else float(wer); r = M.T @ w - wer
    sig = {p: ST.sigma_vec(w, p)[0] for p in ("robust", "canonical")}
    mc = E.suite(r, sig, E.POL); W = pooled(mc); rg = mc[("robust", "G")]
    m1 = E.suite(r - 0.01, sig, E.POL); m2 = E.suite(r - 0.02, sig, E.POL)
    mh = E.suite(H.T @ w - wer, sig, E.POL)
    g1x, g1 = ce_cagr(W, 1); g3x, g3 = ce_cagr(W, 3); g5x, g5 = ce_cagr(W, 5)
    pc = W ** (1 / 40.) - 1
    p5_mean = None
    if abs(rg["P5"] - ST.GATES["P5"]) < ST.P5_BAND:
        p5_mean = float(np.mean([E.run(np.random.default_rng(s), r, L.PI_ROBUST, "G", sig["robust"], E.POL)["P5"] for s in ST.P5_SEEDS]))
    fails = ST.gate_fails_abs(rg, m2[("robust", "G")], mh[("robust", "G")], r, p5_mean=p5_mean)
    st = ST.report_structure(w, d); st["ER %"] = round(wer * 100, 3)
    pi_mean = (np.array(L.PI_ROBUST) + np.array(L.PI_CANON)) / 2
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
        "shortfall6 (bp)": round(ST.shortfall6(r) * 1e4, 1),
        "DRB40@6 (reported)": round(L.drb40(r, h=0.06), 4), "RCEW40 (reported)": round(L.rcew40(r)["robust"], 2),
        "ruin": int(rg["ruin"]), "gates": "PASS" if not fails else "; ".join(fails),
        "constraints": "ok" if not ST.violations(w) else "; ".join(ST.violations(w)),
        "lifetime tax x": round(rg["tax_paid"], 3), "vol48 %": round(S.sigma_of(w) * 100, 2),
        # extras (not in the standing report)
        "pi-mean net CAGR %": round(float(pi_mean @ r) * 100, 2),
        "P10 terminal x": round(float(np.quantile(W, .10)), 3), "P50 terminal x": round(float(np.median(W)), 3),
        "P90 terminal x": round(float(np.quantile(W, .90)), 3),
        "P(W40<1) %": round(float((W < 1.0).mean()) * 100, 1), "P(W40<0.5) %": round(float((W < 0.5).mean()) * 100, 2),
        "P5 W10 / W20 / W30 (robG)": f"{np.quantile(rg['W10'],.05):.2f} / {np.quantile(rg['W20'],.05):.2f} / {np.quantile(rg['W30'],.05):.2f}",
        "yrs reduced draw %": round(rg["yrs_red"] * 100, 1), "roth tapped %": round(rg["roth_tap"] * 100, 2),
        "ruin (any conv)": int(sum(mc[c]["ruin"] for c in CONV)), "-200bp ruin (robG)": int(m2[("robust","G")]["ruin"]),
        "stress sigma ratio": round(ST.sigma_vec(w)[1], 3), "secs": round(time.time() - t0)}
    row.update(st); row["weights"] = " ".join(f"{t}{x*100:g}" for t, x in sorted(d.items(), key=lambda kv: -kv[1]))
    return row
if __name__ == "__main__":
    BK = json.load(open(sys.argv[1])); rows = []
    for nm, spec in BK.items():
        d = {t: v for t, v in spec["w"].items() if v > 0}
        rows.append(profile(nm, d, wer=spec.get("wer"), shift=spec.get("shift")))
        print(nm, rows[-1]["geo x"], rows[-1]["g3 CE-CAGR %"], rows[-1]["P5 terminal x"], rows[-1]["gates"], rows[-1]["secs"], "s", flush=True)
    df = pd.DataFrame(rows).set_index("Port"); df.to_csv(sys.argv[2])
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); print(df.drop(columns=["weights"]).T.to_string())

# ===== books_cmp2.json + books_sens.json =====
"""
{
 "C5_smh3_NUKZ": {
  "w": {
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
  }
 },
 "2FFL_50_TR2070": {
  "w": {
   "VTI": 0.2424,
   "IXUS": 0.1616,
   "AGG": 0.096,
   "AVUV": 0.5
  },
  "wer": 0.00165
 },
 "2FFL_60_TR2070": {
  "w": {
   "VTI": 0.1939,
   "IXUS": 0.1293,
   "AGG": 0.0768,
   "AVUV": 0.6
  },
  "wer": 0.00182
 },
 "2FFL_70_TR2070": {
  "w": {
   "VTI": 0.1454,
   "IXUS": 0.097,
   "AGG": 0.0576,
   "AVUV": 0.7
  },
  "wer": 0.00199
 },
 "2FFL_50_TR2070_90/10": {
  "w": {
   "VTI": 0.27,
   "IXUS": 0.18,
   "AGG": 0.05,
   "AVUV": 0.5
  },
  "wer": 0.00165
 },
 "2FFL_60_TR2070_90/10": {
  "w": {
   "VTI": 0.216,
   "IXUS": 0.144,
   "AGG": 0.04,
   "AVUV": 0.6
  },
  "wer": 0.00182
 },
 "2FFL_70_TR2070_90/10": {
  "w": {
   "VTI": 0.162,
   "IXUS": 0.108,
   "AGG": 0.03,
   "AVUV": 0.7
  },
  "wer": 0.00199
 }
}{
 "2FFL_60_TR2070_SCV+2pp": {
  "w": {
   "VTI": 0.1939,
   "IXUS": 0.1293,
   "AGG": 0.0768,
   "AVUV": 0.6
  },
  "wer": 0.00182,
  "shift": {
   "AVUV": 2.0
  }
 },
 "2FFL_60_TR2070_SCV+3.5pp": {
  "w": {
   "VTI": 0.1939,
   "IXUS": 0.1293,
   "AGG": 0.0768,
   "AVUV": 0.6
  },
  "wer": 0.00182,
  "shift": {
   "AVUV": 3.5
  }
 }
}"""
