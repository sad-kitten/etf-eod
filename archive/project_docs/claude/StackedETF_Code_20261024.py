# ===== build_stack.py =====
"""Synthetic deck rows + monthly series for return-stacked / capital-efficient ETFs (2026-10-24).
Rows are gross-of-ER nominal like every deck row; stacked notional financed at BIL + 0.30 pp implied futures spread."""
import sys, pandas as pd, numpy as np
deck_in, panel_in, deck_out, panel_out, grades = sys.argv[1:6]
D = pd.read_csv(deck_in).set_index("Ticker"); SP = 0.30
L = lambda t: D.loc[t]
tsy = 0.25*L("SHY") + 0.5*L("IEF") + 0.25*L("TLT")
rows = {
 "RSST": L("IVV") + (L("DBMF") - L("BIL")) - SP,
 "RSSB": L("VT") + (tsy - L("BIL")) - SP,
 "NTSX": 0.9*L("IVV") + 0.6*L("IEF") - 0.5*L("BIL") - 0.6*SP,
 "GDE":  0.9*L("IVV") + 0.9*L("GLDM") - 0.8*L("BIL") - 0.9*SP,
 # sensitivity rows: trend excess over cash halved
 "DBMFh": L("BIL") + 0.5*(L("DBMF") - L("BIL")),
 "RSSTh": L("IVV") + 0.5*(L("DBMF") - L("BIL")) - SP,
}
for k, v in rows.items(): D.loc[k] = v
D.reset_index().to_csv(deck_out, index=False)
M = pd.read_csv(panel_in, index_col=0); sp = SP/100/12
tsym = 0.25*M["SHY"] + 0.5*M["IEF"] + 0.25*M["TLT"]
M["RSST"] = M["IVV"] + M["DBMF"] - M["BIL"] - sp
M["RSSB"] = M["VT"] + tsym - M["BIL"] - sp
M["NTSX"] = 0.9*M["IVV"] + 0.6*M["IEF"] - 0.5*M["BIL"] - 0.6*sp
M["GDE"]  = 0.9*M["IVV"] + 0.9*M["GLDM"] - 0.8*M["BIL"] - 0.9*sp
M["DBMFh"] = M["DBMF"]; M["RSSTh"] = M["RSST"]
M.to_csv(panel_out)
G = pd.read_csv(grades)
G = pd.concat([G, pd.DataFrame({"Ticker": list(rows), "Grade": ["B"]*len(rows)})]); G.to_csv(grades, index=False)
print(pd.DataFrame(rows).T.iloc[:, [0, 11, 20, 21]].round(2))
print("48m vol (2022-09..2026-08):", {t: round(M.loc["2022-09":"2026-08", t].std()*np.sqrt(12)*100, 1) for t in ["IVV","DBMF","RSST","RSSB","NTSX","GDE"]})

# ===== lib_pathce.py =====
"""lib_pathce (patch-0001 enginev2/lib_pathce.py, verbatim logic) + ENGINEV2_T horizon env (T50 basis, Addendum 43)."""
import numpy as np, pandas as pd, os, json as _json

WORK = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ENGINEV2_DATA", os.path.join(WORK, "data"))
DECKS = os.path.join(DATA, "decks")
DECK = pd.read_csv(os.path.join(DECKS, os.environ.get("DECK_FILE", "Deck_Recal_R8m_20261006.csv")))
DECK["Ticker"] = DECK["Ticker"].str.strip()
DECK = DECK.set_index("Ticker")
SC = list(DECK.columns)
SCEN = pd.read_csv(os.path.join(DECKS, os.environ.get("SCEN_FILE", "scenarios_25_R8i_20260909.csv")))
W_CAN = SCEN["Weight_Normalized"].values
SEV = SCEN["Severity_1to5"].values.astype(float)
NSC = len(SC)
assert NSC == len(W_CAN), (NSC, len(W_CAN))
DECK.loc["IAI"] = DECK.loc[["CME", "CBOE", "NDAQ"]].mean()   # AR4 benchmark-only proxy (as in every prior run)

P1 = {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.15,'CIBR':.10,'GRID':.10,'CTA':.05,'CME':.05,'JEDI':.05,'MSCI':.05}
EVG11 = {'CIBR':.15,'SHLD':.15,'CTA':.10,'DBMF':.10,'GLDM':.10,'IAI':.10,'NUKZ':.10,'QTUM':.10,'XAR':.10}
ER = {
 'GLDM':0.10,'SHLD':0.50,'DBMF':0.85,'NUKZ':0.85,'CIBR':0.59,'GRID':0.65,'CTA':0.76,'CME':0.00,'JEDI':0.50,'MSCI':0.00,
 'SMH':0.35,'ROKT':0.45,'XAR':0.35,'URNM':0.75,'CEG':0.00,'SETM':0.65,'PPA':0.64,'ITA':0.40,'EUAD':0.75,'PAVE':0.47,'QTUM':0.40,'KMLM':0.90,
 'IEF':0.15,'CBOE':0.00,'ICE':0.00,'SPGI':0.00,'NDAQ':0.00,'NLR':0.61,'XSD':0.35,'SOXX':0.35,'QUAL':0.15,'COWZ':0.49,'AVUV':0.25,'XLE':0.09,
 'HACK':0.60,'IVV':0.03,'MOAT':0.46,'XLV':0.09,'XBI':0.35,'TLT':0.15,'GOVZ':0.10,'IAI':0.42,
 'LTPZ':0.20,'SCHP':0.03,'VTIP':0.04,'GDX':0.51,'WPM':0.00,'FNV':0.00,'PSLV':0.60,'COPX':0.65,'BRKB':0.00,'WM':0.00,'INDA':0.62,'JEPI':0.35,'SPMO':0.13,
 'V':0.00,'IGV':0.41,'FTEC':0.084,'COIN':0.00,'EMXC':0.25,'EWT':0.59,'SIL':0.65,'VTV':0.04,'SCHG':0.04,'IBB':0.45,
}
NP_, T, LAM, DRAW = 60000, int(os.environ.get("ENGINEV2_T", "40")), 0.125, 0.03
_er8 = os.path.join(DECKS, "ER_R8.json")
if os.path.exists(_er8):
    ER.update({k: v for k, v in _json.load(open(_er8)).items() if k not in ER})

def prow(b): return np.array([sum(b[t] * DECK.loc[t, s] for t in b) for s in SC]) / 100.0
def wer(b): return sum(b[t] * ER[t] for t in b) / 100.0
def make_pi_robust(bench=EVG11):
    evg = prow(bench); dsl = np.maximum(0.0, evg[0] - evg)
    st = np.sqrt(W_CAN) * SEV * dsl; st = st / st.sum()
    pi = 0.5 * W_CAN + 0.25 / NSC + 0.25 * st; return pi / pi.sum()
def stress_weights(bench=EVG11):
    evg = prow(bench); dsl = np.maximum(0.0, evg[0] - evg)
    st = np.sqrt(W_CAN) * SEV * dsl; return st / st.sum()
PI_ROBUST = make_pi_robust(); PI_CANON = W_CAN / W_CAN.sum(); ST_W = stress_weights()

class FastEval:
    def __init__(self, npaths=24000, seed=777):
        rng = np.random.default_rng(seed); self.npaths = npaths
        self.Z = rng.standard_normal((npaths, T)); self.S = {}
        for name, pi in (("robust", PI_ROBUST), ("canonical", PI_CANON)):
            s = np.empty((npaths, T), dtype=np.int8); cur = rng.choice(len(pi), size=npaths, p=pi)
            for t in range(T):
                j = rng.random(npaths) < LAM; nj = int(j.sum())
                if nj: cur = cur.copy(); cur[j] = rng.choice(len(pi), size=nj, p=pi)
                s[:, t] = cur
            self.S[name] = s
    def run(self, r_net, pi_name, conv, sigma):
        s = self.S[pi_name]; mu = np.log1p(r_net)[s]
        if conv == "G": mu = mu - sigma * sigma / 2
        W = np.ones(self.npaths); ru = np.zeros(self.npaths, bool); X = np.exp(mu + sigma * self.Z)
        for t in range(T):
            W = W * X[:, t] - DRAW; ru |= (W <= 0); W = np.maximum(W, 1e-9)
        lg = np.log(W)
        return dict(PathCE=float(np.exp(lg.mean())), P50=float(np.median(W)), P10=float(np.quantile(W, .10)),
                    P5=float(np.quantile(W, .05)), ruin=int(ru.sum()))

def tw_after_draw(g, T_=40, d=0.03):
    g = np.asarray(g, dtype=float); growth = (1.0 + g) ** T_
    return np.where(np.abs(g) < 1e-15, 1.0 - d * T_, growth - d * (growth - 1.0) / np.where(np.abs(g) < 1e-15, 1.0, g))
def log_cew(tm, w, eps=1e-12):
    w = np.asarray(w) / np.sum(w); return float(np.exp(np.sum(w * np.log(np.maximum(tm, eps)))))
def rcew40(r_net):
    tm = tw_after_draw(r_net)
    c = log_cew(tm, W_CAN); e = log_cew(tm, np.ones(NSC) / NSC); s = log_cew(tm, ST_W)
    robust = float(np.exp(0.50 * np.log(max(c, 1e-12)) + 0.25 * np.log(max(e, 1e-12)) + 0.25 * np.log(max(s, 1e-12))))
    return dict(canonical=c, equal=e, stress=s, robust=robust)
def drb40(r_net, h=0.05, eps=1e-9):
    tm = np.maximum(tw_after_draw(r_net), eps); twh = tw_after_draw(np.array([h]))[0]
    gap = np.maximum(0.0, np.log(twh / tm))
    leg = lambda w: float(np.sqrt(np.sum((np.asarray(w) / np.sum(w)) * gap ** 2)))
    return 0.50 * leg(W_CAN) + 0.25 * leg(np.ones(NSC) / NSC) + 0.25 * leg(ST_W)
def r3af(r_net, bench_net):
    w = W_CAN.copy(); r = np.asarray(r_net); b = np.asarray(bench_net)
    order = np.argsort(b); xs = b[order]; ws = w[order]; cw = np.cumsum(ws); cw = cw / cw[-1]
    ql = float(np.interp(0.20, cw, xs)); qh = float(np.interp(0.80, cw, xs))
    bad = b <= ql; good = b >= qh; mid = ~(bad | good)
    wm = lambda x, m: float(np.sum(w[m] * x[m]) / np.sum(w[m])) if np.sum(w[m]) > 0 else np.nan
    tail_alpha = wm(r, bad) - wm(b, bad)
    convexity = (wm(r, good) - wm(r, mid)) - (wm(r, mid) - wm(r, bad))
    bw = np.sum(w[bad] * b[bad]) / np.sum(w[bad]); rw = np.sum(w[bad] * r[bad]) / np.sum(w[bad])
    vb = np.sum(w[bad] * (b[bad] - bw) ** 2)
    beta = np.sum(w[bad] * (b[bad] - bw) * (r[bad] - rw)) / vb if vb > 1e-18 else np.nan
    ortho = 1.0 - min(2.0, abs(beta)) if np.isfinite(beta) else np.nan
    return 0.55 * tail_alpha + 0.25 * convexity + 0.20 * ortho
def wer_evg11():
    er_evg = dict(ER); er_evg.setdefault("IAI", 0.33)
    return sum(EVG11[t] * er_evg.get(t, 0.4) for t in EVG11) / 100.0

_MRET = None
def monthly_returns():
    global _MRET
    if _MRET is not None: return _MRET
    df = pd.read_csv(os.environ["ENGINEV2_PANEL"], index_col=0)
    df.columns = [c.replace("-", "") for c in df.columns]
    df.index = pd.PeriodIndex(df.index.astype(str), freq="M")
    _MRET = df.sort_index(); return _MRET

# ===== core.py =====
"""core (patch-0001 enginev2/core.py + ENGINEV2_EXTRA hook from Bench_2FFL_Code_20260919). The P1 FLOOR block is omitted
(unused by v3.4 report/search)."""
import numpy as np, pandas as pd, os as _os
import lib_pathce as L

UNIV = ['GLDM','SHLD','DBMF','NUKZ','CIBR','GRID','CTA','CME','JEDI','MSCI',
        'SMH','ROKT','XAR','URNM','CEG','SETM','PPA','ITA','EUAD','PAVE',
        'QTUM','KMLM','IEF','CBOE','ICE','SPGI','NDAQ','NLR','XSD','SOXX',
        'QUAL','COWZ','AVUV','XLE','HACK','MOAT','XLV','XBI','TLT','GOVZ','IAI','IVV']
UNIV += ['LTPZ','SCHP','VTIP','GDX','WPM','FNV','PSLV','COPX','BRKB','WM','INDA','JEPI','SPMO']
UNIV += ['V','IGV','FTEC','COIN','EMXC','EWT','SIL','VTV','SCHG','IBB']
R8_NEW = ['BIL','SHY','AGG','TIP','HYG','LQD','EMB','VNQ','XLU','XLI','XLF','XLP','XLK','KRE','AMLP','PDBC',
          'GUNR','REMX','URA','IEMG','EWJ','IBIT','TAIL','BTAL','MTUM','STIP','AQWA']
R8_HIST = ['AIRR','BOTZ','KOMP','PRNT','ROBO','SCHD','XT','PSCI','IBKR','VTI']
UNIV += [t for t in R8_NEW + R8_HIST if t not in UNIV]
N_BASE = len(UNIV)
EXTRA = [t for t in _os.environ.get("ENGINEV2_EXTRA", "").split(",") if t and t not in UNIV]
UNIV += EXTRA
N = len(UNIV)
IDX = {t: i for i, t in enumerate(UNIV)}
SINGLES = ['CME','MSCI','CEG','CBOE','ICE','SPGI','NDAQ','BRKB','WM','WPM','FNV','V','COIN','IBKR']
PMEQ = ['GDX','WPM','FNV','PSLV','SIL']
SEMIS = ['SMH','SOXX','XSD']
DEF = ['SHLD','XAR','PPA','ITA','EUAD','ROKT','JEDI']
NUC = ['NUKZ','NLR','URNM','CEG','SETM','GRID','URA']
BAL = ['RSST','RSSB','GLDM','DBMF','CTA','KMLM','IEF','TLT','GOVZ','LTPZ','SCHP','VTIP','BIL','SHY','AGG','TIP','STIP','TAIL','BTAL']

DECKM = np.array([[L.DECK.loc[t, s] for s in L.SC] for t in UNIV]) / 100.0
ERV = np.array([L.ER[t] for t in UNIV]) / 100.0
mr = L.monthly_returns()
END = pd.Period(_os.environ.get("ENGINEV2_END", "2026-08"), "M")
idx48 = pd.period_range(END - 47, END, freq="M")
R48 = mr.reindex(idx48).reindex(columns=UNIV).values
LIVE = ~np.isnan(R48); R48z = np.nan_to_num(R48)

def sigma_of(w):
    Wm = LIVE * w; s = Wm.sum(axis=1, keepdims=True)
    pr = (R48z * Wm).sum(axis=1) / s[:, 0]
    return float(np.std(pr, ddof=1) * np.sqrt(12))

FE = L.FastEval(npaths=int(_os.environ.get("NPATHS", "14000")), seed=777)
CONVS = [("robust","G"),("robust","L"),("canonical","G"),("canonical","L")]

def wvec(d):
    w = np.zeros(N)
    for t, x in d.items(): w[IDX[t]] = x
    return w / w.sum()
P1v = wvec(L.P1)

def feasible(w):
    if w.min() < -1e-12 or abs(w.sum()-1) > 1e-9: return False
    held = w > 1e-9
    if held.sum() > 14: return False
    if np.any(w[held] < 0.02 - 1e-9): return False
    if w.max() > 0.15 + 1e-9: return False
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

# ===== search_B.py =====
"""search_B (reconstruction, verbatim from Bench_2FFL_Code_20260919)."""
import os, numpy as np, pandas as pd
import lib_pathce as L
from core import UNIV, N, N_BASE, IDX, DECKM, ERV, sigma_of, wvec, P1v, CONVS, FE, feasible, SINGLES, idx48, mr

grades = pd.read_csv(os.environ.get("GRADES_FILE", os.path.join(L.DECKS, "Row_Grades_R8j_20260909.csv"))).set_index("Ticker")["Grade"]
colmed = np.median(DECKM[:N_BASE], axis=0); HCC = DECKM.copy()
for i, t in enumerate(UNIV):
    if grades.get(t, "B") == "C": HCC[i] = 0.75 * DECKM[i] + 0.25 * colmed
INFL = 0.025; SS = 45000.0 / 1999798.78; T = L.T

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

def n_singles(w): return int(sum(w[IDX[t]] > 1e-9 for t in SINGLES if t in IDX))
def held(w): return [t for t in UNIV if w[IDX[t]] > 1e-9]
def show(w): return " ".join(f"{t}{int(round(w[IDX[t]]*100))}" for t in sorted(held(w), key=lambda t: -w[IDX[t]]))
POOL = [t for t in UNIV if t not in SINGLES and t != "QTUM"]
S1 = [t for t in SINGLES if t in IDX]

# ===== policy_engine2.py =====
"""policy_engine2 (verbatim from Bench_2FFL_Code_20260919: account-aware tax/draw engine, per-scenario sigma patch)."""
import numpy as np
import lib_pathce as L
from core import CONVS

INFL = 0.025
ACCT = dict(ira=384120.14 + 67216.37, brk=199677.93 + 596544.64 + 129629.35, roth=500318.19, hsa=122292.16)
W0 = float(sum(ACCT.values())); DRAW0 = 0.03 * W0
FLEX = dict(lo=0.80, cut=0.75, hi=1.00)
FED = {"single": dict(std=16100, std65=2050, cg0=49450, cg15=545500,
                      br=[(12400, .10), (50400, .12), (105700, .22), (201775, .24), (256225, .32), (640600, .35), (np.inf, .37)]),
       "mfj": dict(std=32200, std65=3300, cg0=98900, cg15=613700,
                   br=[(24800, .10), (100800, .12), (211400, .22), (403550, .24), (512450, .32), (768700, .35), (np.inf, .37)])}
MO_RATE = 0.047
POL = dict(status="mfj", fill=0.10, ira_years=10, gain0=0.50, y_ord=0.010, y_qual=0.005, age65_year=7, amort_r=0.05,
           rmd_f0=27.1, ss_start=11, ss=45000.0, ss_taxable=0.85, ss_offsets=True)

def bracket_tax(ti, br):
    tax = np.zeros_like(ti); lo = 0.0
    for top, rate in br: tax += np.clip(ti - lo, 0.0, top - lo) * rate; lo = top
    return tax
def fed_tax(ordinc, pref, P, std):
    ti_ord = np.maximum(ordinc - std, 0.0); ti_pref = np.maximum(pref - np.maximum(std - ordinc, 0.0), 0.0)
    lo = ti_ord; hi = ti_ord + ti_pref
    a15 = np.maximum(0.0, np.minimum(hi, P["cg15"]) - np.maximum(lo, P["cg0"])); a20 = np.maximum(0.0, hi - np.maximum(lo, P["cg15"]))
    return bracket_tax(ti_ord, P["br"]) + 0.15 * a15 + 0.20 * a20
def mo_tax(ordinc, qual, std): return MO_RATE * np.maximum(ordinc + qual - std, 0.0)

def run(rng, r_net, pi, conv, sigma, pol=POL):
    NP_, T, LAM = L.NP_, L.T, L.LAM
    P = FED[pol["status"]]; fill_top = [top for top, rate in P["br"] if rate == pol["fill"]][0]
    N = pol["ira_years"]; r = pol["amort_r"]
    sigma = np.asarray(sigma, dtype=float); vec = sigma.ndim == 1
    s = rng.choice(len(pi), size=NP_, p=pi)
    ira = np.full(NP_, ACCT["ira"]); brk = np.full(NP_, ACCT["brk"]); roth = np.full(NP_, ACCT["roth"] + ACCT["hsa"])
    basis = brk * (1.0 - pol["gain0"])
    ru = np.zeros(NP_, bool); red = np.zeros(NP_, bool); brk_out = np.zeros(NP_, bool); roth_tap = np.zeros(NP_, bool)
    yrs_red = np.zeros(NP_); tax_paid = np.zeros(NP_); Wts = []
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
        ss = pol["ss"] if t >= pol["ss_start"] else 0.0; ss_ord = pol["ss_taxable"] * ss
        if t < N:
            n_left = N - t + 1
            room = np.maximum(0.0, std + fill_top - d_ord - ss_ord)
            amort = ira * r / (1.0 - (1.0 + r) ** (-n_left))
            need_gross = np.maximum(0.0, need - d_ord - d_qual - (ss if pol["ss_offsets"] else 0.0)) / (1.0 - pol["fill"] - MO_RATE)
            take = np.minimum(ira, np.maximum(np.maximum(np.maximum(room, amort), need_gross), rmd))
        elif t == N: take = ira.copy()
        else: take = np.zeros(NP_)
        ira = ira - take
        ordinc = d_ord + take
        gf = np.where(brk > 1e-9, np.clip(1.0 - basis / np.maximum(brk, 1e-9), 0.0, 1.0), 0.0)
        base_cash = d_ord + d_qual + take + (ss if pol["ss_offsets"] else 0.0)
        def total_tax(S):
            t_all = fed_tax(ordinc + ss_ord, d_qual + S * gf, P, std) + mo_tax(ordinc, d_qual, std)
            if pol["ss_offsets"] or ss == 0.0: return t_all
            return fed_tax(ordinc, d_qual + S * gf, P, std) + mo_tax(ordinc, d_qual, std)
        S = np.zeros(NP_)
        for _ in range(20): S = np.maximum(0.0, need + total_tax(S) - base_cash)
        S = np.minimum(S, brk); tax = total_tax(S); cash = base_cash + S - tax
        surplus = np.maximum(cash - need, 0.0); short = np.maximum(need - cash, 0.0)
        basis = basis * np.where(brk > 1e-9, 1.0 - S / np.maximum(brk, 1e-9), 1.0); brk = brk - S
        brk = brk + surplus; basis = basis + surplus
        take_r = np.minimum(short, roth); roth = roth - take_r; roth_tap |= take_r > 1.0
        ru |= (short - take_r) > 1.0; brk_out |= brk <= 1.0; tax_paid += tax
        ira = np.maximum(ira, 0.0); brk = np.maximum(brk, 0.0); roth = np.maximum(roth, 0.0); basis = np.minimum(basis, brk)
        if t in (10, 20, 30): Wts.append((ira + brk + roth) / W0)
    W = (ira + brk + roth) / W0; lg = np.log(np.maximum(W, 1e-9))
    return dict(PathCE=float(np.exp(lg.mean())), P10=float(np.quantile(W, .10)), P5=float(np.quantile(W, .05)),
                ruin=int(ru.sum()), yrs_red=float(yrs_red.mean() / T), tax_paid=float(tax_paid.mean() / W0),
                W=W, W10=Wts[0], W20=Wts[1], W30=Wts[2])

def suite(r_net, sigma, pol=POL):
    out = {}
    for pin, pi in (("robust", L.PI_ROBUST), ("canonical", L.PI_CANON)):
        rng = np.random.default_rng(42); sg = sigma[pin] if isinstance(sigma, dict) else sigma
        for conv in ("G", "L"): out[(pin, conv)] = run(rng, r_net, pi, conv, sg, pol)
    return out
def geo(mc): return float(np.exp(np.mean([np.log(mc[c]["PathCE"]) for c in CONVS])))

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
          "gold":    ["GLDM", "FNV", "PSLV", "GDX", "WPM", "SIL"],
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
    return v

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

# ===== env.sh =====
export ENGINEV2_DATA=/home/claude/engine/data DECK_FILE=Deck_R8m_STK_20261024.csv SCEN_FILE=scenarios_25_R8i_20260909.csv GRADES_FILE=/home/claude/engine/data/decks/Row_Grades_R8j_20260909.csv ENGINEV2_PANEL=/home/claude/engine/data/panel_stk.csv ENGINEV2_END=2026-08 ENGINEV2_T=50 NPATHS=2000 STRESS_GATE=1 ETF_CAP=0.20 ENGINEV2_EXTRA=RSST,RSSB,NTSX,GDE,DBMFh,RSSTh PYTHONWARNINGS=ignore

# ===== books_stk.json =====
{
 "C5_SMH3b":   {"GLDM":0.15,"SHLD":0.15,"CIBR":0.15,"DBMF":0.15,"NUKZ":0.08,"BRKB":0.05,"CME":0.05,"FNV":0.05,"MSCI":0.05,"URNM":0.05,"XLE":0.03,"XAR":0.02,"SMH":0.02},
 "STK_RSST15": {"GLDM":0.15,"SHLD":0.15,"CIBR":0.15,"RSST":0.15,"NUKZ":0.08,"BRKB":0.05,"CME":0.05,"FNV":0.05,"MSCI":0.05,"URNM":0.05,"XLE":0.03,"XAR":0.02,"SMH":0.02},
 "STK_RSST20": {"GLDM":0.15,"SHLD":0.15,"CIBR":0.15,"RSST":0.20,"NUKZ":0.03,"BRKB":0.05,"CME":0.05,"FNV":0.05,"MSCI":0.05,"URNM":0.05,"XLE":0.03,"XAR":0.02,"SMH":0.02},
 "STK_R15_D5": {"GLDM":0.15,"SHLD":0.15,"CIBR":0.15,"RSST":0.15,"DBMF":0.05,"NUKZ":0.03,"BRKB":0.05,"CME":0.05,"FNV":0.05,"MSCI":0.05,"URNM":0.05,"XLE":0.03,"SMH":0.02,"XAR":0.02},
 "STK_GDE":    {"GDE":0.15,"SHLD":0.15,"CIBR":0.15,"RSST":0.15,"NUKZ":0.08,"BRKB":0.05,"CME":0.05,"FNV":0.05,"MSCI":0.05,"URNM":0.05,"XLE":0.03,"XAR":0.02,"SMH":0.02},
 "STK_NTSX5":  {"GLDM":0.15,"SHLD":0.15,"CIBR":0.15,"RSST":0.15,"NUKZ":0.03,"NTSX":0.05,"BRKB":0.05,"CME":0.05,"FNV":0.05,"MSCI":0.05,"URNM":0.05,"XLE":0.03,"XAR":0.02,"SMH":0.02}
}

# ===== books_fin.json =====
{
 "STK_CE3":   {"GLDM":0.19,"RSST":0.18,"SHLD":0.14,"DBMF":0.12,"CIBR":0.08,"KMLM":0.08,"CME":0.05,"MSCI":0.05,"BRKB":0.05,"NUKZ":0.02,"URNM":0.02,"FNV":0.02},
 "STK_GEO":   {"GLDM":0.15,"RSST":0.15,"GDE":0.15,"SHLD":0.14,"URNM":0.11,"CIBR":0.09,"CME":0.05,"MSCI":0.05,"BRKB":0.05,"NUKZ":0.02,"FNV":0.02,"XAR":0.02},
 "STK_PREF":  {"RSST":0.20,"GLDM":0.15,"SHLD":0.14,"DBMF":0.12,"KMLM":0.10,"CIBR":0.08,"CME":0.05,"MSCI":0.05,"BRKB":0.05,"URNM":0.02,"NUKZ":0.02,"FNV":0.02},
 "C5_SMH3b_h":{"GLDM":0.15,"SHLD":0.15,"CIBR":0.15,"DBMFh":0.15,"NUKZ":0.08,"BRKB":0.05,"CME":0.05,"FNV":0.05,"MSCI":0.05,"URNM":0.05,"XLE":0.03,"XAR":0.02,"SMH":0.02},
 "STK_CE3_h": {"GLDM":0.19,"RSSTh":0.18,"SHLD":0.14,"DBMFh":0.12,"CIBR":0.08,"KMLM":0.08,"CME":0.05,"MSCI":0.05,"BRKB":0.05,"NUKZ":0.02,"URNM":0.02,"FNV":0.02},
 "STK_PREF_h":{"RSSTh":0.20,"GLDM":0.15,"SHLD":0.14,"DBMFh":0.12,"KMLM":0.10,"CIBR":0.08,"CME":0.05,"MSCI":0.05,"BRKB":0.05,"URNM":0.02,"NUKZ":0.02,"FNV":0.02}
}

