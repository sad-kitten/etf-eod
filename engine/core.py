"""Shared optimization core: universe, class lists, constraints, fast CRN evaluator, P1 floors.
No search is executed at import. Universe = 102 tickers with rows in the R8 deck and history in the panel."""
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
R8_HIST = ['AIRR','BOTZ','KOMP','PRNT','ROBO','SCHD','XT','PSCI','IBKR','VTI']   # deck rows that lacked history until etf-eod run 9
UNIV += [t for t in R8_NEW + R8_HIST if t not in UNIV]                                            # R8 (ER merged in lib)
import os as _os
if _os.environ.get('ENGINEV2_RPRX', '0') == '1': UNIV += ['RPRX']
N_BASE = len(UNIV)
EXTRA = [t for t in _os.environ.get('ENGINEV2_EXTRA', '').split(',') if t and t not in UNIV]
UNIV += EXTRA
N = len(UNIV)
IDX = {t:i for i,t in enumerate(UNIV)}
SINGLES = ['CME','MSCI','CEG','CBOE','ICE','SPGI','NDAQ','BRKB','WM','WPM','FNV','V','COIN','IBKR','UTHR','VRTX','ISRG','ZTS']
PMEQ = ['GDX','WPM','FNV','PSLV','SIL']
SEMIS = ['SMH','SOXX','XSD']
DEF = ['SHLD','XAR','PPA','ITA','EUAD','ROKT','JEDI']
NUC = ['NUKZ','NLR','URNM','CEG','SETM','GRID','URA']
BAL = ['GLDM','DBMF','CTA','KMLM','IEF','TLT','GOVZ','LTPZ','SCHP','VTIP','BIL','SHY','AGG','TIP','STIP','TAIL','BTAL']

DECKM = np.array([[L.DECK.loc[t,s] for s in L.SC] for t in UNIV])/100.0
ERV = np.array([L.ER[t] for t in UNIV])/100.0

import os as _os
mr = L.monthly_returns()
END = pd.Period(_os.environ.get("ENGINEV2_END", "2026-08"), "M")   # last month of the 48-month vol window
idx48 = pd.period_range(END-47, END, freq="M")
_missing = [t for t in UNIV if t not in mr.columns]
if _missing:
    import warnings
    warnings.warn(f"no monthly history for {_missing}; treated as not live in vol48 (books holding them get a NaN sigma)")
R48 = mr.reindex(idx48).reindex(columns=UNIV).values
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
    if np.any(w[held] < float(_os.environ.get('MIN_HOLD', '0.02')) - 1e-9): return False
    if w.max() > 0.15 + 1e-9: return False
    _bc = float(_os.environ.get('BRKB_CAP', '0.05'))   # 2026-10-26 owner: BRKB up to 15%, treated like a fund
    for t in SINGLES:
        if t in IDX and w[IDX[t]] > (_bc if t == 'BRKB' else 0.05) + 1e-9: return False
    ssum = lambda ts: sum(w[IDX[t]] for t in ts if t in IDX)
    if ssum([t for t in SINGLES if not (t == 'BRKB' and _bc > 0.05)]) > float(_os.environ.get('SINGLES_TOT', '0.20')) + 1e-9: return False
    if ssum(PMEQ) > 0.10 + 1e-9: return False
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
            and ev["R3AF"] >= FLOOR["R3AF"] - 0.002)   # RCEW floor removed 2026-10-29 (retired as ranking input; A58)
