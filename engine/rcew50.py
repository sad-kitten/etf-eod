"""RCEW50_rob + full 25-scenario view (closed form, scenario-locked 50 years, 3% draw on initial corpus).
Refresh of lib_pathce.rcew40 (which hard-codes T=40): same robust blend 0.50 canonical / 0.25 equal / 0.25 stress
(stress leg = sqrt(w)*severity*EVG11 downside), same deck rows net of ER. Adds DRB50 at the ratified 6% hurdle,
the stressed-deck RCEW50, and per-scenario regret vs the best book.
Run: source env57.sh; python3 rcew50.py books.json out_prefix
"""
import sys, json, numpy as np, pandas as pd
import lib_pathce as L, standing as ST
from core import DECKM, ERV, wvec

T, D = 50, 0.03
BK = json.load(open(sys.argv[1])); OUT = sys.argv[2]
SC = L.SC; N = L.NSC
WE = np.ones(N) / N

def tw(g, T_=T):
    return L.tw_after_draw(g, T_=T_, d=D)

def rcew(r, T_=T):
    tm = tw(r, T_)
    c = L.log_cew(tm, L.W_CAN); e = L.log_cew(tm, WE); s = L.log_cew(tm, L.ST_W)
    rob = float(np.exp(.5 * np.log(max(c, 1e-12)) + .25 * np.log(max(e, 1e-12)) + .25 * np.log(max(s, 1e-12))))
    return rob, c, e, s

def drb(r, h=0.06, T_=T):
    tm = np.maximum(tw(r, T_), 1e-9); twh = tw(np.array([h]), T_)[0]
    gap = np.maximum(0, np.log(twh / tm))
    leg = lambda w: float(np.sqrt(np.sum(w / w.sum() * gap ** 2)))
    return .5 * leg(L.W_CAN) + .25 * leg(WE) + .25 * leg(L.ST_W)

def cagr_equiv(TW):
    """Net CAGR g such that a 50-y path at constant g with the 3% draw ends at TW (bisection)."""
    lo, hi = -0.05, 0.30
    for _ in range(80):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if tw(np.array([m]))[0] < TW else (lo, m)
    return (lo + hi) / 2

names = list(BK); R = {}; RS = {}
for nm in names:
    w = wvec({t: v for t, v in BK[nm].items() if v > 0}); er = float(ERV @ w)
    R[nm] = DECKM.T @ w - er
    RS[nm] = ST.stressed_deck(DECKM).T @ w - er
M = pd.DataFrame(R, index=SC).T          # net CAGR, decimal
TWm = M.apply(lambda r: pd.Series(tw(r.values), index=SC), axis=1)
best = TWm.max(axis=0)
REG = np.log(best / TWm)                 # log regret vs best book in each scenario

rows = []
for nm in names:
    r = R[nm]; rob, c, e, s = rcew(r); rob40 = rcew(r, 40)[0]; robS = rcew(RS[nm])[0]
    tmv = tw(r)
    rows.append({"Port": nm,
                 "RCEW50_rob x": round(rob, 2), "RCEW50 CE-CAGR %": round(cagr_equiv(rob) * 100, 2),
                 "can / eq / stress x": f"{c:.2f} / {e:.2f} / {s:.2f}",
                 "RCEW40_rob x (old)": round(rob40, 2),
                 "RCEW50 stressed-deck x": round(robS, 2),
                 "DRB50@6": round(drb(r), 4),
                 "scen < 6% net": int((r < 0.06).sum()),
                 "TW50 min x": round(float(tmv.min()), 2),
                 "TW50 max x": round(float(tmv.max()), 2),
                 "max regret %": round(float((1 - np.exp(-REG.loc[nm].max())) * 100), 1),
                 "max-regret scen": REG.loc[nm].idxmax()[:28],
                 "wtd regret %": round(float((1 - np.exp(-np.sum(L.make_pi_robust() * REG.loc[nm].values))) * 100), 1),
                 "scen won": int((TWm.loc[nm] >= best - 1e-12).sum())})
S_ = pd.DataFrame(rows).set_index("Port")
S_.to_csv(OUT + "_summary.csv")
(M * 100).round(2).T.assign(**{"canon wt %": (L.W_CAN * 100).round(2), "sev": L.SEV.astype(int)}).to_csv(OUT + "_scen_netCAGR.csv")
TWm.round(2).T.to_csv(OUT + "_scen_TW50.csv")
print(S_.to_string())
