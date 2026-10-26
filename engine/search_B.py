"""search_B — the parts standing.py / report_v3.py read (verbatim reconstruction from Bench_2FFL_Code_20260919.py)."""
import os, numpy as np, pandas as pd
import lib_pathce as L
from core import UNIV, N, N_BASE, IDX, DECKM, ERV, sigma_of, wvec, P1v, CONVS, FE, feasible, SINGLES, idx48, mr

grades = pd.read_csv(os.environ.get("GRADES_FILE", os.path.join(L.DECKS, "Row_Grades_R8cand_20260903.csv"))).set_index("Ticker")["Grade"]
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
