"""Add back what basis (B) subtracts, to get the raw return.

The reported geo-PathCE is RETAINED REAL wealth: after the 3% real draw (net of the Social Security
offset from year 11), after federal + Missouri tax, after fund expenses, and in today's dollars.
This builds the full ladder from the raw gross return down to that number, per book, all as annual %:

  1  gross nominal asset CAGR      the raw return: what the holdings compound at before anything
  2  - weighted expense ratio      = net-of-fee nominal asset CAGR
  3  - 2.5% inflation              = net-of-fee REAL asset CAGR   (corpus with NO withdrawals)
  4  - draw drag                   = retained real CAGR, taxes switched off
  5  - tax drag                    = retained real CAGR  <- the number reported so far
  +  inflation back on             = retained NOMINAL CAGR (what an account statement would show)

Steps 1-3 come from the engine's own cumulative asset-growth factor (product of the yearly real
growth draws, no withdrawals), pooled over the four conventions -- so it is the same 240k paths and
the same geometric averaging as geo-PathCE, not a separate calculation. Step 4 is a counterfactual
re-run with fed_tax and mo_tax forced to zero. Also shown: the pi-weighted ARITHMETIC mean of the 24
nominal deck rows, whose gap to line 1 is the volatility drag.
"""
import numpy as np, pandas as pd, json, os, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L
import policy_engine2 as E
from core import DECKM, ERV, sigma_of, wvec

INFL = E.INFL
CONV = [("robust","G"),("robust","L"),("canonical","G"),("canonical","L")]
PIS  = {"robust": L.PI_ROBUST, "canonical": L.PI_CANON}
_fed, _mo = E.fed_tax, E.mo_tax

def suite_notax(r, sig):
    E.fed_tax = lambda *a, **k: np.zeros_like(np.asarray(a[0], dtype=float))
    E.mo_tax  = lambda *a, **k: np.zeros_like(np.asarray(a[0], dtype=float))
    try:    return E.suite(r, sig, E.POL)
    finally: E.fed_tax, E.mo_tax = _fed, _mo

def ladder(d):
    w = wvec({a:b for a,b in d.items() if b > 0})
    er = float(ERV @ w)
    row_gross = DECKM.T @ w                      # nominal scenario CAGRs, gross of fees
    r_net = row_gross - er
    sig = sigma_of(w)
    mc = E.suite(r_net, sig, E.POL); mcN = suite_notax(r_net, sig)
    G = np.concatenate([mc[c]["gcum"] for c in CONV])          # real, net of fee, no draws
    W = np.concatenate([np.maximum(mc[c]["W"], 1e-9) for c in CONV])
    WN = np.concatenate([np.maximum(mcN[c]["W"], 1e-9) for c in CONV])
    ann = lambda x: (float(np.exp(np.log(x).mean()))**(1/40.) - 1) * 100
    real_asset = ann(G)                                        # line 3
    nom_asset  = ((1 + real_asset/100) * (1 + INFL) - 1) * 100  # line 2
    gross_nom  = nom_asset + er*100                             # line 1
    ret_notax  = ann(WN)                                        # line 4
    ret_real   = ann(W)                                         # line 5
    ret_nom    = ((1 + ret_real/100) * (1 + INFL) - 1) * 100
    arith = float(np.mean([float(pi @ row_gross) for pi in PIS.values()])) * 100
    return dict(
        deck_arith_nom=round(arith, 2), gross_nom_cagr=round(gross_nom, 2), vol_drag=round(arith - gross_nom, 2),
        ER=round(er*100, 3), net_fee_nom=round(nom_asset, 2), infl=round(INFL*100, 2),
        real_asset_cagr=round(real_asset, 2), draw_drag=round(real_asset - ret_notax, 2),
        ret_real_notax=round(ret_notax, 2), tax_drag=round(ret_notax - ret_real, 2),
        retained_real=round(ret_real, 2), retained_nom=round(ret_nom, 2),
        real_x=round(float(np.exp(np.log(W).mean())), 3),
        nom_x=round(float(np.exp(np.log(W).mean())) * (1+INFL)**40, 2),
        asset_x_real=round(float(np.exp(np.log(G).mean())), 2), vol48=round(sig*100, 2))

if __name__ == "__main__":
    BK = json.load(open("out/finalbooks2.json"))
    ORDER = ["R5c2","N_DBMF_XAR2","N_XLI_XAR2","R7a","C5_balK3","C6_tailK3","C3_balK2","C1_R7a",
             "C4_balK3seed","R3b","R3dK","G2_g3R7a","P1"]
    rows = []
    for nm in ORDER:
        r = ladder(BK[nm]); r["Port"] = nm; rows.append(r); print(nm, r, flush=True)
    df = pd.DataFrame(rows).set_index("Port")
    df.to_csv(os.environ.get("OUT","out/Raw_CAGR_Ladder_20260905.csv"))
    pd.set_option("display.width", 400); print(df.to_string())
