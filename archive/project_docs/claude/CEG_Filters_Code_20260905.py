"""CEG on real history under the Addendum 21 standing rule: every row gets the beta- and
geometric-consistency audit. CEG's row was fundamentals-reviewed (Addendum 12) but never
measured against its own price record; the +2.0 pp contracted premium and the thesis half
are both still judgment."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L
mr = L.monthly_returns()
s = mr['CEG'].dropna(); iv = mr['IVV'].reindex(s.index)
M = pd.concat([s, iv], axis=1).dropna(); a = M.iloc[:,0]; b = M.iloc[:,1]; yrs = len(M)/12
ca = ((1+a).prod())**(1/yrs)-1; cb = ((1+b).prod())**(1/yrs)-1
X = np.column_stack([np.ones(len(a)), b.values])
coef, *_ = np.linalg.lstsq(X, a.values, rcond=None)
resid = a.values - X @ coef; se = resid.std(ddof=1)/np.sqrt(len(a))
cum = (1+a).cumprod(); bad = b <= b.quantile(0.10)
IVV_ROW = float(np.mean([float(L.DECK.loc['IVV', x]) for x in L.SC]))
print(f"CEG real: n={len(M)} {M.index[0]}..{M.index[-1]} ({yrs:.1f} yr)")
print(f"  realised {ca*100:7.2f}%/yr | IVV same {cb*100:7.2f}% | geometric excess {(ca-cb)*100:+7.2f}pp")
print(f"  beta {coef[1]:.2f}  alpha {coef[0]*12*100:+7.2f}%/yr (t={coef[0]/se:+.2f})  vol {a.std(ddof=1)*np.sqrt(12)*100:.1f}%"
      f"  maxDD {(cum/cum.cummax()-1).min()*100:.1f}%  downCapture {a[bad].mean()/b[bad].mean():.2f}  corr {a.corr(b):.2f}")
for t in ['NLR','XLU','URA','NUKZ','URNM','GRID','XLE','GLDM','DBMF']:
    if t in mr.columns:
        o = mr[t].reindex(a.index)
        n = pd.concat([a,o],axis=1).dropna()
        if len(n) > 12:
            print(f"    vs {t:5s} corr {n.iloc[:,0].corr(n.iloc[:,1]):+.2f}  ({len(n)} mo overlap)")
# NLR as sector anchor over the CEG window
nl = mr['NLR'].reindex(a.index).dropna()
aa = a.reindex(nl.index)
yn = len(nl)/12
print(f"\n  over the CEG window: CEG {(((1+aa).prod())**(1/yn)-1)*100:6.2f}%/yr  NLR {(((1+nl).prod())**(1/yn)-1)*100:6.2f}%/yr"
      f"  excess {((((1+aa).prod())**(1/yn)) - (((1+nl).prod())**(1/yn)))*100:+6.2f}pp")
Xn = np.column_stack([np.ones(len(nl)), nl.values]); cn, *_ = np.linalg.lstsq(Xn, aa.values, rcond=None)
rn = aa.values - Xn @ cn; sen = rn.std(ddof=1)/np.sqrt(len(nl))
print(f"  vs NLR: beta {cn[1]:.2f}  alpha {cn[0]*12*100:+6.2f}%/yr (t={cn[0]/sen:+.2f})")
NLR_ROW = float(np.mean([float(L.DECK.loc['NLR', x]) for x in L.SC]))
print(f"\n  deck rows: IVV {IVV_ROW:.2f}  NLR {NLR_ROW:.2f}  CEG (current) {float(np.mean([float(L.DECK.loc['CEG',x]) for x in L.SC])):.2f}")
print(f"  geometric-consistency honest row = {IVV_ROW:.2f} {(ca-cb)*100:+.2f} = {IVV_ROW+(ca-cb)*100:.2f}")
print(f"  beta-consistency (vs IVV) = {coef[1]:.2f} x {IVV_ROW:.2f} = {coef[1]*IVV_ROW:.2f}  (+alpha if significant)")
print(f"  sector-anchored (vs NLR)  = {cn[1]:.2f} x {NLR_ROW:.2f} = {cn[1]*NLR_ROW:.2f}")
"""R8h — CEG's row re-derived on real history under the Addendum 21 standing rule.

Measured (55 months, 2022-02..2026-08): realised 47.64%/yr against IVV 13.89%, beta 1.10 to IVV,
alpha +33.70%/yr but t = +1.68 (not significant), corr 0.37, down-capture 1.06, vol 46.4%.
Against NLR over the same window: beta 0.78, alpha +30.83%/yr, t = +1.63.

The naive geometric-consistency rule gives 7.50 + 33.75 = 41.25 and is obviously invalid. That is the
mirror image of the RPRX case and it shows why the rule was applied ASYMMETRICALLY in Addendum 19
(over-claims only, at half the gap): a realised excess measured over 4.6 years is a re-rating, not an
estimate of forward excess. CEG is up roughly 7x from its 2022 spin-off price; the contracted-power
premium that justified the thesis is now IN the price, so it cannot also be in the forward row.

Construction, identical to the one used for RPRX in Addendum 21:
    base    = beta x IVV_row, alpha set to zero (t = 1.68 < 2)   = 1.10 x 7.50 = 8.28
    premium = +2.0   the Addendum 12 contracted-PPA premium, applied ONCE, not twice
    shape   = 1.00 x deviation of the 0.6*NLR + 0.4*IVV anchor about its mean
Result ~10.3 mean against the R8g row's 12.34: a further -2.0 pp on top of the -2.5 pp taken in
Addendum 12. Grade stays C (4.6 years of history, single stock, 46% vol).
"""
import numpy as np, pandas as pd
import lib_pathce as L

BETA, IVV_ROW, PREM, SHAPE = 1.10, 7.50, 2.0, 1.00
nlr = np.array([float(L.DECK.loc['NLR', s]) for s in L.SC])
ivv = np.array([float(L.DECK.loc['IVV', s]) for s in L.SC])
old = np.array([float(L.DECK.loc['CEG', s]) for s in L.SC])
anchor = 0.6 * nlr + 0.4 * ivv
row = BETA * IVV_ROW + PREM + SHAPE * (anchor - anchor.mean())

d = pd.read_csv("Deck_Recal_R8g_20260904.csv", index_col=0)
d.loc["CEG", list(d.columns)] = np.round(row, 4)
d.to_csv("Deck_Recal_R8h_20260905.csv")

g = pd.read_csv("out/Row_Grades_R8g_20260904.csv")
g.loc[g.Ticker == "CEG", "Meaning"] = ("Constellation, real history 2022-02.. (55 mo); row = beta-consistent base "
    "(1.10 x IVV_row, alpha t=1.68 so set to zero) + 2.0 contracted-PPA premium applied once "
    "+ 1.00 x (0.6*NLR + 0.4*IVV) shape; supersedes the Addendum 12 thesis/anchor blend")
g.to_csv("out/Row_Grades_R8h_20260905.csv", index=False)
print(f"CEG old  mean {old.mean():6.2f}  min {old.min():6.2f}  max {old.max():6.2f}  std {old.std():5.2f}")
print(f"CEG new  mean {row.mean():6.2f}  min {row.min():6.2f}  max {row.max():6.2f}  std {row.std():5.2f}   delta {row.mean()-old.mean():+.2f}")
print(f"  NLR    mean {nlr.mean():6.2f}  min {nlr.min():6.2f}   IVV mean {ivv.mean():6.2f}")
print(f"  history of this row: 14.9 (original) -> 12.34 (Add.12 fundamentals) -> {row.mean():.2f} (Add.26 real data)")
"""Filter columns for the max-wealth table: things the ranking metric cannot see."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, search_B as S
from core import DECKM, ERV, sigma_of, wvec, UNIV, IDX, CONVS
T25 = pd.read_csv("out/Top25_MaxWealth_R8h_20260905.csv")
BK = json.load(open("out/rankbooks.json")); BK.update(json.load(open("out/finalbooks2.json")))
mr = L.monthly_returns()
hist = {t: int(mr[t].notna().sum()) for t in mr.columns}
def geo_on(w, M):
    r = M.T @ w - float(ERV @ w); sig = sigma_of(w)
    return float(np.exp(np.mean([np.log(S.run_B(r, c[0], c[1], sig)["PathCE"]) for c in CONVS])))
rows = []
for _, x in T25.iterrows():
    p = x.Port; d = {t: v for t, v in BK[p].items() if v > 0}; w = wvec(d)
    cw = sum(v for t, v in d.items() if S.grades.get(t, "B") == "C")
    y5 = sum(v for t, v in d.items() if hist.get(t, 999) < 60)
    y3 = sum(v for t, v in d.items() if hist.get(t, 999) < 36)
    g8 = geo_on(w, DECKM); gh = geo_on(w, S.HCC)
    ns, mx = S.risk_shares(w); fails = S.gate_fails(w)
    rows.append(dict(rank=int(x["rank"]), Port=p, geo_x=x.geo_x,
                     hcc_pct=round((gh/g8 - 1)*100, 2), C_wt=round(cw*100), lt5y_wt=round(y5*100), lt3y_wt=round(y3*100),
                     nuc_rs=round(ns*100), name_rs=round(mx*100), ER=round(float(ERV@w)*100, 3),
                     n=len(d), sing=x.singles, gates=("PASS" if not fails else ",".join(sorted(set(f.split(":")[1] for f in fails))))))
df = pd.DataFrame(rows); df.to_csv("out/Top25_Filters_20260905.csv", index=False)
pd.set_option("display.width", 300); print(df.to_string(index=False))
print("\nhistory (months) for the short-history names:")
for t in sorted(set(t for _, x in T25.iterrows() for t in BK[x.Port]), key=lambda t: hist.get(t, 999)):
    if hist.get(t, 999) < 84: print(f"  {t:6s} {hist.get(t,0):4d} mo  ({hist.get(t,0)/12:.1f} y)")
import json, pandas as pd, warnings; warnings.filterwarnings('ignore')
import search_B as S
from core import wvec
BK = json.load(open("out/rankbooks.json")); BK.update(json.load(open("out/finalbooks2.json")))
T = pd.read_csv("out/Top25_MaxWealth_R8h_20260905.csv")
for _, x in T.iterrows():
    w = wvec({t: v for t, v in BK[x.Port].items() if v > 0})
    f = S.gate_fails(w); cf = S.cf_fails(w)
    if f: print(f"{int(x['rank']):2d} {x.Port:13s} FAIL {sorted(set(f))}   closed-form: {sorted(set(cf)) if cf else 'ok'}")
