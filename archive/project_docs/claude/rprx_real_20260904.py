"""RPRX on real history: measure it, then re-derive the deck row under the campaign's own discipline
(geometric-consistency + beta decomposition), replacing the Addendum 20 proxy construction."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import lib_pathce as L

mr = L.monthly_returns()
s = mr['RPRX'].dropna(); iv = mr['IVV'].reindex(s.index)
M = pd.concat([s, iv], axis=1).dropna(); a = M.iloc[:, 0]; b = M.iloc[:, 1]; yrs = len(M) / 12
ca = ((1 + a).prod()) ** (1 / yrs) - 1
cb = ((1 + b).prod()) ** (1 / yrs) - 1
X = np.column_stack([np.ones(len(a)), b.values])
coef, *_ = np.linalg.lstsq(X, a.values, rcond=None)
resid = a.values - X @ coef
se = resid.std(ddof=1) / np.sqrt(len(a))
cum = (1 + a).cumprod()
bad = b <= b.quantile(0.10)
IVV_ROW = float(np.mean([float(L.DECK.loc['IVV', x]) for x in L.SC]))

print(f"RPRX real: n={len(M)} {str(M.index[0])}..{str(M.index[-1])} ({yrs:.1f} yr)")
print(f"  realised {ca*100:6.2f}%/yr | IVV same {cb*100:6.2f}% | geometric excess {(ca-cb)*100:+6.2f}pp")
print(f"  beta {coef[1]:.2f}  alpha {coef[0]*12*100:+6.2f}%/yr (t={coef[0]/se:+.2f})"
      f"  vol {a.std(ddof=1)*np.sqrt(12)*100:.1f}%  maxDD {(cum/cum.cummax()-1).min()*100:.1f}%"
      f"  downCapture {a[bad].mean()/b[bad].mean():.2f}")
print(f"  correlation to IVV {a.corr(b):.2f}")
for t in ['XLV', 'IBB', 'FNV', 'CME', 'GLDM', 'DBMF', 'IEF']:
    if t in mr.columns:
        o = mr[t].reindex(a.index)
        print(f"    corr to {t:5s} {a.corr(o):+.2f}")
print(f"\n  geometric-consistency honest row = {IVV_ROW:.2f} {(ca-cb)*100:+.2f} = {IVV_ROW+(ca-cb)*100:.2f}")
print(f"  beta-consistency (alpha insignificant) = beta x IVV_row = {coef[1]*IVV_ROW:.2f}")
print(f"  proxy-era derived rows for comparison: 8.04 (Add.19), 7.29 (Add.20 proxy)")
