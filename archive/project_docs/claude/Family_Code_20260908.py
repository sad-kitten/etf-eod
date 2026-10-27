"""FNV on real history under the Addendum 21 standing rule: every row gets the beta- and
geometric-consistency audit. FNV's row was fundamentals-reviewed (Addendum 12) but never
measured against its own price record; the +2.0 pp contracted premium and the thesis half
are both still judgment."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L
mr = L.monthly_returns()
s = mr['FNV'].dropna(); iv = mr['IVV'].reindex(s.index)
M = pd.concat([s, iv], axis=1).dropna(); a = M.iloc[:,0]; b = M.iloc[:,1]; yrs = len(M)/12
ca = ((1+a).prod())**(1/yrs)-1; cb = ((1+b).prod())**(1/yrs)-1
X = np.column_stack([np.ones(len(a)), b.values])
coef, *_ = np.linalg.lstsq(X, a.values, rcond=None)
resid = a.values - X @ coef; se = resid.std(ddof=1)/np.sqrt(len(a))
cum = (1+a).cumprod(); bad = b <= b.quantile(0.10)
IVV_ROW = float(np.mean([float(L.DECK.loc['IVV', x]) for x in L.SC]))
print(f"FNV real: n={len(M)} {M.index[0]}..{M.index[-1]} ({yrs:.1f} yr)")
print(f"  realised {ca*100:7.2f}%/yr | IVV same {cb*100:7.2f}% | geometric excess {(ca-cb)*100:+7.2f}pp")
print(f"  beta {coef[1]:.2f}  alpha {coef[0]*12*100:+7.2f}%/yr (t={coef[0]/se:+.2f})  vol {a.std(ddof=1)*np.sqrt(12)*100:.1f}%"
      f"  maxDD {(cum/cum.cummax()-1).min()*100:.1f}%  downCapture {a[bad].mean()/b[bad].mean():.2f}  corr {a.corr(b):.2f}")
for t in ['GLDM','GDX','PSLV','DBMF','NUKZ','SHLD','CIBR','CME','IEF']:
    if t in mr.columns:
        o = mr[t].reindex(a.index)
        n = pd.concat([a,o],axis=1).dropna()
        if len(n) > 12:
            print(f"    vs {t:5s} corr {n.iloc[:,0].corr(n.iloc[:,1]):+.2f}  ({len(n)} mo overlap)")
# GLDM as sector anchor over the FNV window
nl = mr['GLDM'].reindex(a.index).dropna()
aa = a.reindex(nl.index)
yn = len(nl)/12
print(f"\n  over the FNV window: FNV {(((1+aa).prod())**(1/yn)-1)*100:6.2f}%/yr  GLDM {(((1+nl).prod())**(1/yn)-1)*100:6.2f}%/yr"
      f"  excess {((((1+aa).prod())**(1/yn)) - (((1+nl).prod())**(1/yn)))*100:+6.2f}pp")
Xn = np.column_stack([np.ones(len(nl)), nl.values]); cn, *_ = np.linalg.lstsq(Xn, aa.values, rcond=None)
rn = aa.values - Xn @ cn; sen = rn.std(ddof=1)/np.sqrt(len(nl))
print(f"  vs GLDM: beta {cn[1]:.2f}  alpha {cn[0]*12*100:+6.2f}%/yr (t={cn[0]/sen:+.2f})")
GLDM_ROW = float(np.mean([float(L.DECK.loc['GLDM', x]) for x in L.SC]))
print(f"\n  deck rows: IVV {IVV_ROW:.2f}  GLDM {GLDM_ROW:.2f}  FNV (current) {float(np.mean([float(L.DECK.loc['FNV',x]) for x in L.SC])):.2f}")
print(f"  geometric-consistency honest row = {IVV_ROW:.2f} {(ca-cb)*100:+.2f} = {IVV_ROW+(ca-cb)*100:.2f}")
print(f"  beta-consistency (vs IVV) = {coef[1]:.2f} x {IVV_ROW:.2f} = {coef[1]*IVV_ROW:.2f}  (+alpha if significant)")
print(f"  sector-anchored (vs GLDM)  = {cn[1]:.2f} x {GLDM_ROW:.2f} = {cn[1]*GLDM_ROW:.2f}")
import numpy as np, json, warnings; warnings.filterwarnings('ignore')
import search_B as S
from core import wvec, UNIV, IDX, feasible
BK = json.load(open("out/survivors.json"))
FAM = ["F3_xar","F10_setm","F2_bal2","F6_cta","C5_balK3","C4_balK3seed","F11_ivv"]
# counterfactuals inside the family
BK["X1_F2_noFNV"] = {'GLDM':.15,'SHLD':.15,'CIBR':.13,'NUKZ':.10,'DBMF':.14,'URNM':.08,'XLE':.08,'BRKB':.05,'CME':.05,'MSCI':.05,'XAR':.02}
BK["X2_F2_DBMF15"] = {'GLDM':.15,'SHLD':.15,'CIBR':.13,'NUKZ':.10,'DBMF':.15,'URNM':.05,'XLE':.05,'BRKB':.05,'CME':.05,'FNV':.05,'MSCI':.05,'XAR':.02}
for k in ["X1_F2_noFNV","X2_F2_DBMF15"]: assert abs(sum(BK[k].values())-1)<1e-9, k
json.dump({k: BK[k] for k in ["X1_F2_noFNV","X2_F2_DBMF15"]}, open("out/family_cf.json","w"), indent=1)
GOLD = [t for t in ['GLDM','FNV','PSLV','GDX','IAU','SGOL'] if t in IDX]
BAL  = [t for t in ['DBMF','CTA','KMLM','GLDM','FNV','PSLV','IEF','AGG','PDBC'] if t in IDX]
def rs_of(w, names):
    Cv = S.COV @ w; var = float(w @ Cv); mrc = Cv * w / var
    return float(sum(mrc[IDX[t]] for t in names))
print(f"{'book':14s} {'DBMF':>4s} {'ballast wt':>10s} {'gold-cx RS':>10s} {'ballast RS':>10s} {'nuc RS':>6s} {'name RS':>7s} {'gates'}")
W = {}
for k in FAM + ["X1_F2_noFNV","X2_F2_DBMF15"]:
    d = {t:v for t,v in BK[k].items() if v>0}; w = wvec(d); W[k] = w
    ns, mx = S.risk_shares(w); f = S.gate_fails(w)
    bw = sum(d.get(t,0) for t in BAL)
    print(f"{k:14s} {int(round(d.get('DBMF',0)*100)):4d} {bw*100:10.0f} {rs_of(w,GOLD)*100:10.1f} {rs_of(w,BAL)*100:10.1f} {ns*100:6.1f} {mx*100:7.1f} {'PASS' if not f else ','.join(sorted(set(x.split(':')[1] for x in f)))}")
print("\npairwise distance = % of portfolio that must move (L1/2):")
ks = FAM
print(" "*14 + "".join(f"{k[:9]:>10s}" for k in ks))
for a in ks:
    print(f"{a:14s}" + "".join(f"{0.5*np.abs(W[a]-W[b]).sum()*100:10.0f}" for b in ks))
