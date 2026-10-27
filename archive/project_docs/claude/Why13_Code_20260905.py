"""What the record actually delivered, on the panel this campaign uses."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L
from core import ERV, wvec
mr = L.monthly_returns()
def cagr(s):
    s = s.dropna(); n = len(s)/12
    return ((1+s).prod())**(1/n)-1, n
print("=== single tickers, full available window ===")
for t in ['IVV','QQQ','SMH','NUKZ','URNM','CIBR','SHLD','GLDM','DBMF','CTA','XLE','CME','MSCI','BRKB','FNV','GRID','XAR']:
    if t in mr.columns:
        c, n = cagr(mr[t]); s = mr[t].dropna()
        print(f"  {t:5s} {c*100:6.2f}%/yr over {n:4.1f}y  ({s.index[0]}..{s.index[-1]})  vol {s.std(ddof=1)*np.sqrt(12)*100:5.1f}%")
print("\n=== IVV over sub-windows ===")
iv = mr['IVV'].dropna()
for lab, sl in [("full panel", iv), ("last 5y", iv.iloc[-60:]), ("last 3y", iv.iloc[-36:]), ("2020-07..", iv.loc[pd.Period('2020-07','M'):])]:
    c, n = cagr(sl); print(f"  {lab:14s} {c*100:6.2f}%/yr over {n:4.1f}y")
print("\n=== books, live-weight backtest on the same panel ===")
BK = json.load(open("out/finalbooks2.json"))
for nm in ["P1","R7a","C5_balK3","R5c2","R3b"]:
    d = {a:b for a,b in BK[nm].items() if b>0}
    sub = mr[list(d)]; wv = np.array([d[t] for t in sub.columns]); live = sub.notna().values
    Wm = live*wv; ok = Wm.sum(1) > 0.60          # only months where >60% of the book has data
    Wm = Wm/np.maximum(Wm.sum(1,keepdims=True),1e-12)
    ps = pd.Series(np.nansum(sub.values*Wm,1), index=sub.index)[ok]
    er = float(ERV @ wvec(d))
    c, n = cagr(ps)
    print(f"  {nm:10s} {c*100:6.2f}%/yr gross  {(c-er)*100:6.2f}% net-of-fee  over {n:4.1f}y ({ps.index[0]}..{ps.index[-1]})  vol {ps.std(ddof=1)*np.sqrt(12)*100:5.1f}%")
"""How often does the engine actually deliver >=13%? Decade and 40-year, real and nominal."""
import numpy as np, json, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, policy_engine2 as E
from core import DECKM, ERV, sigma_of, wvec
CONV=[("robust","G"),("robust","L"),("canonical","G"),("canonical","L")]
INFL=E.INFL
BK=json.load(open("out/finalbooks2.json"))
def prob(nm):
    d={a:b for a,b in BK[nm].items() if b>0}; w=wvec(d)
    r=DECKM.T@w-float(ERV@w); sig=sigma_of(w)
    mc=E.suite(r,sig,E.POL)
    A=[];D=[];W=[]
    for c in CONV:
        run=mc[c]; s=run["snaps"]; g=run["gcum"]
        A.append(g**(1/40.)-1)                       # asset-only real CAGR, no draws
        W.append(np.maximum(run["W"],1e-9)**(1/40.)-1)
        prev=np.ones(len(g))
        for t in (10,20,30,40):
            cur=np.maximum(s[t],1e-9); D.append((cur/prev)**0.1-1); prev=cur
    A=np.concatenate(A); D=np.concatenate(D); W=np.concatenate(W)
    def pn(x,thr_nom):                                # threshold given as NOMINAL
        thr=(1+thr_nom)/(1+INFL)-1
        return float((x>=thr).mean())*100
    return dict(nm=nm,
        a13=pn(A,0.13), a10=pn(A,0.10), a8=pn(A,0.08),
        d13=pn(D,0.13), d10=pn(D,0.10), d20=pn(D,0.20),
        w13=pn(W,0.13), w10=pn(W,0.10), w8=pn(W,0.08),
        a_best=(np.quantile(A,0.99)+1)*(1+INFL)-1, d_best=(np.quantile(D,0.99)+1)*(1+INFL)-1)
print(f"{'book':10s} {'P(40y asset>=13%n)':>19s} {'>=10%n':>7s} {'>=8%n':>7s} | {'P(decade>=13%n)':>16s} {'>=10%n':>7s} {'>=20%n':>7s} | {'P(40y retained>=13%n)':>22s} {'>=10%n':>7s} | {'99th pct 40y asset':>19s} {'99th decade':>12s}")
for nm in ["R5c2","R7a","C5_balK3","P1"]:
    p=prob(nm)
    print(f"{nm:10s} {p['a13']:19.2f} {p['a10']:7.1f} {p['a8']:7.1f} | {p['d13']:16.1f} {p['d10']:7.1f} {p['d20']:7.1f} | {p['w13']:22.2f} {p['w10']:7.1f} | {p['a_best']*100:18.1f}% {p['d_best']*100:11.1f}%", flush=True)
"""What chasing the trailing 20% actually does, run through the same engine."""
import numpy as np, json, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, policy_engine2 as E, search_B as S
from core import DECKM, ERV, sigma_of, wvec, UNIV, IDX
CONV=[("robust","G"),("robust","L"),("canonical","G"),("canonical","L")]; INFL=E.INFL
BOOKS={
 "HOT_thematic": {'NUKZ':.25,'URNM':.20,'SHLD':.20,'SMH':.15,'CIBR':.10,'GLDM':.10},
 "HOT_max_geo":  {'URNM':.30,'NUKZ':.25,'MSCI':.15,'SHLD':.15,'SMH':.15},
 "R7a":          {'GLDM':.15,'SHLD':.15,'CIBR':.15,'DBMF':.12,'NUKZ':.11,'XLE':.08,'URNM':.07,'CME':.05,'MSCI':.05,'BRKB':.05,'XLI':.02},
}
print(f"{'book':14s} {'deck arith':>10s} {'grossNom':>9s} {'vol48':>7s} {'realAsset':>10s} {'retReal':>8s} {'retNom':>7s} {'P5':>6s} {'ruin/60k':>9s} {'worstScen':>10s} {'gates'}")
for nm,d in BOOKS.items():
    w=wvec(d); er=float(ERV@w); row=DECKM.T@w; r=row-er; sig=sigma_of(w)
    mc=E.suite(r,sig,E.POL)
    G=np.concatenate([mc[c]["gcum"] for c in CONV]); W=np.concatenate([np.maximum(mc[c]["W"],1e-9) for c in CONV])
    ann=lambda x: (float(np.exp(np.log(x).mean()))**(1/40.)-1)*100
    ra=ann(G); rr=ann(W)
    pi=(np.array(L.PI_ROBUST)+np.array(L.PI_CANON))/2
    fails=S.gate_fails(w) if abs(w.sum()-1)<1e-9 else ["n/a"]
    print(f"{nm:14s} {float(pi@row)*100:10.2f} {((1+ra/100)*(1+INFL)-1)*100+er*100:9.2f} {sig*100:7.2f} {ra:10.2f} {rr:8.2f} {((1+rr/100)*(1+INFL)-1)*100:7.2f} "
          f"{mc[('robust','G')]['P5']:6.2f} {mc[('robust','G')]['ruin']:9d} {r.min()*100:10.2f} {'PASS' if not fails else 'FAIL:'+','.join(sorted(set(x.split(':')[1] for x in fails)))}", flush=True)
