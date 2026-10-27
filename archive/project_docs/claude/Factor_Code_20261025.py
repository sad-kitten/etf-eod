# ===== ff.py =====
import pandas as pd, numpy as np, io
def load(path, cols):
    txt = open(path, encoding='latin1').read().replace('\r','').split('\n'); rows=[]
    for l in txt:
        p=[x.strip() for x in l.split(',')]
        if len(p)>=2 and len(p[0])==6 and p[0].isdigit(): rows.append(p[:len(cols)+1])
        elif 'Annual' in l and rows: break
    d=pd.DataFrame(rows).set_index(0).astype(float)/100; d.columns=cols; d.index=pd.PeriodIndex(d.index,freq='M'); return d
F=load('ff5.csv',['MKT','SMB','HML','RMW','CMA','RF']); M=load('mom.csv',['UMD'])
F=F.join(M); F['SV']=F.SMB+F.HML; F['QV']=0.5*F.HML+0.5*F.RMW; F['QSV']=(F.SMB+F.HML+F.RMW)/1.0
print('range',F.index[0],F.index[-1])
def ann(x): return ((1+x).prod()**(12/len(x))-1)*100
def tstat(x): return x.mean()/x.std()*np.sqrt(len(x))
per=[('1963-07','2026-04'),('1963-07','1990-12'),('1991-01','2026-04'),('2000-01','2026-04'),('2007-01','2020-12'),('2010-01','2026-04'),('2021-01','2026-04')]
fs=['MKT','SMB','HML','RMW','CMA','UMD','SV','QV']
out=pd.DataFrame({f'{a[:4]}-{b[:4]}':{f:ann(F.loc[a:b,f]) for f in fs} for a,b in per}).round(2)
print('Geometric annual premium (%/yr; MKT = excess over T-bill; others long-short):'); print(out.to_string())
print('\nt-stat of mean, full and post-1991:'); print(pd.DataFrame({'full':{f:tstat(F[f]) for f in fs},'1991+':{f:tstat(F.loc['1991':,f]) for f in fs}}).round(2).T.to_string())
r=F[fs]; win={}
for y in (10,20):
    roll=(1+r).rolling(12*y).apply(np.prod,raw=True)-1
    win[f'{y}y win%']=(roll>0).mean()*100
print('\nShare of rolling windows with positive premium:'); print(pd.DataFrame(win).round(0).T.to_string())
print('\nCorrelations (monthly, full):'); print(r.corr().round(2).to_string())
# drawdown of factors
dd={f:float(((1+r[f]).cumprod()/(1+r[f]).cumprod().cummax()-1).min()*100) for f in fs}; print('\nMax drawdown %:',{k:round(v,1) for k,v in dd.items()})
F.to_csv('ff_monthly.csv')

# ===== reg.py =====
import pandas as pd, numpy as np
F=pd.read_csv('ff_monthly.csv',index_col=0); F.index=pd.PeriodIndex(F.index,freq='M')
m=pd.read_csv('/home/claude/sad-kitten/etf-eod/monthly_returns.csv',index_col=0); m.index=pd.PeriodIndex(m.index,freq='M'); m.columns=[c.replace('-','') for c in m.columns]
X=['MKT','SMB','HML','RMW','CMA','UMD']
T=['IVV','VTI','RSP','IWM','IJR','IWN','VBR','AVUV','DFAS','AVUS','VTV','IWD','RPV','VLUE','COWZ','QUAL','JQUA','VFQY','VALQ','MOAT','USMV','SPLV','MTUM','SPMO','SCHD','BRKB','XLV','IBB','XBI']
rows={}
for t in T:
    d=pd.concat([m[t],F],axis=1,join='inner').dropna(); d=d.loc[:'2026-04']
    y=d[t]-d['RF']; A=np.column_stack([np.ones(len(d)),d[X].values]); b,res,_,_=np.linalg.lstsq(A,y.values,rcond=None)
    e=y.values-A@b; s2=e@e/(len(d)-len(b)); se=np.sqrt(np.diag(s2*np.linalg.inv(A.T@A)))
    r2=1-e.var()/y.var()
    rows[t]=dict(n=len(d),start=str(d.index[0]),alpha_ann=b[0]*1200,t_alpha=b[0]/se[0],**{x:b[i+1] for i,x in enumerate(X)},R2=r2,te=np.std(e)*np.sqrt(12)*100)
R=pd.DataFrame(rows).T
for c in ['alpha_ann','t_alpha']+X+['R2','te']: R[c]=R[c].astype(float).round(2)
print(R.to_string()); R.to_csv('loadings.csv')

# ===== build_r8p.py =====
"""Deck R8p (Addendum 56, 2026-10-25) = R8n_X + factor-ETF row audit + better UTHR/VRTX sigma proxies.
Implied gross premium over IVV (pi-mean) = (beta_mkt - beta_mkt,IVV) x deck ERP + sum_f (b_f - b_f,IVV) x P_f
   + 0.25 x [(alpha_net - alpha_net,IVV) + (ER - ER_IVV)]     (implementation alpha, gross, shrunk 75% toward 0)
P_f (forward, pp/yr) = half the 1963-2026 Ken French long-short premium: SMB 0.85, HML 1.54, RMW 1.41, CMA 1.35;
UMD 1.6 (a quarter: turnover and crash risk). Loadings: etf-eod monthly returns 2000..2026-04 on FF5 + UMD (loadings.csv).
House rule (2026-09-04): correct HALF the gap, parallel shift in all scenarios, capped at +-1.0 pp. Rows with < 60 months of
history (DFAS) are not corrected. AVDV/AVDE/AVES (ex-US, no ex-US factor file in store) are not corrected."""
import pandas as pd, numpy as np, json
E="/home/claude/engine/data/"
D=pd.read_csv(E+'decks/Deck_R8n_X_20261024.csv').set_index('Ticker'); w=pd.read_csv(E+'decks/scenarios_25_R8i_20260909.csv').Weight_Normalized.values
LD=pd.read_csv('/home/claude/a56/loadings.csv',index_col=0)
ER={'AVUV':.25,'AVUS':.15,'COWZ':.49,'QUAL':.15,'JQUA':.12,'VFQY':.13,'VALQ':.29,'MOAT':.46,'USMV':.15,'SPLV':.25,'MTUM':.15,'SPMO':.13,'SCHD':.06,'VTV':.04,'XLV':.09,'IVV':.03}
P=dict(SMB=0.85,HML=1.54,RMW=1.41,CMA=1.35,UMD=1.6); erp=float(w@(D.loc['IVV']-D.loc['BIL']))
log={}
for t in ER:
    if t=='IVV' or LD.loc[t,'n']<60: continue
    deck=float(w@(D.loc[t]-D.loc['IVV']))
    imp=(LD.loc[t,'MKT']-LD.loc['IVV','MKT'])*erp+sum((LD.loc[t,f]-LD.loc['IVV',f])*p for f,p in P.items())
    imp+=0.25*((LD.loc[t,'alpha_ann']-LD.loc['IVV','alpha_ann'])+(ER[t]-ER['IVV']))
    sh=float(np.clip(0.5*(imp-deck),-1.0,1.0)); D.loc[t]=D.loc[t]+sh
    log[t]=dict(deck=round(deck,2),implied=round(imp,2),shift=round(sh,2))
D.reset_index().to_csv(E+'decks/Deck_R8p_20261025.csv',index=False)
print(pd.DataFrame(log).T.to_string())
# sigma proxies for UTHR/VRTX (no price history reachable): r = rf + beta*(IVV-rf) + 0.5*(XLV - b_XLV*IVV residual) + eps,
# eps scaled to target total vol 30% (UTHR) / 26% (VRTX); betas = stockanalysis 5-y (0.49 / 0.28), floored at 0.6 (health beta).
M=pd.read_csv(E+'panel_x.csv',index_col=0); rf=M['BIL'].fillna(0.0015); iv=M['IVV']; xl=M['XLV']
res=xl-0.73*iv; rng=np.random.default_rng(56)
for t,b,vol in (('UTHR',0.6,0.30),('VRTX',0.6,0.26)):
    base=rf+b*(iv-rf)+0.5*res; e=rng.standard_normal(len(M))
    s=np.sqrt(max(vol**2/12-base.var(),1e-6)); M[t]=(base+s*e).where(iv.notna())
    print(t,'vol %.1f%%'%(M[t].std()*np.sqrt(12)*100),'corr IVV %.2f'%M[t].corr(iv))
M.to_csv(E+'panel_p.csv')
json.dump(log,open('/home/claude/a56/r8p_shifts.json','w'))

# ===== uncertainty.py =====
"""E1 — parameter-uncertainty evaluation (verbatim from R55_Code_20261024.py, Addendum 55b), plus an optional F-extension
(Addendum 56): factor-premium uncertainty for factor ETFs (UNC_FACTOR=1). Off by default => identical to 55b."""
import inspect, os, numpy as np, pandas as pd
import lib_pathce as L, standing as ST, search_B as S, policy_engine2 as E
from core import UNIV, IDX, N, DECKM, ERV, SINGLES, sigma_of

SPEC1 = dict(eq=1.25, them=1.5, block=1.5, gold=1.5, trend=2.0, trend_stress=1.5, A=0.75, B=1.25, C=2.0, single=2.5)
NON_EQUITY = set("IEF TLT GOVZ LTPZ SCHP VTIP BIL SHY AGG TIP STIP HYG LQD EMB TAIL BTAL GLDM GLD PSLV DBMF KMLM CTA "
                 "CTA_OVERLAY PDBC IBIT".split())
GOLD_U = set(ST.GOLD_FAM) | set(ST.BLOCKS["gold"])
TREND_U = {"DBMF": 1.0, "KMLM": 1.0, "CTA": 1.0, "RSST": 1.0, "DBMFh": 1.0, "RSSTh": 1.0}
SLEEVE_SCALE = {"NTSX": dict(eq=0.9), "GDE": dict(eq=0.9, gold=0.9), "DBMFh": dict(eq=0.0)}
BLOCKS_U = dict(nuclear=ST.BLOCKS["nuclear"], defence=ST.BLOCKS["defence"], cyber=ST.BLOCKS["cyber"],
                semis=ST.BLOCKS["semis"], energy=["XLE", "AMLP"])
FACTORS = ["equity", "thematic", "gold", "trend"] + [f"block:{b}" for b in BLOCKS_U]

# ---- Addendum 56 F-extension: factor-premium uncertainty. Loadings = ETF FF5+UMD betas minus IVV's (|.|, regression on
# etf-eod 2000..2026-04, file loadings.csv); sd per unit loading (pp/yr) = forward-premium uncertainty.
UNC_FACTOR = os.environ.get("UNC_FACTOR", "0") == "1"
FSD = dict(SMB=1.0, HML=1.25, RMW=1.0, CMA=0.75, UMD=1.5)
if UNC_FACTOR:
    _LD = pd.read_csv(os.environ.get("LOADINGS_FILE", "/home/claude/a56/loadings.csv"), index_col=0)
    FACTORS = FACTORS + [f"ff:{f}" for f in FSD]

def loadings(spec=SPEC1):
    Ln = np.zeros((N, len(FACTORS))); idio = np.zeros(N)
    for t, i in IDX.items():
        if t not in NON_EQUITY: Ln[i, 0] = spec["eq"]
        if t in ST.THEMATIC: Ln[i, 1] = spec["them"]
        if t in GOLD_U: Ln[i, 2] = spec["gold"]
        Ln[i, 3] = TREND_U.get(t, 0.0) * spec["trend"]
        for k, names in enumerate(BLOCKS_U.values()):
            if t in names: Ln[i, 4 + k] = spec["block"]
        idio[i] = spec["single"] if t in SINGLES else spec.get(str(S.grades.get(t, "B")), spec["B"])
        sc = SLEEVE_SCALE.get(t, {})
        if "eq" in sc: Ln[i, 0] = spec["eq"] * sc["eq"]
        if "gold" in sc: Ln[i, 2] = spec["gold"] * sc["gold"]
        if UNC_FACTOR and t in _LD.index:
            k0 = 4 + len(BLOCKS_U); scale = spec["eq"] / SPEC1["eq"]
            for j, f in enumerate(FSD):
                Ln[i, k0 + j] = (float(_LD.loc[t, f]) - float(_LD.loc["IVV", f])) * FSD[f] * scale
    Ls = Ln.copy(); Ls[:, 3] *= spec["trend_stress"]
    return Ln / 100.0, Ls / 100.0, idio / 100.0

class Draws:
    def __init__(self, npaths, seed, spec=SPEC1):
        rng = np.random.default_rng(seed)
        nf0 = 4 + len(BLOCKS_U)
        F = rng.standard_normal((npaths, nf0)); U = rng.standard_normal((npaths, N)); self.coin = rng.random(npaths)
        if UNC_FACTOR: F = np.hstack([F, np.random.default_rng(seed + 56).standard_normal((npaths, len(FSD)))])
        Ln, Ls, idio = loadings(spec)
        self.En = F @ Ln.T + U * idio; self.Es = F @ Ls.T + U * idio
    def book(self, w, k=1.0, sigmix=True):
        dn = k * (self.En @ w); ds = k * (self.Es @ w)
        if sigmix:
            ratio = float(np.sqrt(max(float(w @ ST.COV120 @ w), 1e-12))) / sigma_of(w)
            m = np.where(self.coin < 0.5, 1.0, ratio)
        else:
            m = np.ones(len(self.coin))
        return dn, ds, m
    def book_sd(self, w, k=1.0):
        return float(np.std(k * (self.En @ w))), float(np.std(k * (self.Es @ w)))

def run_fast_u(r, sigv, pin, conv, dn, ds, m):
    FE = S.FE; s = FE.S[pin]; sig = sigv[s] * m[:, None]
    mu = np.log1p(np.maximum(r[s] + dn[:, None] + (ds - dn)[:, None] * ST.STRESS[s], -0.95))
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
                CE2=float(np.mean(W ** -1.0) ** -1.0), P5=float(np.quantile(W, .05)), P10=float(np.quantile(W, .10)), ruin=int(ru.sum()), pW1=float((W < 1.0).mean()))

CONVS = [("robust", "G"), ("robust", "L"), ("canonical", "G"), ("canonical", "L")]
def evaluate_u(w, D, k=1.0, sigmix=True, deck=None):
    M = DECKM if deck is None else deck
    r = M.T @ w - float(ERV @ w)
    dn, ds, m = (np.zeros(S.FE.npaths), np.zeros(S.FE.npaths), np.ones(S.FE.npaths)) if D is None else D.book(w, k, sigmix)
    out = {}
    for pin in ("robust", "canonical"):
        sigv, _ = ST.sigma_vec(w, pin)
        for conv in ("G", "L"): out[(pin, conv)] = run_fast_u(r, sigv, pin, conv, dn, ds, m)
    mlog = lambda key: float(np.mean([np.log(out[c][key]) for c in CONVS]))
    return dict(geo=np.exp(mlog("PathCE")), ce3=np.exp(mlog("CE3")), ce5=np.exp(mlog("CE5")), P5=out[("robust", "G")]["P5"],
                ruin=out[("robust", "G")]["ruin"], pW1=float(np.mean([out[c]["pW1"] for c in CONVS])))

_src = inspect.getsource(E.run)
_a, _b = "        sg = sigma[s] if vec else sigma\n", "        mu = np.log1p(r_net[s]) - "
assert _src.count(_a) == 1 and _src.count(_b) == 1 and _src.startswith("def run(rng, r_net, pi, conv, sigma, pol=POL):")
_src = (_src.replace("def run(rng, r_net, pi, conv, sigma, pol=POL):",
                     "def run_unc(rng, r_net, pi, conv, sigma, pol=POL, dn=0.0, ds=0.0, m=1.0):", 1)
            .replace(_a, "        sg = (sigma[s] if vec else sigma) * m\n")
            .replace(_b, "        mu = np.log1p(r_net[s] + dn + (ds - dn) * STRESS_U[s]) - "))
_ns = dict(vars(E)); _ns["STRESS_U"] = ST.STRESS
exec(compile(_src, "uncertainty.run_unc", "exec"), _ns); run_unc = _ns["run_unc"]

def suite_u(r_net, sigma, dn=0.0, ds=0.0, m=1.0, pol=E.POL):
    out = {}
    for pin, pi in (("robust", L.PI_ROBUST), ("canonical", L.PI_CANON)):
        rng = np.random.default_rng(42)
        sg = sigma[pin] if isinstance(sigma, dict) else sigma
        for conv in ("G", "L"): out[(pin, conv)] = run_unc(rng, r_net, pi, conv, sg, pol, dn, ds, m)
    return out

def tail_stats(r, pi=None):
    pi = L.PI_ROBUST if pi is None else pi
    o = np.argsort(r); rs, ps = r[o], pi[o] / pi.sum()
    def cvar(q):
        c = np.cumsum(ps); take = np.clip(q - (c - ps), 0.0, ps); return float((take * rs).sum() / q)
    sev5 = L.SEV >= 5; w = L.W_CAN
    return dict(worst=float(rs[0]), worst_scen=L.SC[int(o[0])], second=float(rs[1]), cvar10=cvar(0.10), cvar15=cvar(0.15),
                sev5=float((w[sev5] * r[sev5]).sum() / w[sev5].sum()), worst2=float(rs[:2].mean()))

# ===== report_u.py =====
"""Engine-exact report (55b report_u.py without the stk import; this engine's standing.py carries the A54 wiring natively).
Run: python3 report_u.py books.json out.csv [names]"""
import sys, os, json, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import standing as ST, report_v3 as R, uncertainty as U, policy_engine2 as E, lib_pathce as L
from core import wvec, ERV, IDX
DX = U.Draws(L.NP_, 31415)
BK = json.load(open(sys.argv[1])); keep = sys.argv[3].split(",") if len(sys.argv) > 3 else list(BK)
rows = []
for nm in keep:
    d = {t: v for t, v in BK[nm].items() if v > 0}
    t0 = time.time(); row = R.profile(nm, d)
    w = wvec(d); r = R.M0.T @ w - float(ERV @ w); sig = {p: ST.sigma_vec(w, p)[0] for p in ("robust", "canonical")}
    for tag, k in (("U1", 1.0), ("Uhalf", 0.5), ("U2", 2.0)):
        dn, ds, m = DX.book(w, k, True); mc = U.suite_u(r, sig, dn, ds, m); W = R.pooled(mc); rg = mc[("robust", "G")]
        pc = W ** (1 / float(L.T)) - 1
        row.update({f"{tag} geo x": round(R.ce_cagr(W, 1)[0], 3), f"{tag} real CAGR %": round(R.ce_cagr(W, 1)[1], 2),
                    f"{tag} g3 CE-CAGR %": round(R.ce_cagr(W, 3)[1], 2), f"{tag} P5 x": round(rg["P5"], 3),
                    f"{tag} P5 path CAGR %": round(float(np.quantile(pc, .05)) * 100, 2),
                    f"{tag} P50 x": round(float(np.median(W)), 2), f"{tag} P(W<1) %": round(float((W < 1).mean()) * 100, 1)})
    sdn, sds = DX.book_sd(w); ts = U.tail_stats(r)
    row.update({"err sd pp": round(sdn * 100, 2), "CVaR10 %": round(ts["cvar10"] * 100, 2)})
    row["secs total"] = round(time.time() - t0); rows.append(row)
    print(f"{nm:18s} geo {row['geo x']:.2f} g3 {row['g3 CE-CAGR %']:.2f} P5 {row['P5 terminal x']:.2f} | U1 geo {row['U1 geo x']:.2f} g3 {row['U1 g3 CE-CAGR %']:.2f} P5 {row['U1 P5 x']:.2f} | U2 g3 {row['U2 g3 CE-CAGR %']:.2f} | {row['gates']} | {row['constraints']} ({row['secs total']}s)", flush=True)
pd.DataFrame(rows).set_index("Port").to_csv(sys.argv[2])

# ===== eval_f.py =====
"""Fast screen (Addendum 56): central v3.4 gates (48-mo and M1 sigma) + E1-F metrics. python3 eval_f.py out.csv books.json"""
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import standing as ST, search_B as S, uncertainty as U, lib_pathce as L
from core import wvec, DECKM, ERV, IDX
D = U.Draws(S.FE.npaths, 2718); cg = lambda x: (x ** (1 / float(L.T)) - 1) * 100
rows = []
for nm, d in json.load(open(sys.argv[2])).items():
    d = {t: v for t, v in d.items() if v > 0}; w = wvec(d); e = ST.evaluate_fast(w); v = ST.violations(w)
    sl = ST.SIGMA_LEVEL; ST.SIGMA_LEVEL = "cov120"
    try: e1 = ST.evaluate_fast(w)
    finally: ST.SIGMA_LEVEL = sl
    u = U.evaluate_u(w, D, 1, True); u2 = U.evaluate_u(w, D, 2, True)
    rows.append(dict(Port=nm, geo=round(e["geo"], 3), ce3=round(cg(e["ce3"]), 2), P5=round(e["P5"], 2), U1geo=round(u["geo"], 3),
        U1ce3=round(cg(u["ce3"]), 2), U1P5=round(u["P5"], 3), U2ce3=round(cg(u2["ce3"]), 2), worst=round(e["worst"] * 100, 2),
        gates="PASS" if not e["fails"] else ";".join(e["fails"]), M1="PASS" if not e1["fails"] else ";".join(e1["fails"]),
        cons="ok" if not v else ";".join(v)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).set_index("Port").to_csv(sys.argv[1])

# ===== search_f.py =====
"""Addendum 56 climb: search_v3.climb with (a) E1-F objective (U1 gamma=3 CE, factor-premium uncertainty on, sigma mix),
(b) DUAL_SIGMA co-equal gate (central gates must pass under 48-month and 120-month sigma), (c) LOCK="UTHR:0.05,VRTX:0.05"
floors (owner priority: holdings may not be sold below the floor), (d) ADD_POOL forced into the search pool, RSST/EXCL out.
Run: SEEDS_FILE=... OUTJSON=... python3 search_f.py TAG"""
import os, sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
import standing as ST, search_B as S, uncertainty as U, search_v3 as SV
from core import DECKM, ERV, IDX, UNIV
D = U.Draws(S.FE.npaths, 2718)
def objective(w):
    r = DECKM.T @ w - float(ERV @ w); dn, ds, m = D.book(w, 1.0, True); vals = []
    for pin in ("robust", "canonical"):
        sigv, _ = ST.sigma_vec(w, pin)
        for conv in ("G", "L"): vals.append(np.log(U.run_fast_u(r, sigv, pin, conv, dn, ds, m)["CE3"]))
    return float(np.mean(vals))
SV.objective = objective
_g, _d = SV.gates_ok, SV.deficit
def _m1(f, v):
    s = ST.SIGMA_LEVEL; ST.SIGMA_LEVEL = "cov120"
    try: return f(v)
    finally: ST.SIGMA_LEVEL = s
SV.gates_ok = lambda v: _g(v) and _m1(_g, v)
SV.deficit = lambda v: _d(v) + _m1(_d, v)
LOCK = {k: float(x) for k, x in (p.split(":") for p in os.environ.get("LOCK", "").split(",") if p)}
_nb = SV.neighbours
def neighbours(w, pool):
    out = []
    for v, i, j, d in _nb(w, pool):
        if i in LOCK and v[IDX[i]] < LOCK[i] - 1e-9: continue
        out.append((v, i, j, d))
    return out
SV.neighbours = neighbours
ADD = [t for t in os.environ.get("ADD_POOL", "").split(",") if t]
EXCL = set(os.environ.get("EXCL", "").split(","))
_S1 = list(S.S1); _POOL = list(S.POOL)
S.POOL[:] = [t for t in _POOL if t not in EXCL]; S.S1[:] = [t for t in _S1 if t not in EXCL]
_climb_src = SV.climb
def climb(seed, tag):
    # identical to search_v3.climb except the pool: shortlist by marginal objective + held + ADD_POOL, never RSST/EXCL
    import time
    w = SV.wvec(seed); t0 = time.time()
    pool_all = sorted((set(S.POOL) | {"BRKB"} | set(S.S1)) - EXCL)
    sc = {}
    for j in pool_all:
        v = w * 0.97; v[IDX[j]] += 0.03; v /= v.sum(); sc[j] = objective(v)
    pool = sorted((set(sorted(sc, key=lambda j: -sc[j])[:SV.SHORTLIST]) | set(S.held(w)) | set(ADD)) - EXCL)
    print(f"[{tag}] pool {pool}", flush=True)
    dcur = SV.deficit(w); steps = 0
    while dcur > 1e-9 and steps < 12:
        best = None
        for v, i, j, d in neighbours(w, pool):
            dv = SV.deficit(v)
            if best is None or dv < best[0] - 1e-12: best = (dv, v, i, j, d)
        if best is None or best[0] >= dcur - 1e-12: print(f"[{tag}] repair stalled {dcur:.3f}", flush=True); break
        dcur, w = best[0], best[1]; steps += 1
        print(f"[{tag}] repair {steps}: {best[2]}->{best[3]} {best[4]*100:.0f}% deficit {dcur:.3f} | {S.show(w)}", flush=True)
    cur = objective(w)
    print(f"[{tag}] start {S.show(w)} | U1ce3 {np.exp(cur):.3f}", flush=True)
    for step in range(SV.MAXSTEPS):
        cands = [(o, v, i, j, d) for v, i, j, d in neighbours(w, pool) for o in [objective(v)] if o > cur + 1e-6]
        if not cands: break
        cands.sort(key=lambda x: -x[0]); acc = None
        for c in cands[:20]:
            if SV.gates_ok(c[1]): acc = c; break
        if acc is None: print(f"[{tag}] no improving move clears gates", flush=True); break
        o, v, i, j, d = acc; w, cur = v, o
        print(f"[{tag}] {step+1}: {i}->{j} {d*100:.0f}% | U1ce3 {np.exp(cur):.3f} | {S.show(w)} | {time.time()-t0:.0f}s", flush=True)
    return w
if __name__ == "__main__":
    BK = json.load(open(os.environ["SEEDS_FILE"])); tag = sys.argv[1]
    w = climb({t: v for t, v in BK[tag].items() if v > 0}, tag)
    json.dump({tag: {t: round(float(w[IDX[t]]), 2) for t in S.held(w)}}, open(os.environ["OUTJSON"], "w"), indent=1)
    print(f"[{tag}] FINAL {S.show(w)}", flush=True)

# ===== env_r8p.sh =====
export ENGINEV2_DATA=/home/claude/engine/data DECK_FILE=Deck_R8p_20261025.csv SCEN_FILE=scenarios_25_R8i_20260909.csv GRADES_FILE=/home/claude/engine/data/decks/Row_Grades_R8j_20260909.csv ENGINEV2_PANEL=/home/claude/engine/data/panel_p.csv ENGINEV2_END=2026-08 ENGINEV2_T=50 NPATHS=6000 STRESS_GATE=1 ETF_CAP=0.20 ENGINEV2_EXTRA=RSST,RSSB,NTSX,GDE,DBMFh,RSSTh,UTHR,VRTX,ISRG,DFAS,AVDV,AVUS,AVDE,AVES,USMV,SPLV,JQUA,VFQY,VALQ UNC_FACTOR=1 LOADINGS_FILE=/home/claude/a56/loadings.csv PYTHONWARNINGS=ignore EXCL=DBMFh,RSSTh,QTUM,SPGI,WM,CEG,ICE,CBOE,NDAQ,V,COIN,IBKR,WPM,ISRG,RSST,RSSB,NTSX,GDE

# ===== seeds56.json =====
{"R56_free": {"GLDM": 0.2, "DBMF": 0.2, "SHLD": 0.15, "CIBR": 0.1, "SMH": 0.05, "URNM": 0.05, "CME": 0.05, "MSCI": 0.05, "FNV": 0.05, "BRKB": 0.05, "KMLM": 0.03, "XAR": 0.02}, "R56_UV": {"GLDM": 0.2, "DBMF": 0.2, "SHLD": 0.15, "CIBR": 0.1, "SMH": 0.05, "URNM": 0.05, "CME": 0.05, "BRKB": 0.05, "KMLM": 0.03, "XAR": 0.02, "UTHR": 0.05, "VRTX": 0.05}, "R56_UVx": {"GLDM": 0.2, "DBMF": 0.2, "SHLD": 0.15, "CIBR": 0.05, "SMH": 0.05, "URNM": 0.05, "CME": 0.05, "MSCI": 0.05, "BRKB": 0.05, "KMLM": 0.03, "XAR": 0.02, "UTHR": 0.05, "VRTX": 0.05}, "R56_UVp": {"GLDM": 0.15, "DBMF": 0.15, "KMLM": 0.1, "SHLD": 0.15, "CIBR": 0.1, "SMH": 0.05, "URNM": 0.05, "CME": 0.05, "BRKB": 0.05, "UTHR": 0.05, "VRTX": 0.05, "XAR": 0.05}}
# ===== core.py/standing.py edits: SINGLES_TOT, SINGLES_MAX, CW_CAP env overrides (defaults unchanged) =====
