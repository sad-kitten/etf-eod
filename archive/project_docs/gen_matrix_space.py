#!/usr/bin/env python3
"""Augment 2026Q1 scenario-CAGR matrix with the holdings/benchmark missing from it.
Method (validated): CAGR%_{t,s} = RF%_s + SUMPRODUCT(beta_t, premia%_s) + 12*alpha%_t + delta%_{t,s}
beta,alpha = FF5+Mom OLS on monthly excess returns, shrunk by w=n/(n+60).
Single stocks: matched 120-mo window (multiple-expansion discipline). ETFs: full history."""
import pandas as pd, numpy as np, re, os
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

SRC='/mnt/project'; OUT='/mnt/user-data/outputs'; os.makedirs(OUT, exist_ok=True)
FAC=['Mkt-RF','SMB','HML','RMW','CMA','Mom']

def load_ff():
    def parse(fn,cols):
        rows=[]
        for ln in open(f'{SRC}/{fn}'):
            s=ln.strip().split(',')
            if len(s)>=2 and re.fullmatch(r'\d{6}',s[0].strip()):
                try: rows.append([s[0].strip()]+[float(x) for x in s[1:len(cols)+1]])
                except: pass
        df=pd.DataFrame(rows,columns=['ym']+cols); df['ym']=pd.to_datetime(df['ym'],format='%Y%m')+pd.offsets.MonthEnd(0)
        return df.set_index('ym')
    return parse('F-F_Research_Data_5_Factors_2x3.csv',['Mkt-RF','SMB','HML','RMW','CMA','RF']).join(
                 parse('F-F_Momentum_Factor.csv',['Mom']),how='inner')
ff=load_ff(); man=pd.read_csv(f'{SRC}/manifest.csv'); fmap=dict(zip(man.ticker,man.file))
def mret(t):
    d=pd.read_csv(f"{SRC}/{fmap.get(t,t+'.csv')}"); d['Date']=pd.to_datetime(d['Date'],errors='coerce',utc=True).dt.tz_localize(None)
    d=d.dropna(subset=['Date','Adj_Close']).sort_values('Date'); px=d.set_index('Date')['Adj_Close']; px.index=pd.DatetimeIndex(px.index)
    return px.resample('ME').last().pct_change().dropna()*100

xl=pd.ExcelFile(f'{SRC}/KEEP_Scenario_CAGR_Matrix_2026Q1_plus_CTA_overlay.xlsx')
etf=xl.parse('ETF_Scenario_CAGR').rename(columns={'Unnamed: 0':'Ticker'})
prem=xl.parse('Scenario_Factor_Premia_%').rename(columns={'Unnamed: 0':'Scenario'}).set_index('Scenario')
beta_o=xl.parse('Betas_Shrunk'); dlt_o=xl.parse('Scenario_Deltas_%').rename(columns={'Unnamed: 0':'Ticker'})
scn=xl.parse('Scenarios'); SC=list(scn['Scenario'])

def fit(t,window=None):
    r=mret(t); r=r.iloc[-window:] if window else r
    df=ff.join((r-ff['RF']).rename('ex'),how='inner').dropna()
    X=np.column_stack([np.ones(len(df))]+[df[f].values for f in FAC]); y=df['ex'].values
    b,res,_,_=np.linalg.lstsq(X,y,rcond=None); n=len(df); sw=n/(n+60)
    yh=X@b; r2=1-np.sum((y-yh)**2)/np.sum((y-y.mean())**2)
    return {f:sw*b[1+i] for i,f in enumerate(FAC)}, sw*b[0], n, sw, float(r2)
def cagr_row(beta,a,delta): return {s: prem.loc[s,'RF']+sum(beta[f]*prem.loc[s,f] for f in FAC)+12*a+delta.get(s,0.0) for s in SC}
def dv(d): return {SC[i]:float(d[i]) for i in range(20)}

# disclosed discretionary deltas (caps +/-2). scenario order per SC.
cdf=xl.parse('CTA_Overlay_Compare'); cdf['Overlay_Delta_pts']=pd.to_numeric(cdf['Overlay_Delta_pts'],errors='coerce')
TREND=cdf.drop_duplicates('Scenario').set_index('Scenario')['Overlay_Delta_pts'].reindex(SC).fillna(0).astype(float).to_dict()
GOLD =dv([0,-1,2,2,2,-0.5,0,1,1,0.5,1,1.5,1.5,0,1,0.5,0.5,1.5,1,2])
CYBER=dv([0,0,0,0,1,0,0,1,4,0,0,1,0,0,0,0,0,0,0,0])
EMV  =dv([0,1,-1,0,-2,0,0,-1,-1,0,-2,-1,-1,0,0,0,0,0,0,-1])
QUANT=dv([0,0,0,0,-1,1,0,0,0,0,0,0,0,1.5,0,0,0,0,0,0])
CBOE_D=dv([0,0,1,0.5,2,0,0,1,2,0,0.5,2,1.5,0,1,0,0.5,1,0.5,1.5])
SPGI_D=dv([1,1.5,-0.5,-0.5,-2.5,0.5,0,-0.5,-1,0,-0.5,-1,-1.5,0.5,0,0,0,0,0,-1])
EXCH_D=dv([0,0.5,0.5,0,1,0,0,0.5,1,0,0,1,0.5,0,0.5,0,0,0.5,0,0.5])
ZERO ={s:0.0 for s in SC}
ER={'CME':0,'CBOE':0,'ICE':0,'SPGI':0,'KMLM':.90,'DBMF':.85,'AVDV':.36,'AVES':.36,'IHAK':.47,'QTUM':.40,'GLDM':.10,'IVV':.03}
SPEC=[  # ticker, window, delta, ER, sector, note
 ('CME',120,EXCH_D,0,'Exchange/clearing','Rate-vol + clearing flow; matched 120mo'),
 ('CBOE',120,CBOE_D,0,'Exchange/vol','Long-vol monetizer; + in vol-stress; matched 120mo'),
 ('ICE',120,EXCH_D,0,'Exchange/data','Flow + data; matched 120mo'),
 ('SPGI',120,SPGI_D,0,'Ratings/data','Issuance-geared; - in credit-stress; matched 120mo'),
 ('KMLM',None,TREND,.90,'Managed futures','Trend; CTA_Overlay delta vector'),
 ('DBMF',None,TREND,.85,'Managed futures','Trend; CTA_Overlay delta vector'),
 ('AVDV',None,EMV,.36,'Intl small value','ex-US value; -china/-credit-stress'),
 ('AVES',None,EMV,.36,'EM value','EM value; -china/-credit-stress'),
 ('IHAK',None,CYBER,.47,'Cybersecurity','+cyber-systemic / +geopolitical'),
 ('QTUM',None,QUANT,.40,'Quantum/compute','+AI/+space, -credit-stress'),
 ('GLDM',None,GOLD,.10,'Gold','Debasement/disorder hedge; +stagflation/debt/energy/liquidity'),
 ('IVV',120,ZERO,.03,'Broad beta (benchmark)','S&P500 reference row for R3AF; no discretionary delta'),
]
newC=[]; newB=[]; newD=[]
for t,w,d,er,sec,note in SPEC:
    beta,a,n,sw,r2=fit(t,w); row=cagr_row(beta,a,d)
    newC.append({'Ticker':t, **{s:round(row[s],2) for s in SC}})
    newB.append({'Ticker':t,'n':n,'r2':round(r2,4),'shrink_w':round(sw,4),'alpha_m_shrunk':round(a,4),
                 **{f:round(beta[f],4) for f in FAC}})
    newD.append({'Ticker':t, **{s:round(d[s],2) for s in SC}})
newC=pd.DataFrame(newC); newB=pd.DataFrame(newB); newD=pd.DataFrame(newD)
print("New rows wtd-mean CAGR (sanity vs scoring run):")
Wn=scn.set_index('Scenario')['Weight_Normalized'].reindex(SC).values; Wn=Wn/Wn.sum()
for _,r in newC.iterrows(): print(f"  {r['Ticker']:5s} {float((r[SC].values.astype(float)*Wn).sum()):6.2f}")

# ---- assemble FULL (house 34 + new 12) ----
etf['Source']='house_2026Q1'; newC['Source']='fitted_2026Q2'
beta_o['Source']='house_2026Q1'; newB['Source']='fitted_2026Q2'
dlt_o['Source']='house_2026Q1'; newD['Source']='fitted_2026Q2'
CAGR_FULL=pd.concat([etf[['Ticker']+SC+['Source']], newC[['Ticker']+SC+['Source']]],ignore_index=True)
BETA_FULL=pd.concat([beta_o, newB],ignore_index=True)
DLT_FULL =pd.concat([dlt_o[['Ticker']+SC], newD[['Ticker']+SC]],ignore_index=True)
DLT_FULL['Source']=list(dlt_o['Source'])+list(newD['Source'])
PREM=prem.reset_index()
SCEN=scn.copy()

# ---- CSVs ----
CAGR_FULL.to_csv(f'{OUT}/Scenario_CAGR_Matrix_FULL_2026Q2.csv',index=False)
newC.drop(columns='Source').merge(newB,on='Ticker').to_csv(f'{OUT}/New_Tickers_CAGR_and_Betas_2026Q2.csv',index=False)
newB.drop(columns='Source').merge(newD.drop(columns='Source'),on='Ticker',suffixes=('_beta','_delta')).to_csv(
    f'{OUT}/New_Tickers_Betas_Deltas_2026Q2.csv',index=False)

# ---- workbook ----
ARI='Arial'; HF=Font(name=ARI,bold=True,color='FFFFFF',size=10); BF=Font(name=ARI,size=10)
HEAD=PatternFill('solid',fgColor='1F3864'); NEWF=PatternFill('solid',fgColor='FCE4D6')
KEYF=PatternFill('solid',fgColor='FFF2CC'); thin=Side(style='thin',color='D9D9D9'); BD=Border(thin,thin,thin,thin)
wb=Workbook(); wb.remove(wb.active)

def sheet(name, df, numfmt=None, freeze='B2', widths=None, newmask=None, keycols=None):
    ws=wb.create_sheet(name); cols=list(df.columns)
    for j,c in enumerate(cols,1):
        cell=ws.cell(1,j,c); cell.font=HF; cell.fill=HEAD; cell.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True); cell.border=BD
    for i,(_,r) in enumerate(df.iterrows(),2):
        isnew = newmask[i-2] if newmask is not None else False
        for j,c in enumerate(cols,1):
            v=r[c]; v=None if (isinstance(v,float) and np.isnan(v)) else v
            cell=ws.cell(i,j,v); cell.font=BF; cell.border=BD
            if isinstance(v,(int,float)) and numfmt and c not in ('Ticker','Scenario','Source','Canonical_Parent','Notes_Merged'):
                cell.number_format=numfmt
            if isnew: cell.fill=NEWF
            if keycols and c in keycols and not isnew: cell.fill=KEYF
    ws.freeze_panes=freeze
    W=widths or {}; 
    for j,c in enumerate(cols,1):
        ws.column_dimensions[get_column_letter(j)].width=W.get(c, 11 if c not in ('Ticker','Source') else 13)
    return ws

newmask_cagr=[s=='fitted_2026Q2' for s in CAGR_FULL['Source']]
newmask_beta=[s=='fitted_2026Q2' for s in BETA_FULL['Source']]
newmask_dlt =[s=='fitted_2026Q2' for s in DLT_FULL['Source']]

# README
ws=wb.create_sheet('README'); ws.sheet_view.showGridLines=False
ws.column_dimensions['A'].width=120; ws.cell(1,1,'KEEP — Scenario CAGR Matrix, 2026Q2 AUGMENTED').font=Font(name=ARI,bold=True,size=14)
lines=[
 '',
 'Purpose: extends KEEP_Scenario_CAGR_Matrix_2026Q1 with the 11 portfolio holdings + 1 benchmark that the 2026Q1',
 'matrix had no house rows for (the names the June books turn on). Drop-in replacement: CAGR_Matrix_FULL carries all',
 '34 prior house rows + 12 new fitted rows in one matrix. Source column flags provenance.',
 '',
 'NEW ROWS (fitted_2026Q2): CME, CBOE, ICE, SPGI, KMLM, DBMF, AVDV, AVES, IHAK, QTUM, GLDM, IVV(benchmark).',
 '',
 'CONSTRUCTION (per cell):',
 '   CAGR%[t,s] = RF%[s] + SUMPRODUCT(beta[t,:], premia%[s,:]) + 12 * alpha_m_shrunk%[t] + delta%[t,s]',
 '   beta, alpha = FF5+Mom OLS on monthly excess returns (Ket French), shrunk by w = n/(n+60).',
 '   Single stocks (CME/CBOE/ICE/SPGI/IVV): matched 120-mo window (multiple-expansion discipline).',
 '   ETFs (KMLM/DBMF/AVDV/AVES/IHAK/QTUM/GLDM): full available monthly history.',
 '   Sheets Betas_Shrunk_FULL + Scenario_Factor_Premia_pct + Scenario_Deltas_FULL fully reconstruct every CAGR cell.',
 '   Reconstruction_Check shows 3 live cross-sheet formulas reproducing stored values (diff ~ 0).',
 '',
 'VALIDATION:',
 '   - Refit of an existing matrix name (NUKZ) reproduces its Betas_Shrunk row to 3 decimals.',
 '   - Reconstruction formula reproduces the matrix NUKZ CAGR row to < 0.01 pt.',
 '   - Base-Primary 48-mo realized vol from the same price CSVs = 11.30% (recorded ~11.1%).',
 '',
 'DISCLOSED DISCRETIONARY DELTAS (caps +/-2 pts, per the CTA_Overlay convention):',
 '   GOLD (GLDM): + stagflation/debt-overhang/credit-stress/energy/liquidity-fiscal/geopolitical/taiwan; -1 reflation.',
 '   TREND (KMLM,DBMF): exact CTA_Overlay delta vector (documented managed-futures overlay).',
 '   CBOE: + in vol-stress (credit-stress/cyber/taiwan/liquidity/stagflation) — long-vol monetizer.',
 '   SPGI: - in credit-stress/credit-accident (issuance freezes); + base/reflation.',
 '   CME/ICE: modest + in flow/rate-vol scenarios. IHAK: + cyber-systemic/geopolitical. QTUM: +AI/+space,-credit.',
 '   AVDV/AVES: - china-hard-landing/-credit-stress, + reflation. IVV: none (pure factor benchmark).',
 '',
 'FRAGILITY (symmetric across mods; does not flip the mod ranking):',
 '   - Single-stock CBOE/SPGI alpha is the softest input; their high fitted alpha overstates the central-CAGR cost of',
 '     trimming them. Winning mods fund gold from dust / trend-split, not from gutting franchises, so ranking is robust.',
 '   - GLDM 9.56 fitted CAGR rides recent gold momentum + debasement deltas; if forward gold CAGR is lower the central',
 '     edge shrinks but the vol/tail edge (the reason gold wins) persists.',
 '',
 'LOAD (Python):  import pandas as pd',
 '   cagr  = pd.read_excel("KEEP_Scenario_CAGR_Matrix_2026Q2_AUGMENTED.xlsx","CAGR_Matrix_FULL").set_index("Ticker")',
 '   betas = pd.read_excel(... ,"Betas_Shrunk_FULL").set_index("Ticker")',
 '   Or the flat file: pd.read_csv("Scenario_CAGR_Matrix_FULL_2026Q2.csv")',
 '',
 'UNITS: CAGR, premia, deltas, alpha_m_shrunk in PERCENT. betas dimensionless. Generated 2026-06-28.',
]
for k,t in enumerate(lines,3):
    c=ws.cell(k,1,t); c.font=Font(name=ARI, size=10, bold=t.endswith(':') or t.startswith(('CONSTRUCTION','VALIDATION','DISCLOSED','FRAGILITY','NEW ROWS','PURPOSE','LOAD','UNITS')))
    c.alignment=Alignment(wrap_text=True,vertical='top')

cagr_w={'Ticker':13,'Source':14}; cagr_w.update({s:10 for s in SC})
sheet('CAGR_Matrix_FULL', CAGR_FULL, numfmt='0.00', widths=cagr_w, newmask=newmask_cagr)
sheet('Betas_Shrunk_FULL', BETA_FULL, numfmt='0.0000', widths={'Ticker':13,'Source':14,'n':7,'r2':8,'shrink_w':9,'alpha_m_shrunk':14}, newmask=newmask_beta)
sheet('Scenario_Deltas_FULL', DLT_FULL, numfmt='0.0;-0.0;"-"', widths=cagr_w, newmask=newmask_dlt)
sheet('Scenario_Factor_Premia_pct', PREM, numfmt='0.0', widths={'Scenario':42})
sheet('Scenarios', SCEN, numfmt='0.000', widths={'Scenario':42,'Canonical_Parent':42,'Notes_Merged':60,'Severity_1to5':12,'Weight_%':9,'Weight_Normalized':16,'Weight_Normalized_%':18})

# Reconstruction_Check (live cross-sheet formulas)
def colL(df,name): return get_column_letter(list(df.columns).index(name)+1)
def trow(df,tk): return list(df['Ticker']).index(tk)+2
def srow(scen): return SC.index(scen)+2
ws=wb.create_sheet('Reconstruction_Check')
hdr=['Check (ticker x scenario)','RF','p_MktRF','p_SMB','p_HML','p_RMW','p_CMA','p_Mom','b_MktRF','b_SMB','b_HML','b_RMW','b_CMA','b_Mom','alpha_m','delta','recon RF+SUMPRODUCT(p,b)+12a+d','stored','diff']
for j,h in enumerate(hdr,1):
    c=ws.cell(1,j,h); c.font=HF; c.fill=HEAD; c.alignment=Alignment(horizontal='center',wrap_text=True); c.border=BD
checks=[('CME','Base Case – Managed Re-industrialization'),('GLDM','Stagflation / Fiscal Dominance'),('KMLM','Energy Shock / Supply Constraint')]
bC=colL(BETA_FULL,'Mkt-RF'); # contiguous Mkt-RF..Mom in betas
pC=colL(PREM,'Mkt-RF')
for i,(tk,sc) in enumerate(checks,2):
    pr=srow(sc); br=trow(BETA_FULL,tk); cr=trow(CAGR_FULL,tk); scol=colL(CAGR_FULL,sc); dr=trow(DLT_FULL,tk); dcol=colL(DLT_FULL,sc)
    ws.cell(i,1,f'{tk} x {sc[:28]}').font=BF
    ws.cell(i,2,f"=Scenario_Factor_Premia_pct!B{pr}").font=BF      # RF
    for k in range(6): ws.cell(i,3+k,f"=Scenario_Factor_Premia_pct!{get_column_letter(3+k)}{pr}").font=BF  # premia Mkt-RF..Mom (C..H)
    bcols=[get_column_letter(list(BETA_FULL.columns).index(f)+1) for f in FAC]
    for k,bl in enumerate(bcols): ws.cell(i,9+k,f"=Betas_Shrunk_FULL!{bl}{br}").font=BF
    acol=colL(BETA_FULL,'alpha_m_shrunk')
    ws.cell(i,15,f"=Betas_Shrunk_FULL!{acol}{br}").font=BF
    ws.cell(i,16,f"=Scenario_Deltas_FULL!{dcol}{dr}").font=BF
    ws.cell(i,17,f"=B{i}+SUMPRODUCT(C{i}:H{i},I{i}:N{i})+12*O{i}+P{i}").font=BF; ws.cell(i,17).number_format='0.00'
    ws.cell(i,18,f"=CAGR_Matrix_FULL!{scol}{cr}").font=BF; ws.cell(i,18).number_format='0.00'
    ws.cell(i,19,f"=Q{i}-R{i}").font=BF; ws.cell(i,19).number_format='0.000'
for j in range(1,20): ws.column_dimensions[get_column_letter(j)].width=(30 if j==1 else (34 if j==17 else 9))
ws.freeze_panes='B2'

path=f'{OUT}/KEEP_Scenario_CAGR_Matrix_2026Q2_AUGMENTED.xlsx'; wb.save(path)
print('\nWROTE:', path)
print('rows: house=%d new=%d full=%d' % ((CAGR_FULL['Source']=='house_2026Q1').sum(),(CAGR_FULL['Source']=='fitted_2026Q2').sum(),len(CAGR_FULL)))
print('CSVs:', os.listdir(OUT))
