"""R8n + single-name rows (2026-10-24): base-case = 50-y two-stage FCF IRR at today's price (FCF yield = 1/(EV/FCF), consensus
3-y growth held 5 y then 4.5%, exit 20x FCF) minus 1.0 pp single-stock skew haircut; other scenarios = base + k x (anchor(s) -
anchor(base)). UTHR, VRTX: anchor XLV, k = 0.75 (5-y betas 0.49/0.28 floored); ISRG: anchor IHI, k = 1.5 (beta 1.27 vs IHI 0.83).
Monthly series: ISRG real; UTHR/VRTX proxied by XBI (no history in store; idiosyncratic risk only approximated). Grade C, ER 0."""
import pandas as pd, sys
D=pd.read_csv('data/decks/Deck_R8n_20261024.csv').set_index('Ticker'); b=D.columns[0]
for t,base,anc,k in [('UTHR',10.94-1.0,'XLV',0.75),('VRTX',8.78-1.0,'XLV',0.75),('ISRG',7.98-1.0,'IHI',1.5)]:
    D.loc[t]=base+k*(D.loc[anc]-D.loc[anc,b])
D.reset_index().to_csv('data/decks/Deck_R8n_X_20261024.csv',index=False)
print(D.loc[['UTHR','VRTX','ISRG','IVV','SOXX']].iloc[:,[0,2,11,22,24]].round(2).to_string())
W=pd.read_csv('data/decks/scenarios_25_R8i_20260909.csv').Weight_Normalized.values
print('pi-mean', {t: round(float(W@D.loc[t].values),2) for t in ['UTHR','VRTX','ISRG','IVV','SOXX','BRKB']})
M=pd.read_csv('data/panel_stk.csv',index_col=0); M['UTHR']=M['XBI']; M['VRTX']=M['XBI']; M.to_csv('data/panel_x.csv')
g=pd.read_csv('data/decks/Row_Grades_R8j_20260909.csv'); g=pd.concat([g,pd.DataFrame({'Ticker':['UTHR','VRTX','ISRG'],'Grade':['C']*3})]).drop_duplicates('Ticker',keep='last'); g.to_csv('data/decks/Row_Grades_R8j_20260909.csv',index=False)
import json; e=json.load(open('data/decks/ER_R8.json')); e.update(UTHR=0.0,VRTX=0.0,ISRG=0.0); json.dump(e,open('data/decks/ER_R8.json','w'))
