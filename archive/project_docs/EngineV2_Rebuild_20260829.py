"""Engine v2 REBUILD (2026-08-29) - PathCE reproduction attempt for book #1.
Target: PathCE_v2 12.35 (succession campaign 2026-08-27). Pre-committed bar:
|d|<=0.15 reproduced / <=0.50 consistent / >0.50 mismatch. RESULT: no variant
within bar; target inside variant envelope. Deck consistent in scale; engine
spec underdetermined from store.

ASSUMPTION REGISTER
AR1 ER table absent in store -> prospectus values; WER=0.532% (+/-0.10pp => -/+~0.5 PathCE)
    GLDM.10 SHLD.50 DBMF.85 NUKZ.85 CIBR.59 GRID.65 CTA.76 CME 0 JEDI.50 MSCI 0
AR2 vol48=11.618% monthly, window ..2026-08 (partial Aug; JEDI live 11/48, weights renormalized)
AR3 Markov: yearly stay-or-redraw-from-pi, lambda=1/8 (stationary exact; sojourn ~8y)
AR4 stress tilt = sqrt(w_canon)*severity*max(0, EVGbase-EVGs); benchmark EVG20260811,
    IAI proxied = mean(CME,CBOE,NDAQ) [IAI row absent from ALL in-store decks]
AR5 drift convention unresolved: G (deck=arith mean; mu=ln(1+r)-s^2/2) vs L (deck=log growth)
AR6 PathCE stationary pi unresolved: robust 50/25/25 (registry scopes it to RCEW) vs canonical
AR7 seed=42, 60k paths, 40y, draw 0.03 fixed year-end (per rcew40.py ratified mechanics)
"""
import pandas as pd, numpy as np
rng = np.random.default_rng(42)
deck = pd.read_csv("/mnt/project/Deck_Recal_R1R4_20260822.csv"); deck["Ticker"]=deck["Ticker"].str.strip(); deck=deck.set_index("Ticker")
sc = list(deck.columns)
deck.loc["IAI"] = deck.loc[["CME","CBOE","NDAQ"]].mean()   # AR4 benchmark-only proxy
scen = pd.read_csv("/mnt/project/scenarios_20_canonical.csv")
w_can = scen["Weight_Normalized"].values; sev = scen["Severity_1to5"].values.astype(float)
P1={'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.15,'CIBR':.10,'GRID':.10,'CTA':.05,'CME':.05,'JEDI':.05,'MSCI':.05}
EVG11={'CIBR':.15,'SHLD':.15,'CTA':.10,'DBMF':.10,'GLDM':.10,'IAI':.10,'NUKZ':.10,'QTUM':.10,'XAR':.10}
prow=lambda b: np.array([sum(b[t]*deck.loc[t,s] for t in b) for s in sc])/100.0
def make_pi_robust(bench):
    evg=prow(bench); dsl=np.maximum(0.0,evg[0]-evg)
    st=np.sqrt(w_can)*sev*dsl; st/=st.sum()
    pi=0.5*w_can+0.25/20+0.25*st; return pi/pi.sum()
r_g=prow(P1); WER=0.00532; SIG=0.11618; NP,T,LAM,DRAW=60000,40,0.125,0.03
def run(pi,conv,sigma=SIG,wer=WER):
    rn=r_g-wer; s=rng.choice(20,size=NP,p=pi); W=np.ones(NP); ru=np.zeros(NP,bool)
    for _ in range(T):
        j=rng.random(NP)<LAM
        if j.any(): s[j]=rng.choice(20,size=int(j.sum()),p=pi)
        mu=np.log1p(rn[s])-(sigma*sigma/2 if conv=="G" else 0.0)
        W=W*np.exp(mu+sigma*rng.standard_normal(NP))-DRAW
        ru|=(W<=0); W=np.maximum(W,1e-9)
    return float(np.exp(np.log(W).mean())), float(np.median(W)), float(np.quantile(W,.10)), int(ru.sum())
if __name__=="__main__":
    for name,pi in (("robust",make_pi_robust(EVG11)),("canonical",w_can/w_can.sum())):
        for conv in ("G","L"):
            ce,p50,p10,ru=run(pi,conv)
            print(f"pi={name:9s} conv={conv}: PathCE={ce:.2f} d={ce-12.35:+.2f} P50={p50:.1f} P10={p10:.1f} ruin={ru}")
