# ===== lite.py =====
"""Addendum 57 engine-lite (NOT engine-exact). Deck R8p rows (+ZTS row built here = deck R8q), canonical pi,
48-month realized covariance (prices to 2026-08), 50 y, real, 3% fixed draw (75% when W<0.8 until W>=1), Conv-G/L mean,
persistent-regime Markov scenario paths. Calibrated on Report_T50_R56 (C5_SMH3b, R55_A, R56_A, R56_B); used for DELTAS only.
Deterministic gates are exact: worst scenario (>=5.25), stressed worst (>=3.50, v3.4 haircuts)."""
import glob, json, sys, numpy as np, pandas as pd
A = "/mnt/user-data/attachments/f3a3fc60-1430-546b-b791-1a5b6496ef2c/"
PR = {"CIBR": "project-file-680ff8fa-4e9f-417a-b867-d42e1b2154b7-CIBR.csv", "DBMF": "project-file-87877dbb-e22a-4b81-b05b-e4ed69344aca-DBMF.csv",
      "CME": "project-file-333ec582-8454-4743-a33b-eb3f76f14df3-CME.csv", "CTA": "project-file-237f299f-559f-4c69-afc8-33a99c116de0-CTA.csv"}
TR = "/root/.claude/projects/-home-claude/f3a3fc60-1430-546b-b791-1a5b6496ef2c/tool-results/"
D = pd.read_csv("/home/claude/a57/deck_sub.csv").set_index("Ticker") / 100.0
SC = pd.read_csv("/home/claude/a57/scen.csv"); PI = SC.w.values / SC.w.sum(); SEV = SC.sev.values

# ---- ZTS row (55a method): 50-y two-stage FCF IRR at EV/FCF, g1 for 5 y then 4.5%, exit 20x FCF, minus 1.0 pp skew haircut;
# other scenarios = base + 0.75 x (XLV_s - XLV_base)
def irr(evfcf, g1, g2=0.045, n=50, exit_mult=20.0):
    lo, hi = -0.5, 1.0
    for _ in range(200):
        r = (lo + hi) / 2; f = 1.0; pv = 0.0
        for t in range(1, n + 1):
            f *= (1 + (g1 if t <= 5 else g2)); pv += f / (1 + r) ** t
        pv += exit_mult * f / (1 + r) ** n
        lo, hi = (r, hi) if pv > evfcf else (lo, r)
    return r
ZTS_EVFCF, ZTS_G = float(sys.argv[1]) if len(sys.argv) > 1 else 17.73, float(sys.argv[2]) if len(sys.argv) > 2 else 0.0263
zb = irr(ZTS_EVFCF, ZTS_G) - 0.01
D.loc["ZTS"] = zb + 0.75 * (D.loc["XLV"] - D.loc["XLV", "s1"])
ER = dict(GLDM=.10, SHLD=.50, CIBR=.59, DBMF=.85, NUKZ=.85, BRKB=0, CME=0, FNV=0, MSCI=0, URNM=.75, XLE=.08, XAR=.35, SMH=.35,
          SOXX=.34, KMLM=.90, ITA=.38, GRID=.56, CTA=.76, JEDI=.69, UTHR=0, ZTS=0, VRTX=0, IVV=.03, XLV=.08)
ER = {k: v / 100 for k, v in ER.items()}

# ---- prices -> monthly returns, 48 months to 2026-08
def load(t):
    f = TR + PR[t] if t in PR else glob.glob(A + f"*-{t}.csv")[0]
    x = pd.read_csv(f); x.index = pd.to_datetime(x.Date.str[:10]); p = x["Adj_Close"]
    return p.resample("ME").last().pct_change()
TICK = [t for t in D.index if t not in ("UTHR", "ZTS", "VRTX", "XLV")] + ["XLV", "NLR", "PPA"]
M = pd.DataFrame({t: load(t) for t in TICK}).loc["2022-08":"2026-07"]
for t, a in {"NUKZ": "NLR", "SHLD": "PPA", "JEDI": "PPA", "CTA": "DBMF"}.items(): M[t] = M[a]   # engine ANCHOR rule (<60 months)
rng = np.random.default_rng(57)
iv = M["IVV"]; ivr = iv - iv.mean()
def synth(vol, corr):   # synthetic single-name month series with given vol and corr to IVV
    e = rng.standard_normal(len(M)); e = (e - e.mean()) / e.std()
    b = corr * vol / (iv.std() * np.sqrt(12)); s = np.sqrt(max(vol ** 2 / 12 - (b * iv.std()) ** 2, 1e-8))
    return 0.007 + b * ivr + s * e
M["UTHR"] = synth(0.338, 0.24)   # Addendum 56 postscript: real 120-mo vol 33.8%, corr 0.24
M["VRTX"] = synth(0.274, 0.26)
M["ZTS"] = synth(float(sys.argv[3]) if len(sys.argv) > 3 else 0.28, float(sys.argv[4]) if len(sys.argv) > 4 else 0.40)  # assumption
COV = M.cov().values * 12; COLS = list(M.columns)

GOLD = {"GLDM", "FNV"}; TREND = {"DBMF", "KMLM", "CTA"}
THEM = {"NUKZ", "URNM", "SHLD", "XAR", "ITA", "JEDI", "CIBR", "SMH", "SOXX", "XLE", "GRID"}
def rvec(d, deck=None):
    deck = D if deck is None else deck
    return sum(v * (deck.loc[t].values - ER[t]) for t, v in d.items())
def stressed(d):
    S = D.copy()
    for t in S.index:
        if t in GOLD: S.loc[t] -= 0.015
        if t in TREND: S.loc[t] = S.loc[t] - 0.03 * (SEV >= 4)
        if t in THEM: S.loc[t] -= 0.02
    return rvec(d, S)
def sigma(d):
    w = np.array([d.get(c, 0.0) for c in COLS]); return float(np.sqrt(w @ COV @ w))

NP, T, INFL = 20000, 50, 0.025
def paths(stay, seed=42):
    g = np.random.default_rng(seed); S = np.empty((NP, T), int); S[:, 0] = g.choice(25, NP, p=PI)
    for t in range(1, T):
        new = g.choice(25, NP, p=PI); keep = g.random(NP) < stay; S[:, t] = np.where(keep, S[:, t - 1], new)
    return S, g.standard_normal((NP, T))
def sim(d, stay=0.85, ss=0.0, ratio=1.1, P=None):
    S, Z = P; r = rvec(d); s0 = sigma(d); pst = float(PI @ (SEV >= 4))
    k = 1 / np.sqrt(pst * ratio ** 2 + 1 - pst); sv = s0 * k * np.where(SEV >= 4, ratio, 1.0)
    out = []
    for conv in ("G", "L"):
        mu = np.log1p(r)[S]; sg = sv[S]
        if conv == "G": mu = mu - sg * sg / 2
        X = np.exp(mu + sg * Z) / (1 + INFL); W = np.ones(NP); red = np.zeros(NP, bool)
        for t in range(T):
            W = W * X[:, t]; red = np.where(W < 0.8, True, np.where(W >= 1, False, red))
            W = np.maximum(W - max(0, 1) * np.maximum(0.03 * np.where(red, .75, 1) - (ss if t >= 10 else 0), 0), 1e-9)
        out.append(W)
    W = np.concatenate(out)
    geo = float(np.exp(np.log(W).mean())); ce3 = float(np.mean(np.maximum(W, 0.10) ** -2.0) ** -0.5)   # floor 0.10x: lite ruin tail not engine-faithful
    return dict(geo=geo, g3=(ce3 ** (1 / T) - 1) * 100, P5=float(np.quantile(out[0], .05)), pW1=float((W < 1).mean()) * 100, dep=float((W < 0.01).mean()) * 100)
def det(d):
    r = rvec(d); s = stressed(d)
    return dict(pimean=float(PI @ r) * 100, worst=float(r.min()) * 100, worst_s=int(np.argmin(r)) + 1, sworst=float(s.min()) * 100,
                sev5=float((PI * r)[SEV >= 5].sum() / PI[SEV >= 5].sum()) * 100, sig=sigma(d) * 100, ER=sum(v * ER[t] for t, v in d.items()) * 100)
if __name__ == "__main__":
    print("ZTS row: base %.2f pi-mean %.2f min %.2f (EV/FCF %.2f, g %.2f%%)" % (zb * 100, PI @ D.loc["ZTS"].values * 100, D.loc["ZTS"].min() * 100, ZTS_EVFCF, ZTS_G * 100))
    print("UTHR pi-mean %.2f  IVV %.2f  ITA %.2f  XAR %.2f  SHLD %.2f  SMH=SOXX %.2f" % tuple(PI @ D.loc[t].values * 100 for t in ("UTHR", "IVV", "ITA", "XAR", "SHLD", "SMH")))
    print("rows SMH vs SOXX identical:", bool((D.loc["SMH"] == D.loc["SOXX"]).all()))
    print("48m vol %:", {t: round(float(np.sqrt(COV[COLS.index(t), COLS.index(t)])) * 100, 1) for t in ("SMH", "SOXX", "ITA", "XAR", "SHLD", "GLDM", "FNV", "DBMF", "KMLM")})
    print("corr SMH-SOXX %.3f, ITA-XAR %.3f, ITA-SHLD %.3f" % (M.SMH.corr(M.SOXX), M.ITA.corr(M.XAR), M.ITA.corr(M.SHLD)))

# ===== books.py =====
B = {
 "C5_SMH3b": dict(GLDM=.15,SHLD=.15,CIBR=.15,DBMF=.15,NUKZ=.08,BRKB=.05,CME=.05,FNV=.05,MSCI=.05,URNM=.05,XLE=.03,XAR=.02,SMH=.02),
 "R55_A": dict(GLDM=.20,DBMF=.20,SHLD=.15,CIBR=.10,SMH=.05,URNM=.05,CME=.05,MSCI=.05,FNV=.05,BRKB=.05,KMLM=.03,XAR=.02),
 "R56_A": dict(GLDM=.20,DBMF=.20,SHLD=.15,CIBR=.10,SMH=.05,URNM=.05,CME=.05,MSCI=.05,UTHR=.05,BRKB=.05,KMLM=.03,XAR=.02),
 "R56_B": dict(GLDM=.20,DBMF=.20,SHLD=.15,CIBR=.10,SMH=.05,URNM=.05,CME=.05,UTHR=.05,VRTX=.05,BRKB=.05,KMLM=.03,XAR=.02),
 "K7": dict(GLDM=.20,SHLD=.20,CIBR=.20,DBMF=.20,URNM=.10,MSCI=.05,FNV=.05),
 "M5": dict(GLDM=.30,SHLD=.23,CIBR=.20,DBMF=.17,URNM=.10),
 "P1": dict(GLDM=.15,SHLD=.15,DBMF=.15,NUKZ=.15,CIBR=.10,GRID=.10,CTA=.05,CME=.05,JEDI=.05,MSCI=.05),
}
TGT = {"C5_SMH3b": (7.60, 2.43, 1.29), "R55_A": (7.54, 2.70, 1.53), "R56_A": (7.60, 2.83, 1.64), "R56_B": (7.02, 2.79, 1.66)}

# ===== run57.py =====
import numpy as np, pandas as pd, lite as L
from books import B
P = L.paths(0.8)
def mv(d, src, dst, x):
    d = dict(d); x = min(x, d.get(src, 0)); d[src] = d.get(src, 0) - x; d[dst] = d.get(dst, 0) + x
    return {k: round(v, 4) for k, v in d.items() if v > 1e-9}
SING = {"CME", "MSCI", "FNV", "UTHR", "ZTS", "VRTX"}
def variants(nm, d):
    V = {nm: d}
    g = "FNV" if "FNV" in d else "GLDM"
    V[nm + " +UTHR(" + g + ")"] = mv(d, g, "UTHR", .05) if "UTHR" not in d else None
    V[nm + " GLDM-5>KMLM"] = mv(d, "GLDM", "KMLM", .05)
    V[nm + " GLDM-5>DBMF"] = mv(d, "GLDM", "DBMF", .05)
    if "XAR" in d: V[nm + " XAR>ITA"] = mv(d, "XAR", "ITA", d["XAR"])
    V[nm + " +ITA5(SHLD)"] = mv(d, "SHLD", "ITA", .05)
    V[nm + " +ITA5(CIBR)"] = mv(d, "CIBR", "ITA", .05)
    if "SMH" in d: V[nm + " SMH>SOXX"] = mv(d, "SMH", "SOXX", d["SMH"])
    V[nm + " +ZTS(" + g + ")"] = mv(d, g, "ZTS", .05)
    if "MSCI" in d: V[nm + " +ZTS(MSCI)"] = mv(d, "MSCI", "ZTS", .05)
    if "UTHR" in d: V[nm + " +ZTS(GLDM)"] = mv(d, "GLDM", "ZTS", .05)
    return {k: v for k, v in V.items() if v}
rows = []
for nm in ("C5_SMH3b", "K7", "M5", "R56_A", "P1"):
    for k, d in variants(nm, B[nm]).items():
        s = L.sim(d, ss=0.015, P=P); t = L.det(d)
        ns = sum(1 for x in d if x in SING)
        gl = d.get("GLDM", 0) + d.get("FNV", 0)
        fails = []
        if t["worst"] < 5.25: fails.append(f"worst {t['worst']:.2f}")
        if t["sworst"] < 3.50: fails.append(f"stressed {t['sworst']:.2f}")
        if ns > 3: fails.append(f"{ns} singles")
        rows.append(dict(book=k, geo=round(s["geo"], 2), g3=round(s["g3"], 2), P5=round(s["P5"], 2), pW1=round(s["pW1"], 1),
                         pimean=round(t["pimean"], 2), worst=round(t["worst"], 2), sworst=round(t["sworst"], 2), sev5=round(t["sev5"], 2),
                         sig=round(t["sig"], 2), ER=round(t["ER"], 3), gold=round(gl * 100), gates="PASS" if not fails else "; ".join(fails),
                         w=" ".join(f"{a}{round(b*100)}" for a, b in sorted(d.items(), key=lambda z: -z[1]))))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("Screen_A57_lite.csv", index=False)

# ===== run57b.py =====
import sys, lite as L
from books import B
from run57 import mv
P = L.paths(0.8)
R, K, M, C = B["R56_A"], B["K7"], B["M5"], B["C5_SMH3b"]
X = {
 "R56_A": R,
 "R56_A GLDM-2>KMLM": mv(R, "GLDM", "KMLM", .02), "R56_A GLDM-3>KMLM": mv(R, "GLDM", "KMLM", .03),
 "R56_A GLDM-2>DBMF(cap)>KMLM": mv(mv(R, "GLDM", "KMLM", .02), "XAR", "SHLD", 0),
 "R56_A +ZTS(CME)": mv(R, "CME", "ZTS", .05),
 "C5 +UTHR(FNV) SMH-1>SHLD": mv(mv(C, "FNV", "UTHR", .05), "SMH", "SHLD", .01),
 "C5 +UTHR(FNV) XLE-1>SHLD": mv(mv(C, "FNV", "UTHR", .05), "XLE", "SHLD", .01),
 "K7 +UTHR(FNV) +ZTS(MSCI)": mv(mv(K, "FNV", "UTHR", .05), "MSCI", "ZTS", .05),
 "K7 +UTHR(FNV) GLDM-5>DBMF": mv(mv(K, "FNV", "UTHR", .05), "GLDM", "DBMF", .05),
 "M5 GLDM-5>DBMF": mv(M, "GLDM", "DBMF", .05),
 "M5 GLDM-10>DBMF": mv(M, "GLDM", "DBMF", .10),
}
for k, d in X.items():
    s = L.sim(d, ss=0.015, P=P); t = L.det(d)
    print(f"{k:30s} geo {s['geo']:5.2f} g3 {s['g3']:5.2f} P5 {s['P5']:.2f} worst {t['worst']:.2f} sw {t['sworst']:.2f} sig {t['sig']:5.2f} max {max(d.values())*100:.0f} | " + " ".join(f"{a}{round(b*100)}" for a, b in sorted(d.items(), key=lambda z: -z[1])), flush=True)

# ===== deck_sub.csv =====
Ticker,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25
CIBR,10.1716,11.3016,7.6716,8.3816,5.5116,13.8716,10.2616,8.6316,6.3516,10.2016,6.4016,2.6416,5.0916,10.4716,9.2816,9.4316,7.5816,5.8916,9.4016,8.5016,5.6946,8.3911,5.9046,12.8916,7.4666
GRID,11.5069,13.1069,8.8869,8.6069,3.2369,12.2869,13.0669,7.3269,0.3969,8.6469,4.6469,-0.3931,3.0969,11.5369,6.8369,12.1469,8.3469,4.3369,6.6369,7.1269,3.6319,7.6449,-0.7101,11.9749,3.9789
NUKZ,10.279,10.349,11.789,9.119,6.569,9.959,11.759,8.739,5.539,9.229,7.079,5.399,6.339,9.779,8.109,11.209,8.969,11.089,8.519,9.279,6.176,10.0055,6.079,10.087,5.6505
SHLD,9.1104,8.6904,9.1904,7.8604,7.2904,7.7104,8.3004,10.0704,8.4404,9.1104,7.7204,10.9104,7.4704,11.1304,6.7404,7.8104,7.5604,8.5604,9.9804,10.0504,7.4554,9.7354,11.6994,8.2704,8.2124
SOXX,16.1804,17.2537,11.3944,12.9367,6.9719,22.6359,17.5297,10.4201,3.2073,14.5213,7.8611,-6.3523,5.9792,18.8866,14.7662,15.1117,12.7317,6.0596,11.4809,10.1831,7.0401,10.8666,-2.002,20.0537,7.2994
SMH,16.1804,17.2537,11.3944,12.9367,6.9719,22.6359,17.5297,10.4201,3.2073,14.5213,7.8611,-6.3523,5.9792,18.8866,14.7662,15.1117,12.7317,6.0596,11.4809,10.1831,7.0401,10.8666,-2.002,20.0537,7.2994
XAR,12.4535,14.0935,9.8435,9.5735,4.8835,11.5035,10.7735,10.2035,4.3135,10.4435,6.4935,5.6235,4.8935,14.4735,7.6135,10.0135,8.1935,6.5235,9.5635,9.1535,5.3685,9.477,7.4105,11.8835,8.277
XLE,11.9333,14.4933,13.8033,10.2233,0.7533,9.8933,11.6233,8.5133,0.7833,8.3533,3.1633,1.4233,1.9533,10.5933,4.0433,5.8833,8.2533,14.2733,6.0433,9.4933,0.7163,11.0618,3.8013,10.7093,5.6238
CTA,4.4881,4.924,7.80171246562122,6.252873683654859,5.232337363927723,6.150939085739775,6.84849055988822,6.743144395346436,5.763679534936468,6.952232549648276,5.435399656041712,5.394190974896198,5.623614655987407,6.587739635648708,4.946443683763467,6.337662316815943,6.049361226267624,7.72153406447412,6.973654527466527,8.116636885068477,7.011401528187127,7.342935760548656,7.266765759404487,3.117,6.64583009771416
CME,8.5806,9.9306,9.7306,7.8706,6.4606,8.7306,8.7506,8.4006,6.2406,8.3406,6.1806,5.5406,6.0906,8.6306,7.0106,8.0906,7.4406,8.4806,7.5706,9.0506,6.3026,9.2426,6.3386,7.9206,7.6931
KMLM,4.28,4.924,7.830854062794902,5.301081535455408,6.076859368064243,3.742791336201511,5.310143088150998,6.067066505965753,5.182745895733372,5.696218279599168,5.272878816504872,6.246512102645682,5.8978289775056005,4.53740245368707,3.942280783032862,4.750125274026983,4.922235562564207,8.49728416640615,5.929178164820211,8.337786277671924,6.836857689747732,7.653493469783253,7.394666440913973,3.117,6.219309295695352
DBMF,4.4881,4.924,9.576999248511616,7.412704140005074,7.432638919156336,6.361680638694933,7.43961167805331,7.75815086846346,6.0166869143349375,7.494649203937744,6.553154140957722,6.921980280198673,7.197062287697511,6.787672479019195,5.9908686371426185,6.6452779528309165,6.572315393879592,9.740870280108425,7.52737816371697,10.13469464842798,8.582190122149454,9.1050655036346,8.42738048677274,3.117,7.995123499404846
GLDM,6.9,6.2,10.45,9.07,8.66,6.77,7.43,8.53,7.35,7.95,7.51,7.27,7.91,6.81,7.19,7.39,7.52,9.27,8.36,10.43,8.665,12.523,9.194,5.822,8.8125
IVV,9.7944,10.4944,9.2944,8.7144,4.9544,10.8044,9.7044,8.0044,3.3844,8.8244,5.7944,1.9744,4.5644,9.7444,7.4144,8.8844,7.7844,6.6444,7.3044,8.0844,5.1284,8.3519,3.3314,10.9004,7.2994
ITA,11.5101,12.8883,9.1436,9.0173,3.3174,12.1417,10.4419,8.7599,2.7632,9.6326,5.1879,2.0615,3.5233,13.0243,7.6106,9.4437,7.8243,5.9141,8.2709,8.535,3.9197,8.6952,5.0425,11.8891,7.0588
MSCI,13.8649,15.3975,11.3834,10.8588,5.4816,15.7445,13.4428,10.2564,4.462,12.1923,7.8913,2.7872,5.9316,13.7372,10.5303,12.3753,10.4355,8.6856,10.1953,9.4862,6.2945,10.292,4.4693,14.2427,9.301
URNM,15.2751,16.7231,14.9613,13.078,8.0124,14.7967,15.7009,12.5503,6.5097,13.7418,9.6884,6.1744,8.2449,15.2792,10.5739,15.0047,12.5749,13.2275,11.4203,12.6152,7.3742,14.2286,7.8732,14.9881,7.2994
XLV,8.617156120689655,9.098356120689655,8.317756120689655,7.7957561206896555,4.497556120689655,9.805656120689655,8.608156120689655,7.154456120689655,2.788156120689655,7.8056561206896555,5.0899561206896555,1.363956120689655,3.932056120689655,8.500656120689655,6.661156120689655,7.790856120689655,6.9268561206896555,5.700456120689656,6.420656120689655,7.246456120689655,4.5621561206896555,7.456256120689655,2.658556120689655,9.330256120689656,6.512456120689655
JEDI,8.4909,9.7467,7.8664,6.9863,2.5718,8.4129,8.2307,6.4413,2.1256,7.1054,3.5817,1.3928,2.6781,8.96,5.293,7.2121,5.9483,5.8347,6.0264,6.8823,2.896,7.0556,4.501,8.4441,3.9377
FNV,7.6364,6.8664,12.4564,10.3504,8.6484,7.7574,8.3514,9.3814,6.3434,8.8154,7.2904,5.8084,7.4814,7.4954,7.3284,8.0494,7.9014,10.0094,8.9334,12.0654,8.2076,15.0758,8.9091,6.309,8.8654
BRKB,9.5586,10.2796,10.1406,9.0776,6.9866,9.5656,9.4336,7.6576,3.3776,8.5176,5.4096,1.9216,5.7786,8.9856,7.0126,8.6066,7.7166,6.1856,6.9106,7.6826,7.6219,8.3885,3.2112,9.0628,8.3599
UTHR,9.94,10.3009,9.71545,9.32395,6.850299999999999,10.831375,9.93325,8.842975,5.568249999999999,9.331375,7.2946,4.5001,6.426174999999999,9.852625,8.472999999999999,9.320275,8.672274999999999,7.752475,8.292625,8.911975,6.89875,9.069325,5.471049999999999,10.474825,8.361474999999999
VRTX,7.779999999999999,8.140899999999998,7.555449999999999,7.16395,4.690299999999999,8.671375,7.773249999999999,6.682974999999999,3.408249999999999,7.171374999999999,5.1346,2.3401,4.266174999999999,7.692625,6.312999999999999,7.160275,6.512275,5.592474999999999,6.132624999999999,6.751974999999999,4.73875,6.909324999999999,3.311049999999999,8.314824999999999,6.201474999999999

# ===== scen.csv =====
id,w,sev
1,17.0,2
2,10.0,2
3,7.0,4
4,4.0,4
5,3.0,4
6,2.5,3
7,1.5,3
8,4.0,4
9,4.0,5
10,6.0,3
11,4.0,4
12,4.0,5
13,3.0,4
14,5.0,3
15,3.0,4
16,1.5,3
17,1.5,4
18,6.0,5
19,5.0,4
20,6.0,5
21,4.0,4
22,4.0,5
23,3.0,5
24,1.5,2
25,5.5,3

