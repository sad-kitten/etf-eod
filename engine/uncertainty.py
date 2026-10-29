"""E1 — parameter-uncertainty evaluation (Addendum 55b) with the Addendum 56 F-extension (UNC_FACTOR=1).
Verbatim from Factor_Code_20261025.py uncertainty.py section."""
import inspect, os, numpy as np, pandas as pd
import lib_pathce as L, standing as ST, search_B as S, policy_engine2 as E
from core import UNIV, IDX, N, DECKM, ERV, SINGLES, sigma_of

SPEC1 = dict(eq=1.25, them=float(os.environ.get("UNC_THEM", "1.5")), block=float(os.environ.get("UNC_BLOCK", "1.5")),
             gold=float(os.environ.get("UNC_GOLD", "1.5")), trend=float(os.environ.get("UNC_TREND", "2.0")), trend_stress=1.5,
             A=0.75, B=1.25, C=2.0, single=2.5)   # A61: env overrides (defaults = ratified E1 spec)
NON_EQUITY = set("IEF TLT GOVZ LTPZ SCHP VTIP BIL SHY AGG TIP STIP HYG LQD EMB TAIL BTAL GLDM GLD PSLV DBMF KMLM CTA "
                 "CTA_OVERLAY PDBC IBIT".split())
GOLD_U = set(ST.GOLD_FAM) | set(ST.BLOCKS["gold"])
TREND_U = {"DBMF": 1.0, "KMLM": 1.0, "CTA": 1.0, "RSST": 1.0, "DBMFh": 1.0, "RSSTh": 1.0}
SLEEVE_SCALE = {"NTSX": dict(eq=0.9), "GDE": dict(eq=0.9, gold=0.9), "DBMFh": dict(eq=0.0)}
BLOCKS_U = dict(nuclear=ST.BLOCKS["nuclear"], defence=ST.BLOCKS["defence"], cyber=ST.BLOCKS["cyber"],
                semis=ST.BLOCKS["semis"], energy=["XLE", "AMLP"])
FACTORS = ["equity", "thematic", "gold", "trend"] + [f"block:{b}" for b in BLOCKS_U]

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
                CE2=float(np.mean(W ** -1.0) ** -1.0), CE15=float(np.mean(W ** -0.5) ** -2.0), P5=float(np.quantile(W, .05)), P10=float(np.quantile(W, .10)), ruin=int(ru.sum()), pW1=float((W < 1.0).mean()))

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
    return dict(geo=np.exp(mlog("PathCE")), ce3=np.exp(mlog("CE3")), ce5=np.exp(mlog("CE5")), ce2=np.exp(mlog("CE2")), ce15=np.exp(mlog("CE15")), P5=out[("robust", "G")]["P5"],
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
