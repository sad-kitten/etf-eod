"""Account-aware policy engine (policy_engine2_20260904.py) with the Addendum 31 item-11 patch: per-scenario sigma vector.
Verbatim reconstruction from Bench_2FFL_Code_20260919.py."""
import json, sys
import numpy as np, pandas as pd
import lib_pathce as L
from core import DECKM, ERV, sigma_of, wvec, CONVS

INFL = 0.025
ACCT = dict(ira=384120.14 + 67216.37, brk=199677.93 + 596544.64 + 129629.35, roth=500318.19, hsa=122292.16)
W0 = float(sum(ACCT.values()))
DRAW0 = 0.03 * W0
FLEX = dict(lo=0.80, cut=0.75, hi=1.00)
FED = {
    "single": dict(std=16100, std65=2050, cg0=49450, cg15=545500,
                   br=[(12400, .10), (50400, .12), (105700, .22), (201775, .24), (256225, .32), (640600, .35), (np.inf, .37)]),
    "mfj":    dict(std=32200, std65=3300, cg0=98900, cg15=613700,
                   br=[(24800, .10), (100800, .12), (211400, .22), (403550, .24), (512450, .32), (768700, .35), (np.inf, .37)])}
MO_RATE = 0.047
POL = dict(status="mfj", fill=0.10, ira_years=10, gain0=0.50, y_ord=0.010, y_qual=0.005, age65_year=7, amort_r=0.05,
           rmd_f0=27.1, ss_start=11, ss=45000.0, ss_taxable=0.85, ss_offsets=True)

def bracket_tax(ti, br):
    tax = np.zeros_like(ti); lo = 0.0
    for top, rate in br:
        tax += np.clip(ti - lo, 0.0, top - lo) * rate; lo = top
    return tax

def fed_tax(ordinc, pref, P, std):
    ti_ord = np.maximum(ordinc - std, 0.0)
    ti_pref = np.maximum(pref - np.maximum(std - ordinc, 0.0), 0.0)
    lo = ti_ord; hi = ti_ord + ti_pref
    a15 = np.maximum(0.0, np.minimum(hi, P["cg15"]) - np.maximum(lo, P["cg0"]))
    a20 = np.maximum(0.0, hi - np.maximum(lo, P["cg15"]))
    return bracket_tax(ti_ord, P["br"]) + 0.15 * a15 + 0.20 * a20

def mo_tax(ordinc, qual, std):
    return MO_RATE * np.maximum(ordinc + qual - std, 0.0)

def run(rng, r_net, pi, conv, sigma, pol=POL):
    NP_, T, LAM = L.NP_, L.T, L.LAM
    P = FED[pol["status"]]; fill_top = [top for top, rate in P["br"] if rate == pol["fill"]][0]
    N = pol["ira_years"]; r = pol["amort_r"]
    sigma = np.asarray(sigma, dtype=float)
    vec = sigma.ndim == 1
    s = rng.choice(len(pi), size=NP_, p=pi)
    ira = np.full(NP_, ACCT["ira"]); brk = np.full(NP_, ACCT["brk"]); roth = np.full(NP_, ACCT["roth"] + ACCT["hsa"])
    basis = brk * (1.0 - pol["gain0"])
    ru = np.zeros(NP_, bool); red = np.zeros(NP_, bool); brk_out = np.zeros(NP_, bool); roth_tap = np.zeros(NP_, bool)
    yrs_red = np.zeros(NP_); tax_paid = np.zeros(NP_); tax_yr = np.zeros(T); ira_yr = np.zeros(T)
    Wts = []
    for t in range(1, T + 1):
        j = rng.random(NP_) < L.LAMV[s]
        if j.any(): s[j] = rng.choice(len(pi), size=int(j.sum()), p=L.entry(pi))
        sg = sigma[s] if vec else sigma
        mu = np.log1p(r_net[s]) - (sg * sg / 2 if conv == "G" else 0.0)
        g = np.exp(mu + sg * rng.standard_normal(NP_)) / (1.0 + INFL)
        d_ord = pol["y_ord"] * brk; d_qual = pol["y_qual"] * brk
        rmd = ira / max(pol["rmd_f0"] - (t - 1), 1.0)
        ira = ira * g; roth = roth * g; brk = np.maximum(brk * g - d_ord - d_qual, 0.0)
        basis = basis / (1.0 + INFL)
        W = ira + brk + roth
        red = np.where(W < FLEX["lo"] * W0, True, np.where(W >= FLEX["hi"] * W0, False, red))
        need = DRAW0 * np.where(red, FLEX["cut"], 1.0); yrs_red += red
        std = P["std"] + (P["std65"] if t >= pol["age65_year"] else 0.0)
        ss = pol["ss"] if t >= pol["ss_start"] else 0.0
        ss_ord = pol["ss_taxable"] * ss
        if t < N:
            n_left = N - t + 1
            room = np.maximum(0.0, std + fill_top - d_ord - ss_ord)
            amort = ira * r / (1.0 - (1.0 + r) ** (-n_left))
            need_gross = np.maximum(0.0, need - d_ord - d_qual - (ss if pol["ss_offsets"] else 0.0)) / (1.0 - pol["fill"] - MO_RATE)
            take = np.minimum(ira, np.maximum(np.maximum(np.maximum(room, amort), need_gross), rmd))
        elif t == N:
            take = ira.copy()
        else:
            take = np.zeros(NP_)
        ira = ira - take; ira_yr[t - 1] = take.mean()
        ordinc = d_ord + take
        gf = np.where(brk > 1e-9, np.clip(1.0 - basis / np.maximum(brk, 1e-9), 0.0, 1.0), 0.0)
        base_cash = d_ord + d_qual + take + (ss if pol["ss_offsets"] else 0.0)
        def total_tax(S):
            t_all = fed_tax(ordinc + ss_ord, d_qual + S * gf, P, std) + mo_tax(ordinc, d_qual, std)
            if pol["ss_offsets"] or ss == 0.0: return t_all
            return fed_tax(ordinc, d_qual + S * gf, P, std) + mo_tax(ordinc, d_qual, std)
        S = np.zeros(NP_)
        for _ in range(20):
            S = np.maximum(0.0, need + total_tax(S) - base_cash)
        S = np.minimum(S, brk)
        tax = total_tax(S)
        cash = base_cash + S - tax
        surplus = np.maximum(cash - need, 0.0); short = np.maximum(need - cash, 0.0)
        basis = basis * np.where(brk > 1e-9, 1.0 - S / np.maximum(brk, 1e-9), 1.0); brk = brk - S
        brk = brk + surplus; basis = basis + surplus
        take_r = np.minimum(short, roth); roth = roth - take_r; roth_tap |= take_r > 1.0
        ru |= (short - take_r) > 1.0
        brk_out |= brk <= 1.0
        tax_paid += tax; tax_yr[t - 1] = tax.mean()
        ira = np.maximum(ira, 0.0); brk = np.maximum(brk, 0.0); roth = np.maximum(roth, 0.0); basis = np.minimum(basis, brk)
        if t in (10, 20, 30): Wts.append((ira + brk + roth) / W0)
    W = (ira + brk + roth) / W0
    lg = np.log(np.maximum(W, 1e-9))
    return dict(PathCE=float(np.exp(lg.mean())), P10=float(np.quantile(W, .10)), P5=float(np.quantile(W, .05)),
                ruin=int(ru.sum()), yrs_red=float(yrs_red.mean() / T), tax_paid=float(tax_paid.mean() / W0),
                tax_yr=tax_yr, ira_yr=ira_yr, brk_out=float(brk_out.mean()), roth_tap=float(roth_tap.mean()),
                roth_share=float((roth / W0 / np.maximum(W, 1e-9)).mean()), W=W, W10=Wts[0], W20=Wts[1], W30=Wts[2])

def suite(r_net, sigma, pol=POL):
    out = {}
    for pin, pi in (("robust", L.PI_ROBUST), ("canonical", L.PI_CANON)):
        rng = np.random.default_rng(42)
        sg = sigma[pin] if isinstance(sigma, dict) else sigma
        for conv in ("G", "L"):
            out[(pin, conv)] = run(rng, r_net, pi, conv, sg, pol)
    return out

def geo(mc): return float(np.exp(np.mean([np.log(mc[c]["PathCE"]) for c in CONVS])))
