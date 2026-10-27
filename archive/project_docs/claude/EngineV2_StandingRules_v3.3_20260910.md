# EngineV2 Standing Rules v3.3 (2026-09-10) — v3.2 plus the averaged risk-share estimator and 3-sd tie bands

Supersedes `EngineV2_StandingRules_v3.2_20260910.md`. Changes from v3.2 are marked **[v3.3]**; everything else is restated so this is the single current statement. Code: `standing.py`, `search_v3.py`, `report_v3.py` / `report_v33.py`, `build_r8k.py`, `Reaudit_Schedule_20260909.md`. Every rule is enforced in code. Reporting deck **R8k**.

## 1. Objective (unchanged)
Primary criterion: **CE-CAGR at γ = 3** on basis (B), engine-exact, 4 conventions × 60k paths, reported beside γ = 1 (expected wealth) and γ = 5. **Decision rule: a book must be non-dominated on the (γ = 1, γ = 3) pair among admissible, gate-passing books; the owner picks the point.** Every search climbs on the criterion it reports (`OBJ=CE3` or `OBJ=GEO`) from all eight seeds at the fast evaluator's calibration path count (6,000), under §2 and §3 simultaneously; every endpoint is confirmed engine-exact before it is called admissible.

## 2. Absolute gates (unchanged from v3.2) — `standing.GATES`, `FAST_P5`, `P5_BAND`
| gate | threshold | basis |
|---|---|---|
| ruin paths | 0 | tripwire |
| P5 terminal | ≥ 1.00 × corpus, real, engine-exact; ≥ 1.06 on the fast evaluator at search time; a single-seed exact P5 within ±0.02 of the line is decided by the 6-seed exact mean | tail |
| worst-scenario net CAGR | ≥ 5.25% (5.5 reported as "comfortable") | regime floor |
| R3AF (re-based) | ≥ 0.05 — bad states = deck severity 5, good = severity 2, reference IVV net; 0.55 tail alpha + 0.25 convexity + 0.20 (1 − |β|) | anti-fragility vs the equity market in severe states |
| −200 bp deck: ruin | 0 | tripwire |
Reported, not gated: −200 bp P5, HCC P5 and ruin, DRB40@6%, RCEW40, shortfall6, P10, legacy R3AF (one more revision). `FAST_P5` = 1.00 + mean fast-vs-exact P5 offset + 1 sd, rounded up to 0.01 (R8k: 1.053 → 1.06); re-derived at every deck revision.

## 3. Standing constraints — `standing.violations`
| constraint | value |
|---|---|
| feasibility | long-only, ETF ≤ 15%, singles ≤ 5% / 20% aggregate, semis ≤ 12%, defence ≤ 35%, nuclear ≤ 35% weight, ballast set ≥ 20%, PM-equity ≤ 10%, ≤ 14 names, 1% grid, 2% minimum |
| nuclear complex risk share | ≤ 38% of book variance |
| any single holding risk share | ≤ 24% |
| any theme block (defence, gold, cyber, semis) | ≤ 35% of variance |
| grade-C weight | ≤ 20% |
| weight in funds with < 60 months of history | ≤ 30% |
| single names (BRKB exempt) | ≤ 3 |
| ballast floor: DBMF + CTA + KMLM | ≥ 9% (a book-calibrated number — Addendum 35 §3) |
| duplicates | two books are one book if < 3% of the portfolio must move |
**[v3.3] Estimator.** Every risk share is the **average of the marginal risk contributions** from the standing 48-month covariance and from a 120-month panel in which short-history names are represented by their sector anchors (the regime-σ panel logic), caps unchanged. Month-resampling sd ≈ 2.5 pp instead of 3.4–4.4, and one regime no longer sets the number. The 48-month statistic is reported beside it for one revision. Effect: R7a and its neighbourhood become constraint-admissible (R7a 36.2 vs 39.9); all are dominated; the frontier is unchanged.

## 4. Deck and rows (unchanged) — `build_r8k.py`, `Reaudit_Schedule_20260909.md`
Row admission as in v3; AI-bust de-rating multiplicative on the premium (hit = max(additive pp, premium over IVV); ballast +0.5 additive). Deck R8k. Every deck revision re-runs gates on all standing books, re-derives `FAST_P5`, and **[v3.3] re-measures the noise floor** that sets §6's bands.

## 5. Method (unchanged) — `standing.sigma_vec`
Regime-conditional volatility as in v3.

## 6. Reporting set and resolution — `report_v3.py`
As in v3.2. **[v3.3] Tie bands are 3 sd of the regime-σ noise floor: 0.02 × on geo, 0.04 pp/yr on γ = 3 CE-CAGR, 0.015 × on P5** (floor on R8k: geo sd 0.006–0.007, γ = 3 sd 0.011–0.014 pp, P5 sd ≈ 0.005). A non-dominated book inside the γ = 3 band of a higher-wealth non-dominated book is reported as that book's tie-band neighbour: a legitimate choice on structure, not a separate frontier point. Inside a band, choose on cost and structure, never on the number.
