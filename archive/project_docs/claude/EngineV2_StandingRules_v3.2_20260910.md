# EngineV2 Standing Rules v3.2 (2026-09-10) — v3.1 plus the Addendum 34 items the owner accepted

Supersedes `EngineV2_StandingRules_v3.1_20260910.md`. Changes from v3.1 are marked **[v3.2]**; everything else is restated so this is the single current statement. Code: `standing.py`, `search_v3.py`, `report_v3.py`, `build_r8k.py`, `constraint_assess.py`, `Reaudit_Schedule_20260909.md`. Every rule is enforced in code. Reporting deck **R8k**.

## 1. Objective (unchanged)
Primary criterion: **CE-CAGR at γ = 3** on basis (B), engine-exact, 4 conventions × 60k paths, reported beside γ = 1 (expected wealth) and γ = 5. **Decision rule: a book must be non-dominated on the (γ = 1, γ = 3) pair among admissible, gate-passing books; the owner picks the point.** Every search climbs on the criterion it reports (`OBJ=CE3` or `OBJ=GEO`) from all eight seeds, at the fast evaluator's calibration path count (6,000), under §2 and §3 simultaneously; every endpoint is confirmed engine-exact before it is called admissible.

## 2. Absolute gates — `standing.GATES`, `standing.FAST_P5`, `standing.P5_BAND`
On basis (B), robust-G convention, current deck:
| gate | threshold | basis |
|---|---|---|
| ruin paths | 0 | base deck — tripwire |
| P5 terminal | ≥ 1.00 × corpus, real, engine-exact; ≥ 1.06 on the fast evaluator at search time. **[v3.2] A single-seed exact P5 within ±0.02 of the line is decided by the mean of six exact seeds** (42, 7, 101, 2024, 555, 31337); the exact seed sd is 0.005–0.009, so the band is ≥ 2 sd | tail |
| worst-scenario net CAGR | ≥ 5.25% (5.5 reported as "comfortable") | regime floor, at the deck's resolution |
| **[v3.2] R3AF, re-based** | **≥ 0.05**. Bad states = the deck's **severity-5** scenarios (23% of canonical mass), good = severity 2 (25%), mid = 3–4; reference for tail alpha and bad-state beta = **IVV net of ER**; formula unchanged (0.55 tail alpha + 0.25 convexity + 0.20 (1 − |β|)), canonical weights. 0.05 is what a book scores that beats the equity market by 2 pp in severity-5 states with a beta of 0.8 to it there. The legacy EVG11-quintile form (threshold −0.002) is reported for one more revision, then retired | anti-fragility vs the equity market in the deck's severe states |
| −200 bp deck: ruin | 0 | tripwire |
Reported, not gated: −200 bp P5, HCC P5 and ruin, DRB40@6%, RCEW40, shortfall6, P10, legacy R3AF. Gates are for safety only — a return floor is never a gate.

`FAST_P5` is a calibrated constant: fast-evaluator P5 runs **×1.038** (offset +0.044, sd 0.009 across 28 books on R8k) and fast geo ×1.056 against the exact engine; the threshold is 1.00 + mean offset + 1 sd, rounded **up** to 0.01 (R8k: 1.053 → 1.06, unchanged). Re-derived at every deck revision (`fastcal.py`).

## 3. Standing constraints (unchanged values) — `standing.violations`
| constraint | value |
|---|---|
| feasibility | long-only, ETF ≤ 15%, singles ≤ 5% / 20% aggregate, semis ≤ 12%, defence ≤ 35%, nuclear ≤ 35% weight, ballast set ≥ 20%, PM-equity ≤ 10%, ≤ 14 names, 1% grid, 2% minimum |
| nuclear complex risk share | ≤ 38% of book variance (48-month covariance) |
| any single holding risk share | ≤ 24% |
| any theme block (defence, gold, cyber, semis) | ≤ 35% of variance |
| grade-C weight | ≤ 20% |
| weight in funds with < 60 months of history | ≤ 30% |
| single names (BRKB exempt) | ≤ 3 |
| ballast floor: DBMF + CTA + KMLM | ≥ 9% |
| duplicates | two books are one book if < 3% of the portfolio must move |
**[v3.2] Assessed (Addendum 35 §3):** the nuclear cap is the working constraint (15 unique fails of 78), all within 2.4 pp of the line, on a statistic whose month-resampling sd is 3.4–4.4 pp and which moves 5–8 pp with the covariance window. A lower-noise estimator is **proposed** (Addendum 35 §3), not applied; values unchanged pending the owner's decision.

## 4. Deck and rows — `build_r8k.py`, `Reaudit_Schedule_20260909.md`
Row admission unchanged (beta/geometric audit; alpha zeroed unless t ≥ 2; over-claims halved; under-claims reported as the R8j_sym sensitivity; < 36-month rows grade C unless anchored; premium capped at anchor + 1.0 pp). **[v3.2] AI Buildout Disappointment de-rating is multiplicative on the premium:** hit = max(additive pp, premium over IVV's blended row) — additive floor, premium cap; a name whose thematic premium exceeds the additive hit loses the whole premium and lands at the market in that scenario. Ballast +0.5 pp stays additive. Deck **R8k** (six rows changed in one column: SOXX −2.06, URNM −1.30, SMH −1.06, XSD −0.79, QTUM −0.44, URA −0.29). Every deck revision re-runs gates on all standing books and re-derives `FAST_P5`.

## 5. Method (unchanged) — `standing.sigma_vec`
Regime-conditional volatility as in v3 (bottom-tercile equity months vs the rest; short-history names via sector anchors; ratio shrunk halfway to 1; severity ≥ 4 uses the stress σ; π-weighted variance preserved).

## 6. Reporting set and resolution — `report_v3.py`
As in v3, plus **[v3.2]** the 6-seed P5 mean when invoked, and both R3AF forms. **Noise floor re-measured under regime σ on R8k** (6 seeds, engine-exact, X2 and V3_geoX3c): geo sd 0.006–0.007, γ = 3 CE-CAGR sd 0.011–0.014 pp, P5 sd 0.0045–0.0054. Tie bands stay at 0.03 × geo and 0.02 pp γ = 3 pending the owner's decision on the proposal in Addendum 35 §4 (3-sd bands: 0.02 × geo, 0.04 pp γ = 3, 0.015 × P5).
