# Compliant-Frontier Note — "Does any compliant US-listed book beat the current ones?" (2026-08-31)

Question tested against the R6 standing battery (R6 deck, C-grade-haircut deck, −200bp level stress, ×4 engine conventions, like-for-like vs P1) plus the session's concentration-stress discipline. Universe: 41 modeled names incl. IAI (added: proxy row, C-grade, ER 0.42%).

## 1. ETF-only books (no single stocks): NO
Six constructions tested — R2/R1b/P2 with CME/MSCI/CBOE→IAI and CEG→NLR, including duration-repaired variants (TLT 3 / GOVZ 3). **All fail P10/P5/ruin floors.** The single-stock sleeve is load-bearing: CME/CBOE/MSCI carry the book's most tail-resilient rows at 0% ER; IAI substitutes a C-haircut proxy row plus 42bp of drag. No compliant all-ETF portfolio found that even matches the floors.

## 2. Letter-of-the-gates beat: exists, fails discipline
Multi-seed search (incl. perturbation restarts) found one book at 60k geo-PathCE **14.53** (+8% over R1b_final): GLDM 15 · DBMF 15 · CIBR **15** · SHLD 14 · XLE **8** · MSCI 5 · XAR 5 · URNM 5 · CEG 5 · NUKZ 4 · CME 3 · CBOE 3 · SPGI 3 (CTA and GRID zeroed). It passes the codified gates but **fails the concentration stress** (CIBR+XLE rows 25%→median: R3AF fails, while P1/R2/R1b all pass) and reintroduces sub-6% scenarios (worst 5.81%, DRB40@6 = 0.023). Same failure mode that retired QTUM — a row-precision bet, not a better portfolio. Not adopted.

## 3. Tempered survivor: R1c (marginal, legitimate)
**R1c_tempered** = GLDM 15 · DBMF 15 · SHLD 14 · CIBR 12 · NUKZ 9 · XLE 6 · MSCI 5 · XAR 5 · URNM 4 · CEG 4 · CME 3 · CBOE 3 · CTA 3 · GRID 2 (24% turnover vs P1). Passes the full battery **and** the concentration stress. 60k: geo 13.81 (HCC 12.69), LS-P5 0.23, worst 5.98% (DRB40@6 = 0.0026, vs 0.0000 for R2/R1b). Verdict: +2.9% over R1b_final — inside the row-uncertainty noise band, at more turnover and a hair-below-6% worst case. A flavor on the flat frontier, not a material beat.

## 4. Ceiling context
Greedy maximization with feasibility only (no floors): geo ≈ 14.9, failing ruin and R3AF, LS-P5 0.16. Compliance costs ~10% of geo-PathCE; that is the price of dominating P1 on every floor under every deck reading. Everything above the current books is thin and consists almost entirely of row-concentration risk.

## 5. Boundary
Judgment applies to the modeled universe (77 deck rows). The ~4,000 unmodeled US-listed ETFs cannot be scored without rows, but the unmodeled classes are structurally disfavored under 40-year log-wealth with fixed draws: leverage amplifies variance drag and ruin; covered-call/buffer funds cap the right tail that funds the draw; single-country and active funds are dominated by factor exposures already represented. Any specific candidate can be added via the established row pipeline.

## Conclusion
No robust compliant beat found. P2 → R2_final → R2d → R1b_final remain the frontier; R1c_tempered is an optional higher-octane flavor (+0.4 geo, discipline-surviving, worst 5.98%). Metrics/weights CSVs updated with R1c.
