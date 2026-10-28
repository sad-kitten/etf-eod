# Decision record — default book moves from X2 to C5_bal (2026-09-10)

Owner asked whether C5_bal should be the default; the author's answer was yes, and per the owner's standing instruction (default to the author's recommendations) the decision is recorded here. Rules unchanged (v3.3, deck R8k); this changes only which frontier point the author names as the default.

**C5_bal** — CIBR 15 · DBMF 15 · GLDM 15 · SHLD 15 · NUKZ 9 · BRKB 5 · CME 5 · FNV 5 · MSCI 5 · XLE 5 · URNM 4 · HACK 2 (12 holdings, 3 singles). From X2 it is a 4% move: CIBR 13 → 15, HACK 0 → 2, NUKZ 10 → 9, URNM 5 → 4, XAR 2 → 0.

**Why (R8k, v3.3, engine-exact, X2 → C5_bal):** γ = 3 CE-CAGR 2.55 → **2.64** (+0.09, twice the tie band — the primary criterion prefers it); γ = 5 0.90 → 1.12; P5 terminal 1.12 → **1.17** (+4.5% in the bad decile); −200 bp P5 0.56 → 0.59; C-haircut P5 1.08 → 1.14; nuclear risk share 29 → 25; grade-C weight 17 → 13; vol48 11.4 → 10.9; lifetime tax 0.118 → 0.114. **Price:** expected wealth 5.347 → 5.253 × (−0.09 ×, −1.8% of expected terminal wealth, −0.05 pp/yr); worst scenario 5.78 → 5.64 (both comfortable); shortfall6 1.2 → 1.8 bp; XAR leaves the book. For a corpus being drawn, 4.5% more in the bad decile for 1.8% less in expectation is the direction the draw itself argues for.

**Robustness of the choice:** C5_bal has been on every frontier since Addendum 23 (2026-09-05) — decks R8h, R8i, R8j, R8k; filter stack, SACE, v3, v3.1, v3.2, v3.3 — and was the only book to survive the full filter stack in Addendum 26. Its γ = 3 edge over X2 (0.08–0.09) held through the AI-bust scenario, the NUKZ cap and the multiplicative de-rating, and it widens under the stress views (C-haircut, −200 bp), because less of the book rests on grade-C rows.

**Standing:** default **C5_bal**; X2 remains the wealth-leaning alternative one step up the frontier (+0.09 ×, −0.09 γ = 3); the recommended set is unchanged — V3_geoX3c · V31_C5g · X2 · C5_bal · V3_reliab.
