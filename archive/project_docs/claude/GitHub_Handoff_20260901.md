# GitHub handoff — how to give a Claude session push access to the three repos (2026-09-01)

## Why this session can't do it itself
Repositories are bound to a cloud session **when the session is created** (the repo picked on the New task screen at claude.ai/code). This session was started from the mobile app against the *investing* Project with no repository selected, so its git proxy has an empty authorized set — it can read public `etf-eod` but cannot push, and cannot see the private repos. There is no in-session tool here to attach a repository after the fact.

## Do this (about two minutes, from a browser)
1. **Connect GitHub once** (skip if done): open https://claude.ai/code → if prompted, authorize the Claude GitHub App for the `sad-kitten` account; the session can then reach any repo that account can see. (Installing the App *on* a repo is only needed for PR auto-fix.)
2. **Start a new task bound to the repo**: claude.ai/code → **New task** → repository **`sad-kitten/etf-eod`** → paste the message in the box below. Do the same later with **`sad-kitten/investment-lab`** (and `sovereign_portfolio_app`) — one repo per task is the documented model.
3. That's it. The new session clones with credentials injected by the proxy, applies the prepared patch, pushes, opens the PR, and the Action refreshes prices automatically.

## Message to paste into the etf-eod task
```
Apply the patch stored in the investing Project as
claude/etf-eod_patch_0001_tickers_validation_20260901.patch (git am), push it as branch
claude/tickers-file-validation, open a PR against main titled
"Refresh flow: tickers.txt-driven universe, validation, monthly-returns panel", then trigger the
refresh-prices workflow (workflow_dispatch) and report pull_status.csv / validation.csv once it
finishes. Read claude/EngineV2_Addendum5_GitHub_Audit_20260901.md for context.
```

## Step 5 — investment-lab (2026-09-03): the package is prepared and verified; the task only applies it
`claude/investment-lab_patch_0001_enginev2_20260903.patch` (261 KB, one commit, adds `enginev2/` — 26 files, nothing
modified) applies with `git am` onto any tree that has no `enginev2/` directory. Verified here: applies onto a non-empty
repo; `python enginev2/reproduce.py --check` reproduces `reference/R8_Books_Metrics_20260903.csv` to the digit in ~1 min.

**Transfer channel (learned 2026-09-03):** the investment-lab task (session 01TMvdGR…) cannot read Project docs; it can read
Google Drive, Gmail and its repo clones (it has all three repos). The patch was therefore delivered to the user as a file
card to be saved into Drive → `PathCE` folder. Verify before applying: `sha256sum` must equal
`eed259824e6f98902594011c99606fbc233bde3dadb1f753552834e78c68ba5c` (1,829 lines, 260,722 bytes).
investment-lab layout (from that task): flat Python 3.10+ project, `main.py` → `src/investment_lab/` (analysis, reporting,
pipeline, config, knowledge, feedback), `data/input`, `reports/{current,archive}`, `state/*.json`, `bok`, `principles`, `docs`,
`scripts`; setuptools src layout, no tests, no CI; other branch `improve/resilience-and-run-feedback`. `enginev2/` sits beside `src/`.

Message to paste into the **existing investment-lab task** once the patch is in Drive:
```
The patch is now in Google Drive, folder PathCE, file investment-lab_patch_0001_enginev2_20260903.patch. Download it,
confirm sha256 eed259824e6f98902594011c99606fbc233bde3dadb1f753552834e78c68ba5c (1,829 lines, 260,722 bytes) — if it
differs, stop and report — then apply it verbatim with git am on branch claude/enginev2-package, run
pip install -r enginev2/requirements.txt and python enginev2/reproduce.py --check (must print "reference reproduced to
the digit"), push, and open the PR titled "Add enginev2: versioned PathCE-first portfolio engine with reproduce.py",
quoting the --check output.
```

**Step 5 done (2026-09-03):** PR sad-kitten/investment-lab#3 **merged** into main (merge commit 4bad722; enginev2 commit e943b11) — opened on `claude/enginev2-package` — sha256/bytes/lines matched,
`git am` clean (26 files under `enginev2/`, nothing else touched), `reproduce.py --check`: all 11 books EXACT, exit 0
(output quoted in the PR body). The repo has no CI, so merging is a plain review decision.
Follow-up for the wiring PR: running `python enginev2/reproduce.py` from the repo root writes `metrics.csv` at the root
(the package .gitignore only covers `enginev2/metrics.csv`) — make `--out` default to a path inside `enginev2/` (or
`reports/`) when wiring it into the pipeline. That session (01TMvdGR…) has clones of all three repos, so it is also the
cheapest place to get the `sovereign_portfolio_app` layout description for Step 6.

## Step 6 — sovereign_portfolio_app (surveyed 2026-09-03 by the same task; nothing changed)
Windows-first Rust CLI, crate `equities-app` 0.50.0: deterministic, human-gated research console for the CritMin/PathCE
workflow — ranks candidates by CritMin_PathCE, then RCEW40 (net of ER and 3% draw), DRB40, R3AF, downside floors; optional
OpenAI adversarial review / frontier governance layered on local calculations that stay authoritative. Flat `src/` (15 files;
`main.rs` ~4,600 lines of clap subcommands; `db.rs` SQLite; `engine.rs`/`rust_engine.rs` EvaluationEngine boundary + native
proxy metrics; `canonical.rs` + large `canonical_reference.rs` = frozen metric contracts and reference data; `research.rs`
champion-challenger loop; `python_bridge.rs` shells out to nine scripts under `python/` for *authoritative* reference scoring
and stress tournaments "pending Rust parity"; `ai.rs`, `ai_counsel.rs`, `frontier_model.rs`, `adversary.rs`; adversary-report
JSON schema under `schemas/`). Inputs: portfolio JSON {id, ticker→weight} (fractions or percents, normalized), scenarios CSV
(id, name, weight, definition); outputs to a SQLite workspace outside the repo (`%USERPROFILE%\Projects\sovereign_portfolio_workspace`).
Repo carries sample inputs + one R3AF parity fixture; no price panels, no deck files. Nine inline `#[test]`s, no `tests/`, no CI.

Plan (gated on the parity survey below): the campaign moved past what the app freezes — 24-scenario R8 deck with grades and
haircut deck, DRB40 hurdle 6% (ratified), −200bp like-for-like gate, engine-exact MC conventions, flexible-draw rule, books
R3b/R3dK. Because the app already treats `python/` as the authoritative scorer, the low-risk integration is an enginev2 adapter
in `python/` (app portfolio JSON in → enginev2 metrics in the app's JSON shape out; `--enginev2-root` pointing at the
investment-lab checkout; deck/ER/panel from there), a version bump with a parity fixture, and canonical_reference updated only
where the Project ratified a change. First step is read-only:

Message to paste into the existing task (it has all three clones):
```
In sovereign_portfolio_app, change nothing, and report: (1) the nine python/ scripts, one line each — what they compute and
which reference data they embed or read (scenario deck version and count, ticker rows, DRB40 hurdle, ER table, draw rate);
(2) the implemented definitions of CritMin_PathCE, RCEW40, DRB40 and R3AF (formula + parameters) and the deck stamp in
canonical_reference.rs; (3) score P1, R3b and R3dK from investment-lab/enginev2/books.json through the app's authoritative
Python reference scoring and print its metrics beside enginev2/reference/R8_Books_Metrics_20260903.csv for the same books,
with the differences.
```

**Parity survey done (2026-09-03, Addendum 7):** the app's authoritative scorer cannot score P1/R3b/R3dK (its 73-row 2026-07
decks lack JEDI, BRKB, FNV); its engine differs from EngineV2 (hold-one-scenario, 3-vector log blend, median drift, no ER
table, DRB hurdle 5%) but ranks every book identically; net-of-ER it is consistent (12.79 vs 12.35) with the 2026-08-27
store target that the EngineV2 rebuild could not reproduce. Step 6 = generated support root (R8cand rows net of ER, 102
tickers + IVV, scenarios_24, Canonical20 stress formula, etf-eod archive, MANIFEST, ratification JSON pending) + generator
script + parity fixture; DRB 6% as a separate governance PR. Message for the task, if approved:
```
In sovereign_portfolio_app, on branch claude/enginev2-support-root: add python/enginev2_support_root.py that builds a
CritMin24-layout support root named EngineV2_R8cand_24_Support_2026-09-03 from an enginev2 checkout (investment-lab
PR #3): matrix = enginev2/data/decks/Deck_Recal_R8cand_20260903.csv rows MINUS each ticker's expense ratio
(lib_pathce.ER merged with ER_R8.json), all 102 universe tickers plus IVV; scenarios = scenarios_24_R6_20260831.csv
(id, name, weight, severity, definition); stress/blend weights by the Canonical20 formula, not the CritMin24 precomputed
column; price archive zipped from etf-eod prices/ (main); MANIFEST with sha256s; a ratification JSON in the
CM24_RATIFICATION schema with status pending. If root validation requires registering the root in canonical_reference.rs,
add it as a minimal versioned entry (0.50.0 -> 0.51.0) without altering existing entries or any frozen contract (DRB hurdle
stays 5%). Verify: the app's doctor accepts the root, and canonical_reference_score.py scores P1, R3b, R3dK from it
within 0.1 of the net columns in App_Parity_20260903.csv (I will put that file in Drive/PathCE). Add that check as a parity
fixture. Open a PR quoting the doctor and parity output. Change nothing else.
```

Original message (for a **new** task, if one is ever needed instead):
```
Read the investing Project doc claude/investment-lab_patch_0001_enginev2_20260903.patch and apply it verbatim with
git am on a branch claude/enginev2-package (do not re-derive or restructure anything; if the repo already has an
enginev2/ directory, stop and report). Then: pip install -r enginev2/requirements.txt; python enginev2/reproduce.py
--check — it must print "reference reproduced to the digit". Push and open a PR against the default branch titled
"Add enginev2: versioned PathCE-first portfolio engine with reproduce.py", quoting the --check output in the PR body.
Report the repo's existing layout in one paragraph so the package can be wired into it later.
```

## No-Claude fallback (from any machine with git)
```
git clone https://github.com/sad-kitten/etf-eod && cd etf-eod
git am <path-to>/etf-eod_patch_0001_tickers_validation_20260901.patch
git push -u origin claude/tickers-file-validation      # push triggers the refresh workflow on the branch
```
Then open the PR on GitHub; merging runs it on main.

## What the etf-eod patch does
tickers.txt-driven universe (131 → 168 tickers incl. the 10 deck rows lacking history and the asset-class gaps), scripts/pull.py with a dispatch-time subset, scripts/validate.py → validation.csv + monthly_returns.csv, path-scoped push trigger, concurrency guard, README. Validation dry-ran clean on the current data.

## Status 2026-09-03 (Step 3 check)
- Step 1 done (GitHub connected). Step 2 done: the etf-eod task (a separate Claude session, 01UNJrhC…, which does not see this conversation) pushed `claude/tickers-file-validation-ujo0xd` and opened PR #1.
- Run 8 (14:35 UTC 2026-09-02) failed silently: 0/168 pulled, every `pull_status.csv` row `No module named 'scipy'` — patch 0001 dropped `scipy scikit-learn` from the pip line that main had (yfinance `repair=True` needs them); the job still committed because nothing gated the commit step.
- Both sessions diagnosed it independently. The etf-eod session's own fix (de1acd9, 23:54 UTC: `pip install "yfinance[repair]"`, real yfinance error text in `err`, exit 1 on total failure) landed first and supersedes this session's patch 0002, which was withdrawn to avoid a conflicting `git am`.
- Re-run (a6102f0, 00:00 UTC 2026-09-03) is green: **168/168 ok**, all 37 new tickers present, `validation.csv` no stale/nonpositive/gap flags, `monthly_returns.csv` 321 × 168 through 2026-09 (2026-09 is a partial month — exclude from vol48). Short histories: JEDI 235 rows, EUAD 467, NUKZ 655, IBIT 663, SHLD 745.
- **Merged 2026-09-02/03** (PR #1, then PR #2 "Refresh flow follow-ups" from the same etf-eod task: run-time ref checkout + rebase-and-retry on push, yfinance errors visible in logs, README corrections). main's latest refresh 25ea5be (00:42 UTC 2026-09-03) is green: 168/168 ok, no stale, last_date 2026-09-02; feature branches deleted. The weekly schedule (Mondays 09:20 UTC) now maintains the 168-ticker universe. Step 4 was completed from the branch (Addendum 6).

Division of labor from here: the repo-bound tasks own repository changes (they hold push access); this session owns analysis and only reads the public repo.
