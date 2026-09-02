#!/usr/bin/env python3
"""Pull daily end-of-day history from Yahoo Finance for every symbol in tickers.txt.

Usage:
    python scripts/pull.py                 # pull the whole universe
    SUBSET=VTI,BRK-B python scripts/pull.py  # pull only those symbols

Outputs:
    prices/<SYMBOL>.csv   one file per symbol, dashes removed (BRK-B -> BRKB.csv)
    pull_status.csv       ticker,ok,rows,err  (rows for tickers not pulled in a
                          SUBSET run are carried over from the previous file)
"""
import os
import sys
import time

import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKERS_FILE = os.path.join(ROOT, "tickers.txt")
PRICES_DIR = os.path.join(ROOT, "prices")
STATUS_FILE = os.path.join(ROOT, "pull_status.csv")

START = "2000-01-01"
COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj_Close", "Volume", "Capital_Gains", "Repaired?"]
STATUS_COLUMNS = ["ticker", "ok", "rows", "err"]
ATTEMPTS = 3            # total tries per ticker
BACKOFF_SECONDS = 3     # sleep 3s, then 6s between failed attempts
PAUSE_SECONDS = 0.7     # polite pause between tickers


def read_tickers(path=TICKERS_FILE):
    """Return the ordered, de-duplicated list of symbols in tickers.txt."""
    symbols, seen = [], set()
    with open(path, encoding="utf-8-sig") as fh:
        for raw in fh:
            sym = raw.split("#", 1)[0].strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                symbols.append(sym)
    return symbols


def parse_subset(value):
    """Parse the SUBSET env var (comma/whitespace separated) into an ordered list."""
    if value is None:
        return []
    parts, seen = [], set()
    for tok in value.replace("\n", ",").replace(" ", ",").split(","):
        sym = tok.strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            parts.append(sym)
    return parts


def csv_name(ticker):
    return ticker.replace("-", "") + ".csv"


def pull_one(ticker):
    """Download one symbol and write prices/<SYMBOL>.csv. Returns (ok, rows, err)."""
    ok, rows, err = False, 0, ""
    for attempt in range(ATTEMPTS):
        try:
            df = yf.Ticker(ticker).history(start=START, auto_adjust=False, actions=True, repair=True)
            if df is None or df.empty:
                raise RuntimeError("empty")
            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.strftime("%Y-%m-%d")
            df = df.drop(columns=[c for c in ("Dividends", "Stock Splits") if c in df.columns])
            df = df.rename(columns={"Adj Close": "Adj_Close", "Capital Gains": "Capital_Gains"})
            if "Adj_Close" not in df.columns:
                raise RuntimeError("no adj close")
            df["Volume"] = df["Volume"].fillna(0).astype("int64")
            cols = [c for c in COLUMNS if c in df.columns]
            rows = len(df)
            df[cols].to_csv(os.path.join(PRICES_DIR, csv_name(ticker)), index=False)
            ok, err = True, ""
            break
        except Exception as exc:  # noqa: BLE001 - any failure is retried, then recorded
            err = str(exc)[:80]
            if attempt < ATTEMPTS - 1:
                time.sleep(BACKOFF_SECONDS * (attempt + 1))
    return ok, rows, err


def read_prior_status(path=STATUS_FILE):
    """Return {ticker: (ok, rows, err)} from an existing pull_status.csv, or {}."""
    if not os.path.exists(path):
        return {}
    prior = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for rec in prior.to_dict("records"):
        t = str(rec.get("ticker", "")).strip().upper()
        if not t:
            continue
        ok = str(rec.get("ok", "")).strip().lower() == "true"
        try:
            rows = int(float(rec.get("rows", "") or 0))
        except ValueError:
            rows = 0
        out[t] = (ok, rows, str(rec.get("err", "")))
    return out


def write_status(status, universe, subset_run, path=STATUS_FILE):
    """Write pull_status.csv.

    Full run: one row per symbol in tickers.txt (symbols dropped from the file
    disappear). Subset run: prior rows are kept and only the pulled symbols are
    updated; prior rows for symbols no longer in tickers.txt are appended last.
    """
    merged = read_prior_status(path) if subset_run else {}
    merged.update(status)
    ordered = [t for t in universe if t in merged]
    ordered += [t for t in merged if t not in set(universe)]
    frame = pd.DataFrame(
        [(t, merged[t][0], merged[t][1], merged[t][2]) for t in ordered],
        columns=STATUS_COLUMNS,
    )
    frame.to_csv(path, index=False)
    return frame


def write_step_summary(frame, pulled):
    """Append a short markdown report to the GitHub Actions job summary, if any."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    bad = frame[(frame["ticker"].isin(pulled)) & (~frame["ok"])]
    lines = [
        "## pull",
        f"- pulled {len(pulled)} symbol(s), {len(pulled) - len(bad)} ok, {len(bad)} failed",
    ]
    if len(bad):
        lines.append("")
        lines.append("| ticker | err |")
        lines.append("|---|---|")
        lines += [f"| {r.ticker} | {r.err} |" for r in bad.itertuples()]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n\n")


def main():
    universe = read_tickers()
    if not universe:
        print(f"no symbols found in {TICKERS_FILE}", file=sys.stderr)
        return 2

    subset = parse_subset(os.environ.get("SUBSET"))
    if subset:
        unknown = [t for t in subset if t not in set(universe)]
        if unknown:
            print(f"SUBSET contains symbols not in tickers.txt: {unknown}", file=sys.stderr)
            return 2
        targets = [t for t in universe if t in set(subset)]
        print(f"subset run: {len(targets)} of {len(universe)} symbols")
    else:
        targets = universe
        print(f"full run: {len(targets)} symbols")

    os.makedirs(PRICES_DIR, exist_ok=True)
    status = {}
    for i, ticker in enumerate(targets, 1):
        ok, rows, err = pull_one(ticker)
        status[ticker] = (ok, rows, err)
        print(f"[{i}/{len(targets)}] {ticker}: {'ok' if ok else 'FAIL'} rows={rows} {err}".rstrip(), flush=True)
        time.sleep(PAUSE_SECONDS)

    frame = write_status(status, universe, subset_run=bool(subset))
    failed = [t for t, (ok, _, _) in status.items() if not ok]
    print(f"{len(status) - len(failed)}/{len(status)} ok; failed: {failed}")
    write_step_summary(frame, set(status))
    return 0


if __name__ == "__main__":
    sys.exit(main())
