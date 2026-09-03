#!/usr/bin/env python3
"""Validate every prices/*.csv and build a monthly-returns panel.

Usage:
    python scripts/validate.py

Outputs (repo root):
    validation.csv       one row per price file:
                         ticker, first_date, last_date, rows, max_gap_days,
                         stale, adj_close_nonpositive, repaired_rows
    monthly_returns.csv  wide panel indexed by month (YYYY-MM): month-end
                         Adj_Close percent change per ticker
"""
import datetime as dt
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKERS_FILE = os.path.join(ROOT, "tickers.txt")
PRICES_DIR = os.path.join(ROOT, "prices")
VALIDATION_FILE = os.path.join(ROOT, "validation.csv")
RETURNS_FILE = os.path.join(ROOT, "monthly_returns.csv")

STALE_DAYS = 10                 # last_date older than this many days => stale
VALIDATION_COLUMNS = [
    "ticker", "first_date", "last_date", "rows", "max_gap_days",
    "stale", "adj_close_nonpositive", "repaired_rows",
]


def symbol_map():
    """Map file stem (dashes removed) -> Yahoo symbol from tickers.txt, e.g. BRKB -> BRK-B."""
    out = {}
    if not os.path.exists(TICKERS_FILE):
        return out
    with open(TICKERS_FILE, encoding="utf-8-sig") as fh:
        for raw in fh:
            sym = raw.split("#", 1)[0].strip().upper()
            if sym:
                out.setdefault(sym.replace("-", ""), sym)
    return out


def load_prices(path):
    """Read one price file. Returns (clean_df, raw_rows).

    raw_rows is the number of data rows in the file, which is what validation.csv
    reports. The frame used for the other metrics is sorted by date, with
    unparseable dates dropped and duplicate dates collapsed (last wins).
    """
    df = pd.read_csv(path)
    raw_rows = len(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date", kind="stable")
    df = df.drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    return df, raw_rows


def validate_one(ticker, df, today, rows):
    dates = df["Date"]
    if len(df) == 0:
        return dict(ticker=ticker, first_date="", last_date="", rows=rows, max_gap_days=0,
                    stale=True, adj_close_nonpositive=0, repaired_rows=0)
    gaps = dates.diff().dt.days.dropna()
    max_gap = int(gaps.max()) if len(gaps) else 0
    last = dates.iloc[-1].date()
    adj = pd.to_numeric(df.get("Adj_Close"), errors="coerce") if "Adj_Close" in df.columns else pd.Series(dtype=float)
    nonpos = int((adj <= 0).sum())
    if "Repaired?" in df.columns:
        repaired = int(df["Repaired?"].astype(str).str.strip().str.lower().eq("true").sum())
    else:
        repaired = 0
    return dict(
        ticker=ticker,
        first_date=dates.iloc[0].strftime("%Y-%m-%d"),
        last_date=last.strftime("%Y-%m-%d"),
        rows=rows,
        max_gap_days=max_gap,
        stale=(today - last).days > STALE_DAYS,
        adj_close_nonpositive=nonpos,
        repaired_rows=repaired,
    )


def monthly_returns(ticker, df):
    """Month-end Adj_Close percent change as a Series indexed by YYYY-MM strings."""
    if "Adj_Close" not in df.columns:
        return None
    adj = pd.Series(pd.to_numeric(df["Adj_Close"], errors="coerce").values, index=df["Date"]).dropna()
    if adj.empty:
        return None
    month_end = adj.groupby(adj.index.to_period("M")).last()
    rets = month_end.pct_change()
    rets.index = rets.index.strftime("%Y-%m")
    rets.name = ticker
    return rets


def write_step_summary(val):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    stale = val[val["stale"]]
    gappy = val.sort_values("max_gap_days", ascending=False).head(5)
    lines = [
        "## validate",
        f"- {len(val)} files, {int(val['stale'].sum())} stale (> {STALE_DAYS} days), "
        f"{int((val['adj_close_nonpositive'] > 0).sum())} with non-positive Adj_Close, "
        f"{int((val['repaired_rows'] > 0).sum())} with repaired rows",
    ]
    if len(stale):
        lines += ["", "| stale ticker | last_date |", "|---|---|"]
        lines += [f"| {r.ticker} | {r.last_date} |" for r in stale.itertuples()]
    lines += ["", "| largest gaps | max_gap_days | rows |", "|---|---|---|"]
    lines += [f"| {r.ticker} | {r.max_gap_days} | {r.rows} |" for r in gappy.itertuples()]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n\n")


def main():
    files = sorted(glob.glob(os.path.join(PRICES_DIR, "*.csv")))
    if not files:
        print(f"no price files found in {PRICES_DIR}", file=sys.stderr)
        return 2

    names = symbol_map()
    today = dt.datetime.now(dt.timezone.utc).date()
    records, panels = [], []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        ticker = names.get(stem.upper(), stem)
        try:
            df, raw_rows = load_prices(path)
        except Exception as exc:  # noqa: BLE001 - report unreadable files, keep going
            print(f"{ticker}: unreadable ({exc})", file=sys.stderr)
            records.append(dict(ticker=ticker, first_date="", last_date="", rows=0, max_gap_days=0,
                                stale=True, adj_close_nonpositive=0, repaired_rows=0))
            continue
        records.append(validate_one(ticker, df, today, raw_rows))
        rets = monthly_returns(ticker, df)
        if rets is not None:
            panels.append(rets)

    val = pd.DataFrame(records, columns=VALIDATION_COLUMNS).sort_values("ticker").reset_index(drop=True)
    val.to_csv(VALIDATION_FILE, index=False)

    panel = pd.concat(panels, axis=1).sort_index() if panels else pd.DataFrame()
    panel = panel.reindex(columns=sorted(panel.columns))
    panel.index.name = "month"
    panel.round(6).to_csv(RETURNS_FILE)

    stale = val[val["stale"]]["ticker"].tolist()
    print(f"{len(val)} files validated; stale: {stale}")
    print(f"largest gaps: {val.nlargest(5, 'max_gap_days')[['ticker', 'max_gap_days']].to_dict('records')}")
    print(f"non-positive Adj_Close: {val[val['adj_close_nonpositive'] > 0]['ticker'].tolist()}")
    print(f"monthly_returns.csv: {panel.shape[0]} months x {panel.shape[1]} tickers")
    write_step_summary(val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
