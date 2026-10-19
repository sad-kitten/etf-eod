#!/usr/bin/env python3
"""Merge Value Scout universe additions into tickers.txt.

Reads the published-to-web CSV of the "Universe" tab of the Value Scout sheet
(URL in env VS_ADDITIONS_CSV_URL). Rows whose Status is "queued" are appended to
tickers.txt as "SYMBOL  # VS <date> <exposure>". Additions only: existing lines
are never changed or removed. At most MAX_ADD symbols per run; symbols must look
like plain US tickers (letters, digits, dot or dash; max 10 chars).
"""
import csv
import io
import os
import re
import sys
import urllib.request

MAX_ADD = 25
SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def main():
    url = os.environ.get("VS_ADDITIONS_CSV_URL", "").strip()
    if not url.startswith("https://docs.google.com/"):
        print("VS_ADDITIONS_CSV_URL not set or not a Google Sheets URL; nothing to merge")
        return 0
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    with open("tickers.txt", encoding="utf-8-sig") as fh:
        lines = fh.read().splitlines()
    have = {l.split("#", 1)[0].strip().upper() for l in lines}
    added = []
    for r in rows:
        sym = (r.get("Ticker") or "").strip().upper()
        status = (r.get("Status") or "").strip().lower()
        if status != "queued" or not SYMBOL.match(sym) or sym in have:
            continue
        note = re.sub(r"[^A-Za-z0-9 &/().,'-]", "", (r.get("Exposure") or ""))[:60]
        date = re.sub(r"[^0-9-]", "", (r.get("Date queued") or ""))[:10]
        lines.append(f"{sym}  # VS {date} {note}".rstrip())
        have.add(sym)
        added.append(sym)
        if len(added) >= MAX_ADD:
            break
    if added:
        with open("tickers.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    print(f"added {len(added)}: {', '.join(added)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
