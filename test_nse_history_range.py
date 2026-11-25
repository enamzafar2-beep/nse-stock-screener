import io
from datetime import date, timedelta

import pandas as pd
import requests

BASE = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{}.csv"
TARGETS = [
    date(2026, 11, 24),
    date(2026, 8, 25),
    date(2026, 5, 26),
    date(2026, 2, 24),
    date(2025, 11, 25),
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.nseindia.com/all-reports",
}

session = requests.Session()
session.headers.update(HEADERS)
try:
    session.get("https://www.nseindia.com/", timeout=20)
except requests.RequestException as exc:
    print(f"Session note: {exc}")

results = []
for target in TARGETS:
    found = False
    for back in range(0, 6):
        day = target - timedelta(days=back)
        tag = day.strftime("%d%m%Y")
        url = BASE.format(tag)
        try:
            response = session.get(url, timeout=35)
            if response.status_code == 404:
                print(f"{day}: HTTP 404 (not published)")
                continue
            response.raise_for_status()
            frame = pd.read_csv(io.BytesIO(response.content))
            frame.columns = [str(c).strip().upper() for c in frame.columns]
            needed = {"SYMBOL", "SERIES", "DATE1", "CLOSE_PRICE", "TTL_TRD_QNTY"}
            missing = sorted(needed - set(frame.columns))
            if missing:
                raise ValueError(f"missing columns {missing}")
            eq = frame[frame["SERIES"].astype(str).str.strip().str.upper() == "EQ"]
            if len(eq) < 100:
                raise ValueError(f"only {len(eq)} EQ rows")
            print(f"PASS: target {target}, actual file date {day}, EQ rows {len(eq):,}, bytes {len(response.content):,}")
            results.append((target.isoformat(), day.isoformat(), len(eq), len(response.content)))
            found = True
            break
        except Exception as exc:
            print(f"ERROR {day}: {exc}")
            break
    if not found:
        results.append((target.isoformat(), "NOT FOUND", 0, 0))

print("\nHISTORY RANGE TEST SUMMARY")
for item in results:
    print(" | ".join(map(str, item)))
passed = sum(1 for item in results if item[2] >= 100)
print(f"\nDates found: {passed}/{len(TARGETS)}")
if passed < 3:
    raise SystemExit("FAIL: historical archive access is too limited for the planned backfill")
