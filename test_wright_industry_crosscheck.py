from pathlib import Path
from io import StringIO
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent
universe = pd.read_csv(ROOT / "universe.csv")
series_col = next((c for c in universe.columns if str(c).strip().lower() == "series"), None)
symbol_col = next(c for c in universe.columns if str(c).strip().lower() == "symbol")
if series_col:
    universe[series_col] = universe[series_col].astype(str).str.strip().str.upper()
    universe = universe[universe[series_col] == "EQ"].copy()
universe[symbol_col] = universe[symbol_col].astype(str).str.strip().str.upper()
symbols = set(universe[symbol_col])

# Reproduce the current exact-ticker Annexure coverage test.
headers = {"User-Agent": "Mozilla/5.0 (compatible; NSEIndustryCrosscheck/1.0)"}
annexure_url = "https://theannexure.com/companies"
annexure_response = requests.get(annexure_url, headers=headers, timeout=60)
annexure_response.raise_for_status()
annexure_tables = pd.read_html(StringIO(annexure_response.text))
annexure = max(annexure_tables, key=len)
annexure.columns = [str(c).strip() for c in annexure.columns]
if "Ticker" not in annexure.columns:
    raise SystemExit("FAIL: Annexure Ticker column unavailable")
annexure_tickers = set(annexure["Ticker"].astype(str).str.strip().str.upper())
unmapped = sorted(symbols - annexure_tickers)

session = requests.Session()
session.headers.update(headers)
records = []
pages = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
for letter in pages:
    url = f"https://www.wrightresearch.in/stocks/directory/{letter}/"
    response = session.get(url, timeout=45)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    # Directory entries are rendered as list items. Match only complete ticker tokens.
    candidates = []
    for node in soup.find_all(["li", "tr", "p", "div"]):
        txt = " ".join(node.get_text(" ", strip=True).split())
        if txt and len(txt) < 500:
            candidates.append(txt)
    # Keep the shortest containing entry per ticker to avoid capturing whole page containers.
    for symbol in unmapped:
        pat = re.compile(rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])", re.I)
        hits = [x for x in candidates if pat.search(x)]
        if not hits:
            continue
        entry = min(hits, key=len)
        match = pat.search(entry)
        category = entry[match.end():].strip(" -|:")
        records.append({
            "Symbol": symbol,
            "Company": universe.loc[universe[symbol_col] == symbol, "Company"].iloc[0] if "Company" in universe.columns else "",
            "Wright directory entry": entry,
            "Wright category text": category,
            "Source URL": url,
            "Mapping status": "Third-party cross-check only; not an NSE classification",
        })

result = pd.DataFrame(records)
if not result.empty:
    result = result.drop_duplicates("Symbol", keep="first").sort_values("Symbol")
result.to_csv(ROOT / "wright_industry_crosscheck.csv", index=False)
found = set(result["Symbol"]) if not result.empty else set()
summary = [
    "WRIGHT RESEARCH DIRECTORY CROSS-CHECK — NOT AN OFFICIAL NSE MAPPING",
    f"NSE EQ universe: {len(symbols):,}",
    f"Unmapped by exact Annexure ticker: {len(unmapped):,}",
    f"Unmapped symbols found in Wright directory: {len(found):,}",
    f"Still absent from Wright directory: {len(set(unmapped) - found):,}",
    "",
    "The Wright directory labels are third-party industry labels. They are research leads only and must not be inserted into the NSE 22-sector / 59-industry taxonomy without a reviewed crosswalk and company-disclosure verification.",
    "",
    "FOUND SYMBOLS:",
    *sorted(found),
    "",
    "STILL NOT FOUND:",
    *sorted(set(unmapped) - found),
]
(ROOT / "wright_industry_crosscheck_summary.txt").write_text("\n".join(summary), encoding="utf-8")
print("\n".join(summary))
