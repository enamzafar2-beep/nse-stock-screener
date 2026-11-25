import io
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).parent
UNIVERSE = ROOT / "universe.csv"
OUT = ROOT / "nse_industry_mapping_test.csv"
SUMMARY = ROOT / "nse_industry_mapping_summary.txt"

URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "text/csv,*/*",
    "Referer": "https://www.nseindia.com/",
}
session = requests.Session()
session.headers.update(HEADERS)

universe = pd.read_csv(UNIVERSE)
symbol_col = next(c for c in universe.columns if str(c).strip().lower() == "symbol")
symbols = set(universe[symbol_col].dropna().astype(str).str.strip().str.upper())
mapping = {}
source_counts = []
errors = []

for url in URLS:
    try:
        response = session.get(url, timeout=45)
        response.raise_for_status()
        frame = pd.read_csv(io.BytesIO(response.content))
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        if not {"symbol", "industry"}.issubset(frame.columns):
            raise ValueError("CSV does not contain both Symbol and Industry columns")
        frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
        frame["industry"] = frame["industry"].astype(str).str.strip()
        frame = frame[frame["symbol"].isin(symbols) & frame["industry"].ne("")]
        added = 0
        for _, row in frame.iterrows():
            sym = row["symbol"]
            if sym not in mapping:
                mapping[sym] = {
                    "Symbol": sym,
                    "NSE Industry": row["industry"],
                    "Source list": url.rsplit("/", 1)[-1],
                }
                added += 1
        source_counts.append(f"{url.rsplit('/',1)[-1]}: {len(frame)} universe matches; {added} new mappings")
    except Exception as exc:
        errors.append(f"{url}: {type(exc).__name__}: {exc}")

result = pd.DataFrame(mapping.values(), columns=["Symbol", "NSE Industry", "Source list"])
if not result.empty:
    result = result.sort_values(["NSE Industry", "Symbol"])
result.to_csv(OUT, index=False)

covered = set(result["Symbol"]) if not result.empty else set()
missing = sorted(symbols - covered)
lines = [
    "NSE INDUSTRY MAPPING COVERAGE TEST",
    f"Universe symbols: {len(symbols):,}",
    f"Symbols mapped from official NSE constituent lists: {len(covered):,}",
    f"Coverage: {len(covered) / max(len(symbols), 1) * 100:.1f}%",
    f"Distinct industry labels returned: {result['NSE Industry'].nunique() if not result.empty else 0}",
    f"Unmapped symbols: {len(missing):,}",
    "",
    "SOURCE RESULTS:",
    *(source_counts or ["No source list succeeded."]),
    "",
    "LIMITATION: This is a coverage experiment using official NSE index constituent CSVs. It does not prove that the lists cover every NSE-listed company or every industry in the 59-industry classification. Do not use this partial mapping as a production sector filter.",
]
if errors:
    lines += ["", "SOURCE ERRORS:", *errors]
lines += ["", "FIRST 100 UNMAPPED SYMBOLS:", *missing[:100]]
SUMMARY.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:30]))
if result.empty:
    raise SystemExit("FAIL: no NSE industry mappings retrieved")
print(f"\nSaved mapping sample: {OUT}")
