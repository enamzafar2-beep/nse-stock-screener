from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).parent
UNIVERSE = pd.read_csv(ROOT / "universe.csv")
symbol_col = next(c for c in UNIVERSE.columns if str(c).strip().lower() == "symbol")
symbols = set(UNIVERSE[symbol_col].dropna().astype(str).str.strip().str.upper())

url = "https://theannexure.com/companies"
headers = {"User-Agent": "Mozilla/5.0 (compatible; NSEIndustryCoverageTest/1.0)", "Accept": "text/html"}
response = requests.get(url, headers=headers, timeout=60)
response.raise_for_status()
tables = pd.read_html(response.text)
if not tables:
    raise SystemExit("FAIL: no company table found on The Annexure page")

table = max(tables, key=len)
table.columns = [str(c).strip() for c in table.columns]
required = ["Ticker", "Sector", "Industry", "Basic industry"]
if not set(required).issubset(table.columns):
    raise SystemExit(f"FAIL: expected columns missing; found {list(table.columns)}")

table["Ticker"] = table["Ticker"].astype(str).str.strip().str.upper()
for c in ["Sector", "Industry", "Basic industry"]:
    table[c] = table[c].astype(str).str.strip()
table = table[table["Ticker"].isin(symbols)].copy()
table = table.drop_duplicates("Ticker", keep="first")
table = table.rename(columns={"Ticker": "Symbol"})
table["Mapping source"] = "The Annexure / BSE equity metadata reference"
table = table[["Symbol", "Sector", "Industry", "Basic industry", "Mapping source"]]
table.to_csv(ROOT / "annexure_industry_mapping_test.csv", index=False)

mapped = set(table["Symbol"])
unmapped = sorted(symbols - mapped)
summary = [
    "THE ANNEXURE INDUSTRY MAPPING COVERAGE TEST",
    f"Universe symbols: {len(symbols):,}",
    f"Companies on source page: {len(max(tables, key=len)):,}",
    f"Universe symbols mapped: {len(mapped):,}",
    f"Coverage: {len(mapped) / max(len(symbols), 1) * 100:.1f}%",
    f"Distinct sectors in mapped universe: {table['Sector'].nunique()}",
    f"Distinct industries in mapped universe: {table['Industry'].nunique()}",
    f"Distinct basic industries in mapped universe: {table['Basic industry'].nunique()}",
    f"Unmapped symbols: {len(unmapped):,}",
    "",
    "SAMPLE CHECKS:",
]
for sym in ["3BBLACKBIO", "AASTHA", "ABANSENT", "BLUSPRING"]:
    x = table[table["Symbol"] == sym]
    summary.append(f"{sym}: " + (f"{x.iloc[0]['Sector']} / {x.iloc[0]['Industry']} / {x.iloc[0]['Basic industry']}" if not x.empty else "not mapped"))
summary += ["", "LIMITATION: Third-party reference, not an official NSE feed. Its published classification counts (58 industries / 192 basic industries) differ from NSE's published 59 / 197 framework. Verify mapping quality and currency before production use.", "", "FIRST 100 UNMAPPED SYMBOLS:", *unmapped[:100]]
(ROOT / "annexure_industry_mapping_summary.txt").write_text("\n".join(summary), encoding="utf-8")
print("\n".join(summary[:20]))
if table.empty:
    raise SystemExit("FAIL: no universe symbols mapped")
