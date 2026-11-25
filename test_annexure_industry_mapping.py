from pathlib import Path
from io import StringIO
from difflib import SequenceMatcher
import pandas as pd
import requests
import re

ROOT = Path(__file__).parent
UNIVERSE = pd.read_csv(ROOT / "universe.csv")
symbol_col = next(c for c in UNIVERSE.columns if str(c).strip().lower() == "symbol")
symbols_frame = UNIVERSE.copy()
series_col = next((c for c in UNIVERSE.columns if str(c).strip().lower() == "series"), None)
if series_col:
    symbols_frame[series_col] = symbols_frame[series_col].astype(str).str.strip().str.upper()
    symbols_frame = symbols_frame[symbols_frame[series_col] == "EQ"]
symbols = set(symbols_frame[symbol_col].dropna().astype(str).str.strip().str.upper())

url = "https://theannexure.com/companies"
headers = {"User-Agent": "Mozilla/5.0 (compatible; NSEIndustryCoverageTest/1.0)", "Accept": "text/html"}
response = requests.get(url, headers=headers, timeout=60)
response.raise_for_status()
tables = pd.read_html(StringIO(response.text))
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

# Pass 1: exact ticker. Pass 2: exact normalized company-name match only;
# ambiguous or fuzzy matches are deliberately not assigned.
def norm_name(value):
    value = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()
    value = re.sub(r"\b(LIMITED|LTD|PRIVATE|PVT|PUBLIC|PLC|INDIA)\b", " ", value)
    return " ".join(value.split())

table["Company_key"] = table["Company"].map(norm_name)
symbols_frame["Symbol_key"] = symbols_frame[symbol_col].astype(str).str.strip().str.upper()
ticker_matches = table[table["Ticker"].isin(symbols)].drop_duplicates("Ticker", keep="first").copy()
ticker_matches["Symbol"] = ticker_matches["Ticker"]
ticker_matches["Mapping source"] = "The Annexure / BSE equity metadata reference (exact ticker)"
covered = set(ticker_matches["Symbol"])

if "Company" in symbols_frame.columns:
    universe_names = symbols_frame[[symbol_col, "Company"]].copy()
    universe_names["Company_key"] = universe_names["Company"].map(norm_name)
    source_name_counts = table.groupby("Company_key")["Ticker"].nunique()
    universe_name_counts = universe_names.groupby("Company_key")[symbol_col].nunique()
    unique_keys = set(source_name_counts[source_name_counts == 1].index) & set(universe_name_counts[universe_name_counts == 1].index)
    name_candidates = universe_names[universe_names["Company_key"].isin(unique_keys)].merge(
        table, on="Company_key", how="inner", suffixes=("_nse", "_source")
    )
    name_candidates = name_candidates[~name_candidates[symbol_col].isin(covered)].copy()
    name_candidates["Symbol"] = name_candidates[symbol_col].astype(str).str.strip().str.upper()
    name_candidates["Mapping source"] = "The Annexure / BSE equity metadata reference (unique exact normalized company name)"
    name_matches = name_candidates[name_candidates["Symbol"].isin(symbols)]
else:
    name_matches = pd.DataFrame(columns=["Symbol", "Sector", "Industry", "Basic industry", "Mapping source"])

table = pd.concat([
    ticker_matches[["Symbol", "Sector", "Industry", "Basic industry", "Mapping source"]],
    name_matches[["Symbol", "Sector", "Industry", "Basic industry", "Mapping source"]],
], ignore_index=True).drop_duplicates("Symbol", keep="first")
table.to_csv(ROOT / "annexure_industry_mapping_test.csv", index=False)


mapped = set(table["Symbol"])
unmapped = sorted(symbols - mapped)

# Exploratory fuzzy candidates for manual review only. These are never auto-mapped.
candidate_rows = []
unmapped_names = symbols_frame[symbols_frame[symbol_col].astype(str).str.strip().str.upper().isin(unmapped)].copy()
source_names = max(tables, key=len).copy()
source_names["Company_key"] = source_names["Company"].map(norm_name)
source_names["Ticker"] = source_names["Ticker"].astype(str).str.strip().str.upper()
source_names = source_names.drop_duplicates("Ticker", keep="first")
for _, company_row in unmapped_names.iterrows():
    symbol = str(company_row[symbol_col]).strip().upper()
    company_name = str(company_row["Company"])
    key = norm_name(company_name)
    scored = []
    for _, source_row in source_names.iterrows():
        source_key = source_row["Company_key"]
        if not key or not source_key:
            continue
        score = SequenceMatcher(None, key, source_key).ratio()
        scored.append((score, source_row))
    scored.sort(key=lambda item: item[0], reverse=True)
    for score, source_row in scored[:3]:
        if score >= 0.50:
            candidate_rows.append({
                "NSE symbol": symbol,
                "NSE company name": company_name,
                "Candidate source ticker": source_row["Ticker"],
                "Candidate source company": source_row["Company"],
                "Name similarity": round(score, 3),
                "Sector": source_row["Sector"],
                "Industry": source_row["Industry"],
                "Basic industry": source_row["Basic industry"],
                "Review status": "MANUAL REVIEW ONLY — NOT MAPPED",
            })
pd.DataFrame(candidate_rows).to_csv(ROOT / "annexure_unmapped_name_candidates.csv", index=False)
sector_values = table["Sector"].dropna().astype(str).str.strip()
sector_values = sector_values[~sector_values.str.lower().isin(["", "nan", "unclassified"])]
industry_values = table["Industry"].dropna().astype(str).str.strip()
basic_values = table["Basic industry"].dropna().astype(str).str.strip()
unclassified = int(table["Sector"].astype(str).str.strip().str.lower().eq("unclassified").sum())
summary = [
    "THE ANNEXURE INDUSTRY MAPPING COVERAGE TEST — NSE EQ UNIVERSE",
    "Official NSE Indices reference structure: 12 macro-economic sectors / 22 sectors / 59 industries / 197 basic industries",
    f"Universe EQ symbols: {len(symbols):,}",
    f"Companies on source page: {len(max(tables, key=len)):,}",
    f"Universe EQ symbols mapped: {len(mapped):,}",
    f"Coverage: {len(mapped) / max(len(symbols), 1) * 100:.1f}%",
    f"Distinct mapped sector labels (excluding Unclassified): {sector_values.nunique()}",
    f"Distinct mapped industry labels: {industry_values.nunique()}",
    f"Distinct mapped basic-industry labels: {basic_values.nunique()}",
    f"Rows labelled Unclassified: {unclassified:,}",
    f"Unmapped EQ symbols: {len(unmapped):,}",
    "",
    "SAMPLE CHECKS:",
]
for sym in ["3BBLACKBIO", "AASTHA", "ABANSENT", "BLUSPRING"]:
    x = table[table["Symbol"] == sym]
    summary.append(f"{sym}: " + (f"{x.iloc[0]['Sector']} / {x.iloc[0]['Industry']} / {x.iloc[0]['Basic industry']}" if not x.empty else "not mapped"))
summary += [
    "",
    "LIMITATION: The Annexure is a third-party BSE equity metadata reference, not an official NSE company-classification feed. The official NSE Indices taxonomy contains 22 sectors, 59 industries and 197 basic industries. Observed distinct labels in this universe are coverage counts, not proof that every label matches the official taxonomy. Do not enable this as a production filter until unmapped symbols and taxonomy differences are reviewed.",
    "",
    "FIRST 100 UNMAPPED EQ SYMBOLS:",
    *unmapped[:100],
]
(ROOT / "annexure_industry_mapping_summary.txt").write_text("\n".join(summary), encoding="utf-8")
print("\n".join(summary[:22]))
if table.empty:
    raise SystemExit("FAIL: no EQ universe symbols mapped")
