from pathlib import Path
import pandas as pd

root = Path(__file__).parent
universe = pd.read_csv(root / "universe.csv", dtype=str).fillna("")
mapping = pd.read_csv(root / "industry_classification.csv", dtype=str).fillna("")
symbol_col = next(c for c in universe.columns if c.strip().lower() == "symbol")
series_col = next((c for c in universe.columns if c.strip().lower() == "series"), None)
if series_col:
    universe = universe[universe[series_col].str.strip().str.upper() == "EQ"]
expected = set(universe[symbol_col].str.strip().str.upper())
required = {"Symbol", "Sector", "Industry", "Basic industry"}
missing = required - set(mapping.columns)
if missing:
    raise SystemExit(f"FAIL: classification columns missing: {sorted(missing)}")
mapping["Symbol"] = mapping["Symbol"].str.strip().str.upper()
if mapping["Symbol"].duplicated().any():
    raise SystemExit("FAIL: duplicate symbols in industry_classification.csv")
actual = set(mapping["Symbol"])
if expected != actual:
    raise SystemExit(
        f"FAIL: universe coverage mismatch; missing={len(expected-actual)}, "
        f"extra={len(actual-expected)}"
    )
for column in ("Sector", "Industry", "Basic industry"):
    values = mapping[column].str.strip()
    invalid = values.eq("") | values.str.lower().isin(["nan", "unknown", "unclassified"])
    if invalid.any():
        raise SystemExit(f"FAIL: {int(invalid.sum())} blank/unclassified values in {column}")
print(f"PASS: {len(actual):,}/{len(expected):,} EQ symbols uniquely classified")
print(f"Sectors: {mapping['Sector'].nunique()}; industries: {mapping['Industry'].nunique()}; basic industries: {mapping['Basic industry'].nunique()}")
