from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
DB = ROOT / "nse_daily_history.csv.gz"
UNIVERSE = ROOT / "universe.csv"
OUT = ROOT / "nse_scanner_test_report.csv"
MIN_HISTORY = 252

history = pd.read_csv(DB, parse_dates=["DATE"])
for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "TURNOVER_CRORES"]:
    history[col] = pd.to_numeric(history[col], errors="coerce")
history = history.dropna(subset=["DATE", "SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "TURNOVER_CRORES"])
history = history.sort_values(["SYMBOL", "DATE"])

universe = pd.read_csv(UNIVERSE)
universe["Symbol"] = universe["Symbol"].astype(str).str.strip().str.upper()
eq_symbols = set(universe.loc[universe["Series"].astype(str).str.upper().eq("EQ"), "Symbol"])
history = history[history["SYMBOL"].isin(eq_symbols)]

rows = []
short_history = 0
for symbol, g in history.groupby("SYMBOL", sort=False):
    g = g.sort_values("DATE").reset_index(drop=True)
    if len(g) < MIN_HISTORY:
        short_history += 1
        continue
    close, high, low, volume = (g[c].astype(float) for c in ["CLOSE", "HIGH", "LOW", "VOLUME"])
    turnover = g["TURNOVER_CRORES"].astype(float)
    last = float(close.iloc[-1])
    low_52w = float(low.tail(252).min())
    ret_6m = (last / float(close.iloc[-127]) - 1) * 100
    avg_turnover = float(turnover.tail(30).mean())
    adr_pct = float((((high.tail(14) - low.tail(14)) / close.tail(14)) * 100).mean())
    prior_vol = float(volume.iloc[-31:-1].mean())
    volume_ratio = float(volume.iloc[-1] / prior_vol * 100) if prior_vol > 0 else 0.0
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50_series = close.ewm(span=50, adjust=False).mean()
    ema50 = float(ema50_series.iloc[-1])
    ema50_rising = float(ema50_series.iloc[-1]) > float(ema50_series.iloc[-11])
    swing_high, swing_low = float(high.tail(127).max()), float(low.tail(127).min())
    swing_advance = (swing_high / swing_low - 1) * 100 if swing_low > 0 else 0.0
    near20 = abs(last / ema20 - 1) * 100 <= 2.0
    near50 = abs(last / ema50 - 1) * 100 <= 2.0
    holding50 = last >= ema50 * .98
    pullback = swing_advance >= 50 and ema50_rising and (near20 or near50) and holding50
    common_before_adr = last > 90 and low_52w > 0 and last / low_52w >= 1.5 and avg_turnover >= 7
    common = common_before_adr and adr_pct >= 3.0
    momentum = common and ret_6m >= 50
    pullback_match = common and pullback
    rows.append({
        "Symbol": symbol, "As of": g["DATE"].iloc[-1].date().isoformat(),
        "History sessions": len(g), "Price (₹)": last, "6M return (%)": ret_6m,
        "52-week low (₹)": low_52w, "Price / 52-week low (x)": last / low_52w,
        "30D avg NSE turnover (₹ Cr)": avg_turnover, "ADR% (14 sessions)": adr_pct,
        "ADR% >= 3%": adr_pct >= 3.0, "Latest volume": int(volume.iloc[-1]),
        "Today's volume vs prior 30D avg (%)": volume_ratio,
        "EMA20 (₹)": ema20, "EMA50 (₹)": ema50,
        "Prior 6M swing advance (%)": swing_advance, "EMA50 rising": ema50_rising,
        "Momentum core filters (pre-mcap/RS/sector)": momentum,
        "Pullback core filters (pre-mcap/RS/sector)": pullback_match,
        "Momentum + 150% volume": momentum and volume_ratio >= 150,
        "Pullback + 70% volume": pullback_match and volume_ratio <= 70,
    })

report = pd.DataFrame(rows)
if report.empty:
    raise SystemExit("FAIL: no EQ symbols have a full 252-session history")
report = report.sort_values(
    ["Momentum core filters (pre-mcap/RS/sector)", "Pullback core filters (pre-mcap/RS/sector)"],
    ascending=False,
)
report.to_csv(OUT, index=False)
print("NSE SCANNER TEST SUMMARY — STRICT 252-SESSION VERSION")
print(f"Database date range: {history['DATE'].min().date()} to {history['DATE'].max().date()}")
print(f"EQ symbols in live universe: {len(eq_symbols):,}")
print(f"EQ symbols with at least 252 sessions: {len(report):,}")
print(f"EQ symbols excluded for insufficient history: {short_history:,}")
print(f"Stocks passing ADR% >= 3%: {int(report['ADR% >= 3%'].sum()):,}")
print(f"Momentum core matches after ADR (before market cap, RS and sector): {int(report['Momentum core filters (pre-mcap/RS/sector)'].sum()):,}")
print(f"Momentum + 150% volume: {int(report['Momentum + 150% volume'].sum()):,}")
print(f"Pullback core matches (before market cap, RS and sector): {int(report['Pullback core filters (pre-mcap/RS/sector)'].sum()):,}")
print(f"Pullback + 70% volume: {int(report['Pullback + 70% volume'].sum()):,}")
print("\nRequested example symbols:")
for symbol in ["3BBLACKBIO", "AASTHA", "ABANSENT"]:
    x = report[report["Symbol"] == symbol]
    if x.empty:
        n = int((history["SYMBOL"] == symbol).sum())
        print(f"{symbol}: {n} sessions; excluded from strict 52-week test")
    else:
        r = x.iloc[0]
        print(f"{symbol}: {int(r['History sessions'])} sessions; 6M return {r['6M return (%)']:.1f}%; 30D NSE turnover ₹{r['30D avg NSE turnover (₹ Cr)']:.2f} Cr")
print("\nLIMITATIONS: Preliminary technical matches only. Market-cap, PEG, sector-strength and benchmark-relative RS filters are not applied because those fields are not in the NSE bhavcopy database. This test uses only symbols with a complete 252-session lookback and official NSE turnover.")
