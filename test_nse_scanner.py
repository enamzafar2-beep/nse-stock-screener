from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
DB = ROOT / "nse_daily_history.csv.gz"
UNIVERSE = ROOT / "universe.csv"
OUT = ROOT / "nse_scanner_test_report.csv"
# Re-run audit whenever the refreshed NSE database changes; latest verified session is reported below.

MIN_HISTORY = 127
MIN_PRICE = 90.0
MIN_ABOVE_52W_LOW = 1.5
MIN_TURNOVER_CR = 7.0
MIN_ADR_PCT = 3.0
ADR_SESSIONS = 14
EMA_PROXIMITY_PCT = 2.0
PULLBACK_DAILY_VOLUME_MAX_PCT = 70.0
MOMENTUM_DAILY_VOLUME_MIN_PCT = 150.0

history = pd.read_csv(DB, compression="gzip", parse_dates=["DATE"])
required = {"DATE", "SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "TURNOVER_CRORES"}
missing = required - set(history.columns)
if missing:
    raise SystemExit(f"FAIL: NSE database missing columns: {sorted(missing)}")

for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "TURNOVER_CRORES"]:
    history[col] = pd.to_numeric(history[col], errors="coerce")
history["SYMBOL"] = history["SYMBOL"].astype(str).str.strip().str.upper()
history = history.dropna(subset=["DATE", "SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "TURNOVER_CRORES"])
history = history.sort_values(["SYMBOL", "DATE"])

universe = pd.read_csv(UNIVERSE)
symbol_col = next((c for c in universe.columns if str(c).strip().lower() == "symbol"), None)
series_col = next((c for c in universe.columns if str(c).strip().lower() == "series"), None)
if symbol_col is None:
    raise SystemExit("FAIL: universe.csv has no Symbol column")
if series_col is not None:
    universe = universe[universe[series_col].astype(str).str.upper().str.strip() == "EQ"]
eq_symbols = set(universe[symbol_col].dropna().astype(str).str.strip().str.upper())
history = history[history["SYMBOL"].isin(eq_symbols)]

rows = []
short_history = 0
for symbol, g in history.groupby("SYMBOL", sort=False):
    g = g.sort_values("DATE").reset_index(drop=True)
    if len(g) < MIN_HISTORY:
        short_history += 1
        continue

    close = g["CLOSE"].astype(float)
    high = g["HIGH"].astype(float)
    low = g["LOW"].astype(float)
    open_price = g["OPEN"].astype(float)
    volume = g["VOLUME"].astype(float)
    turnover = g["TURNOVER_CRORES"].astype(float)

    last = float(close.iloc[-1])
    low_52w = float(low.tail(252).min())
    ret_6m = float((last / float(close.iloc[-127]) - 1) * 100)
    avg_turnover = float(turnover.tail(30).mean())
    adr14_pct = float((((high - low) / close) * 100).tail(ADR_SESSIONS).mean())

    prior_30d_avg_volume = float(volume.iloc[-31:-1].mean())
    latest_volume = float(volume.iloc[-1])
    volume_ratio_pct = latest_volume / prior_30d_avg_volume * 100 if prior_30d_avg_volume > 0 else 0.0

    range_pct = ((high - low) / close) * 100
    avg_volume_5d = float(volume.tail(5).mean())
    prior_20d_avg_volume = float(volume.iloc[-25:-5].mean())
    avg_range_5d = float(range_pct.tail(5).mean())
    prior_20d_range = float(range_pct.iloc[-25:-5].mean())
    volume_contracting = prior_20d_avg_volume > 0 and avg_volume_5d < prior_20d_avg_volume
    range_contracting = prior_20d_range > 0 and avg_range_5d < prior_20d_range
    latest_day_quiet = volume_ratio_pct <= PULLBACK_DAILY_VOLUME_MAX_PCT
    pullback_volume_and_price_contracting = (
        latest_day_quiet and volume_contracting and range_contracting
    )

    ema20_series = close.ewm(span=20, adjust=False).mean()
    ema50_series = close.ewm(span=50, adjust=False).mean()
    ema20 = float(ema20_series.iloc[-1])
    ema50 = float(ema50_series.iloc[-1])
    ema50_rising = float(ema50_series.iloc[-1]) > float(ema50_series.iloc[-11])
    near_ema20 = abs(last / ema20 - 1) * 100 <= EMA_PROXIMITY_PCT
    near_ema50 = abs(last / ema50 - 1) * 100 <= EMA_PROXIMITY_PCT
    holding_ema50 = last >= ema50 * 0.98

    swing_high = float(high.tail(127).max())
    swing_low = float(low.tail(127).min())
    prior_advance = (swing_high / swing_low - 1) * 100 if swing_low > 0 else 0.0
    pullback_setup = (
        prior_advance >= 50 and ema50_rising and (near_ema20 or near_ema50) and holding_ema50
    )

    adr_pass = pd.notna(adr14_pct) and adr14_pct >= MIN_ADR_PCT
    common_pass = (
        last > MIN_PRICE and low_52w > 0
        and last / low_52w >= MIN_ABOVE_52W_LOW
        and avg_turnover >= MIN_TURNOVER_CR
    )
    momentum_core = adr_pass and common_pass and ret_6m >= 50
    momentum_with_150_volume = momentum_core and volume_ratio_pct >= MOMENTUM_DAILY_VOLUME_MIN_PCT
    pullback_setup_before_contraction = adr_pass and common_pass and pullback_setup
    pullback_default = pullback_setup_before_contraction and pullback_volume_and_price_contracting

    rows.append({
        "Symbol": symbol,
        "As of": g["DATE"].iloc[-1].date().isoformat(),
        "History sessions": len(g),
        "Price (₹)": last,
        "6M return (%)": ret_6m,
        "52W low (₹)": low_52w,
        "Price / 52W low (x)": last / low_52w if low_52w > 0 else None,
        "30D avg NSE turnover (₹ Cr)": avg_turnover,
        "ADR 14D (%)": adr14_pct,
        "ADR >= 3%": adr_pass,
        "Latest volume": int(latest_volume),
        "Today's volume vs prior 30D avg (%)": volume_ratio_pct,
        "5D avg volume": avg_volume_5d,
        "Prior 20D avg volume": prior_20d_avg_volume,
        "5D vs prior 20D volume (%)": avg_volume_5d / prior_20d_avg_volume * 100 if prior_20d_avg_volume > 0 else None,
        "5D avg daily range (%)": avg_range_5d,
        "Prior 20D avg daily range (%)": prior_20d_range,
        "5D vs prior 20D range (%)": avg_range_5d / prior_20d_range * 100 if prior_20d_range > 0 else None,
        "Latest day volume <= 70%": latest_day_quiet,
        "5D volume contracting": volume_contracting,
        "5D price range contracting": range_contracting,
        "Pullback contraction filter passes": pullback_volume_and_price_contracting,
        "EMA20 (₹)": ema20,
        "EMA50 (₹)": ema50,
        "EMA50 rising": ema50_rising,
        "Near EMA20": near_ema20,
        "Near EMA50": near_ema50,
        "Holding EMA50": holding_ema50,
        "Prior 6M swing advance (%)": prior_advance,
        "Common price/low/turnover filters": common_pass,
        "Momentum core (pre market-cap/RS/sector)": momentum_core,
        "Momentum + optional 150% volume": momentum_with_150_volume,
        "Pullback setup before contraction": pullback_setup_before_contraction,
        "Pullback default (all 3 contraction checks)": pullback_default,
    })

report = pd.DataFrame(rows)
if report.empty:
    raise SystemExit("FAIL: no EQ symbols have at least 127 sessions in the NSE database")

# Fetch Yahoo metadata only for technical candidates, matching app.py's metadata stage.
MIN_MARKET_CAP_CR = 1000.0
MIN_RS_SCORE = 80.0
BENCHMARK = "^CRSLDX"

candidate_symbols = set(report.loc[
    report["Momentum core (pre market-cap/RS/sector)"]
    | report["Pullback default (all 3 contraction checks)"], "Symbol"
])

def fetch_metadata(symbol):
    try:
        info = yf.Ticker(f"{symbol}.NS").get_info() or {}
        market_cap = info.get("marketCap")
        market_cap_cr = float(market_cap) / 1e7 if market_cap else None
        sector = info.get("sector") or "Unknown"
        if market_cap_cr is not None and not math.isfinite(market_cap_cr):
            market_cap_cr = None
        return symbol, market_cap_cr, sector
    except Exception:
        return symbol, None, "Unknown"

metadata = {}
with ThreadPoolExecutor(max_workers=6) as pool:
    futures = [pool.submit(fetch_metadata, symbol) for symbol in sorted(candidate_symbols)]
    for future in as_completed(futures):
        symbol, market_cap_cr, sector = future.result()
        metadata[symbol] = (market_cap_cr, sector)

report["Yahoo market cap (₹ Cr)"] = report["Symbol"].map(
    lambda s: metadata.get(s, (None, "Unknown"))[0]
)
report["Yahoo sector"] = report["Symbol"].map(
    lambda s: metadata.get(s, (None, "Unknown"))[1]
)
report["Market cap >= ₹1,000 Cr"] = report["Yahoo market cap (₹ Cr)"].gt(MIN_MARKET_CAP_CR)

benchmark_return = None
try:
    benchmark = yf.download(
        BENCHMARK, period="6mo", interval="1d", auto_adjust=True,
        progress=False, threads=False, timeout=25
    )
    if benchmark is not None and not benchmark.empty:
        if isinstance(benchmark.columns, pd.MultiIndex):
            benchmark.columns = benchmark.columns.get_level_values(0)
        benchmark_close = benchmark["Close"].dropna()
        if len(benchmark_close) >= 100:
            benchmark_return = float(
                (benchmark_close.iloc[-1] / benchmark_close.iloc[0] - 1) * 100
            )
except Exception as exc:
    print(f"Benchmark retrieval warning: {str(exc)[:160]}")

if benchmark_return is None:
    raise SystemExit("FAIL: Yahoo Nifty 500 proxy (^CRSLDX) benchmark unavailable; RS cannot be validated")

report["Nifty 500 6M return (%)"] = benchmark_return
report["Excess return vs Nifty 500 (%)"] = report["6M return (%)"] - benchmark_return

def apply_relative_strength_and_sector(frame, technical_column, prefix):
    eligible = frame[frame[technical_column] & frame["Market cap >= ₹1,000 Cr"]].copy()
    if eligible.empty:
        frame[f"{prefix} RS score"] = None
        frame[f"{prefix} strong sector"] = False
        frame[f"{prefix} final pass"] = False
        return frame

    eligible["RS"] = (
        eligible["Excess return vs Nifty 500 (%)"].rank(pct=True) * 100
    ).round(1)

    sector_perf = (
        eligible[eligible["Yahoo sector"] != "Unknown"]
        .groupby("Yahoo sector")["6M return (%)"].median()
    )
    if sector_perf.empty:
        strong_names = set()
        eligible["Sector median 6M return (%)"] = None
    else:
        strong_names = set(
            sector_perf[sector_perf >= sector_perf.quantile(0.70)].index
        )
        strong_names = {
            name for name in strong_names if sector_perf[name] > benchmark_return
        }
        eligible["Sector median 6M return (%)"] = eligible["Yahoo sector"].map(sector_perf)

    eligible["Strong sector"] = eligible["Yahoo sector"].isin(strong_names)
    eligible["Final pass"] = (
        (eligible["RS"] >= MIN_RS_SCORE) & eligible["Strong sector"]
    )

    rs_map = eligible.set_index("Symbol")["RS"].to_dict()
    sector_map = eligible.set_index("Symbol")["Strong sector"].to_dict()
    median_map = eligible.set_index("Symbol")["Sector median 6M return (%)"].to_dict()
    final_map = eligible.set_index("Symbol")["Final pass"].to_dict()

    frame[f"{prefix} RS score"] = frame["Symbol"].map(rs_map)
    frame[f"{prefix} strong sector"] = frame["Symbol"].map(sector_map).fillna(False)
    frame[f"{prefix} sector median 6M return (%)"] = frame["Symbol"].map(median_map)
    frame[f"{prefix} final pass"] = frame["Symbol"].map(final_map).fillna(False)
    return frame

report = apply_relative_strength_and_sector(
    report, "Momentum core (pre market-cap/RS/sector)", "Momentum"
)
report = apply_relative_strength_and_sector(
    report, "Pullback default (all 3 contraction checks)", "Pullback"
)

report = report.sort_values(
    ["Pullback final pass", "Momentum final pass",
     "Pullback default (all 3 contraction checks)",
     "Momentum core (pre market-cap/RS/sector)"],
    ascending=False,
)
report.to_csv(OUT, index=False)

print("NSE SCANNER VALIDATION — TECHNICAL + MARKET CAP + RS + SECTOR")
print(f"Database date range: {history['DATE'].min().date()} to {history['DATE'].max().date()}")
print(f"EQ symbols in universe: {len(eq_symbols):,}")
print(f"EQ symbols with at least {MIN_HISTORY} sessions: {len(report):,}")
print(f"EQ symbols excluded for insufficient history: {short_history:,}")
print(f"ADR 14D >= {MIN_ADR_PCT:.1f}%: {int(report['ADR >= 3%'].sum()):,}")
print(f"Common price/52W-low/30D-turnover filters: {int(report['Common price/low/turnover filters'].sum()):,}")
print(f"Yahoo metadata candidates requested: {len(candidate_symbols):,}")
print(f"Yahoo metadata candidates with market cap > ₹{MIN_MARKET_CAP_CR:,.0f} Cr: {int(report['Market cap >= ₹1,000 Cr'].sum()):,}")
print(f"Nifty 500 proxy 6M return: {benchmark_return:.2f}%")
print(f"Momentum core (pre market-cap/RS/sector): {int(report['Momentum core (pre market-cap/RS/sector)'].sum()):,}")
print(f"Momentum with optional >= {MOMENTUM_DAILY_VOLUME_MIN_PCT:.0f}% volume: {int(report['Momentum + optional 150% volume'].sum()):,}")
print(f"Momentum final after market cap + RS >= {MIN_RS_SCORE:.0f} + strong sector: {int(report['Momentum final pass'].sum()):,}")
print(f"Pullback setups before contraction: {int(report['Pullback setup before contraction'].sum()):,}")
print(f"Pullback after all 3 default contraction checks: {int(report['Pullback default (all 3 contraction checks)'].sum()):,}")
print(f"Pullback final after market cap + RS >= {MIN_RS_SCORE:.0f} + strong sector: {int(report['Pullback final pass'].sum()):,}")
print("Pullback contraction breakdown:")
print(f"  Latest day volume <= {PULLBACK_DAILY_VOLUME_MAX_PCT:.0f}% of prior 30D average: {int(report['Latest day volume <= 70%'].sum()):,}")
print(f"  5D average volume < prior 20D average: {int(report['5D volume contracting'].sum()):,}")
print(f"  5D average daily range < prior 20D average: {int(report['5D price range contracting'].sum()):,}")
print()
print("AUDIT NOTE: Technical filters use official NSE bhavcopy history. Market cap, sector and benchmark-relative RS use Yahoo Finance metadata/data, matching app.py's current sources and default thresholds. Yahoo metadata may be unavailable for some symbols; those are excluded as in app.py.")
