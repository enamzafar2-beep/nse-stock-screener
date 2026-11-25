import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import math
from functools import lru_cache

import pandas as pd
import streamlit as st
import yfinance as yf  # Retained only for benchmark and company metadata

ROOT = Path(__file__).parent
HISTORY_FILE = ROOT / "daily_scan_results.csv"
UNIVERSE_FILE = ROOT / "universe.csv"
IST = "Asia/Kolkata"
BENCHMARK = "^CRSLDX"  # Yahoo Finance proxy for Nifty 500; NSE stock OHLCV comes from the local database
NSE_HISTORY_FILE = ROOT / "nse_daily_history.csv.gz"
INDUSTRY_FILE = ROOT / "industry_classification.csv"
ADR_SESSIONS = 14
MIN_ADR_MOMENTUM_PCT = 3.0
MIN_ADR_PULLBACK_PCT = 2.0

# The GitHub daily pipeline runs both scan modes and records each separately.

@lru_cache(maxsize=1)
def load_nse_history():
    if not NSE_HISTORY_FILE.exists():
        raise FileNotFoundError("NSE historical database nse_daily_history.csv.gz is missing.")
    frame = pd.read_csv(NSE_HISTORY_FILE, compression="gzip", parse_dates=["DATE"])
    required = {"DATE", "SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "TURNOVER_CRORES"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"NSE history database is missing columns: {sorted(missing)}")
    frame["SYMBOL"] = frame["SYMBOL"].astype(str).str.strip().str.upper()
    frame = frame.sort_values(["SYMBOL", "DATE"])
    return {symbol: group.set_index("DATE")[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "TURNOVER_CRORES"]].astype(float)
            for symbol, group in frame.groupby("SYMBOL", sort=False)}

def load_symbols():
    if UNIVERSE_FILE.exists():
        try:
            frame = pd.read_csv(UNIVERSE_FILE)
            col = next((c for c in frame.columns if str(c).strip().lower() == "symbol"), None)
            if col:
                series_col = next((c for c in frame.columns if str(c).strip().lower() == "series"), None)
                if series_col:
                    frame = frame[frame[series_col].astype(str).str.upper().str.strip() == "EQ"]
                symbols = sorted(set(frame[col].dropna().astype(str).str.upper()
                                     .str.replace(".NS", "", regex=False).str.strip()))
                if symbols:
                    return symbols
        except Exception:
            pass
    return ["RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "INFY", "TCS",
            "LT", "BEL", "HAL", "MAZDOCK", "CDSL", "MCX"]

@lru_cache(maxsize=1)
def load_industry_classification():
    if not INDUSTRY_FILE.exists():
        raise FileNotFoundError("Industry classification database industry_classification.csv is missing.")
    frame = pd.read_csv(INDUSTRY_FILE, dtype={"Symbol": str})
    required = {"Symbol", "Sector", "Industry", "Basic industry"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Industry classification database is missing columns: {sorted(missing)}")
    frame["Symbol"] = frame["Symbol"].astype(str).str.strip().str.upper()
    if frame["Symbol"].duplicated().any():
        duplicates = frame.loc[frame["Symbol"].duplicated(), "Symbol"].head(10).tolist()
        raise ValueError(f"Duplicate symbols in industry classification database: {duplicates}")
    for column in ("Sector", "Industry", "Basic industry"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    return frame.set_index("Symbol")[["Sector", "Industry", "Basic industry"]].to_dict("index")

def download_history(symbol):
    # Prices, volume and turnover are sourced from the verified NSE bhavcopy database.
    return load_nse_history().get(str(symbol).upper())

def series(hist, name):
    value = hist[name]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value.astype(float)

def benchmark_return():
    try:
        hist = yf.download(BENCHMARK, period="6mo", interval="1d",
                           auto_adjust=True, progress=False, threads=False, timeout=25)
        if hist is None or hist.empty:
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        close = series(hist, "Close").dropna()
        if len(close) < 100:
            return None
        return float((close.iloc[-1] / close.iloc[0] - 1) * 100)
    except Exception:
        return None

def build_universe_strength_context(symbols, classification_level, nifty500_return,
                                    min_adr_pct=MIN_ADR_PULLBACK_PCT):
    """
    Calculate RS ranks and group medians from the broad NSE EQ universe,
    before scan-specific EMA, volume, liquidity and market-cap filters.
    Only symbols with 127 sessions and ADR14 >= the shared 2% floor enter this strength universe.
    """
    classification = load_industry_classification()
    group_column = classification_level if classification_level in (
        "Sector", "Industry", "Basic industry"
    ) else "Sector"
    records = []
    histories = load_nse_history()
    for symbol in symbols:
        hist = histories.get(str(symbol).strip().upper())
        if hist is None or len(hist) < 127:
            continue
        close = series(hist, "CLOSE").dropna()
        if len(close) < 127:
            continue
        high = series(hist, "HIGH").reindex(close.index)
        low = series(hist, "LOW").reindex(close.index)
        last = float(close.iloc[-1])
        adr = float((((high - low) / close) * 100).tail(ADR_SESSIONS).mean())
        if not math.isfinite(adr) or adr < min_adr_pct:
            continue
        ret_6m = float((last / float(close.iloc[-127]) - 1) * 100)
        group = classification.get(str(symbol).strip().upper(), {}).get(group_column, "")
        records.append({"Symbol": str(symbol).strip().upper(),
                        "6M return (%)": ret_6m, group_column: group})
    universe = pd.DataFrame(records)
    if universe.empty:
        return {}, {}, {}
    universe["Excess return vs Nifty 500 (%)"] = universe["6M return (%)"] - nifty500_return
    universe["RS score"] = (
        universe["Excess return vs Nifty 500 (%)"].rank(pct=True) * 100
    ).round(1)
    valid_groups = universe[
        universe[group_column].notna()
        & ~universe[group_column].astype(str).str.strip().str.lower().isin(
            ["", "unknown", "unclassified", "nan"]
        )
    ]
    group_medians = valid_groups.groupby(group_column)["6M return (%)"].median()
    if group_medians.empty:
        strong_groups = set()
    else:
        strong_groups = set(group_medians[
            (group_medians >= group_medians.quantile(0.70))
            & (group_medians > nifty500_return)
        ].index)
    rs_map = universe.set_index("Symbol")["RS score"].to_dict()
    return rs_map, group_medians.to_dict(), strong_groups



def build_sector_strength_context(symbols, nifty500_return):
    """
    Sector RS uses every NSE EQ constituent with at least 127 sessions,
    independent of ADR and scan-specific technical/liquidity filters.
    Sector return is the median constituent 6M return; Sector RS is the
    percentile rank of sector excess return versus the Nifty 500 proxy.
    """
    classification = load_industry_classification()
    histories = load_nse_history()
    records = []
    for symbol in symbols:
        key = str(symbol).strip().upper()
        hist = histories.get(key)
        if hist is None or len(hist) < 127:
            continue
        close = series(hist, "CLOSE").dropna()
        if len(close) < 127:
            continue
        sector = classification.get(key, {}).get("Sector", "")
        if not sector or str(sector).strip().lower() in ("unknown", "unclassified", "nan"):
            continue
        ret_6m = float((close.iloc[-1] / float(close.iloc[-127]) - 1) * 100)
        records.append({"Symbol": key, "Sector": sector, "6M return (%)": ret_6m})
    constituents = pd.DataFrame(records)
    if constituents.empty:
        return {}, pd.DataFrame(columns=[
            "Sector", "Constituents", "Median 6M return (%)",
            "Excess return vs Nifty 500 (%)", "Sector RS score"
        ])
    ranking = constituents.groupby("Sector").agg(
        Constituents=("Symbol", "nunique"),
        **{"Median 6M return (%)": ("6M return (%)", "median")}
    ).reset_index()
    ranking["Excess return vs Nifty 500 (%)"] = (
        ranking["Median 6M return (%)"] - nifty500_return
    )
    ranking["Sector RS score"] = (
        ranking["Excess return vs Nifty 500 (%)"].rank(pct=True) * 100
    ).round(1)
    ranking = ranking.sort_values(
        ["Sector RS score", "Excess return vs Nifty 500 (%)"],
        ascending=[False, False]
    ).reset_index(drop=True)
    sector_map = ranking.set_index("Sector")["Sector RS score"].to_dict()
    return sector_map, ranking

def inspect_symbol(symbol, scan_type, min_mcap, min_return, min_price,
                   min_above_low, min_turnover, proximity, use_peg, max_peg,
                   use_volume_filter, volume_threshold_pct,
                   min_adr_pct=MIN_ADR_MOMENTUM_PCT):
    try:
        hist = download_history(symbol)
        # Symbols with fewer than 127 verified sessions are expected exclusions:
        # they cannot support six-month return/RS calculations, but this is not
        # a data-fetch or calculation error and should not inflate the error count.
        if hist is None or len(hist) < 127:
            return None, None
        close = series(hist, "CLOSE").dropna()
        high = series(hist, "HIGH").reindex(close.index)
        low = series(hist, "LOW").reindex(close.index)
        open_price = series(hist, "OPEN").reindex(close.index)
        volume = series(hist, "VOLUME").reindex(close.index).fillna(0)
        last = float(close.iloc[-1])
        low_52w = float(low.tail(252).min())
        ret_6m = float((last / float(close.iloc[-127]) - 1) * 100)
        avg_turnover_cr = float(hist["TURNOVER_CRORES"].reindex(close.index).tail(30).mean())
        latest_volume = int(volume.iloc[-1])
        avg_volume_30d = int(volume.tail(30).mean())
        adr_pct = float((((high - low) / close) * 100).tail(ADR_SESSIONS).mean())
        if not math.isfinite(adr_pct) or adr_pct < min_adr_pct:
            return None, None
        prior_30d_avg_volume = float(volume.iloc[-31:-1].mean())
        volume_ratio_pct = (
            latest_volume / prior_30d_avg_volume * 100
            if prior_30d_avg_volume > 0 else 0.0
        )

        # Compare the latest five sessions with the immediately preceding
        # twenty sessions. These are separate contraction checks from the
        # latest-day 70% volume test.
        avg_volume_5d = float(volume.tail(5).mean())
        prior_20d_avg_volume = float(volume.iloc[-25:-5].mean())
        range_pct = ((high - low) / close) * 100
        avg_range_5d_pct = float(range_pct.tail(5).mean())
        prior_20d_avg_range_pct = float(range_pct.iloc[-25:-5].mean())
        volume_contracting = (
            prior_20d_avg_volume > 0 and avg_volume_5d < prior_20d_avg_volume
        )
        price_range_contracting = (
            prior_20d_avg_range_pct > 0 and avg_range_5d_pct < prior_20d_avg_range_pct
        )

        # Pullbacks require all three quiet/contraction conditions when the
        # pullback volume filter is enabled. Momentum retains its separate
        # latest-day expansion threshold.
        if use_volume_filter:
            if scan_type == "Pullback":
                if (volume_ratio_pct > volume_threshold_pct or
                        not volume_contracting or not price_range_contracting):
                    return None, None
            elif scan_type == "Momentum" and volume_ratio_pct < volume_threshold_pct:
                return None, None

        if last <= min_price or low_52w <= 0 or last / low_52w < min_above_low:
            return None, None
        if avg_turnover_cr < min_turnover:
            return None, None
        # EMA calculations use NSE end-of-day closing prices (raw, not corporate-action adjusted).
        ema = {n: close.ewm(span=n, adjust=False).mean() for n in (10, 20, 50)}
        ema_values = {n: float(v.iloc[-1]) for n, v in ema.items()}
        ema_near = [n for n, value in ema_values.items()
                    if abs(last / value - 1) * 100 <= proximity]
        # Pullback qualification: a substantial prior six-month advance,
        # rising EMA50, price holding close to EMA20/EMA50, and not deeply
        # below EMA50. This deliberately does NOT require today's 6M return
        # to remain above the momentum scanner's 50% threshold.
        six_month_high = float(high.tail(127).max())
        six_month_low = float(low.tail(127).min())
        prior_advance_pct = ((six_month_high / six_month_low) - 1) * 100 if six_month_low > 0 else 0
        ema50_rising = len(ema[50]) >= 11 and float(ema[50].iloc[-1]) > float(ema[50].iloc[-11])
        near_ema20 = abs(last / ema_values[20] - 1) * 100 <= proximity
        near_ema50 = abs(last / ema_values[50] - 1) * 100 <= proximity
        holding_ema50 = last >= ema_values[50] * 0.98
        pullback_ok = (prior_advance_pct >= 50 and ema50_rising and
                       (near_ema20 or near_ema50) and holding_ema50)
        if scan_type == "Momentum":
            if ret_6m < min_return:
                return None, None
        elif not pullback_ok:
            return None, None
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.get_info() or {}
        market_cap = info.get("marketCap")
        market_cap_cr = float(market_cap) / 1e7 if market_cap else None
        if market_cap_cr is None or market_cap_cr <= min_mcap:
            return None, None
        classification = load_industry_classification().get(str(symbol).strip().upper(), {})
        sector = classification.get("Sector") or "Unclassified"
        industry = classification.get("Industry") or "Unclassified"
        basic_industry = classification.get("Basic industry") or "Unclassified"
        peg = info.get("pegRatio")
        try:
            peg = float(peg) if peg is not None else None
            if peg is not None and (not math.isfinite(peg) or peg <= 0):
                peg = None
        except (TypeError, ValueError):
            peg = None
        if use_peg and (peg is None or peg > max_peg):
            return None, None
        if scan_type == "Pullback":
            # A first-pass recovery label, not a guaranteed reversal signal.
            turning_up = (last > float(close.iloc[-2]) and
                          last > float(open_price.iloc[-1]))
            setup = "Pullback turning up" if turning_up else "Near EMA support"
        else:
            setup = "Momentum"
        return {
            "Symbol": symbol,
            "Setup": setup,
            "Price (₹)": last,
            "Market cap (₹ Cr)": market_cap_cr,
            "6M return (%)": ret_6m,
            f"ADR {ADR_SESSIONS}D (%)": round(adr_pct, 2),
            "Prior 6M swing advance (%)": prior_advance_pct,
            "52W low (₹)": low_52w,
            "Above 52W low (x)": last / low_52w,
            "Latest volume (shares)": latest_volume,
            "30D avg volume (shares)": avg_volume_30d,
            "Today's volume vs prior 30D avg (%)": round(volume_ratio_pct, 1),
            "5D avg volume (shares)": round(avg_volume_5d, 0),
            "Prior 20D avg volume (shares)": round(prior_20d_avg_volume, 0),
            "5D vs prior 20D volume (%)": round(avg_volume_5d / prior_20d_avg_volume * 100, 1)
                if prior_20d_avg_volume > 0 else None,
            "5D avg daily range (%)": round(avg_range_5d_pct, 2),
            "Prior 20D avg daily range (%)": round(prior_20d_avg_range_pct, 2),
            "5D vs prior 20D range (%)": round(avg_range_5d_pct / prior_20d_avg_range_pct * 100, 1)
                if prior_20d_avg_range_pct > 0 else None,
            "30D avg turnover (₹ Cr)": avg_turnover_cr,
            "Sector": sector,
            "Industry": industry,
            "Basic industry": basic_industry,
            "PEG": peg,
            "EMA match": ", ".join(str(n) for n, matched in ((20, near_ema20), (50, near_ema50)) if matched)
                         if (near_ema20 or near_ema50) else "—",
            "EMA 10 (₹)": ema_values[10],
            "EMA 20 (₹)": ema_values[20],
            "EMA 50 (₹)": ema_values[50],
            "As of": str(close.index[-1].date()),
        }, None
    except Exception as exc:
        return None, f"{symbol}: {str(exc)[:160]}"

def run_scan(symbols, scan_type, min_mcap=1000, min_return=50, min_price=90,
             min_above_low=1.5, min_turnover=7, min_rs=80,
             strong_sector_only=True, proximity=2, use_peg=False,
             max_peg=2.0, workers=6, use_volume_filter=False,
             volume_threshold_pct=None, classification_level="Sector",
             selected_sectors=None, selected_industries=None,
             min_adr_pct=None, use_sector_rs_filter=False, min_sector_rs=70):
    if volume_threshold_pct is None:
        volume_threshold_pct = 70.0 if scan_type == "Pullback" else 150.0
    if min_adr_pct is None:
        min_adr_pct = MIN_ADR_PULLBACK_PCT if scan_type == "Pullback" else MIN_ADR_MOMENTUM_PCT
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(inspect_symbol, symbol, scan_type, min_mcap, min_return,
                        min_price, min_above_low, min_turnover, proximity,
                        use_peg, max_peg, use_volume_filter,
                        volume_threshold_pct, min_adr_pct): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            row, err = future.result()
            if row:
                rows.append(row)
            if err:
                errors.append(err)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, errors
    if selected_sectors:
        df = df[df["Sector"].isin(selected_sectors)]
    if selected_industries:
        df = df[df["Industry"].isin(selected_industries)]
    if df.empty:
        return df, errors

    nifty500_return = benchmark_return()
    df["Nifty 500 6M return (%)"] = nifty500_return
    if nifty500_return is None:
        errors.append("Nifty 500 benchmark data unavailable; RS score cannot be verified.")
        return pd.DataFrame(), errors

    # Rank against the broad ADR/history-qualified NSE universe, not merely
    # stocks that already passed this scan's technical and liquidity filters.
    rs_map, group_medians, strong_names = build_universe_strength_context(
        symbols, classification_level, nifty500_return, MIN_ADR_PULLBACK_PCT
    )
    sector_rs_map, sector_ranking = build_sector_strength_context(symbols, nifty500_return)
    group_column = classification_level if classification_level in (
        "Sector", "Industry", "Basic industry"
    ) else "Sector"
    df["Excess return vs Nifty 500 (%)"] = df["6M return (%)"] - nifty500_return
    df["RS score"] = df["Symbol"].map(rs_map)
    df["Sector RS score"] = df["Sector"].map(sector_rs_map)
    df["Classification group median 6M return (%)"] = df[group_column].map(group_medians)
    df["Strong group"] = df[group_column].isin(strong_names)
    df = df[df["RS score"].notna() & (df["RS score"] >= min_rs)]
    if use_sector_rs_filter:
        df = df[df["Sector RS score"].notna() & (df["Sector RS score"] >= min_sector_rs)]
    if strong_sector_only:
        df = df[df["Strong group"]]
    df.attrs["sector_rs_ranking"] = sector_ranking
    if not df.empty:
        sort_cols = ["RS score", "Prior 6M swing advance (%)", "30D avg turnover (₹ Cr)"]
        df = df.sort_values(sort_cols, ascending=[False, False, False], na_position="last")
    return df, errors

def save_daily_results(df, symbols, errors, scan_type="Momentum"):
    now = pd.Timestamp.now(tz=IST)
    scan_date = now.strftime("%Y-%m-%d")
    saved = df.copy()
    if saved.empty:
        saved = pd.DataFrame(columns=["Symbol", "Setup", "Price (₹)", "Market cap (₹ Cr)",
                                      "6M return (%)", "RS score", "As of"])
    saved.insert(0, "Scan date", scan_date)
    saved.insert(1, "Scan time IST", now.strftime("%H:%M:%S"))
    saved.insert(2, "Scan type", scan_type)
    saved["Symbols checked"] = len(symbols)
    saved["Scan errors"] = len(errors)
    old = pd.read_csv(HISTORY_FILE) if HISTORY_FILE.exists() else pd.DataFrame()
    if not old.empty and "Scan date" in old:
        # Keep Momentum and Pullback records independently for the same session.
        if "Scan type" not in old:
            old["Scan type"] = "Momentum"
        old = old[~((old["Scan date"].astype(str) == scan_date) &
                    (old["Scan type"].astype(str) == scan_type))]
    pd.concat([old, saved], ignore_index=True).to_csv(HISTORY_FILE, index=False)

# Scheduled GitHub Actions runs both independent strategies against the same
# verified NSE end-of-day database. Each scan is saved separately per date.
if os.getenv("NSE_DAILY_SCAN") == "1":
    symbols = load_symbols()
    failures = []
    for daily_scan_type in ("Momentum", "Pullback"):
        results, errors = run_scan(symbols, daily_scan_type)
        save_daily_results(results, symbols, errors, daily_scan_type)
        failures.extend(errors)
        print(
            f"Daily {daily_scan_type} scan: {len(symbols)} symbols, "
            f"{len(results)} matches, {len(errors)} errors"
        )
    if failures:
        print(f"Total data errors across both scans: {len(failures)}")
    raise SystemExit(0)

st.set_page_config(page_title="NSE Strength & EMA Pullback Scanner",
                   page_icon="📈", layout="wide")
st.title("📈 NSE Strength & EMA Pullback Scanner")
st.caption("End-of-day research scanner for swing and positional setups. Not a live intraday scanner.")
with st.sidebar:
    st.header("Scanner type")
    scan_type = st.radio("Choose what to find", ["Momentum", "Pullback"],
                         help="Pullback finds stocks with a prior strong advance that are now near a rising EMA20 or EMA50.")
    st.header("Core filters")
    min_adr_pct = st.number_input(
        f"Minimum ADR {ADR_SESSIONS}D (%) — {scan_type}",
        min_value=0.0, max_value=15.0,
        value=MIN_ADR_MOMENTUM_PCT if scan_type == "Momentum" else MIN_ADR_PULLBACK_PCT,
        step=0.25,
        help="Separate volatility floor for each scan mode. Pullback defaults lower because the setup already requires price and volume contraction."
    )
    min_mcap = st.number_input("Minimum market cap (₹ crore)", min_value=1, value=1000, step=100)
    min_return = st.number_input("Minimum 6-month return (%)", min_value=0.0, value=50.0, step=5.0,
                                 disabled=(scan_type == "Pullback"))
    min_price = st.number_input("Minimum share price (₹)", min_value=1.0, value=90.0, step=5.0)
    min_above_low = st.number_input("Minimum price / 52-week low", min_value=1.0, value=1.5, step=0.1)
    min_turnover = st.number_input("Minimum 30-day average turnover (₹ crore)", min_value=0.0, value=7.0, step=1.0)
    min_rs = st.slider("Minimum Relative Strength score", 0, 100, 80)
    use_sector_rs_filter = st.checkbox(
        "Apply minimum Sector RS filter",
        value=False,
        help="Sector RS ranks each sector's median 6-month return relative to the Nifty 500 proxy. The ranking is shown even when this filter is off."
    )
    min_sector_rs = st.slider(
        "Minimum Sector RS score", 0, 100, 70,
        disabled=not use_sector_rs_filter
    )
    strong_sector_only = st.checkbox("Only strong groups", value=True)
    classification_level = st.selectbox(
        "Classification level for strength filter",
        ["Sector", "Industry", "Basic industry"],
        index=0
    )
    classification_data = load_industry_classification()
    sector_options = sorted({v["Sector"] for v in classification_data.values() if v["Sector"] and v["Sector"] != "Unclassified"})
    industry_options = sorted({v["Industry"] for v in classification_data.values() if v["Industry"] and v["Industry"] != "Unclassified"})
    selected_sectors = st.multiselect("Limit to sectors (empty = all)", sector_options, default=[])
    selected_industries = st.multiselect("Limit to industries (empty = all)", industry_options, default=[])
    st.header("EMA pullback settings")
    proximity = st.number_input("Distance from EMA20/EMA50 (%)", min_value=0.5, max_value=8.0,
                                value=2.0, step=0.5)
    use_peg = st.checkbox("Enable PEG filter (missing PEG values excluded)", value=False)
    max_peg = st.number_input("Maximum PEG ratio", min_value=0.1, value=2.0, step=0.1)

    st.header("Volume confirmation")
    if scan_type == "Pullback":
        use_volume_filter = st.checkbox(
            "Require pullback volume and price contraction",
            value=True,
            help="When enabled, all three must hold: today's volume <= the selected % of the prior 30-session average; 5-session average volume below the preceding 20-session average; and 5-session average daily range below the preceding 20-session average."
        )
        volume_threshold_pct = st.number_input(
            "Maximum today's volume (% of prior 30-day average)",
            min_value=10.0, max_value=100.0, value=70.0, step=5.0,
            disabled=not use_volume_filter
        )
    else:
        use_volume_filter = st.checkbox(
            "Require momentum volume expansion (optional)",
            value=False,
            help="When enabled, today's volume must be at least the selected percentage of the previous 30 trading days' average."
        )
        volume_threshold_pct = st.number_input(
            "Minimum today's volume (% of prior 30-day average)",
            min_value=100.0, max_value=500.0, value=150.0, step=10.0,
            disabled=not use_volume_filter
        )
    workers = st.slider("Concurrent requests", 1, 12, 6)

st.subheader("1. NSE universe")
symbols = load_symbols()
st.write(f"**{len(symbols):,} symbols loaded** from `universe.csv`.")
classification_count = len(load_industry_classification())
classified_universe = sum(1 for symbol in symbols if symbol in load_industry_classification())
st.caption(f"Sector / industry classification: {classified_universe:,} of {len(symbols):,} loaded symbols ({classified_universe / max(len(symbols), 1) * 100:.1f}%).")
if st.button("↻ Reload saved NSE universe", use_container_width=True):
    st.rerun()
if st.button("🔎 Run end-of-day scanner", type="primary", use_container_width=True):
    if not symbols:
        st.warning("No symbols found. Check universe.csv in your GitHub repository.")
    else:
        with st.spinner(f"Scanning {len(symbols):,} symbols. This may take some time..."):
            results, errors = run_scan(
                symbols, scan_type, min_mcap, min_return, min_price, min_above_low,
                min_turnover, min_rs, strong_sector_only, proximity, use_peg,
                max_peg, workers, use_volume_filter, volume_threshold_pct,
                classification_level, selected_sectors, selected_industries,
                min_adr_pct, use_sector_rs_filter, min_sector_rs
            )
            sector_ranking = results.attrs.get("sector_rs_ranking", pd.DataFrame())
            save_daily_results(results, symbols, errors, scan_type)
            st.session_state["latest_results"] = results
            st.session_state["latest_sector_rs"] = sector_ranking
            st.session_state["latest_errors"] = errors
            st.session_state["latest_scan_type"] = scan_type
        st.success(f"{scan_type} scan completed: {len(results)} matches; {len(errors)} data errors.")

if "latest_results" in st.session_state:
    sector_ranking = st.session_state.get("latest_sector_rs", pd.DataFrame())
    st.subheader("Sector Relative Strength ranking")
    if not sector_ranking.empty:
        st.dataframe(sector_ranking, use_container_width=True, hide_index=True)
    st.subheader(f"2. Latest {st.session_state.get('latest_scan_type', '')} scan results")
    latest = st.session_state["latest_results"]
    if latest.empty:
        st.info("No stocks matched all selected filters. You can review the filters, or try a later scan.")
    else:
        st.dataframe(latest, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download latest results CSV",
                           latest.to_csv(index=False).encode("utf-8"),
                           "nse_scanner_results.csv", "text/csv")
    errs = st.session_state.get("latest_errors", [])
    if errs:
        with st.expander(f"Data errors ({len(errs)})"):
            st.write(errs[:100])

st.divider()
st.subheader("3. Saved daily scan history")
if HISTORY_FILE.exists():
    try:
        history = pd.read_csv(HISTORY_FILE)
        if not history.empty and "Scan date" in history:
            dates = sorted(history["Scan date"].dropna().astype(str).unique(), reverse=True)
            selected = st.selectbox("Choose scan date", dates)
            day = history[history["Scan date"].astype(str) == selected].copy()
            a, b, c = st.columns(3)
            a.metric("Matching stocks", len(day))
            b.metric("Symbols checked", int(day["Symbols checked"].max()) if "Symbols checked" in day and len(day) else 0)
            c.metric("Scan date", selected)
            if not day.empty:
                st.dataframe(day.drop(columns=["Scan date", "Scan time IST"], errors="ignore"),
                             use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download selected day's results",
                                   day.to_csv(index=False).encode("utf-8"),
                                   f"nse_scan_{selected}.csv", "text/csv")
        else:
            st.info("No saved daily scans yet.")
    except Exception as exc:
        st.warning(f"Could not read saved history: {exc}")
else:
    st.info("No saved daily scans yet.")

st.divider()
st.subheader("4. How the scanner works")
st.markdown("""
- **Momentum mode:** keeps the existing minimum six-month return of 50% by default.
- **Pullback mode:** does not require the current six-month return to remain above 50%. Instead, the stock must have made at least a 50% rise from its six-month low to its six-month high, have a rising EMA50, and be within the chosen distance (2% by default) of EMA20 or EMA50 while remaining no more than 2% below EMA50.
- **Pullback labels:** “Near EMA support” means the stock is in the pullback zone. “Pullback turning up” means the latest close is above the previous close and above the day's open. These are screening labels, not confirmed buy signals.
- **Sector Relative Strength:** each sector's median six-month constituent return minus the Nifty 500 proxy return, ranked by percentile across sectors. It uses all history-qualified NSE EQ constituents (not the ADR filter) and is displayed strongest to weakest; its minimum-score filter is optional and off by default.\n- **Relative Strength:** percentile rank of six-month excess return versus the Nifty 500 proxy series `^CRSLDX`, among the broad NSE EQ universe with at least 127 sessions and ADR14 of at least 2%, before scan-specific technical and liquidity filters.
- **Sector and industry classification:** sourced from the reviewed `industry_classification.csv` database. Optional sector and industry selections narrow the universe; the strong-group filter can evaluate Sector, Industry, or Basic industry median six-month returns (top 30% and above the benchmark).
- **Liquidity and quality:** close above ₹90, at least 1.5× the 52-week low, 30-day average daily turnover at least ₹7 crore, and market cap above ₹1,000 crore by default.
- **ADR:** average daily high-low range as a percentage of close over 14 sessions. The default minimum is 3% for Momentum and 2% for Pullback; each is independently adjustable. Pullback has a lower floor because its separate setup and contraction checks already seek quieter price/volume behaviour. ADR is a historical volatility eligibility filter, not a requirement for today's pullback candle.
- **PEG:** optional; missing PEG values are excluded when the filter is enabled.
- **Pullback contraction confirmation (enabled by default):** all three conditions must hold: today's volume is no more than 70% by default of the preceding 30-session average (latest day excluded from that baseline); average volume over the latest 5 sessions is below the preceding 20-session average; and average daily high-low range over the latest 5 sessions is below the preceding 20-session average. Disable this option only if you want to see EMA pullbacks without these contraction filters. The range comparison measures price contraction, not direction.
- **Momentum volume confirmation:** optional and separate. When enabled, today's volume must be at least 150% by default of the preceding 30 trading days' average.
- All calculations use daily end-of-day data. Intraday screening and backtesting remain future stages.
**Data sources:** Daily stock OHLCV and turnover are read from the saved NSE Full Bhavcopy historical database. The Nifty 500 proxy return and company market-cap, sector and PEG metadata still use Yahoo Finance; these fields can be missing or stale. Verify candidates with NSE and your broker. This is a research screener, not investment advice.
""")
