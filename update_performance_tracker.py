"""Maintain a forward-performance ledger for historical NSE scanner candidates.

The ledger is rebuilt from the append-only scan history and verified NSE OHLCV
database on every daily pipeline run. Outcomes use only sessions after the
candidate's signal date; incomplete horizons remain pending.
"""
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).parent
SCANS = ROOT / "daily_scan_results.csv"
HISTORY = ROOT / "nse_daily_history.csv.gz"
OUTPUT = ROOT / "performance_tracker.csv"
BENCHMARK = "^CRSLDX"


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def load_benchmark(start, end):
    """Best-effort historical Nifty 500 proxy; never fail the NSE pipeline."""
    try:
        import yfinance as yf
        frame = yf.download(BENCHMARK, start=start, end=end, auto_adjust=True,
                            progress=False, threads=False, timeout=20)
        if frame is None or frame.empty:
            return {}
        close = frame["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        return close.dropna().astype(float).to_dict()
    except Exception as exc:
        print(f"Benchmark history unavailable; relative returns left blank: {exc}")
        return {}


def main():
    if not SCANS.exists() or not HISTORY.exists():
        raise SystemExit("Required scan results or verified NSE history file is missing.")

    scans = pd.read_csv(SCANS)
    required_scan = {"Symbol", "Scan type", "As of", "Scan date", "Price (₹)"}
    missing = required_scan - set(scans.columns)
    if missing:
        raise SystemExit(f"Scan history missing columns: {sorted(missing)}")

    scans["Symbol"] = scans["Symbol"].astype(str).str.strip().str.upper()
    scans["Scan type"] = scans["Scan type"].astype(str).str.strip()
    scans["Signal date"] = pd.to_datetime(scans["As of"], errors="coerce")
    fallback = pd.to_datetime(scans["Scan date"], errors="coerce")
    scans["Signal date"] = scans["Signal date"].fillna(fallback).dt.normalize()
    scans = scans[scans["Scan type"].isin(["Momentum", "Pullback"])]
    scans = scans[scans["Signal date"].notna() & scans["Symbol"].ne("")]
    scans = scans.drop_duplicates(["Signal date", "Scan type", "Symbol"], keep="first")

    history = pd.read_csv(HISTORY, compression="gzip", parse_dates=["DATE"])
    required_history = {"DATE", "SYMBOL", "HIGH", "LOW", "CLOSE"}
    missing = required_history - set(history.columns)
    if missing:
        raise SystemExit(f"NSE history missing columns: {sorted(missing)}")
    history["SYMBOL"] = history["SYMBOL"].astype(str).str.strip().str.upper()
    history["DATE"] = pd.to_datetime(history["DATE"]).dt.normalize()
    history = history.sort_values(["SYMBOL", "DATE"]).drop_duplicates(
        ["SYMBOL", "DATE"], keep="last"
    )
    by_symbol = {s: g.set_index("DATE").sort_index()
                 for s, g in history.groupby("SYMBOL", sort=False)}

    latest = history["DATE"].max()
    first_signal = scans["Signal date"].min()
    benchmark = load_benchmark(
        str(first_signal.date()), str((latest + pd.Timedelta(days=3)).date())
    )
    records = []
    for row in scans.to_dict("records"):
        symbol = row["Symbol"]
        signal_date = row["Signal date"]
        prices = by_symbol.get(symbol)
        entry = None
        entry_source = "Missing"
        future = pd.DataFrame()
        if prices is not None:
            if signal_date in prices.index:
                entry = _number(prices.loc[signal_date, "CLOSE"])
                entry_source = "NSE close"
            future = prices.loc[prices.index > signal_date].head(20)
        if entry is None:
            entry = _number(row.get("Price (₹)"))
            if entry is not None:
                entry_source = "Scanner price fallback"

        item = {
            "Signal date": signal_date.date().isoformat(),
            "Scan type": row["Scan type"],
            "Symbol": symbol,
            "Entry price (₹)": entry,
            "Entry price source": entry_source,
            "Sector": row.get("Sector", ""),
            "Industry": row.get("Industry", ""),
            "RS score": _number(row.get("RS score")),
            "Sector RS score": _number(row.get("Sector RS score")),
            "Strong sector": row.get("Strong sector", ""),
            "Latest available NSE session": latest.date().isoformat(),
        }
        if entry is not None and entry > 0 and not future.empty:
            for horizon in (5, 10, 20):
                subset = future.head(horizon)
                complete = len(subset) >= horizon
                prefix = f"{horizon}D"
                item[f"{prefix} status"] = "Complete" if complete else "Pending"
                item[f"{prefix} session date"] = (
                    subset.index[-1].date().isoformat() if complete else ""
                )
                if complete:
                    close_return = (float(subset["CLOSE"].iloc[-1]) / entry - 1) * 100
                    mfe = (float(subset["HIGH"].max()) / entry - 1) * 100
                    mae = (float(subset["LOW"].min()) / entry - 1) * 100
                    item[f"{prefix} return (%)"] = round(close_return, 2)
                    item[f"{prefix} max favourable (%)"] = round(mfe, 2)
                    item[f"{prefix} max adverse (%)"] = round(mae, 2)
                    b_entry = benchmark.get(signal_date)
                    b_exit = benchmark.get(subset.index[-1])
                    if b_entry and b_exit:
                        b_return = (float(b_exit) / float(b_entry) - 1) * 100
                        item[f"{prefix} Nifty 500 proxy (%)"] = round(b_return, 2)
                        item[f"{prefix} excess vs Nifty 500 (%)"] = round(
                            close_return - b_return, 2
                        )
                    else:
                        item[f"{prefix} Nifty 500 proxy (%)"] = None
                        item[f"{prefix} excess vs Nifty 500 (%)"] = None
                else:
                    for suffix in ("return (%)", "max favourable (%)",
                                   "max adverse (%)", "Nifty 500 proxy (%)",
                                   "excess vs Nifty 500 (%)"):
                        item[f"{prefix} {suffix}"] = None
            item["Sessions elapsed"] = len(future)
        else:
            item["Sessions elapsed"] = 0
            for horizon in (5, 10, 20):
                prefix = f"{horizon}D"
                item[f"{prefix} status"] = "Pending"
                item[f"{prefix} session date"] = ""
                for suffix in ("return (%)", "max favourable (%)",
                               "max adverse (%)", "Nifty 500 proxy (%)",
                               "excess vs Nifty 500 (%)"):
                    item[f"{prefix} {suffix}"] = None
        records.append(item)

    output = pd.DataFrame(records)
    if not output.empty:
        output = output.sort_values(
            ["Signal date", "Scan type", "Symbol"], ascending=[False, True, True]
        )
    output.to_csv(OUTPUT, index=False)
    complete_5 = int((output["5D status"] == "Complete").sum()) if not output.empty else 0
    complete_10 = int((output["10D status"] == "Complete").sum()) if not output.empty else 0
    complete_20 = int((output["20D status"] == "Complete").sum()) if not output.empty else 0
    print("FORWARD PERFORMANCE TRACKER")
    print(f"Unique candidate appearances: {len(output):,}")
    print(f"Completed 5/10/20-session outcomes: {complete_5}/{complete_10}/{complete_20}")
    print(f"Latest verified NSE session: {latest.date()}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
