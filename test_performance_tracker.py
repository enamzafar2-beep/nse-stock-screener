import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import update_performance_tracker as tracker


class PerformanceTrackerTests(unittest.TestCase):
    def test_deduplicates_signals_and_computes_forward_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scans_path = root / "daily_scan_results.csv"
            history_path = root / "nse_daily_history.csv.gz"
            output_path = root / "performance_tracker.csv"

            scans = pd.DataFrame([
                {"Scan date": "2026-01-02", "As of": "2026-01-02",
                 "Scan type": "Momentum", "Symbol": "ABC", "Price (₹)": 100,
                 "Sector": "Industrials", "Industry": "Engineering",
                 "RS score": 90, "Sector RS score": 80},
                {"Scan date": "2026-01-02", "As of": "2026-01-02",
                 "Scan type": "Momentum", "Symbol": "ABC", "Price (₹)": 100,
                 "Sector": "Industrials", "Industry": "Engineering",
                 "RS score": 90, "Sector RS score": 80},
            ])
            scans.to_csv(scans_path, index=False)

            dates = pd.bdate_range("2026-01-02", periods=21)
            rows = []
            for i, day in enumerate(dates):
                rows.append({"DATE": day, "SYMBOL": "ABC",
                             "HIGH": 101 + i, "LOW": 99 - i * 0.1,
                             "CLOSE": 100 + i})
            pd.DataFrame(rows).to_csv(history_path, index=False, compression="gzip")

            with patch.object(tracker, "SCANS", scans_path), \
                 patch.object(tracker, "HISTORY", history_path), \
                 patch.object(tracker, "OUTPUT", output_path), \
                 patch.object(tracker, "load_benchmark", return_value={}):
                tracker.main()

            result = pd.read_csv(output_path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.loc[0, "5D status"], "Complete")
            self.assertEqual(result.loc[0, "5D return (%)"], 5.0)
            self.assertEqual(result.loc[0, "5D max favourable (%)"], 6.0)
            self.assertEqual(result.loc[0, "5D max adverse (%)"], -1.4)
            self.assertEqual(result.loc[0, "20D status"], "Complete")

    def test_incomplete_horizon_is_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scans_path = root / "daily_scan_results.csv"
            history_path = root / "nse_daily_history.csv.gz"
            output_path = root / "performance_tracker.csv"
            pd.DataFrame([{
                "Scan date": "2026-01-02", "As of": "2026-01-02",
                "Scan type": "Pullback", "Symbol": "XYZ", "Price (₹)": 50,
            }]).to_csv(scans_path, index=False)
            pd.DataFrame([
                {"DATE": day, "SYMBOL": "XYZ", "HIGH": 51,
                 "LOW": 49, "CLOSE": 50}
                for day in pd.bdate_range("2026-01-02", periods=4)
            ]).to_csv(history_path, index=False, compression="gzip")

            with patch.object(tracker, "SCANS", scans_path), \
                 patch.object(tracker, "HISTORY", history_path), \
                 patch.object(tracker, "OUTPUT", output_path), \
                 patch.object(tracker, "load_benchmark", return_value={}):
                tracker.main()

            result = pd.read_csv(output_path)
            self.assertEqual(result.loc[0, "5D status"], "Pending")
            self.assertTrue(pd.isna(result.loc[0, "5D return (%)"]))


if __name__ == "__main__":
    unittest.main()
