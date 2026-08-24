"""
Unit and integration tests for the reporting and Telegram notification module (reporting.py).
Can be executed on-demand via:
    python -m unittest test_reporting.py
"""

import io
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# Add pipeline directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_core.db import create_all_tables
from reporting import (
    fetch_recent_portfolio_values,
    fetch_recent_var_metrics,
    fetch_top_risk_contributors,
    fetch_dividends_for_date,
    fetch_top_position_movers,
    fetch_available_dates,
    generate_portfolio_report_chart,
    format_telegram_caption,
    send_telegram_photo,
    send_telegram_report,
    generate_report,
    generate_reports_for_dates,
)


class TestReportingModule(unittest.TestCase):
    """Unit tests for reporting functions, chart generation, and Telegram notification delivery."""

    def setUp(self):
        # Create an in-memory SQLite database for isolated database tests
        self.engine = create_engine("sqlite:///:memory:")
        create_all_tables(self.engine)

        # Populate sample portfolio values
        dates = pd.date_range("2026-08-01", periods=10, freq="B").strftime("%Y-%m-%d")
        with self.engine.begin() as conn:
            for i, d in enumerate(dates):
                val = 100000.0 + (i * 1500.0)
                stk = val * 0.95
                csh = val * 0.05
                conn.execute(
                    text("INSERT INTO PORTFOLIO_VALUES (DATE, TOTAL_VALUE, STOCKS, CASH, CURRENCY) VALUES (:d, :tot, :stk, :csh, 'GBP')"),
                    {"d": d, "tot": val, "stk": stk, "csh": csh}
                )

                # Populate sample VaR records for 1%, 5%, 95%, 99% percentiles
                for method in ["Historical Simulation", "Vol-Scaled VaR (EWMA Volatility (λ=0.94))"]:
                    for cl in [0.01, 0.05, 0.95, 0.99]:
                        if cl in [0.01, 0.05]:
                            var_gbp = 1500.0 * (2.0 if cl == 0.01 else 1.2)
                        else:
                            var_gbp = -1500.0 * (2.0 if cl == 0.99 else 1.0)
                        conn.execute(
                            text("INSERT INTO PORTFOLIO_VAR (DATE, METHOD, CONFIDENCE_LEVEL, HORIZON_DAYS, PORTFOLIO_VALUE_GBP, VAR_GBP, VAR_PCT) VALUES (:d, :m, :cl, 1, :pv, :v, :vp)"),
                            {"d": d, "m": method, "cl": cl, "pv": val, "v": var_gbp, "vp": (var_gbp / val) * 100.0}
                        )

            # Populate sample risk contributions for latest date
            latest_d = dates[-1]
            sample_tickers = [("NVDA", 40000.0, 40.0, -800.0, 50.0), ("STAN.L", 30000.0, 30.0, -500.0, 30.0), ("EMIM.L", 15000.0, 15.0, -200.0, 12.0), ("IUIT.L", 10000.0, 10.0, -100.0, 6.0), ("SJPA.L", 5000.0, 5.0, -40.0, 2.0)]
            for ticker, pos_v, wt, s_var, s_pct in sample_tickers:
                conn.execute(
                    text("""
                        INSERT INTO PORTFOLIO_RISK_CONTRIBUTIONS (
                            DATE, TICKER, METHOD, CONFIDENCE_LEVEL, HORIZON_DAYS,
                            POSITION_VALUE_GBP, WEIGHT_PCT, SHAPLEY_VAR_GBP, SHAPLEY_VAR_PCT,
                            SHAPLEY_CVAR_GBP, SHAPLEY_CVAR_PCT, STANDALONE_VAR_GBP, DIVERSIFICATION_BENEFIT_GBP
                        ) VALUES (
                            :d, :t, 'Vol-Scaled VaR (EWMA Volatility (λ=0.94))', 0.99, 1,
                            :pv, :wt, :sv, :spct, :scv, :spct, :st, :div
                        )
                    """),
                    {
                        "d": latest_d, "t": ticker, "pv": pos_v, "wt": wt,
                        "sv": s_var, "spct": s_pct, "scv": s_var * 1.5,
                        "st": s_var * 1.3, "div": abs(s_var) * 0.3
                    }
                )

    def test_fetch_recent_portfolio_values(self):
        """Test retrieving recent portfolio valuation records chronologically."""
        df = fetch_recent_portfolio_values(days=5, engine=self.engine)
        self.assertEqual(len(df), 5)
        self.assertIn("TOTAL_VALUE", df.columns)
        self.assertIn("STOCKS", df.columns)
        self.assertIn("CASH", df.columns)
        self.assertTrue(df["DATE"].is_monotonic_increasing)

    def test_fetch_recent_var_metrics(self):
        """Test retrieving recent VaR metrics filtered by confidence levels."""
        df = fetch_recent_var_metrics(days=7, confidence_levels=[0.01, 0.05, 0.95, 0.99], engine=self.engine)
        self.assertGreater(len(df), 0)
        self.assertTrue(set(df["CONFIDENCE_LEVEL"].unique()).issubset({0.01, 0.05, 0.95, 0.99}))
        self.assertIn("Historical Simulation", df["METHOD"].values)
        self.assertIn("Vol-Scaled VaR (EWMA Volatility (λ=0.94))", df["METHOD"].values)

    def test_fetch_top_risk_contributors(self):
        """Test retrieving top risk contributors ordered by absolute Shapley contribution."""
        top_df = fetch_top_risk_contributors(top_n=5, engine=self.engine)
        self.assertEqual(len(top_df), 5)
        self.assertEqual(top_df["TICKER"].iloc[0], "NVDA")
        self.assertIn("SHAPLEY_VAR_GBP", top_df.columns)
        self.assertIn("WEIGHT_PCT", top_df.columns)

    def test_generate_portfolio_report_chart_with_table(self):
        """Test 3-panel chart generation including top risk contributors table produces valid PNG image bytes."""
        p_df = fetch_recent_portfolio_values(days=7, engine=self.engine)
        v_df = fetch_recent_var_metrics(days=10, engine=self.engine)
        top_df = fetch_top_risk_contributors(top_n=5, engine=self.engine)

        chart_bytes = generate_portfolio_report_chart(p_df, v_df, top_contributors_df=top_df)
        self.assertIsInstance(chart_bytes, bytes)
        self.assertGreater(len(chart_bytes), 1000)
        # PNG signature check (\x89PNG\r\n\x1a\n)
        self.assertTrue(chart_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_generate_chart_empty_data(self):
        """Test chart generation handles empty DataFrames gracefully without raising errors."""
        empty_p = pd.DataFrame(columns=["DATE", "TOTAL_VALUE", "STOCKS", "CASH", "CURRENCY"])
        empty_v = pd.DataFrame(columns=["DATE", "METHOD", "CONFIDENCE_LEVEL", "PORTFOLIO_VALUE_GBP", "VAR_GBP", "VAR_PCT"])

        chart_bytes = generate_portfolio_report_chart(empty_p, empty_v)
        self.assertIsInstance(chart_bytes, bytes)
        self.assertTrue(chart_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_format_telegram_caption_with_top_contributors(self):
        """Test HTML caption formatting includes stock holdings valuation, changes, VaR breakdown, and top contributors."""
        p_df = fetch_recent_portfolio_values(days=7, engine=self.engine)
        v_df = fetch_recent_var_metrics(days=7, engine=self.engine)
        top_df = fetch_top_risk_contributors(top_n=5, engine=self.engine)

        caption = format_telegram_caption(p_df, v_df, top_contributors_df=top_df)
        self.assertIn("Daily Stock Holdings & Risk Report", caption)
        self.assertIn("Stock Holdings Valuation:", caption)
        self.assertIn("Stock Holdings:", caption)
        self.assertIn("Historical Simulation Percentiles:", caption)
        self.assertIn("Vol-Scaled VaR Percentiles:", caption)
        self.assertIn("1% Percentile:", caption)
        self.assertIn("5% Percentile:", caption)
        self.assertIn("95% Percentile:", caption)
        self.assertIn("99% Percentile:", caption)
        self.assertIn("Top 5 Vol-Scaled VaR Risk Contributors:", caption)
        self.assertIn("NVDA", caption)
        self.assertIn("STAN.L", caption)

    def test_fetch_dividends_for_date(self):
        """Test retrieving dividend cashflows for a specific date from CASHFLOWS table."""
        # Insert sample dividend records
        test_date = "2026-08-14"
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO CASHFLOWS (DATE, TICKER, TYPE, SHARES, DIVIDEND_PER_SHARE, AMOUNT, CURRENCY, AMOUNT_GBP)
                    VALUES
                    (:d, 'AAPL', 'DIVIDEND', 100.0, 0.25, 25.00, 'USD', 19.50),
                    (:d, 'BP.L', 'DIVIDEND', 500.0, 6.00, 3000.00, 'GBp', 30.00)
                """),
                {"d": test_date}
            )

        div_df = fetch_dividends_for_date(asof_date=test_date, engine=self.engine)
        self.assertEqual(len(div_df), 2)
        self.assertIn("TICKER", div_df.columns)
        self.assertIn("AMOUNT_GBP", div_df.columns)
        self.assertEqual(div_df.iloc[0]["TICKER"], "BP.L")  # Ordered by AMOUNT_GBP desc
        self.assertEqual(div_df.iloc[1]["TICKER"], "AAPL")

    def test_fetch_dividends_for_date_empty(self):
        """Test retrieving dividends for a date with no dividend payments returns empty DataFrame."""
        div_df = fetch_dividends_for_date(asof_date="1999-01-01", engine=self.engine)
        self.assertTrue(div_df.empty)
        self.assertIn("TICKER", div_df.columns)

    def test_format_telegram_caption_with_dividends(self):
        """Test HTML caption formatting includes dividend section with translations into GBP."""
        p_df = fetch_recent_portfolio_values(days=7, engine=self.engine)
        v_df = fetch_recent_var_metrics(days=7, engine=self.engine)
        div_df = pd.DataFrame([
            {"DATE": "2026-08-14", "TICKER": "AAPL", "TYPE": "DIVIDEND", "SHARES": 100.0, "DIVIDEND_PER_SHARE": 0.25, "AMOUNT": 25.0, "CURRENCY": "USD", "AMOUNT_GBP": 19.50},
            {"DATE": "2026-08-14", "TICKER": "BP.L", "TYPE": "DIVIDEND", "SHARES": 500.0, "DIVIDEND_PER_SHARE": 6.00, "AMOUNT": 3000.0, "CURRENCY": "GBp", "AMOUNT_GBP": 30.00},
            {"DATE": "2026-08-14", "TICKER": "SAN.MC", "TYPE": "DIVIDEND", "SHARES": 200.0, "DIVIDEND_PER_SHARE": 0.10, "AMOUNT": 20.0, "CURRENCY": "EUR", "AMOUNT_GBP": 17.20},
            {"DATE": "2026-08-14", "TICKER": "AZN.L", "TYPE": "DIVIDEND", "SHARES": 50.0, "DIVIDEND_PER_SHARE": 1.50, "AMOUNT": 75.0, "CURRENCY": "GBP", "AMOUNT_GBP": 75.00},
        ])

        caption = format_telegram_caption(p_df, v_df, dividends_df=div_df)
        self.assertIn("Dividends Paid Today:", caption)
        self.assertIn("AAPL", caption)
        self.assertIn("$25.00", caption)
        self.assertIn("£19.50", caption)
        self.assertIn("BP.L", caption)
        self.assertIn("3,000p", caption)
        self.assertIn("£30.00", caption)
        self.assertIn("SAN.MC", caption)
        self.assertIn("€20.00", caption)
        self.assertIn("£17.20", caption)
        self.assertIn("AZN.L", caption)
        self.assertIn("£75.00", caption)
        self.assertIn("Total Dividends:", caption)
        self.assertIn("£141.70", caption)

    def test_format_telegram_caption_without_dividends(self):
        """Test HTML caption formatting displays 'None' when no dividends were paid on reporting day."""
        p_df = fetch_recent_portfolio_values(days=7, engine=self.engine)
        v_df = fetch_recent_var_metrics(days=7, engine=self.engine)
        empty_div = pd.DataFrame()

        caption = format_telegram_caption(p_df, v_df, dividends_df=empty_div)
        self.assertIn("Dividends Paid Today:", caption)
        self.assertIn("None", caption)

    def test_fetch_top_position_movers(self):
        """Test retrieving top daily position value movers ranked by absolute change in value."""
        # Insert sample transactions and asset prices for 2 dates
        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO TRANSACTIONS (TICKER, TRANSACTION_DATE, QUANTITY)
                    VALUES
                    ('NVDA', '2026-08-01', 100.0),
                    ('TSLA', '2026-08-01', 50.0),
                    ('AAPL', '2026-08-01', 200.0)
                """)
            )
            # Date 1: 2026-08-13
            conn.execute(
                text("""
                    INSERT INTO ASSET_PRICES (DATE, TICKER, CLOSE, CURRENCY)
                    VALUES
                    ('2026-08-13', 'NVDA', 100.0, 'GBP'),
                    ('2026-08-13', 'TSLA', 200.0, 'GBP'),
                    ('2026-08-13', 'AAPL', 50.0, 'GBP')
                """)
            )
            # Date 2: 2026-08-14
            conn.execute(
                text("""
                    INSERT INTO ASSET_PRICES (DATE, TICKER, CLOSE, CURRENCY)
                    VALUES
                    ('2026-08-14', 'NVDA', 110.0, 'GBP'),
                    ('2026-08-14', 'TSLA', 180.0, 'GBP'),
                    ('2026-08-14', 'AAPL', 51.0, 'GBP')
                """)
            )

        movers_df = fetch_top_position_movers(top_n=5, asof_date="2026-08-14", engine=self.engine)
        self.assertEqual(len(movers_df), 3)
        # TSLA moved -£1,000 (abs £1,000), NVDA moved +£1,000 (abs £1,000), AAPL moved +£200 (abs £200)
        self.assertEqual(movers_df["ABS_DIFF_GBP"].iloc[0], 1000.0)
        self.assertEqual(movers_df["ABS_DIFF_GBP"].iloc[1], 1000.0)
        self.assertEqual(movers_df["ABS_DIFF_GBP"].iloc[2], 200.0)
        self.assertIn("DIFF_PCT", movers_df.columns)
        self.assertIn("VALUE_TODAY_GBP", movers_df.columns)

    def test_fetch_top_position_movers_empty(self):
        """Test retrieving top movers with insufficient price history returns empty DataFrame."""
        empty_engine = create_engine("sqlite:///:memory:")
        create_all_tables(empty_engine)
        movers_df = fetch_top_position_movers(top_n=5, engine=empty_engine)
        self.assertTrue(movers_df.empty)
        self.assertIn("TICKER", movers_df.columns)

    def test_format_telegram_caption_with_top_movers(self):
        """Test HTML caption formatting includes top daily value movers section."""
        p_df = fetch_recent_portfolio_values(days=7, engine=self.engine)
        v_df = fetch_recent_var_metrics(days=7, engine=self.engine)
        movers_df = pd.DataFrame([
            {"TICKER": "NVDA", "VALUE_TODAY_GBP": 11000.0, "VALUE_PREV_GBP": 10000.0, "DIFF_GBP": 1000.0, "DIFF_PCT": 10.0, "ABS_DIFF_GBP": 1000.0},
            {"TICKER": "TSLA", "VALUE_TODAY_GBP": 9000.0, "VALUE_PREV_GBP": 10000.0, "DIFF_GBP": -1000.0, "DIFF_PCT": -10.0, "ABS_DIFF_GBP": 1000.0},
        ])

        caption = format_telegram_caption(p_df, v_df, top_movers_df=movers_df)
        self.assertIn("Top 2 Daily Value Movers (|ΔValue|):", caption)
        self.assertIn("NVDA", caption)
        self.assertIn("£11,000.00 (+£1,000.00 / +10.00%)", caption)
        self.assertIn("TSLA", caption)
        self.assertIn("£9,000.00 (-£1,000.00 / -10.00%)", caption)

    @patch("requests.post")
    def test_send_telegram_photo_success(self, mock_post):
        """Test successful HTTP POST request to Telegram sendPhoto API."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": {"message_id": 42}}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        res = send_telegram_photo(
            token="dummy_token_123",
            chat_id="987654321",
            photo_bytes=b"\x89PNG\r\n\x1a\nfake_image_content",
            caption="<b>Test Report</b>"
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["result"]["message_id"], 42)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn("https://api.telegram.org/botdummy_token_123/sendPhoto", args[0])
        self.assertEqual(kwargs["data"]["chat_id"], "987654321")

    @patch("requests.post")
    def test_send_telegram_photo_long_caption(self, mock_post):
        """Test caption exceeding 1024 chars sends photo and delivers full caption via sendMessage."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": {"message_id": 43}}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        long_caption = "<b>Long Report</b>\n" + ("x" * 1100)
        res = send_telegram_photo(
            token="dummy_token_123",
            chat_id="987654321",
            photo_bytes=b"\x89PNG\r\n\x1a\nfake_image_content",
            caption=long_caption
        )
        self.assertTrue(res["ok"])
        self.assertEqual(mock_post.call_count, 2)
        # First call was sendPhoto, second call was sendMessage
        self.assertIn("sendPhoto", mock_post.call_args_list[0][0][0])
        self.assertIn("sendMessage", mock_post.call_args_list[1][0][0])

    @patch("requests.post")
    def test_send_telegram_report_broadcast(self, mock_post):
        """Test full send_telegram_report broadcasting to multiple recipient IDs."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": {"message_id": 101}}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        recipients = ["111222333", "444555666"]
        result = send_telegram_report(
            recipients=recipients,
            token="test_token_xyz",
            engine=self.engine
        )

        self.assertEqual(result["status"], "Delivered")
        self.assertEqual(result["recipients_total"], 2)
        self.assertEqual(len(result["delivered_recipients"]), 2)
        self.assertEqual(len(result["failed_recipients"]), 0)
        self.assertGreaterEqual(mock_post.call_count, 2)

    def test_fetch_available_dates(self):
        """Test retrieving all distinct valuation dates available in the database."""
        dates = fetch_available_dates(engine=self.engine)
        self.assertEqual(len(dates), 10)
        self.assertEqual(dates[0], "2026-08-03")  # First weekday in 2026-08-01 range

    def test_fetch_portfolio_and_var_with_asof_date(self):
        """Test retrieving historical valuation and VaR records up to a specific asof_date."""
        target_date = "2026-08-07"
        p_df = fetch_recent_portfolio_values(days=14, asof_date=target_date, engine=self.engine)
        self.assertFalse(p_df.empty)
        # All returned dates should be <= target_date
        self.assertTrue(all(p_df["DATE"].dt.strftime("%Y-%m-%d") <= target_date))
        self.assertEqual(p_df.iloc[-1]["DATE"].strftime("%Y-%m-%d"), target_date)

        v_df = fetch_recent_var_metrics(days=130, asof_date=target_date, engine=self.engine)
        self.assertFalse(v_df.empty)
        self.assertTrue(all(v_df["DATE"].dt.strftime("%Y-%m-%d") <= target_date))
        self.assertEqual(v_df.iloc[-1]["DATE"].strftime("%Y-%m-%d"), target_date)

    def test_generate_report_standalone(self):
        """Test standalone generate_report for a specific historical date without sending to Telegram."""
        target_date = "2026-08-07"
        report = generate_report(asof_date=target_date, engine=self.engine)
        self.assertEqual(report["asof_date"], target_date)
        self.assertIsInstance(report["chart_bytes"], bytes)
        self.assertTrue(report["chart_bytes"].startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn("Daily Stock Holdings & Risk Report", report["caption"])
        self.assertIn("**", report["clean_markdown_caption"])

    def test_generate_reports_for_dates_batch(self):
        """Test generating multiple reports for a batch of historical dates."""
        target_dates = ["2026-08-05", "2026-08-06", "2026-08-07"]
        reports = generate_reports_for_dates(dates=target_dates, send_telegram=False, engine=self.engine)
        self.assertEqual(len(reports), 3)
        for d in target_dates:
            self.assertIn(d, reports)
            self.assertEqual(reports[d]["asof_date"], d)
            self.assertIsInstance(reports[d]["chart_bytes"], bytes)

    @patch("requests.post")
    def test_send_telegram_report_historical_date(self, mock_post):
        """Test send_telegram_report for a historical asof_date broadcasts the correct historical snapshot."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": {"message_id": 202}}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        target_date = "2026-08-07"
        result = send_telegram_report(
            recipients=["123456789"],
            token="test_token_xyz",
            asof_date=target_date,
            engine=self.engine
        )
        self.assertEqual(result["status"], "Delivered")
        self.assertEqual(result["asof_date"], target_date)
        self.assertEqual(len(result["delivered_recipients"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
