"""Unit tests for Yahoo Finance client."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from murban_copilot.domain.entities import MarketData
from murban_copilot.domain.exceptions import MarketDataFetchError
from murban_copilot.infrastructure.market_data.yahoo_client import YahooFinanceClient


class TestYahooFinanceClient:
    """Tests for YahooFinanceClient."""

    @pytest.fixture
    def client(self):
        """Return a YahooFinanceClient instance."""
        return YahooFinanceClient()

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame with market data."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="B")
        return pd.DataFrame(
            {
                "Open": [85.0 + i * 0.1 for i in range(10)],
                "High": [86.0 + i * 0.1 for i in range(10)],
                "Low": [84.0 + i * 0.1 for i in range(10)],
                "Close": [85.5 + i * 0.1 for i in range(10)],
                "Volume": [1000000 + i * 10000 for i in range(10)],
            },
            index=dates,
        )

    def test_ticker_mapping(self, client):
        """Test ticker mapping for convenience names."""
        assert client.TICKER_MAPPING["murban"] == "CL=F"  # WTI as proxy
        assert client.TICKER_MAPPING["brent"] == "BZ=F"

    @patch("yfinance.Ticker")
    def test_fetch_historical_data_success(self, mock_ticker, client, sample_df):
        """Test successful data fetch."""
        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_df
        mock_ticker.return_value = mock_instance

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 15)

        result = client.fetch_historical_data("BZ=F", start_date, end_date)

        assert len(result) == 10
        assert all(isinstance(r, MarketData) for r in result)
        mock_instance.history.assert_called_once()

    @patch("yfinance.Ticker")
    def test_fetch_historical_data_empty(self, mock_ticker, client):
        """Test fetch with empty response."""
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_instance

        with pytest.raises(MarketDataFetchError, match="No data returned"):
            client.fetch_historical_data(
                "INVALID",
                datetime(2024, 1, 1),
                datetime(2024, 1, 15),
            )

    @patch("yfinance.Ticker")
    def test_fetch_historical_data_with_murban_shortcut(self, mock_ticker, client, sample_df):
        """Test fetch using 'murban' shortcut."""
        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_df
        mock_ticker.return_value = mock_instance

        client.fetch_historical_data(
            "murban",
            datetime(2024, 1, 1),
            datetime(2024, 1, 15),
        )

        mock_ticker.assert_called_with("CL=F")  # WTI as proxy for Murban

    @patch("yfinance.Ticker")
    def test_fetch_historical_data_exception(self, mock_ticker, client):
        """Test fetch with exception."""
        mock_instance = MagicMock()
        mock_instance.history.side_effect = Exception("Network error")
        mock_ticker.return_value = mock_instance

        with pytest.raises(MarketDataFetchError, match="Failed to fetch"):
            client.fetch_historical_data(
                "BZ=F",
                datetime(2024, 1, 1),
                datetime(2024, 1, 15),
            )

    @patch("yfinance.Ticker")
    def test_get_latest_price(self, mock_ticker, client, sample_df):
        """Test getting latest price."""
        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_df
        mock_ticker.return_value = mock_instance

        result = client.get_latest_price("BZ=F")

        assert isinstance(result, MarketData)
        # Should return the most recent date
        assert result.date == sample_df.index[-1].to_pydatetime()

    def test_fill_missing_data_no_gaps(self, client, sample_df):
        """Test fill_missing_data with no gaps."""
        result = client._fill_missing_data(sample_df)

        assert len(result) >= len(sample_df)
        assert not result["Close"].isna().any()

    def test_fill_missing_data_with_gaps(self, client):
        """Test fill_missing_data with gaps."""
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"])
        df = pd.DataFrame(
            {
                "Open": [85.0, 85.1, 85.3],
                "High": [86.0, 86.1, 86.3],
                "Low": [84.0, 84.1, 84.3],
                "Close": [85.5, 85.6, 85.8],
                "Volume": [1000000, 1010000, 1030000],
            },
            index=dates,
        )

        result = client._fill_missing_data(df)

        # Should have business days filled in
        assert not result["Close"].isna().any()

    def test_convert_to_market_data(self, client, sample_df):
        """Test DataFrame to MarketData conversion."""
        result = client._convert_to_market_data(sample_df, "TEST")

        assert len(result) == 10
        assert all(r.ticker == "TEST" for r in result)
        assert result[0].close == 85.5

    def test_convenience_methods(self, client):
        """Test convenience methods exist."""
        assert hasattr(client, "fetch_murban_data")
        assert hasattr(client, "fetch_brent_data")
