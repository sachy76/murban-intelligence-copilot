"""Integration tests for market data fetching."""

from datetime import datetime, timedelta

import pytest

from murban_copilot.domain.entities import MarketData
from murban_copilot.domain.exceptions import MarketDataFetchError
from murban_copilot.infrastructure.market_data.yahoo_client import YahooFinanceClient


@pytest.mark.integration
@pytest.mark.slow
class TestMarketDataIntegration:
    """Integration tests for real market data fetching."""

    @pytest.fixture
    def client(self):
        """Return a real YahooFinanceClient."""
        return YahooFinanceClient(timeout=30)

    def test_fetch_brent_data(self, client):
        """Test fetching real Brent crude data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        try:
            data = client.fetch_historical_data("BZ=F", start_date, end_date)

            assert len(data) > 0
            assert all(isinstance(d, MarketData) for d in data)

            for d in data:
                assert d.close > 0
                assert d.high >= d.low
        except MarketDataFetchError as e:
            pytest.skip(f"Market data unavailable: {e}")

    def test_fetch_wti_data(self, client):
        """Test fetching real WTI crude data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        try:
            data = client.fetch_historical_data("wti", start_date, end_date)

            if len(data) > 0:
                assert all(isinstance(d, MarketData) for d in data)
        except MarketDataFetchError:
            pytest.skip("WTI data not available in Yahoo Finance")

    def test_get_latest_price(self, client):
        """Test getting latest price."""
        try:
            price = client.get_latest_price("BZ=F")

            assert isinstance(price, MarketData)
            assert price.close > 0
        except MarketDataFetchError as e:
            pytest.skip(f"Market data unavailable: {e}")

    def test_data_date_range(self, client):
        """Test that returned data is within requested range."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        try:
            data = client.fetch_historical_data("BZ=F", start_date, end_date)

            for d in data:
                # Strip timezone for comparison (Yahoo returns tz-aware dates)
                date_naive = d.date.replace(tzinfo=None) if d.date.tzinfo else d.date
                assert date_naive >= start_date - timedelta(days=1)
                assert date_naive <= end_date + timedelta(days=1)
        except MarketDataFetchError as e:
            pytest.skip(f"Market data unavailable: {e}")
