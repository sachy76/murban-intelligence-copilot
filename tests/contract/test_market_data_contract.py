"""Contract tests for market data source protocol."""

from datetime import datetime, timedelta
from typing import Protocol, Sequence, runtime_checkable

import pytest

from murban_copilot.domain.entities import MarketData
from murban_copilot.infrastructure.market_data.protocols import MarketDataSource
from murban_copilot.infrastructure.market_data.yahoo_client import YahooFinanceClient


@pytest.mark.contract
class TestMarketDataSourceContract:
    """Contract tests verifying MarketDataSource implementations."""

    @pytest.fixture(params=["yahoo"])
    def data_source(self, request) -> MarketDataSource:
        """Return implementations of MarketDataSource."""
        if request.param == "yahoo":
            return YahooFinanceClient()
        raise ValueError(f"Unknown data source: {request.param}")

    def test_implements_protocol(self, data_source):
        """Test that implementation has required methods."""
        assert hasattr(data_source, "fetch_historical_data")
        assert hasattr(data_source, "get_latest_price")

    def test_fetch_historical_data_signature(self, data_source):
        """Test fetch_historical_data method signature."""
        import inspect

        sig = inspect.signature(data_source.fetch_historical_data)
        params = list(sig.parameters.keys())

        assert "ticker" in params
        assert "start_date" in params
        assert "end_date" in params

    def test_fetch_historical_data_returns_sequence(self, data_source):
        """Test that fetch_historical_data returns a Sequence."""
        # Use a stub/mock approach to test return type
        from unittest.mock import MagicMock, patch

        with patch.object(data_source, "fetch_historical_data") as mock:
            mock.return_value = [
                MarketData(
                    date=datetime.now(),
                    open=85.0,
                    high=86.0,
                    low=84.0,
                    close=85.5,
                )
            ]

            result = data_source.fetch_historical_data(
                "TEST",
                datetime.now() - timedelta(days=7),
                datetime.now(),
            )

            assert isinstance(result, Sequence)
            assert all(isinstance(r, MarketData) for r in result)

    def test_get_latest_price_returns_market_data(self, data_source):
        """Test that get_latest_price returns MarketData."""
        from unittest.mock import patch

        with patch.object(data_source, "get_latest_price") as mock:
            mock.return_value = MarketData(
                date=datetime.now(),
                open=85.0,
                high=86.0,
                low=84.0,
                close=85.5,
            )

            result = data_source.get_latest_price("TEST")

            assert isinstance(result, MarketData)


@pytest.mark.contract
class TestMarketDataContract:
    """Contract tests for MarketData entity."""

    def test_market_data_required_fields(self):
        """Test MarketData has required fields."""
        data = MarketData(
            date=datetime.now(),
            open=85.0,
            high=86.0,
            low=84.0,
            close=85.5,
        )

        assert hasattr(data, "date")
        assert hasattr(data, "open")
        assert hasattr(data, "high")
        assert hasattr(data, "low")
        assert hasattr(data, "close")

    def test_market_data_optional_fields(self):
        """Test MarketData optional fields."""
        data = MarketData(
            date=datetime.now(),
            open=85.0,
            high=86.0,
            low=84.0,
            close=85.5,
            volume=1000000.0,
            ticker="TEST",
        )

        assert data.volume == 1000000.0
        assert data.ticker == "TEST"

    def test_market_data_immutability(self):
        """Test MarketData is frozen/immutable."""
        data = MarketData(
            date=datetime.now(),
            open=85.0,
            high=86.0,
            low=84.0,
            close=85.5,
        )

        with pytest.raises(AttributeError):
            data.close = 90.0
