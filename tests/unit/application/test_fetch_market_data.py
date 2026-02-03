"""Unit tests for fetch market data use case."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from murban_copilot.application.fetch_market_data import FetchMarketDataUseCase
from murban_copilot.domain.entities import MarketData
from murban_copilot.domain.exceptions import MarketDataFetchError


class TestFetchMarketDataUseCase:
    """Tests for FetchMarketDataUseCase."""

    @pytest.fixture
    def mock_data_source(self, sample_wti_data, sample_brent_data):
        """Create a mock market data source."""
        mock = MagicMock()
        mock.fetch_historical_data.side_effect = lambda ticker, *args: (
            sample_wti_data if "wti" in ticker.lower() else sample_brent_data
        )
        return mock

    @pytest.fixture
    def use_case(self, mock_data_source):
        """Create a FetchMarketDataUseCase instance."""
        return FetchMarketDataUseCase(mock_data_source)

    def test_execute_fetches_both_tickers(self, use_case, mock_data_source):
        """Test that execute fetches both WTI and Brent data."""
        wti, brent = use_case.execute(days=30)

        assert len(wti) > 0
        assert len(brent) > 0
        assert mock_data_source.fetch_historical_data.call_count == 2

    def test_execute_returns_market_data(self, use_case):
        """Test that execute returns MarketData sequences."""
        wti, brent = use_case.execute(days=30)

        assert all(isinstance(d, MarketData) for d in wti)
        assert all(isinstance(d, MarketData) for d in brent)

    def test_execute_with_custom_days(self, use_case, mock_data_source):
        """Test execute with custom days parameter."""
        use_case.execute(days=60)

        call_args = mock_data_source.fetch_historical_data.call_args_list
        for call in call_args:
            start_date = call[0][1]
            end_date = call[0][2]
            # Start should be at least 60 days before end
            assert (end_date - start_date).days >= 60

    def test_execute_with_end_date(self, use_case, mock_data_source):
        """Test execute with custom end date."""
        end_date = datetime(2024, 1, 15)
        use_case.execute(days=30, end_date=end_date)

        call_args = mock_data_source.fetch_historical_data.call_args_list
        for call in call_args:
            assert call[0][2] == end_date

    def test_execute_raises_on_empty_wti(self, mock_data_source):
        """Test that execute raises when WTI data is empty."""
        mock_data_source.fetch_historical_data.side_effect = lambda ticker, *args: (
            [] if "wti" in ticker.lower() else [MagicMock()]
        )
        use_case = FetchMarketDataUseCase(mock_data_source)

        with pytest.raises(MarketDataFetchError, match="No WTI data"):
            use_case.execute()

    def test_execute_raises_on_empty_brent(
        self, mock_data_source, sample_wti_data
    ):
        """Test that execute raises when Brent data is empty."""
        mock_data_source.fetch_historical_data.side_effect = lambda ticker, *args: (
            sample_wti_data if "wti" in ticker.lower() else []
        )
        use_case = FetchMarketDataUseCase(mock_data_source)

        with pytest.raises(MarketDataFetchError, match="No BRENT data"):
            use_case.execute()

    def test_fetch_single_ticker(self, use_case, mock_data_source):
        """Test fetching a single ticker."""
        data = use_case.fetch_single_ticker("BZ=F", days=30)

        assert len(data) > 0
        mock_data_source.fetch_historical_data.assert_called()
