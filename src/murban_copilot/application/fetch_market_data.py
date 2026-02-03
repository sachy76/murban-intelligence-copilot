"""Use case for fetching market data."""

from datetime import datetime, timedelta
from typing import Sequence

from murban_copilot.domain.entities import MarketData
from murban_copilot.domain.exceptions import MarketDataFetchError
from murban_copilot.infrastructure.logging import get_logger
from murban_copilot.infrastructure.market_data.protocols import MarketDataSource

logger = get_logger(__name__)


class FetchMarketDataUseCase:
    """Use case for fetching and preparing market data."""

    DEFAULT_BUFFER_DAYS = 10

    def __init__(
        self,
        market_data_source: MarketDataSource,
        buffer_days: int = DEFAULT_BUFFER_DAYS,
    ) -> None:
        """
        Initialize the use case.

        Args:
            market_data_source: Data source for market data
            buffer_days: Extra days to fetch for moving average calculations
        """
        self.data_source = market_data_source
        self.buffer_days = buffer_days

    def execute(
        self,
        days: int = 30,
        end_date: datetime | None = None,
    ) -> tuple[Sequence[MarketData], Sequence[MarketData]]:
        """
        Fetch WTI and Brent market data.

        Args:
            days: Number of days of historical data
            end_date: End date for data range (default: today)

        Returns:
            Tuple of (wti_data, brent_data)

        Raises:
            MarketDataFetchError: If fetching fails
        """
        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=days + self.buffer_days)

        logger.info(
            f"Fetching market data from {start_date.date()} to {end_date.date()}"
        )

        wti_data = self._fetch_ticker("wti", start_date, end_date)
        brent_data = self._fetch_ticker("brent", start_date, end_date)

        return wti_data, brent_data

    def _fetch_ticker(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[MarketData]:
        """
        Fetch data for a single ticker with error handling.

        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            Sequence of MarketData

        Raises:
            MarketDataFetchError: If fetching fails or no data available
        """
        try:
            data = self.data_source.fetch_historical_data(ticker, start_date, end_date)
            logger.info(f"Fetched {len(data)} {ticker.upper()} records")
        except MarketDataFetchError:
            raise
        except Exception as e:
            raise MarketDataFetchError(
                f"Failed to fetch {ticker.upper()} data: {str(e)}",
                ticker=ticker,
                original_error=e,
            )

        if not data:
            raise MarketDataFetchError(
                f"No {ticker.upper()} data available",
                ticker=ticker,
            )

        return data

    def fetch_single_ticker(
        self,
        ticker: str,
        days: int = 30,
        end_date: datetime | None = None,
    ) -> Sequence[MarketData]:
        """
        Fetch data for a single ticker.

        Args:
            ticker: Ticker symbol
            days: Number of days of historical data
            end_date: End date for data range

        Returns:
            Sequence of MarketData

        Raises:
            MarketDataFetchError: If fetching fails
        """
        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=days + self.buffer_days)

        return self.data_source.fetch_historical_data(ticker, start_date, end_date)
