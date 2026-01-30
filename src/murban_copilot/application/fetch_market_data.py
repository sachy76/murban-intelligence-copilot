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

    def __init__(self, market_data_source: MarketDataSource) -> None:
        """
        Initialize the use case.

        Args:
            market_data_source: Data source for market data
        """
        self.data_source = market_data_source

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

        start_date = end_date - timedelta(days=days + 10)

        logger.info(
            f"Fetching market data from {start_date.date()} to {end_date.date()}"
        )

        try:
            wti_data = self.data_source.fetch_historical_data(
                "wti", start_date, end_date
            )
            logger.info(f"Fetched {len(wti_data)} WTI records")
        except MarketDataFetchError:
            raise
        except Exception as e:
            raise MarketDataFetchError(
                f"Failed to fetch WTI data: {str(e)}",
                ticker="wti",
                original_error=e,
            )

        try:
            brent_data = self.data_source.fetch_historical_data(
                "brent", start_date, end_date
            )
            logger.info(f"Fetched {len(brent_data)} Brent records")
        except MarketDataFetchError:
            raise
        except Exception as e:
            raise MarketDataFetchError(
                f"Failed to fetch Brent data: {str(e)}",
                ticker="brent",
                original_error=e,
            )

        if not wti_data:
            raise MarketDataFetchError(
                "No WTI data available",
                ticker="wti",
            )

        if not brent_data:
            raise MarketDataFetchError(
                "No Brent data available",
                ticker="brent",
            )

        return wti_data, brent_data

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

        start_date = end_date - timedelta(days=days + 10)

        return self.data_source.fetch_historical_data(ticker, start_date, end_date)
