"""Protocols for market data sources."""

from datetime import datetime
from typing import Protocol, Sequence

from murban_copilot.domain.entities import MarketData


class MarketDataSource(Protocol):
    """Protocol for market data sources."""

    def fetch_historical_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[MarketData]:
        """
        Fetch historical market data for a given ticker.

        Args:
            ticker: The ticker symbol
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            Sequence of MarketData objects

        Raises:
            MarketDataFetchError: If fetching fails
        """
        ...

    def get_latest_price(self, ticker: str) -> MarketData:
        """
        Get the latest available price for a ticker.

        Args:
            ticker: The ticker symbol

        Returns:
            Most recent MarketData

        Raises:
            MarketDataFetchError: If fetching fails
        """
        ...
