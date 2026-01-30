"""Market data infrastructure."""

from .protocols import MarketDataSource
from .yahoo_client import YahooFinanceClient

__all__ = ["MarketDataSource", "YahooFinanceClient"]
