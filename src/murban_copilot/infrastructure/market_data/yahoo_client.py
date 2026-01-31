"""Yahoo Finance client for fetching market data."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence, TYPE_CHECKING

import pandas as pd
import yfinance as yf
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from murban_copilot.infrastructure.logging import get_logger
from murban_copilot.domain.entities import MarketData
from murban_copilot.domain.exceptions import MarketDataFetchError

if TYPE_CHECKING:
    from murban_copilot.domain.config import MarketDataConfig

logger = get_logger(__name__)


def _is_retryable_error(exception: BaseException) -> bool:
    """Determine if an exception is retryable."""
    # Retry on network errors, timeouts, and temporary API failures
    retryable_errors = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    if isinstance(exception, retryable_errors):
        return True

    # Check for HTTP errors that are retryable (5xx, 429)
    error_msg = str(exception).lower()
    if any(code in error_msg for code in ["500", "502", "503", "504", "429", "timeout", "connection"]):
        return True

    return False

class YahooFinanceClient:
    """Client for fetching market data from Yahoo Finance."""

    # Default ticker symbols
    DEFAULT_BRENT_TICKER = "BZ=F"
    DEFAULT_WTI_TICKER = "CL=F"

    # Default retry configuration
    DEFAULT_TIMEOUT = 60
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_MIN_WAIT = 1  # seconds
    DEFAULT_MAX_WAIT = 10  # seconds
    DEFAULT_LATEST_LOOKBACK_DAYS = 7

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        min_retry_wait: int = DEFAULT_MIN_WAIT,
        max_retry_wait: int = DEFAULT_MAX_WAIT,
        wti_ticker: str = DEFAULT_WTI_TICKER,
        brent_ticker: str = DEFAULT_BRENT_TICKER,
        latest_price_lookback_days: int = DEFAULT_LATEST_LOOKBACK_DAYS,
    ) -> None:
        """
        Initialize the Yahoo Finance client.

        Args:
            timeout: Request timeout in seconds (default 60)
            max_retries: Maximum number of retry attempts (default 3)
            min_retry_wait: Minimum wait between retries in seconds (default 1)
            max_retry_wait: Maximum wait between retries in seconds (default 10)
            wti_ticker: WTI crude futures ticker symbol
            brent_ticker: Brent crude futures ticker symbol
            latest_price_lookback_days: Days to look back for latest price
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_retry_wait = min_retry_wait
        self.max_retry_wait = max_retry_wait
        self.wti_ticker = wti_ticker
        self.brent_ticker = brent_ticker
        self.latest_price_lookback_days = latest_price_lookback_days

        # Build ticker mapping from instance values
        self.ticker_mapping = {
            "wti": self.wti_ticker,
            "brent": self.brent_ticker,
        }

    @classmethod
    def from_config(cls, config: "MarketDataConfig") -> "YahooFinanceClient":
        """
        Create a YahooFinanceClient from MarketDataConfig.

        Args:
            config: Market data configuration

        Returns:
            Configured YahooFinanceClient instance
        """
        return cls(
            timeout=config.timeout,
            max_retries=config.max_retries,
            min_retry_wait=config.min_retry_wait,
            max_retry_wait=config.max_retry_wait,
            wti_ticker=config.wti_ticker,
            brent_ticker=config.brent_ticker,
            latest_price_lookback_days=config.latest_price_lookback_days,
        )

    def fetch_historical_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[MarketData]:
        """
        Fetch historical market data for a given ticker.

        Args:
            ticker: The ticker symbol (can use 'wti' or 'brent' shortcuts)
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            Sequence of MarketData objects

        Raises:
            MarketDataFetchError: If fetching fails after all retries
        """
        resolved_ticker = self.ticker_mapping.get(ticker.lower(), ticker)

        return self._fetch_with_retry(resolved_ticker, start_date, end_date)

    def _fetch_with_retry(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[MarketData]:
        """
        Internal method that performs the fetch with retry logic.

        Uses tenacity for exponential backoff retry on transient failures.
        """

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self.min_retry_wait,
                max=self.max_retry_wait,
            ),
            retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
            before_sleep=before_sleep_log(logger, log_level=20),  # INFO level
            reraise=True,
        )
        def _do_fetch() -> Sequence[MarketData]:
            try:
                yf_ticker = yf.Ticker(ticker)
                df = yf_ticker.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    timeout=self.timeout,
                )
                logger.debug(f"Fetched data for {ticker}: {len(df)} rows")

                if df.empty:
                    raise MarketDataFetchError(
                        f"No data returned for ticker {ticker}",
                        ticker=ticker,
                    )

                df = self._fill_missing_data(df)

                return self._convert_to_market_data(df, ticker)

            except MarketDataFetchError:
                raise
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.warning(f"Retryable error fetching {ticker}: {e}")
                raise
            except Exception as e:
                # Check if this is a retryable error based on message
                if _is_retryable_error(e):
                    logger.warning(f"Retryable error fetching {ticker}: {e}")
                    raise ConnectionError(str(e)) from e
                raise MarketDataFetchError(
                    f"Failed to fetch data for {ticker}: {str(e)}",
                    ticker=ticker,
                    original_error=e,
                )

        try:
            return _do_fetch()
        except (ConnectionError, TimeoutError, OSError) as e:
            # All retries exhausted
            raise MarketDataFetchError(
                f"Failed to fetch data for {ticker} after {self.max_retries} retries: {str(e)}",
                ticker=ticker,
                original_error=e,
            )

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
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.latest_price_lookback_days)

        data = self.fetch_historical_data(ticker, start_date, end_date)

        if not data:
            raise MarketDataFetchError(
                f"No recent data available for {ticker}",
                ticker=ticker,
            )

        return max(data, key=lambda x: x.date)

    def _fill_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing data using forward fill and interpolation.

        Strategy:
        - Gaps of 2 days or less: Forward fill
        - Longer gaps: Linear interpolation

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with filled missing values
        """
        if df.empty:
            return df

        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="B")
        df = df.reindex(full_date_range)

        mask = df["Close"].isna()
        if not mask.any():
            return df

        consecutive_gaps = mask.astype(int).groupby((~mask).cumsum()).cumsum()

        short_gap_mask = (consecutive_gaps <= 2) & mask
        df.loc[short_gap_mask] = df.ffill().loc[short_gap_mask]

        if df["Close"].isna().any():
            numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].interpolate(method="linear")

        return df

    def _convert_to_market_data(
        self,
        df: pd.DataFrame,
        ticker: str,
    ) -> list[MarketData]:
        """
        Convert DataFrame to list of MarketData objects.

        Args:
            df: DataFrame with OHLCV data
            ticker: The ticker symbol

        Returns:
            List of MarketData objects
        """
        result = []
        for idx, row in df.iterrows():
            if pd.isna(row["Close"]):
                continue

            result.append(
                MarketData(
                    date=pd.Timestamp(idx).to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]) if pd.notna(row.get("Volume")) else None,
                    ticker=ticker,
                )
            )

        return result

    def fetch_wti_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[MarketData]:
        """Convenience method to fetch WTI crude data."""
        return self.fetch_historical_data("wti", start_date, end_date)

    def fetch_brent_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[MarketData]:
        """Convenience method to fetch Brent crude data."""
        return self.fetch_historical_data("brent", start_date, end_date)
