"""Yahoo Finance client for fetching market data."""

from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd
import yfinance as yf

from murban_copilot.infrastructure.logging import get_logger
from murban_copilot.domain.entities import MarketData
from murban_copilot.domain.exceptions import MarketDataFetchError

logger = get_logger(__name__)

class YahooFinanceClient:
    """Client for fetching market data from Yahoo Finance."""

    BRENT_TICKER = "BZ=F"
    MURBAN_TICKER = "CL=F"  # Using WTI as proxy (Murban ADM=F not available on Yahoo Finance)

    TICKER_MAPPING = {
        "murban": MURBAN_TICKER,
        "brent": BRENT_TICKER,
    }

    def __init__(self, timeout: int = 60) -> None:
        """
        Initialize the Yahoo Finance client.

        Args:
            timeout: Request timeout in seconds (default 60)
        """
        self.timeout = timeout

    def fetch_historical_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[MarketData]:
        """
        Fetch historical market data for a given ticker.

        Args:
            ticker: The ticker symbol (can use 'murban' or 'brent' shortcuts)
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            Sequence of MarketData objects

        Raises:
            MarketDataFetchError: If fetching fails
        """
        resolved_ticker = self.TICKER_MAPPING.get(ticker.lower(), ticker)

        try:
            yf_ticker = yf.Ticker(resolved_ticker)
            df = yf_ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                timeout=self.timeout,
            )
            logger.info(
                f"Market Data DF {yf_ticker.info}"
            )

            if df.empty:
                raise MarketDataFetchError(
                    f"No data returned for ticker {resolved_ticker}",
                    ticker=resolved_ticker,
                )

            df = self._fill_missing_data(df)

            return self._convert_to_market_data(df, resolved_ticker)

        except MarketDataFetchError:
            raise
        except Exception as e:
            raise MarketDataFetchError(
                f"Failed to fetch data for {resolved_ticker}: {str(e)}",
                ticker=resolved_ticker,
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
        start_date = end_date - timedelta(days=7)

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

    def fetch_murban_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[MarketData]:
        """Convenience method to fetch Murban crude data."""
        return self.fetch_historical_data("murban", start_date, end_date)

    def fetch_brent_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[MarketData]:
        """Convenience method to fetch Brent crude data."""
        return self.fetch_historical_data("brent", start_date, end_date)
