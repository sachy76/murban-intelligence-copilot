"""Spread calculation and moving average computations."""

from datetime import datetime
from typing import Sequence

import numpy as np
import pandas as pd

from .entities import MarketData, MovingAverages, SpreadData
from .exceptions import InsufficientDataError, SpreadCalculationError


class SpreadCalculator:
    """Calculator for crude oil spread and moving averages."""

    def __init__(
        self,
        short_ma_window: int = 5,
        long_ma_window: int = 20,
        outlier_threshold: float = 3.0,
    ) -> None:
        """
        Initialize the spread calculator.

        Args:
            short_ma_window: Window size for short-term MA (default 5 days)
            long_ma_window: Window size for long-term MA (default 20 days)
            outlier_threshold: Number of MAD units for outlier detection
        """
        self.short_ma_window = short_ma_window
        self.long_ma_window = long_ma_window
        self.outlier_threshold = outlier_threshold

    def calculate_spread(
        self,
        murban_data: Sequence[MarketData],
        brent_data: Sequence[MarketData],
    ) -> list[SpreadData]:
        """
        Calculate the spread between Murban and Brent crude prices.

        Args:
            murban_data: Sequence of Murban market data
            brent_data: Sequence of Brent market data

        Returns:
            List of SpreadData sorted by date

        Raises:
            SpreadCalculationError: If calculation fails
        """
        if not murban_data or not brent_data:
            raise SpreadCalculationError(
                "Cannot calculate spread with empty data",
                murban_data=murban_data,
                brent_data=brent_data,
            )

        murban_by_date = {d.date.date(): d for d in murban_data}
        brent_by_date = {d.date.date(): d for d in brent_data}

        common_dates = sorted(set(murban_by_date.keys()) & set(brent_by_date.keys()))

        if not common_dates:
            raise SpreadCalculationError(
                "No common dates between Murban and Brent data",
                murban_data=murban_data,
                brent_data=brent_data,
            )

        spreads = []
        for date in common_dates:
            murban = murban_by_date[date]
            brent = brent_by_date[date]
            spread = murban.close - brent.close

            spreads.append(
                SpreadData(
                    date=datetime.combine(date, datetime.min.time()),
                    murban_close=murban.close,
                    brent_close=brent.close,
                    spread=spread,
                )
            )

        return spreads

    def calculate_moving_averages(
        self,
        spread_data: Sequence[SpreadData],
    ) -> list[MovingAverages]:
        """
        Calculate moving averages for spread data.

        Uses robust median-based calculations for outlier resistance.

        Args:
            spread_data: Sequence of spread data points

        Returns:
            List of MovingAverages sorted by date

        Raises:
            InsufficientDataError: If not enough data for calculations
        """
        if not spread_data:
            raise InsufficientDataError(
                "Cannot calculate moving averages with empty data",
                required=1,
                available=0,
            )

        sorted_data = sorted(spread_data, key=lambda x: x.date)
        spreads = [d.spread for d in sorted_data]

        cleaned_spreads = self._handle_outliers(spreads)

        results = []
        for i, data in enumerate(sorted_data):
            ma_5 = None
            ma_20 = None

            if i >= self.short_ma_window - 1:
                window = cleaned_spreads[i - self.short_ma_window + 1 : i + 1]
                ma_5 = float(np.mean(window))

            if i >= self.long_ma_window - 1:
                window = cleaned_spreads[i - self.long_ma_window + 1 : i + 1]
                ma_20 = float(np.mean(window))

            results.append(
                MovingAverages(
                    date=data.date,
                    spread=cleaned_spreads[i],
                    ma_5=ma_5,
                    ma_20=ma_20,
                )
            )

        return results

    def _handle_outliers(self, values: list[float]) -> list[float]:
        """
        Handle outliers using Median Absolute Deviation (MAD).

        Replaces outliers with the median value for robustness.

        Args:
            values: List of spread values

        Returns:
            List with outliers replaced by median
        """
        if len(values) < 3:
            return values.copy()

        arr = np.array(values)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))

        if mad == 0:
            return values.copy()

        modified_z_scores = 0.6745 * (arr - median) / mad

        result = arr.copy()
        outlier_mask = np.abs(modified_z_scores) > self.outlier_threshold
        result[outlier_mask] = median

        return result.tolist()

    def get_trend_summary(
        self,
        moving_averages: Sequence[MovingAverages],
    ) -> dict[str, object]:
        """
        Generate a summary of the spread trend.

        Args:
            moving_averages: Sequence of moving average data

        Returns:
            Dictionary with trend summary statistics
        """
        if not moving_averages:
            return {
                "current_spread": None,
                "trend": None,
                "spread_change_5d": None,
                "spread_change_20d": None,
            }

        sorted_data = sorted(moving_averages, key=lambda x: x.date)
        latest = sorted_data[-1]

        spread_change_5d = None
        spread_change_20d = None

        if len(sorted_data) >= 5:
            spread_change_5d = latest.spread - sorted_data[-5].spread

        if len(sorted_data) >= 20:
            spread_change_20d = latest.spread - sorted_data[-20].spread

        return {
            "current_spread": latest.spread,
            "trend": latest.trend_signal,
            "ma_5": latest.ma_5,
            "ma_20": latest.ma_20,
            "spread_change_5d": spread_change_5d,
            "spread_change_20d": spread_change_20d,
        }
