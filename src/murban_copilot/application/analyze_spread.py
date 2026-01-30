"""Use case for analyzing spread data."""

from typing import Sequence

from murban_copilot.domain.entities import MarketData, MovingAverages, SpreadData
from murban_copilot.domain.exceptions import InsufficientDataError, SpreadCalculationError
from murban_copilot.domain.spread_calculator import SpreadCalculator
from murban_copilot.domain.validators import validate_spread_data
from murban_copilot.infrastructure.logging import get_logger

logger = get_logger(__name__)


class AnalyzeSpreadUseCase:
    """Use case for analyzing Murban-Brent spread."""

    def __init__(
        self,
        spread_calculator: SpreadCalculator | None = None,
    ) -> None:
        """
        Initialize the use case.

        Args:
            spread_calculator: Calculator for spread and MA (default: SpreadCalculator())
        """
        self.calculator = spread_calculator or SpreadCalculator()

    def execute(
        self,
        murban_data: Sequence[MarketData],
        brent_data: Sequence[MarketData],
    ) -> tuple[list[SpreadData], list[MovingAverages], dict[str, object]]:
        """
        Analyze the spread between Murban and Brent crude.

        Args:
            murban_data: Murban market data
            brent_data: Brent market data

        Returns:
            Tuple of (spread_data, moving_averages, trend_summary)

        Raises:
            SpreadCalculationError: If calculation fails
            InsufficientDataError: If not enough data
        """
        logger.info(
            f"Analyzing spread with {len(murban_data)} Murban and "
            f"{len(brent_data)} Brent records"
        )

        validate_spread_data(murban_data, brent_data)

        spread_data = self.calculator.calculate_spread(murban_data, brent_data)
        logger.info(f"Calculated {len(spread_data)} spread data points")

        if len(spread_data) < 5:
            raise InsufficientDataError(
                f"Need at least 5 days of spread data, got {len(spread_data)}",
                required=5,
                available=len(spread_data),
            )

        moving_averages = self.calculator.calculate_moving_averages(spread_data)
        logger.info("Calculated moving averages")

        trend_summary = self.calculator.get_trend_summary(moving_averages)
        logger.info(f"Trend signal: {trend_summary.get('trend', 'unknown')}")

        return spread_data, moving_averages, trend_summary

    def get_latest_spread(
        self,
        spread_data: Sequence[SpreadData],
    ) -> SpreadData | None:
        """
        Get the most recent spread data point.

        Args:
            spread_data: Sequence of spread data

        Returns:
            Latest SpreadData or None if empty
        """
        if not spread_data:
            return None
        return max(spread_data, key=lambda x: x.date)

    def get_spread_statistics(
        self,
        spread_data: Sequence[SpreadData],
    ) -> dict[str, float | None]:
        """
        Calculate basic statistics for the spread.

        Args:
            spread_data: Sequence of spread data

        Returns:
            Dictionary with min, max, mean, std, current spread
        """
        if not spread_data:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "current": None,
            }

        spreads = [d.spread for d in spread_data]
        sorted_data = sorted(spread_data, key=lambda x: x.date)

        import numpy as np

        return {
            "min": float(np.min(spreads)),
            "max": float(np.max(spreads)),
            "mean": float(np.mean(spreads)),
            "std": float(np.std(spreads)),
            "current": sorted_data[-1].spread,
        }
