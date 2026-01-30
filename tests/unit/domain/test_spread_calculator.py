"""Unit tests for spread calculator."""

from datetime import datetime, timedelta

import pytest

from murban_copilot.domain.entities import MarketData, SpreadData
from murban_copilot.domain.exceptions import InsufficientDataError, SpreadCalculationError
from murban_copilot.domain.spread_calculator import SpreadCalculator


class TestSpreadCalculator:
    """Tests for SpreadCalculator."""

    def test_calculate_spread_basic(self, sample_murban_data, sample_brent_data):
        """Test basic spread calculation."""
        calculator = SpreadCalculator()
        spreads = calculator.calculate_spread(sample_murban_data, sample_brent_data)

        assert len(spreads) > 0
        assert all(isinstance(s, SpreadData) for s in spreads)

        for spread in spreads:
            expected = spread.murban_close - spread.brent_close
            assert abs(spread.spread - expected) < 0.0001

    def test_calculate_spread_empty_murban(self, sample_brent_data):
        """Test spread calculation with empty Murban data."""
        calculator = SpreadCalculator()

        with pytest.raises(SpreadCalculationError, match="empty data"):
            calculator.calculate_spread([], sample_brent_data)

    def test_calculate_spread_empty_brent(self, sample_murban_data):
        """Test spread calculation with empty Brent data."""
        calculator = SpreadCalculator()

        with pytest.raises(SpreadCalculationError, match="empty data"):
            calculator.calculate_spread(sample_murban_data, [])

    def test_calculate_spread_no_common_dates(self):
        """Test spread calculation with no common dates."""
        calculator = SpreadCalculator()

        murban = [
            MarketData(
                date=datetime(2024, 1, 15),
                open=85.0, high=86.0, low=84.0, close=85.0,
            )
        ]
        brent = [
            MarketData(
                date=datetime(2024, 2, 15),
                open=82.0, high=83.0, low=81.0, close=82.0,
            )
        ]

        with pytest.raises(SpreadCalculationError, match="No common dates"):
            calculator.calculate_spread(murban, brent)

    def test_calculate_spread_sorted_by_date(self, sample_murban_data, sample_brent_data):
        """Test that spreads are sorted by date."""
        calculator = SpreadCalculator()
        spreads = calculator.calculate_spread(sample_murban_data, sample_brent_data)

        dates = [s.date for s in spreads]
        assert dates == sorted(dates)

    def test_calculate_moving_averages(self, sample_spread_data):
        """Test moving average calculation."""
        calculator = SpreadCalculator()
        mas = calculator.calculate_moving_averages(sample_spread_data)

        assert len(mas) == len(sample_spread_data)

        # First 4 entries should not have 5-day MA
        for i in range(4):
            assert mas[i].ma_5 is None

        # From index 4 onwards should have 5-day MA
        for i in range(4, len(mas)):
            assert mas[i].ma_5 is not None

    def test_calculate_moving_averages_empty(self):
        """Test moving averages with empty data."""
        calculator = SpreadCalculator()

        with pytest.raises(InsufficientDataError):
            calculator.calculate_moving_averages([])

    def test_calculate_moving_averages_20_day(self, sample_spread_data):
        """Test 20-day moving average calculation."""
        calculator = SpreadCalculator()
        mas = calculator.calculate_moving_averages(sample_spread_data)

        # First 19 entries should not have 20-day MA
        for i in range(19):
            assert mas[i].ma_20 is None

        # From index 19 onwards should have 20-day MA
        for i in range(19, len(mas)):
            assert mas[i].ma_20 is not None

    def test_outlier_handling(self):
        """Test outlier detection and handling."""
        calculator = SpreadCalculator()

        # Create data with an outlier
        values = [1.0, 1.1, 1.0, 0.9, 1.0, 100.0, 1.0, 1.1]  # 100.0 is outlier
        cleaned = calculator._handle_outliers(values)

        # The outlier should be replaced with median
        assert cleaned[5] != 100.0
        assert abs(cleaned[5] - 1.0) < 0.5

    def test_outlier_handling_small_data(self):
        """Test that outlier handling doesn't modify small datasets."""
        calculator = SpreadCalculator()

        values = [1.0, 100.0]  # Too small for outlier detection
        cleaned = calculator._handle_outliers(values)

        assert cleaned == values

    def test_get_trend_summary(self, sample_moving_averages):
        """Test trend summary generation."""
        calculator = SpreadCalculator()
        summary = calculator.get_trend_summary(sample_moving_averages)

        assert "current_spread" in summary
        assert "trend" in summary
        assert "ma_5" in summary
        assert "ma_20" in summary
        assert "spread_change_5d" in summary
        assert "spread_change_20d" in summary

    def test_get_trend_summary_empty(self):
        """Test trend summary with empty data."""
        calculator = SpreadCalculator()
        summary = calculator.get_trend_summary([])

        assert summary["current_spread"] is None
        assert summary["trend"] is None

    def test_custom_window_sizes(self, sample_spread_data):
        """Test calculator with custom window sizes."""
        calculator = SpreadCalculator(short_ma_window=3, long_ma_window=10)
        mas = calculator.calculate_moving_averages(sample_spread_data)

        # Check 3-day MA starts at index 2
        assert mas[1].ma_5 is None  # Using ma_5 attribute but it's actually 3-day
        assert mas[2].ma_5 is not None

    def test_moving_average_values(self):
        """Test actual moving average values."""
        calculator = SpreadCalculator(short_ma_window=3, long_ma_window=5)

        spreads = []
        base_date = datetime(2024, 1, 1)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for i, val in enumerate(values):
            spreads.append(
                SpreadData(
                    date=base_date + timedelta(days=i),
                    murban_close=val + 80,
                    brent_close=80.0,
                    spread=val,
                )
            )

        mas = calculator.calculate_moving_averages(spreads)

        # 3-day MA at index 2 should be (1+2+3)/3 = 2.0
        assert abs(mas[2].ma_5 - 2.0) < 0.01

        # 5-day MA at index 4 should be (1+2+3+4+5)/5 = 3.0
        assert abs(mas[4].ma_20 - 3.0) < 0.01
