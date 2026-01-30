"""Unit tests for analyze spread use case."""

import pytest

from murban_copilot.application.analyze_spread import AnalyzeSpreadUseCase
from murban_copilot.domain.entities import MovingAverages, SpreadData
from murban_copilot.domain.exceptions import InsufficientDataError, ValidationError
from murban_copilot.domain.spread_calculator import SpreadCalculator


class TestAnalyzeSpreadUseCase:
    """Tests for AnalyzeSpreadUseCase."""

    @pytest.fixture
    def use_case(self):
        """Create an AnalyzeSpreadUseCase instance."""
        return AnalyzeSpreadUseCase()

    def test_execute_returns_tuple(
        self, use_case, sample_wti_data, sample_brent_data
    ):
        """Test that execute returns a tuple of spreads, MAs, and summary."""
        result = use_case.execute(sample_wti_data, sample_brent_data)

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_execute_returns_spread_data(
        self, use_case, sample_wti_data, sample_brent_data
    ):
        """Test that execute returns SpreadData list."""
        spreads, _, _ = use_case.execute(sample_wti_data, sample_brent_data)

        assert all(isinstance(s, SpreadData) for s in spreads)

    def test_execute_returns_moving_averages(
        self, use_case, sample_wti_data, sample_brent_data
    ):
        """Test that execute returns MovingAverages list."""
        _, mas, _ = use_case.execute(sample_wti_data, sample_brent_data)

        assert all(isinstance(m, MovingAverages) for m in mas)

    def test_execute_returns_trend_summary(
        self, use_case, sample_wti_data, sample_brent_data
    ):
        """Test that execute returns trend summary dict."""
        _, _, summary = use_case.execute(sample_wti_data, sample_brent_data)

        assert isinstance(summary, dict)
        assert "current_spread" in summary
        assert "trend" in summary

    def test_execute_with_custom_calculator(
        self, sample_wti_data, sample_brent_data
    ):
        """Test execute with custom spread calculator."""
        custom_calc = SpreadCalculator(short_ma_window=3, long_ma_window=10)
        use_case = AnalyzeSpreadUseCase(spread_calculator=custom_calc)

        spreads, mas, _ = use_case.execute(sample_wti_data, sample_brent_data)

        # Custom calculator should produce different MA start indices
        assert len(mas) == len(spreads)

    def test_execute_validates_data(self, use_case, sample_wti_data):
        """Test that execute validates input data."""
        with pytest.raises(ValidationError):
            use_case.execute(sample_wti_data, [])

    def test_get_latest_spread(self, use_case, sample_spread_data):
        """Test getting latest spread."""
        latest = use_case.get_latest_spread(sample_spread_data)

        assert isinstance(latest, SpreadData)
        # Should be the most recent date
        assert latest.date == max(s.date for s in sample_spread_data)

    def test_get_latest_spread_empty(self, use_case):
        """Test getting latest spread from empty data."""
        result = use_case.get_latest_spread([])
        assert result is None

    def test_get_spread_statistics(self, use_case, sample_spread_data):
        """Test getting spread statistics."""
        stats = use_case.get_spread_statistics(sample_spread_data)

        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "current" in stats
        assert all(v is not None for v in stats.values())

    def test_get_spread_statistics_empty(self, use_case):
        """Test getting spread statistics from empty data."""
        stats = use_case.get_spread_statistics([])

        assert all(v is None for v in stats.values())
