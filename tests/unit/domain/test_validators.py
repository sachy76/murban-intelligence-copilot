"""Unit tests for validators."""

from datetime import datetime

import pytest

from murban_copilot.domain.entities import MarketData, SpreadData
from murban_copilot.domain.exceptions import ValidationError
from murban_copilot.domain.validators import (
    validate_llm_input,
    validate_ohlc,
    validate_spread_data,
)


class TestValidateOHLC:
    """Tests for validate_ohlc function."""

    def test_valid_ohlc(self):
        """Test validation of valid OHLC data."""
        data = {
            "open": 85.0,
            "high": 86.0,
            "low": 84.0,
            "close": 85.5,
        }

        assert validate_ohlc(data) is True

    def test_missing_field(self):
        """Test validation with missing field."""
        data = {
            "open": 85.0,
            "high": 86.0,
            "low": 84.0,
            # Missing close
        }

        with pytest.raises(ValidationError, match="Missing required field"):
            validate_ohlc(data)

    def test_none_value(self):
        """Test validation with None value."""
        data = {
            "open": 85.0,
            "high": 86.0,
            "low": 84.0,
            "close": None,
        }

        with pytest.raises(ValidationError, match="cannot be None"):
            validate_ohlc(data)

    def test_non_numeric_value(self):
        """Test validation with non-numeric value."""
        data = {
            "open": "85.0",
            "high": 86.0,
            "low": 84.0,
            "close": 85.5,
        }

        with pytest.raises(ValidationError, match="must be numeric"):
            validate_ohlc(data)

    def test_negative_value(self):
        """Test validation with negative value."""
        data = {
            "open": -85.0,
            "high": 86.0,
            "low": 84.0,
            "close": 85.5,
        }

        with pytest.raises(ValidationError, match="cannot be negative"):
            validate_ohlc(data)

    def test_high_less_than_low(self):
        """Test validation when high < low."""
        data = {
            "open": 85.0,
            "high": 83.0,
            "low": 84.0,
            "close": 85.0,
        }

        with pytest.raises(ValidationError, match="cannot be less than low"):
            validate_ohlc(data)


class TestValidateSpreadData:
    """Tests for validate_spread_data function."""

    def test_valid_spread_data(self, sample_murban_data, sample_brent_data):
        """Test validation of valid spread data."""
        assert validate_spread_data(sample_murban_data, sample_brent_data) is True

    def test_empty_murban_data(self, sample_brent_data):
        """Test validation with empty Murban data."""
        with pytest.raises(ValidationError, match="Murban data cannot be empty"):
            validate_spread_data([], sample_brent_data)

    def test_empty_brent_data(self, sample_murban_data):
        """Test validation with empty Brent data."""
        with pytest.raises(ValidationError, match="Brent data cannot be empty"):
            validate_spread_data(sample_murban_data, [])

    def test_no_common_dates(self):
        """Test validation with no common dates."""
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

        with pytest.raises(ValidationError, match="No common dates"):
            validate_spread_data(murban, brent)


class TestValidateLLMInput:
    """Tests for validate_llm_input function."""

    def test_valid_llm_input(self, sample_spread_data):
        """Test validation of valid LLM input."""
        assert validate_llm_input(sample_spread_data) is True

    def test_empty_spread_data(self):
        """Test validation with empty spread data."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_llm_input([])

    def test_insufficient_data(self):
        """Test validation with insufficient data."""
        spread_data = [
            SpreadData(
                date=datetime(2024, 1, i),
                murban_close=85.0,
                brent_close=82.0,
                spread=3.0,
            )
            for i in range(1, 4)  # Only 3 days
        ]

        with pytest.raises(ValidationError, match="Insufficient data"):
            validate_llm_input(spread_data, min_days=5)

    def test_custom_min_days(self):
        """Test validation with custom minimum days."""
        spread_data = [
            SpreadData(
                date=datetime(2024, 1, i),
                murban_close=85.0,
                brent_close=82.0,
                spread=3.0,
            )
            for i in range(1, 4)
        ]

        # Should pass with min_days=3
        assert validate_llm_input(spread_data, min_days=3) is True
