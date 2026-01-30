"""Unit tests for domain entities."""

from datetime import datetime

import pytest

from murban_copilot.domain.entities import (
    MarketData,
    MarketSignal,
    MovingAverages,
    SpreadData,
)


class TestMarketData:
    """Tests for MarketData entity."""

    def test_create_valid_market_data(self):
        """Test creating valid MarketData."""
        data = MarketData(
            date=datetime(2024, 1, 15),
            open=85.0,
            high=86.0,
            low=84.0,
            close=85.5,
            volume=1000000.0,
            ticker="WTI",
        )

        assert data.open == 85.0
        assert data.high == 86.0
        assert data.low == 84.0
        assert data.close == 85.5
        assert data.volume == 1000000.0
        assert data.ticker == "WTI"

    def test_market_data_immutable(self):
        """Test that MarketData is immutable."""
        data = MarketData(
            date=datetime(2024, 1, 15),
            open=85.0,
            high=86.0,
            low=84.0,
            close=85.5,
        )

        with pytest.raises(AttributeError):
            data.close = 90.0

    def test_high_less_than_low_raises_error(self):
        """Test that high < low raises ValueError."""
        with pytest.raises(ValueError, match="High.*cannot be less than low"):
            MarketData(
                date=datetime(2024, 1, 15),
                open=85.0,
                high=83.0,
                low=84.0,
                close=85.0,
            )

    def test_high_less_than_open_raises_error(self):
        """Test that high < open raises ValueError."""
        with pytest.raises(ValueError, match="High.*must be >= open"):
            MarketData(
                date=datetime(2024, 1, 15),
                open=87.0,
                high=86.0,
                low=84.0,
                close=85.0,
            )

    def test_low_greater_than_close_raises_error(self):
        """Test that low > close raises ValueError."""
        with pytest.raises(ValueError, match="Low.*must be <="):
            MarketData(
                date=datetime(2024, 1, 15),
                open=85.0,
                high=87.0,
                low=86.0,
                close=85.0,
            )

    def test_optional_volume(self):
        """Test that volume is optional."""
        data = MarketData(
            date=datetime(2024, 1, 15),
            open=85.0,
            high=86.0,
            low=84.0,
            close=85.5,
        )

        assert data.volume is None


class TestSpreadData:
    """Tests for SpreadData entity."""

    def test_create_valid_spread_data(self):
        """Test creating valid SpreadData."""
        data = SpreadData(
            date=datetime(2024, 1, 15),
            wti_close=85.0,
            brent_close=82.0,
            spread=3.0,
        )

        assert data.wti_close == 85.0
        assert data.brent_close == 82.0
        assert data.spread == 3.0

    def test_spread_mismatch_raises_error(self):
        """Test that incorrect spread raises ValueError."""
        with pytest.raises(ValueError, match="Spread.*does not match"):
            SpreadData(
                date=datetime(2024, 1, 15),
                wti_close=85.0,
                brent_close=82.0,
                spread=5.0,  # Should be 3.0
            )

    def test_negative_spread(self):
        """Test that negative spreads are valid."""
        data = SpreadData(
            date=datetime(2024, 1, 15),
            wti_close=80.0,
            brent_close=82.0,
            spread=-2.0,
        )

        assert data.spread == -2.0


class TestMovingAverages:
    """Tests for MovingAverages entity."""

    def test_create_with_all_mas(self):
        """Test creating MovingAverages with all MAs."""
        ma = MovingAverages(
            date=datetime(2024, 1, 15),
            spread=3.0,
            ma_5=2.8,
            ma_20=2.5,
        )

        assert ma.spread == 3.0
        assert ma.ma_5 == 2.8
        assert ma.ma_20 == 2.5

    def test_has_short_term_ma(self):
        """Test has_short_term_ma property."""
        ma_with = MovingAverages(
            date=datetime(2024, 1, 15),
            spread=3.0,
            ma_5=2.8,
        )
        ma_without = MovingAverages(
            date=datetime(2024, 1, 15),
            spread=3.0,
        )

        assert ma_with.has_short_term_ma is True
        assert ma_without.has_short_term_ma is False

    def test_trend_signal_bullish(self):
        """Test bullish trend signal."""
        ma = MovingAverages(
            date=datetime(2024, 1, 15),
            spread=3.0,
            ma_5=3.0,
            ma_20=2.5,
        )

        assert ma.trend_signal == "bullish"

    def test_trend_signal_bearish(self):
        """Test bearish trend signal."""
        ma = MovingAverages(
            date=datetime(2024, 1, 15),
            spread=3.0,
            ma_5=2.0,
            ma_20=2.5,
        )

        assert ma.trend_signal == "bearish"

    def test_trend_signal_neutral(self):
        """Test neutral trend signal."""
        ma = MovingAverages(
            date=datetime(2024, 1, 15),
            spread=3.0,
            ma_5=2.5,
            ma_20=2.5,
        )

        assert ma.trend_signal == "neutral"

    def test_trend_signal_none_when_missing_ma(self):
        """Test trend signal is None when MAs missing."""
        ma = MovingAverages(
            date=datetime(2024, 1, 15),
            spread=3.0,
            ma_5=2.8,
        )

        assert ma.trend_signal is None


class TestMarketSignal:
    """Tests for MarketSignal entity."""

    def test_create_valid_signal(self):
        """Test creating valid MarketSignal."""
        signal = MarketSignal(
            timestamp=datetime(2024, 1, 15),
            signal="bullish",
            confidence=0.75,
            analysis="Market looks strong.",
        )

        assert signal.signal == "bullish"
        assert signal.confidence == 0.75
        assert "financial advice" in signal.disclaimer.lower()

    def test_invalid_signal_raises_error(self):
        """Test that invalid signal raises ValueError."""
        with pytest.raises(ValueError, match="Invalid signal"):
            MarketSignal(
                timestamp=datetime(2024, 1, 15),
                signal="strong_buy",
                confidence=0.75,
                analysis="Market looks strong.",
            )

    def test_confidence_out_of_range_raises_error(self):
        """Test that confidence out of range raises ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            MarketSignal(
                timestamp=datetime(2024, 1, 15),
                signal="bullish",
                confidence=1.5,
                analysis="Market looks strong.",
            )

    def test_all_signal_types(self):
        """Test all valid signal types."""
        for signal_type in ["bullish", "bearish", "neutral"]:
            signal = MarketSignal(
                timestamp=datetime(2024, 1, 15),
                signal=signal_type,
                confidence=0.5,
                analysis="Analysis.",
            )
            assert signal.signal == signal_type
