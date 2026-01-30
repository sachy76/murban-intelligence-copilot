"""Domain entities for market data and analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class MarketData:
    """Represents OHLCV market data for a single day."""

    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    ticker: str = ""

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be less than low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"High ({self.high}) must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"Low ({self.low}) must be <= open and close")


@dataclass(frozen=True)
class SpreadData:
    """Represents the spread between two crude oil benchmarks."""

    date: datetime
    murban_close: float
    brent_close: float
    spread: float

    def __post_init__(self) -> None:
        expected_spread = self.murban_close - self.brent_close
        if abs(self.spread - expected_spread) > 0.0001:
            raise ValueError(
                f"Spread ({self.spread}) does not match "
                f"murban_close - brent_close ({expected_spread})"
            )


@dataclass(frozen=True)
class MovingAverages:
    """Moving average calculations for spread data."""

    date: datetime
    spread: float
    ma_5: Optional[float] = None
    ma_20: Optional[float] = None

    @property
    def has_short_term_ma(self) -> bool:
        return self.ma_5 is not None

    @property
    def has_long_term_ma(self) -> bool:
        return self.ma_20 is not None

    @property
    def trend_signal(self) -> Optional[str]:
        """Returns trend based on MA crossover."""
        if self.ma_5 is None or self.ma_20 is None:
            return None
        if self.ma_5 > self.ma_20:
            return "bullish"
        elif self.ma_5 < self.ma_20:
            return "bearish"
        return "neutral"


@dataclass
class MarketSignal:
    """AI-generated market signal with analysis."""

    timestamp: datetime
    signal: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    analysis: str
    disclaimer: str = field(
        default="This is AI-generated analysis for informational purposes only. "
        "Not financial advice. Always conduct your own research and consult "
        "with qualified financial advisors before making trading decisions."
    )

    def __post_init__(self) -> None:
        if self.signal not in ("bullish", "bearish", "neutral"):
            raise ValueError(f"Invalid signal: {self.signal}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
