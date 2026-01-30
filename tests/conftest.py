"""Shared test fixtures for the Murban Copilot test suite."""

from datetime import datetime, timedelta
from typing import Sequence

import pytest

from murban_copilot.domain.entities import (
    MarketData,
    MarketSignal,
    MovingAverages,
    SpreadData,
)
from murban_copilot.domain.spread_calculator import SpreadCalculator
from murban_copilot.infrastructure.llm.llm_client import MockLlamaClient


@pytest.fixture
def sample_date() -> datetime:
    """Return a sample date for testing."""
    return datetime(2024, 1, 15, 12, 0, 0)


@pytest.fixture
def sample_wti_data(sample_date: datetime) -> list[MarketData]:
    """Generate sample WTI market data."""
    data = []
    base_price = 85.0

    for i in range(30):
        date = sample_date - timedelta(days=29 - i)
        price = base_price + (i * 0.1) + ((-1) ** i * 0.5)

        data.append(
            MarketData(
                date=date,
                open=price - 0.3,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                volume=1000000.0 + i * 10000,
                ticker="WTI",
            )
        )

    return data


@pytest.fixture
def sample_brent_data(sample_date: datetime) -> list[MarketData]:
    """Generate sample Brent market data."""
    data = []
    base_price = 82.0

    for i in range(30):
        date = sample_date - timedelta(days=29 - i)
        price = base_price + (i * 0.08) + ((-1) ** i * 0.4)

        data.append(
            MarketData(
                date=date,
                open=price - 0.25,
                high=price + 0.45,
                low=price - 0.45,
                close=price,
                volume=2000000.0 + i * 20000,
                ticker="BRENT",
            )
        )

    return data


@pytest.fixture
def sample_spread_data(
    sample_wti_data: list[MarketData],
    sample_brent_data: list[MarketData],
) -> list[SpreadData]:
    """Generate sample spread data from market data."""
    calculator = SpreadCalculator()
    return calculator.calculate_spread(sample_wti_data, sample_brent_data)


@pytest.fixture
def sample_moving_averages(
    sample_spread_data: list[SpreadData],
) -> list[MovingAverages]:
    """Generate sample moving averages from spread data."""
    calculator = SpreadCalculator()
    return calculator.calculate_moving_averages(sample_spread_data)


@pytest.fixture
def sample_trend_summary(
    sample_moving_averages: list[MovingAverages],
) -> dict[str, object]:
    """Generate sample trend summary."""
    calculator = SpreadCalculator()
    return calculator.get_trend_summary(sample_moving_averages)


@pytest.fixture
def sample_market_signal(sample_date: datetime) -> MarketSignal:
    """Generate a sample market signal."""
    return MarketSignal(
        timestamp=sample_date,
        signal="bullish",
        confidence=0.75,
        analysis="The spread shows bullish momentum with MA crossover.",
    )


@pytest.fixture
def mock_llm_client() -> MockLlamaClient:
    """Return a mock LLM client."""
    return MockLlamaClient()


@pytest.fixture
def spread_calculator() -> SpreadCalculator:
    """Return a spread calculator instance."""
    return SpreadCalculator()


# Markers for test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "contract: marks tests as contract tests")
