"""Validation functions for domain entities."""

from typing import Any, Sequence

import pandas as pd

from .entities import MarketData, SpreadData
from .exceptions import ValidationError


def validate_ohlc(data: dict[str, Any]) -> bool:
    """
    Validate OHLC data dictionary.

    Args:
        data: Dictionary with open, high, low, close keys

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    required_fields = ["open", "high", "low", "close"]

    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Missing required field: {field}", field=field)

        value = data[field]
        if value is None:
            raise ValidationError(f"Field {field} cannot be None", field=field, value=value)

        if not isinstance(value, (int, float)):
            raise ValidationError(
                f"Field {field} must be numeric, got {type(value).__name__}",
                field=field,
                value=value,
            )

        if pd.isna(value):
            raise ValidationError(f"Field {field} cannot be NaN", field=field, value=value)

        if value < 0:
            raise ValidationError(
                f"Field {field} cannot be negative: {value}",
                field=field,
                value=value,
            )

    if data["high"] < data["low"]:
        raise ValidationError(
            f"High ({data['high']}) cannot be less than low ({data['low']})",
            field="high",
            value=data["high"],
        )

    return True


def validate_spread_data(
    wti_data: Sequence[MarketData],
    brent_data: Sequence[MarketData],
) -> bool:
    """
    Validate spread calculation input data.

    Args:
        wti_data: Sequence of WTI market data
        brent_data: Sequence of Brent market data

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not wti_data:
        raise ValidationError("WTI data cannot be empty", field="wti_data")

    if not brent_data:
        raise ValidationError("Brent data cannot be empty", field="brent_data")

    wti_dates = {d.date.date() for d in wti_data}
    brent_dates = {d.date.date() for d in brent_data}

    common_dates = wti_dates & brent_dates
    if not common_dates:
        raise ValidationError(
            "No common dates between WTI and Brent data",
            field="dates",
        )

    return True


def validate_llm_input(
    spread_data: Sequence[SpreadData],
    min_days: int = 5,
) -> bool:
    """
    Validate input data for LLM analysis.

    Args:
        spread_data: Sequence of spread data points
        min_days: Minimum required days of data

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if not spread_data:
        raise ValidationError("Spread data cannot be empty", field="spread_data")

    if len(spread_data) < min_days:
        raise ValidationError(
            f"Insufficient data: need at least {min_days} days, got {len(spread_data)}",
            field="spread_data",
            value=len(spread_data),
        )

    for i, data in enumerate(spread_data):
        if pd.isna(data.spread):
            raise ValidationError(
                f"Spread value at index {i} is NaN",
                field="spread",
                value=data.spread,
            )

    return True
