"""Unit tests for prompt templates."""

from datetime import datetime

import pytest

from murban_copilot.domain.entities import MovingAverages, SpreadData
from murban_copilot.infrastructure.llm.prompt_templates import (
    SignalExtractionTemplate,
    TraderTalkTemplate,
)


class TestTraderTalkTemplate:
    """Tests for TraderTalkTemplate."""

    def test_disclaimer_exists(self):
        """Test that disclaimer exists and contains key phrases."""
        assert "financial advice" in TraderTalkTemplate.DISCLAIMER.lower()
        assert "informational" in TraderTalkTemplate.DISCLAIMER.lower()

    def test_system_prompt_exists(self):
        """Test that system prompt exists and sets context."""
        assert "trader" in TraderTalkTemplate.SYSTEM_PROMPT.lower()
        assert "crude oil" in TraderTalkTemplate.SYSTEM_PROMPT.lower()

    def test_format_analysis_prompt(
        self,
        sample_spread_data,
        sample_moving_averages,
        sample_trend_summary,
    ):
        """Test analysis prompt formatting."""
        prompt = TraderTalkTemplate.format_analysis_prompt(
            sample_spread_data,
            sample_moving_averages,
            sample_trend_summary,
        )

        assert "Current Spread" in prompt
        assert "Moving Average" in prompt
        assert "Recent Spread History" in prompt

    def test_format_analysis_prompt_with_none_values(self):
        """Test prompt formatting with None values in summary."""
        spread_data = [
            SpreadData(
                date=datetime(2024, 1, i),
                murban_close=85.0,
                brent_close=82.0,
                spread=3.0,
            )
            for i in range(1, 6)
        ]
        moving_averages = [
            MovingAverages(
                date=datetime(2024, 1, i),
                spread=3.0,
            )
            for i in range(1, 6)
        ]
        trend_summary = {
            "current_spread": None,
            "ma_5": None,
            "ma_20": None,
            "trend": None,
            "spread_change_5d": None,
            "spread_change_20d": None,
        }

        prompt = TraderTalkTemplate.format_analysis_prompt(
            spread_data,
            moving_averages,
            trend_summary,
        )

        assert "N/A" in prompt

    def test_get_full_prompt(
        self,
        sample_spread_data,
        sample_moving_averages,
        sample_trend_summary,
    ):
        """Test full prompt generation."""
        prompt = TraderTalkTemplate.get_full_prompt(
            sample_spread_data,
            sample_moving_averages,
            sample_trend_summary,
        )

        # Should include both system prompt and analysis
        assert "trader" in prompt.lower()
        assert "Current Market Data" in prompt


class TestSignalExtractionTemplate:
    """Tests for SignalExtractionTemplate."""

    def test_format_extraction_prompt(self):
        """Test extraction prompt formatting."""
        analysis = "The market shows bullish momentum."
        prompt = SignalExtractionTemplate.format_extraction_prompt(analysis)

        assert "bullish momentum" in prompt
        assert "SIGNAL:" in prompt
        assert "CONFIDENCE:" in prompt
