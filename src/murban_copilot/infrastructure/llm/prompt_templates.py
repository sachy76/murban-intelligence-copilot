"""Prompt templates for LLM-based market analysis."""

from dataclasses import dataclass
from typing import Sequence

from murban_copilot.domain.entities import MovingAverages, SpreadData


@dataclass
class TraderTalkTemplate:
    """Template for generating trader-style market analysis."""

    DISCLAIMER = (
        "DISCLAIMER: This is AI-generated analysis for informational purposes only. "
        "It does not constitute financial advice. Always conduct your own research "
        "and consult with qualified financial advisors before making trading decisions. "
        "Past performance is not indicative of future results."
    )

    SYSTEM_PROMPT = """You are an Senior Quantitative Strategist crude oil trader providing market analysis.
    Your style is professional but accessible, using common trading terminology.
    Provide concise, actionable insights based on the data provided.
    Always maintain a balanced perspective and acknowledge uncertainty.
    Focus on the WTI-Brent spread dynamics and what they might indicate.
"""

    ANALYSIS_TEMPLATE = """Analyze the following WTI-Brent crude oil spread data and provide comprehensive market analysis.

## Current Market Data
- Current Spread: ${current_spread:.2f}/barrel
- 5-Day Moving Average: {ma_5}
- 20-Day Moving Average: {ma_20}
- Trend Signal: {trend}
- 5-Day Spread Change: {spread_change_5d}
- 20-Day Spread Change: {spread_change_20d}

## Recent Spread History (last 5 days)
{recent_history}

## Your Task
Generate a comprehensive "Executive Trading Brief" with the following sections:

### 1. Market Overview
- Current spread positioning relative to historical norms
- Assessment of current market regime (contango/backwardation implications)

### 2. Technical Analysis
- Moving average crossover analysis (MA5 vs MA20)
- Spread momentum and velocity assessment
- Key support/resistance levels for the spread
- Identify any chart patterns or divergences

### 3. Fundamental Drivers
- What macro factors could be driving the current spread dynamics
- Regional supply/demand imbalances (US vs North Sea)
- Refinery margin implications
- Seasonal patterns and their current impact

### 4. Risk Assessment
- Downside risks to current positioning
- Potential catalysts for spread widening/narrowing
- Volatility outlook

### 5. Trading Implications
- Outlook: bullish/bearish/neutral with clear reasoning
- Confidence level: low/medium/high (with justification)
- Potential tactical opportunities (hedging, arbitrage, timing)
- Recommended position sizing considerations

### 6. Strategic Considerations
- How does this market state impact trading strategy
- Key levels to watch for position adjustments

## OUTPUT FORMAT
- Use professional trading terminology
- Be specific with numbers and levels where possible
- Acknowledge uncertainty and provide probability-weighted scenarios
- End with: SIGNAL: [bullish/bearish/neutral], CONFIDENCE: [0.0-1.0]
"""

    @classmethod
    def format_analysis_prompt(
        cls,
        spread_data: Sequence[SpreadData],
        moving_averages: Sequence[MovingAverages],
        trend_summary: dict[str, object],
    ) -> str:
        """
        Format the analysis prompt with market data.

        Args:
            spread_data: Recent spread data
            moving_averages: Moving average calculations
            trend_summary: Summary statistics

        Returns:
            Formatted prompt string
        """
        sorted_spreads = sorted(spread_data, key=lambda x: x.date, reverse=True)[:5]
        recent_history = "\n".join(
            f"- {s.date.strftime('%Y-%m-%d')}: WTI ${s.wti_close:.2f}, "
            f"Brent ${s.brent_close:.2f}, Spread ${s.spread:+.2f}"
            for s in sorted_spreads
        )

        def format_value(val: object, prefix: str = "$", suffix: str = "") -> str:
            if val is None:
                return "N/A (insufficient data)"
            if isinstance(val, float):
                return f"{prefix}{val:+.2f}{suffix}" if prefix == "$" else f"{val:.2f}"
            return str(val)

        return cls.ANALYSIS_TEMPLATE.format(
            current_spread=trend_summary.get("current_spread", 0) or 0,
            ma_5=format_value(trend_summary.get("ma_5")),
            ma_20=format_value(trend_summary.get("ma_20")),
            trend=trend_summary.get("trend") or "undetermined",
            spread_change_5d=format_value(trend_summary.get("spread_change_5d")),
            spread_change_20d=format_value(trend_summary.get("spread_change_20d")),
            recent_history=recent_history,
        )

    @classmethod
    def get_full_prompt(
        cls,
        spread_data: Sequence[SpreadData],
        moving_averages: Sequence[MovingAverages],
        trend_summary: dict[str, object],
    ) -> str:
        """
        Get the complete prompt with system context.

        Args:
            spread_data: Recent spread data
            moving_averages: Moving average calculations
            trend_summary: Summary statistics

        Returns:
            Complete prompt with system instructions
        """
        analysis_prompt = cls.format_analysis_prompt(
            spread_data, moving_averages, trend_summary
        )
        return f"{cls.SYSTEM_PROMPT}\n\n{analysis_prompt}"


@dataclass
class SignalExtractionTemplate:
    """Template for extracting structured signals from LLM output."""

    EXTRACTION_PROMPT = """Based on the following market analysis, extract the key signal information.

    Analysis:
    {analysis}

    Respond in exactly this format:
    SIGNAL: [bullish/bearish/neutral]
    CONFIDENCE: [0.0-1.0]
    SUMMARY: [One sentence summary]
    """

    @classmethod
    def format_extraction_prompt(cls, analysis: str) -> str:
        """Format the signal extraction prompt."""
        return cls.EXTRACTION_PROMPT.format(analysis=analysis)
