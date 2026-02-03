"""Prompt templates for LLM-based market analysis."""

from typing import Sequence

from murban_copilot.domain.entities import MovingAverages, SpreadData


class TraderTalkTemplate:
    """Template for generating trader-style market analysis.

    Note: This is a utility class with only class attributes and classmethods.
    No instantiation needed.
    """

    DISCLAIMER = (
        "DISCLAIMER: This is AI-generated analysis for informational purposes only. "
        "It does not constitute financial advice. Always conduct your own research "
        "and consult with qualified financial advisors before making trading decisions. "
        "Past performance is not indicative of future results."
    )

    SYSTEM_PROMPT = """
    ROLE & CONTEXT
        You are a Senior Quantitative Strategist and Crude Oil Trader providing institutional-grade market analysis.
        Your tone is professional, concise, and accessible, using standard energy trading terminology.
        Your objective is to deliver actionable insights while maintaining a balanced, risk-aware perspective.
        You must explicitly acknowledge uncertainty and avoid over-confidence.

        Your primary analytical focus is on WTI–Brent crude oil spread dynamics, interpreting what the spread signals about:
            -   Regional supply–demand imbalances
            -   Logistics and infrastructure constraints
            -   Macro and geopolitical influences
            -   Short-term vs medium-term trading implications
    """

    ANALYSIS_TEMPLATE = """
    TASK
        Analyze the WTI–Brent crude oil spread data provided below and generate a comprehensive Executive Trading Brief suitable for senior traders, portfolio managers, and risk committees.
    
    INPUT DATA
        -   Current Spread: ${current_spread:.2f} per barrel
        -   5-Day Moving Average: {ma_5}
        -   20-Day Moving Average: {ma_20}
        -   Trend Signal: {trend}
        -   5-Day Spread Change: {spread_change_5d}
        -   20-Day Spread Change: {spread_change_20d}

        Recent Spread History (Last 5 Trading Days):
            {recent_history}

    REQUIRED OUTPUT
        Generate an Executive Trading Brief with clear section headers and concise, decision-oriented commentary.
        At a minimum, include the following sections (add others where relevant):
        -   Market Overview
            -   Current spread level in historical and recent context
            -   Immediate market interpretation
            -   Assessment of current market regime (contango/backwardation implications)
        -   Technical Analysis
            -   Relationship between current spread, 5-day MA, and 20-day MA
            -   Momentum, mean-reversion, or trend-continuation signals and velocity assessment
            -   Key technical inflection levels
            -   Identify any chart patterns or divergences
        -   Fundamental Drivers
            -   Supply-side dynamics (US production, exports, OPEC+, inventories, US vs North Sea)
            -   Demand-side factors (regional demand, refinery utilization, US vs North Sea)
            -   Logistics, infrastructure, or quality differentials
            -   Seasonal patterns and their current impact
        -   Risk Assessment
            -   Key upside and downside risks to the spread
            -   Event risks (data releases, OPEC decisions, geopolitical events)
            -   Volatility considerations
        -   Trading Implications
            -   Near-term and medium-term implications for spread traders
            -   Directional bias with supporting rationale
            -   Confidence level: low/medium/high (with justification)
            -   Potential tactical opportunities (hedging, arbitrage, timing)
            -   Alternative interpretations if signals fail
        -   Strategic Considerations
            -   Positioning guidance (e.g., tactical vs structural)
            -   Hedging or optionality considerations
            -   Time-horizon alignment

    OUTPUT CONSTRAINTS
        -   Use professional trading terminology
        -   Be specific with numbers, levels, and directional bias where possible
        -   Clearly acknowledge uncertainty
        -   Present probability-weighted scenarios (e.g., base case, bull case, bear case)

    FINAL LINE (MANDATORY)
        -   End the brief with the following format:
        -   SIGNAL: bullish / bearish / neutral
        -   CONFIDENCE: numeric value between 0.0 and 1.0
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


class SignalExtractionTemplate:
    """Template for extracting structured signals from LLM output.

    Note: This is a utility class with only class attributes and classmethods.
    No instantiation needed.
    """

    EXTRACTION_PROMPT = """
    ROLE
        You are a Senior Quantitative Sentiment Analysis Engine specialized in energy markets and crude oil spreads.
        Your task is to perform sentiment analysis on the provided market commentary and classify the overall directional sentiment of the analysis as it relates to the WTI–Brent spread.
        You are not generating new market views.
        You are inferring sentiment expressed by the analyst and converting it into a trading signal.
    
    INPUT
        Market Analysis Text: {analysis}

    SENTIMENT INTERPRETATION RULES
        -   Evaluate language tone, directional bias, and risk framing
        -   Identify whether sentiment reflects:
            -   Positive / constructive outlook → bullish
            -   Negative / deteriorating outlook → bearish
            -   Balanced / conflicting / range-bound outlook → neutral
        -   Weigh base-case scenarios more heavily than tail risks
        -   Convert the strength of sentiment into a confidence score
        -   Do not infer conviction beyond what the text supports

    SIGNAL MAPPING
        -   Positive sentiment → SIGNAL: bullish
        -   Negative sentiment → SIGNAL: bearish
        -   Mixed or balanced sentiment → SIGNAL: neutral


    OUTPUT FORMAT (STRICT — NO VARIATIONS ALLOWED)
        -   SIGNAL: [bullish / bearish / neutral]
        -   CONFIDENCE: [0.0–1.0]
        -   SUMMARY: [One sentence, professional trading summary]

    CONSTRAINTS
        -   Perform sentiment classification only
        -   Do not restate data or introduce new analysis
        -   One sentence only in SUMMARY
        -   Use professional trading language
        -   No additional text or formatting
    """

    @classmethod
    def format_extraction_prompt(cls, analysis: str) -> str:
        """Format the signal extraction prompt."""
        return cls.EXTRACTION_PROMPT.format(analysis=analysis)
