"""Use case for generating market signals using LLM."""

import re
from datetime import datetime
from typing import Sequence

from murban_copilot.domain.entities import MarketSignal, MovingAverages, SpreadData
from murban_copilot.domain.exceptions import LLMInferenceError, ValidationError
from murban_copilot.domain.validators import validate_llm_input
from murban_copilot.infrastructure.llm.prompt_templates import TraderTalkTemplate
from murban_copilot.infrastructure.llm.protocols import LLMInference
from murban_copilot.infrastructure.logging import get_logger

logger = get_logger(__name__)


class GenerateSignalUseCase:
    """Use case for generating AI-powered market signals."""

    def __init__(self, llm_client: LLMInference) -> None:
        """
        Initialize the use case.

        Args:
            llm_client: LLM client for inference
        """
        self.llm = llm_client

    def execute(
        self,
        spread_data: Sequence[SpreadData],
        moving_averages: Sequence[MovingAverages],
        trend_summary: dict[str, object],
    ) -> MarketSignal:
        """
        Generate a market signal based on spread analysis.

        Args:
            spread_data: Recent spread data
            moving_averages: Moving average calculations
            trend_summary: Trend summary statistics

        Returns:
            MarketSignal with analysis

        Raises:
            ValidationError: If input data is invalid
            LLMInferenceError: If LLM inference fails
        """
        logger.info("Generating market signal")

        validate_llm_input(spread_data)

        prompt = TraderTalkTemplate.get_full_prompt(
            spread_data,
            moving_averages,
            trend_summary,
        )

        logger.debug(f"Generated prompt length: {len(prompt)} chars")

        try:
            analysis = self.llm.generate(
                prompt,
                max_tokens=512,
                temperature=0.7,
            )
        except Exception as e:
            raise LLMInferenceError(
                f"Failed to generate analysis: {str(e)}",
                original_error=e,
            )

        signal, confidence = self._extract_signal(analysis)

        return MarketSignal(
            timestamp=datetime.utcnow(),
            signal=signal,
            confidence=confidence,
            analysis=analysis,
            disclaimer=TraderTalkTemplate.DISCLAIMER,
        )

    def _extract_signal(self, analysis: str) -> tuple[str, float]:
        """
        Extract signal and confidence from LLM analysis.

        Args:
            analysis: Raw LLM output

        Returns:
            Tuple of (signal, confidence)
        """
        signal = "neutral"
        confidence = 0.5

        signal_match = re.search(
            r"SIGNAL:\s*(bullish|bearish|neutral)",
            analysis,
            re.IGNORECASE,
        )
        if signal_match:
            signal = signal_match.group(1).lower()

        confidence_match = re.search(
            r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)",
            analysis,
            re.IGNORECASE,
        )
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass

        if not signal_match:
            analysis_lower = analysis.lower()
            bullish_keywords = ["bullish", "upward", "positive", "buy"]
            bearish_keywords = ["bearish", "downward", "negative", "sell"]

            bullish_count = sum(1 for kw in bullish_keywords if kw in analysis_lower)
            bearish_count = sum(1 for kw in bearish_keywords if kw in analysis_lower)

            if bullish_count > bearish_count:
                signal = "bullish"
            elif bearish_count > bullish_count:
                signal = "bearish"

        return signal, confidence

    def is_llm_available(self) -> bool:
        """
        Check if the LLM is available.

        Returns:
            True if LLM is ready for inference
        """
        return self.llm.is_available()
