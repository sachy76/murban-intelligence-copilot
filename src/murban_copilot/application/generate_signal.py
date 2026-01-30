"""Use case for generating market signals using LLM."""

import re
from datetime import datetime
from typing import Sequence

from murban_copilot.domain.entities import MarketSignal, MovingAverages, SpreadData
from murban_copilot.domain.exceptions import LLMInferenceError, ValidationError
from murban_copilot.domain.validators import validate_llm_input
from murban_copilot.infrastructure.llm.prompt_templates import (
    SignalExtractionTemplate,
    TraderTalkTemplate,
)
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
            # Step 1: Generate free-form analysis
            analysis = self.llm.generate(
                prompt,
                max_tokens=1024,
                temperature=0.7,
            )
            logger.debug("Step 1 complete: Generated analysis")

            # Step 2: Extract structured signal using dedicated prompt
            extraction_prompt = SignalExtractionTemplate.format_extraction_prompt(analysis)
            extraction_response = self.llm.generate(
                extraction_prompt,
                max_tokens=128,
                temperature=0.3,  # Lower temperature for more consistent extraction
            )
            logger.debug("Step 2 complete: Extracted signal")

        except Exception as e:
            raise LLMInferenceError(
                f"Failed to generate analysis: {str(e)}",
                original_error=e,
            )

        signal, confidence = self._extract_signal(extraction_response, analysis)

        return MarketSignal(
            timestamp=datetime.utcnow(),
            signal=signal,
            confidence=confidence,
            analysis=analysis,
            disclaimer=TraderTalkTemplate.DISCLAIMER,
        )

    def _extract_signal(
        self, extraction_response: str, original_analysis: str
    ) -> tuple[str, float]:
        """
        Extract signal and confidence from LLM extraction response.

        Uses the structured extraction response first, falls back to
        keyword analysis of the original response if extraction fails.

        Args:
            extraction_response: Response from extraction prompt
            original_analysis: Original free-form analysis (fallback)

        Returns:
            Tuple of (signal, confidence)
        """
        signal = "neutral"
        confidence = 0.5

        # Try to extract from structured extraction response first
        signal_match = re.search(
            r"SIGNAL:\s*(bullish|bearish|neutral)",
            extraction_response,
            re.IGNORECASE,
        )
        if signal_match:
            signal = signal_match.group(1).lower()
            logger.debug(f"Extracted signal from extraction response: {signal}")

        confidence_match = re.search(
            r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)",
            extraction_response,
            re.IGNORECASE,
        )
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
                logger.debug(f"Extracted confidence: {confidence}")
            except ValueError:
                pass

        # Fallback: keyword analysis on original analysis if extraction failed
        if not signal_match:
            logger.debug("Extraction failed, falling back to keyword analysis")
            analysis_lower = original_analysis.lower()
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
