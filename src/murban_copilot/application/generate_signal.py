"""Use case for generating market signals using LLM."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Sequence, TYPE_CHECKING

from murban_copilot.domain.entities import MarketSignal, MovingAverages, SpreadData
from murban_copilot.domain.exceptions import LLMInferenceError, ValidationError
from murban_copilot.domain.validators import validate_llm_input
from murban_copilot.infrastructure.llm.prompt_templates import (
    SignalExtractionTemplate,
    TraderTalkTemplate,
)
from murban_copilot.infrastructure.llm.protocols import LLMInference
from murban_copilot.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from murban_copilot.domain.config import SignalConfig

logger = get_logger(__name__)

# Default fallback values
DEFAULT_SIGNAL = "neutral"
DEFAULT_CONFIDENCE = 0.5
DEFAULT_BULLISH_KEYWORDS = ["bullish", "upward", "positive", "buy"]
DEFAULT_BEARISH_KEYWORDS = ["bearish", "downward", "negative", "sell"]


class GenerateSignalUseCase:
    """Use case for generating AI-powered market signals."""

    def __init__(
        self,
        llm_client: LLMInference,
        extraction_client: LLMInference | None = None,
        analysis_max_tokens: int = 2048,
        analysis_temperature: float = 0.7,
        extraction_max_tokens: int = 1024,
        extraction_temperature: float = 0.3,
        signal_config: "SignalConfig | None" = None,
    ) -> None:
        """
        Initialize the use case.

        Args:
            llm_client: LLM client for analysis inference
            extraction_client: Optional separate LLM client for extraction.
                               Defaults to llm_client if not provided.
            analysis_max_tokens: Max tokens for analysis generation
            analysis_temperature: Temperature for analysis generation
            extraction_max_tokens: Max tokens for extraction
            extraction_temperature: Temperature for extraction
            signal_config: Optional signal configuration for defaults and keywords
        """
        self.llm = llm_client
        self.extraction_llm = extraction_client or llm_client
        self.analysis_max_tokens = analysis_max_tokens
        self.analysis_temperature = analysis_temperature
        self.extraction_max_tokens = extraction_max_tokens
        self.extraction_temperature = extraction_temperature

        # Signal extraction configuration
        if signal_config:
            self.default_signal = signal_config.default_signal
            self.default_confidence = signal_config.default_confidence
            self.bullish_keywords = signal_config.bullish_keywords
            self.bearish_keywords = signal_config.bearish_keywords
        else:
            self.default_signal = DEFAULT_SIGNAL
            self.default_confidence = DEFAULT_CONFIDENCE
            self.bullish_keywords = DEFAULT_BULLISH_KEYWORDS
            self.bearish_keywords = DEFAULT_BEARISH_KEYWORDS

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
            # Step 1: Generate comprehensive analysis
            analysis = self.llm.generate(
                prompt,
                max_tokens=self.analysis_max_tokens,
                temperature=self.analysis_temperature,
            )
            logger.debug("Step 1 complete: Generated analysis")

            # Step 2: Extract structured signal using dedicated prompt
            extraction_prompt = SignalExtractionTemplate.format_extraction_prompt(analysis)
            extraction_response = self.extraction_llm.generate(
                extraction_prompt,
                max_tokens=self.extraction_max_tokens,
                temperature=self.extraction_temperature,
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
        signal = self.default_signal
        confidence = self.default_confidence

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

            bullish_count = sum(1 for kw in self.bullish_keywords if kw in analysis_lower)
            bearish_count = sum(1 for kw in self.bearish_keywords if kw in analysis_lower)

            if bullish_count > bearish_count:
                signal = "bullish"
            elif bearish_count > bullish_count:
                signal = "bearish"

        return signal, confidence

    def is_llm_available(self) -> bool:
        """
        Check if the LLM is available.

        Returns:
            True if both analysis and extraction LLMs are ready for inference
        """
        return self.llm.is_available() and self.extraction_llm.is_available()
