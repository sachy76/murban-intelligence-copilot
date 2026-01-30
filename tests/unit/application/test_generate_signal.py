"""Unit tests for generate signal use case."""

from datetime import datetime

import pytest

from murban_copilot.application.generate_signal import GenerateSignalUseCase
from murban_copilot.domain.entities import MarketSignal
from murban_copilot.domain.exceptions import LLMInferenceError, ValidationError
from murban_copilot.infrastructure.llm.llm_client import MockLlamaClient


class TestGenerateSignalUseCase:
    """Tests for GenerateSignalUseCase."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        return MockLlamaClient()

    @pytest.fixture
    def use_case(self, mock_llm):
        """Create a GenerateSignalUseCase instance."""
        return GenerateSignalUseCase(mock_llm)

    def test_execute_returns_market_signal(
        self,
        use_case,
        sample_spread_data,
        sample_moving_averages,
        sample_trend_summary,
    ):
        """Test that execute returns a MarketSignal."""
        signal = use_case.execute(
            sample_spread_data,
            sample_moving_averages,
            sample_trend_summary,
        )

        assert isinstance(signal, MarketSignal)

    def test_execute_signal_has_required_fields(
        self,
        use_case,
        sample_spread_data,
        sample_moving_averages,
        sample_trend_summary,
    ):
        """Test that signal has required fields."""
        signal = use_case.execute(
            sample_spread_data,
            sample_moving_averages,
            sample_trend_summary,
        )

        assert signal.signal in ("bullish", "bearish", "neutral")
        assert 0.0 <= signal.confidence <= 1.0
        assert len(signal.analysis) > 0
        assert len(signal.disclaimer) > 0

    def test_execute_validates_input(
        self,
        use_case,
        sample_moving_averages,
        sample_trend_summary,
    ):
        """Test that execute validates input data."""
        with pytest.raises(ValidationError):
            use_case.execute(
                [],  # Empty spread data
                sample_moving_averages,
                sample_trend_summary,
            )

    def test_execute_calls_llm(
        self,
        mock_llm,
        use_case,
        sample_spread_data,
        sample_moving_averages,
        sample_trend_summary,
    ):
        """Test that execute calls the LLM."""
        use_case.execute(
            sample_spread_data,
            sample_moving_averages,
            sample_trend_summary,
        )

        assert mock_llm.call_count == 2  # Analysis + extraction
        assert mock_llm.last_prompt is not None

    def test_extract_signal_bullish(self, use_case):
        """Test signal extraction for bullish analysis."""
        extraction_response = "SIGNAL: bullish CONFIDENCE: 0.8"
        original_analysis = "Market is bullish."
        signal, confidence = use_case._extract_signal(extraction_response, original_analysis)

        assert signal == "bullish"
        assert confidence == 0.8

    def test_extract_signal_bearish(self, use_case):
        """Test signal extraction for bearish analysis."""
        extraction_response = "SIGNAL: bearish\nCONFIDENCE: 0.65"
        original_analysis = "Market is bearish."
        signal, confidence = use_case._extract_signal(extraction_response, original_analysis)

        assert signal == "bearish"
        assert confidence == 0.65

    def test_extract_signal_neutral(self, use_case):
        """Test signal extraction for neutral analysis."""
        extraction_response = "SIGNAL: neutral CONFIDENCE: 0.5"
        original_analysis = "Market is neutral."
        signal, confidence = use_case._extract_signal(extraction_response, original_analysis)

        assert signal == "neutral"
        assert confidence == 0.5

    def test_extract_signal_from_keywords(self, use_case):
        """Test signal extraction from keywords when explicit signal missing."""
        extraction_response = "Unable to determine signal"  # No SIGNAL: in extraction
        original_analysis = "The market shows bullish upward positive momentum"
        signal, _ = use_case._extract_signal(extraction_response, original_analysis)

        assert signal == "bullish"

    def test_extract_signal_defaults(self, use_case):
        """Test signal extraction defaults."""
        extraction_response = "Unable to parse"
        original_analysis = "Unclear market conditions"
        signal, confidence = use_case._extract_signal(extraction_response, original_analysis)

        assert signal == "neutral"
        assert confidence == 0.5

    def test_is_llm_available(self, use_case):
        """Test LLM availability check."""
        assert use_case.is_llm_available() is True

    def test_execute_handles_llm_error(
        self,
        sample_spread_data,
        sample_moving_averages,
        sample_trend_summary,
    ):
        """Test that execute handles LLM errors."""
        class FailingLLM:
            def generate(self, *args, **kwargs):
                raise Exception("LLM failed")

            def is_available(self):
                return True

        use_case = GenerateSignalUseCase(FailingLLM())

        with pytest.raises(LLMInferenceError):
            use_case.execute(
                sample_spread_data,
                sample_moving_averages,
                sample_trend_summary,
            )
