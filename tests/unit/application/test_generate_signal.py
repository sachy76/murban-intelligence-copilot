"""Unit tests for generate signal use case."""

from datetime import datetime

import pytest

from murban_copilot.application.generate_signal import GenerateSignalUseCase
from murban_copilot.domain.entities import MarketSignal
from murban_copilot.domain.exceptions import LLMInferenceError, ValidationError
from murban_copilot.infrastructure.llm.mock_client import MockLlamaClient


class TrackingMockLLM:
    """Mock LLM that tracks which step it was called for."""

    def __init__(self, name: str):
        self.name = name
        self.call_count = 0
        self.prompts = []

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, use_cache: bool = True) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        return f"SIGNAL: neutral\nCONFIDENCE: 0.5\n{self.name} response"

    def is_available(self) -> bool:
        return True


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


class TestDualLLMConfiguration:
    """Tests for dual-LLM configuration support."""

    def test_uses_separate_clients_for_analysis_and_extraction(
        self,
        sample_spread_data,
        sample_moving_averages,
        sample_trend_summary,
    ):
        """Test that separate clients are used for analysis and extraction."""
        analysis_llm = TrackingMockLLM("analysis")
        extraction_llm = TrackingMockLLM("extraction")

        use_case = GenerateSignalUseCase(
            llm_client=analysis_llm,
            extraction_client=extraction_llm,
        )

        use_case.execute(
            sample_spread_data,
            sample_moving_averages,
            sample_trend_summary,
        )

        assert analysis_llm.call_count == 1
        assert extraction_llm.call_count == 1

    def test_defaults_extraction_to_llm_client(
        self,
        sample_spread_data,
        sample_moving_averages,
        sample_trend_summary,
    ):
        """Test that extraction defaults to llm_client if not provided."""
        mock_llm = MockLlamaClient()

        use_case = GenerateSignalUseCase(llm_client=mock_llm)

        use_case.execute(
            sample_spread_data,
            sample_moving_averages,
            sample_trend_summary,
        )

        # Both calls go to the same client
        assert mock_llm.call_count == 2

    def test_custom_inference_parameters(
        self,
        sample_spread_data,
        sample_moving_averages,
        sample_trend_summary,
    ):
        """Test that custom inference parameters are used."""
        analysis_llm = TrackingMockLLM("analysis")
        extraction_llm = TrackingMockLLM("extraction")

        use_case = GenerateSignalUseCase(
            llm_client=analysis_llm,
            extraction_client=extraction_llm,
            analysis_max_tokens=4096,
            analysis_temperature=0.8,
            extraction_max_tokens=512,
            extraction_temperature=0.2,
        )

        # Just verify the use case can be created with custom params
        assert use_case.analysis_max_tokens == 4096
        assert use_case.analysis_temperature == 0.8
        assert use_case.extraction_max_tokens == 512
        assert use_case.extraction_temperature == 0.2

    def test_is_llm_available_checks_both_clients(self):
        """Test LLM availability check works with dual clients."""
        analysis_llm = TrackingMockLLM("analysis")
        extraction_llm = TrackingMockLLM("extraction")

        use_case = GenerateSignalUseCase(
            llm_client=analysis_llm,
            extraction_client=extraction_llm,
        )

        assert use_case.is_llm_available() is True

    def test_is_llm_available_false_if_extraction_unavailable(self):
        """Test availability is False if extraction client unavailable."""
        class UnavailableLLM:
            def generate(self, *args, **kwargs):
                return "response"

            def is_available(self):
                return False

        analysis_llm = TrackingMockLLM("analysis")
        extraction_llm = UnavailableLLM()

        use_case = GenerateSignalUseCase(
            llm_client=analysis_llm,
            extraction_client=extraction_llm,
        )

        assert use_case.is_llm_available() is False
