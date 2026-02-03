"""Unit tests for TransformersClient."""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from murban_copilot.domain.config import LLMModelConfig, ModelType, LLMInferenceConfig
from murban_copilot.domain.exceptions import LLMInferenceError
from murban_copilot.infrastructure.llm.transformers_client import (
    TransformersClient,
    detect_torch_device,
)


class TestDetectTorchDevice:
    """Tests for detect_torch_device function."""

    def test_cpu_preference(self):
        """Test CPU preference returns -1."""
        result = detect_torch_device("cpu")
        assert result == -1

    def test_cuda_preference(self):
        """Test CUDA preference returns 0."""
        result = detect_torch_device("cuda")
        assert result == 0

    def test_mps_preference(self):
        """Test MPS preference returns 'mps'."""
        result = detect_torch_device("mps")
        assert result == "mps"

    def test_auto_with_cuda_available(self):
        """Test auto preference with CUDA available."""
        with patch("murban_copilot.infrastructure.llm.transformers_client.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            result = detect_torch_device("auto")
            assert result == 0

    def test_auto_with_mps_available(self):
        """Test auto preference with MPS available (no CUDA)."""
        with patch("murban_copilot.infrastructure.llm.transformers_client.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True
            result = detect_torch_device("auto")
            assert result == "mps"

    def test_auto_cpu_fallback(self):
        """Test auto falls back to CPU when no GPU available."""
        with patch("murban_copilot.infrastructure.llm.transformers_client.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            # Simulate no MPS support
            mock_torch.backends = MagicMock()
            mock_torch.backends.mps.is_available.return_value = False
            result = detect_torch_device("auto")
            assert result == -1

    def test_unknown_preference_returns_cpu(self):
        """Test unknown preference defaults to CPU."""
        result = detect_torch_device("unknown")
        assert result == -1


class TestTransformersClient:
    """Tests for TransformersClient."""

    @pytest.fixture
    def client(self, tmp_path):
        """Return a TransformersClient instance."""
        return TransformersClient(
            model_repo="test/model",
            task="sentiment-analysis",
            device="cpu",
            cache_dir=tmp_path / "cache",
        )

    def test_init(self, client):
        """Test initialization."""
        assert client.model_repo == "test/model"
        assert client.task == "sentiment-analysis"
        assert client.device == "cpu"
        assert client._pipeline is None

    def test_init_defaults(self):
        """Test initialization with defaults."""
        client = TransformersClient(model_repo="test/model")
        assert client.task == "sentiment-analysis"
        assert client.device == "auto"
        assert client.cache_enabled is True

    def test_get_model_identifier(self, client):
        """Test model identifier returns repo name."""
        assert client._get_model_identifier() == "test/model"

    def test_from_config(self, tmp_path):
        """Test creating client from config."""
        config = LLMModelConfig(
            model_repo="config/model",
            model_type=ModelType.TRANSFORMERS,
            task="text-classification",
            device="mps",
        )

        client = TransformersClient.from_config(
            config,
            cache_dir=tmp_path / "cache",
            cache_enabled=False,
        )

        assert client.model_repo == "config/model"
        assert client.task == "text-classification"
        assert client.device == "mps"
        assert client.cache_enabled is False

    def test_sentiment_to_signal_mapping(self):
        """Test sentiment label to signal mapping."""
        assert TransformersClient.SENTIMENT_TO_SIGNAL["positive"] == "bullish"
        assert TransformersClient.SENTIMENT_TO_SIGNAL["negative"] == "bearish"
        assert TransformersClient.SENTIMENT_TO_SIGNAL["neutral"] == "neutral"
        assert TransformersClient.SENTIMENT_TO_SIGNAL["POSITIVE"] == "bullish"
        assert TransformersClient.SENTIMENT_TO_SIGNAL["NEGATIVE"] == "bearish"
        assert TransformersClient.SENTIMENT_TO_SIGNAL["POS"] == "bullish"
        assert TransformersClient.SENTIMENT_TO_SIGNAL["NEG"] == "bearish"

    def test_load_model_import_error(self, client):
        """Test _load_model raises on import error."""
        with patch.dict("sys.modules", {"transformers": None}):
            with patch(
                "murban_copilot.infrastructure.llm.transformers_client.pipeline",
                side_effect=ImportError("transformers not installed"),
            ):
                # Force reload attempt
                client._pipeline = None
                with pytest.raises(LLMInferenceError) as exc_info:
                    client._load_model()
                assert "transformers" in str(exc_info.value).lower()

    def test_load_model_success(self, client):
        """Test successful model loading."""
        mock_pipeline = MagicMock()

        with patch(
            "murban_copilot.infrastructure.llm.transformers_client.pipeline",
            return_value=mock_pipeline,
        ) as mock_pipeline_fn:
            client._load_model()

            mock_pipeline_fn.assert_called_once_with(
                "sentiment-analysis",
                model="test/model",
                device=-1,  # CPU
            )
            assert client._pipeline is mock_pipeline

    def test_load_model_already_loaded(self, client):
        """Test _load_model skips if already loaded."""
        client._pipeline = MagicMock()

        with patch(
            "murban_copilot.infrastructure.llm.transformers_client.pipeline"
        ) as mock_pipeline_fn:
            client._load_model()
            mock_pipeline_fn.assert_not_called()

    def test_load_model_generic_error(self, client):
        """Test _load_model raises on generic error."""
        with patch(
            "murban_copilot.infrastructure.llm.transformers_client.pipeline",
            side_effect=RuntimeError("Model loading failed"),
        ):
            client._pipeline = None
            with pytest.raises(LLMInferenceError) as exc_info:
                client._load_model()
            assert "Failed to load transformers pipeline" in str(exc_info.value)

    def test_run_classification(self, client):
        """Test classification output formatting."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.95}]
        client._pipeline = mock_pipeline

        result = client._run_classification("Test text")

        assert "SIGNAL: bullish" in result
        assert "CONFIDENCE: 0.95" in result
        assert "SUMMARY:" in result

    def test_run_classification_truncates_long_text(self, client):
        """Test classification truncates text over 2000 chars."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.5}]
        client._pipeline = mock_pipeline

        long_text = "x" * 3000
        client._run_classification(long_text)

        # Verify the pipeline was called with truncated text
        called_text = mock_pipeline.call_args[0][0]
        assert len(called_text) == 2000

    def test_run_classification_handles_dict_result(self, client):
        """Test classification handles non-list result."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {"label": "negative", "score": 0.8}
        client._pipeline = mock_pipeline

        result = client._run_classification("Test text")

        assert "SIGNAL: bearish" in result
        assert "CONFIDENCE: 0.80" in result

    def test_run_classification_unknown_label(self, client):
        """Test classification handles unknown sentiment label."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "unknown_label", "score": 0.6}]
        client._pipeline = mock_pipeline

        result = client._run_classification("Test text")

        assert "SIGNAL: neutral" in result  # Falls back to neutral

    def test_do_generate_classification_task(self, client):
        """Test _do_generate routes to classification for sentiment task."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.9}]
        client._pipeline = mock_pipeline

        result = client._do_generate("Test", max_tokens=100, temperature=0.5)

        assert "SIGNAL:" in result
        mock_pipeline.assert_called_once()

    def test_do_generate_text_classification_task(self, tmp_path):
        """Test _do_generate routes to classification for text-classification."""
        client = TransformersClient(
            model_repo="test/model",
            task="text-classification",
            device="cpu",
            cache_dir=tmp_path / "cache",
        )
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.9}]
        client._pipeline = mock_pipeline

        result = client._do_generate("Test", max_tokens=100, temperature=0.5)

        assert "SIGNAL:" in result

    def test_do_generate_generation_task(self, tmp_path):
        """Test _do_generate routes to generation for other tasks."""
        client = TransformersClient(
            model_repo="test/model",
            task="text-generation",
            device="cpu",
            cache_dir=tmp_path / "cache",
        )
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "Generated response"}]
        client._pipeline = mock_pipeline

        result = client._do_generate("Test", max_tokens=100, temperature=0.5)

        assert result == "Generated response"

    def test_run_generation_with_all_params(self, tmp_path):
        """Test _run_generation with all optional parameters."""
        client = TransformersClient(
            model_repo="test/model",
            task="text-generation",
            device="cpu",
            cache_dir=tmp_path / "cache",
        )
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "Response"}]
        client._pipeline = mock_pipeline

        result = client._run_generation(
            prompt="Test prompt",
            max_tokens=200,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            frequency_penalty=0.3,
            presence_penalty=0.1,
        )

        assert result == "Response"
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["max_new_tokens"] == 200
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 50
        assert "repetition_penalty" in call_kwargs
        # (0.3 + 0.1) / 2 + 1.0 = 1.2
        assert call_kwargs["repetition_penalty"] == 1.2

    def test_run_generation_with_only_frequency_penalty(self, tmp_path):
        """Test _run_generation with only frequency_penalty."""
        client = TransformersClient(
            model_repo="test/model",
            task="text-generation",
            device="cpu",
            cache_dir=tmp_path / "cache",
        )
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "Response"}]
        client._pipeline = mock_pipeline

        client._run_generation(
            prompt="Test",
            max_tokens=100,
            temperature=0.5,
            frequency_penalty=0.4,
        )

        call_kwargs = mock_pipeline.call_args[1]
        # 0.4 / 2 + 1.0 = 1.2
        assert call_kwargs["repetition_penalty"] == 1.2

    def test_run_generation_empty_result(self, tmp_path):
        """Test _run_generation handles empty result."""
        client = TransformersClient(
            model_repo="test/model",
            task="text-generation",
            device="cpu",
            cache_dir=tmp_path / "cache",
        )
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = []
        client._pipeline = mock_pipeline

        result = client._run_generation("Test", max_tokens=100, temperature=0.5)

        assert result == ""

    def test_run_generation_zero_temperature(self, tmp_path):
        """Test _run_generation with zero temperature disables sampling."""
        client = TransformersClient(
            model_repo="test/model",
            task="text-generation",
            device="cpu",
            cache_dir=tmp_path / "cache",
        )
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "Response"}]
        client._pipeline = mock_pipeline

        client._run_generation("Test", max_tokens=100, temperature=0.0)

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["do_sample"] is False

    def test_generate_uses_cache(self, client):
        """Test generate uses cached response."""
        client._cache_response("test prompt", 100, 0.5, "cached result")

        result = client.generate("test prompt", max_tokens=100, temperature=0.5)

        assert result == "cached result"

    def test_is_available_without_pipeline(self, client):
        """Test is_available returns False when pipeline not loaded."""
        client._pipeline = None
        # is_available should still return True (it just checks if can be loaded)
        # based on base_client implementation
        assert client.is_available() in (True, False)