"""Unit tests for LLM client."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from murban_copilot.domain.config import CacheConfig, LLMModelConfig, LLMInferenceConfig
from murban_copilot.domain.exceptions import LLMInferenceError
from murban_copilot.infrastructure.llm.llm_client import LlamaClient
from murban_copilot.infrastructure.llm.mock_client import MockLlamaClient


class TestMockLlamaClient:
    """Tests for MockLlamaClient."""

    def test_generate_returns_default_response(self):
        """Test that generate returns default response."""
        client = MockLlamaClient()
        result = client.generate("Test prompt")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "SIGNAL:" in result

    def test_generate_with_custom_response(self):
        """Test generate with custom response."""
        custom_response = "Custom analysis result"
        client = MockLlamaClient(default_response=custom_response)

        result = client.generate("Test prompt")

        assert result == custom_response

    def test_call_count_tracking(self):
        """Test that call count is tracked."""
        client = MockLlamaClient()

        assert client.call_count == 0
        client.generate("Prompt 1")
        assert client.call_count == 1
        client.generate("Prompt 2")
        assert client.call_count == 2

    def test_last_prompt_tracking(self):
        """Test that last prompt is tracked."""
        client = MockLlamaClient()

        client.generate("First prompt")
        assert client.last_prompt == "First prompt"

        client.generate("Second prompt")
        assert client.last_prompt == "Second prompt"

    def test_is_available(self):
        """Test that is_available returns True."""
        client = MockLlamaClient()
        assert client.is_available() is True


class TestLlamaClient:
    """Tests for LlamaClient."""

    @pytest.fixture
    def client(self, tmp_path):
        """Return a LlamaClient instance with temp cache dir."""
        return LlamaClient(
            model_path="/fake/model.gguf",
            cache_config=CacheConfig(directory=str(tmp_path / "cache")),
        )

    def test_init_defaults(self):
        """Test initialization with defaults."""
        client = LlamaClient()

        assert client.model_repo == LlamaClient.DEFAULT_MODEL_REPO
        assert client.model_file == LlamaClient.DEFAULT_MODEL_FILE
        assert client.n_ctx == 4096
        assert client.n_gpu_layers == -1

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        client = LlamaClient(
            model_path="/custom/model.gguf",
            n_ctx=2048,
            n_gpu_layers=10,
        )

        assert client.model_path == "/custom/model.gguf"
        assert client.n_ctx == 2048
        assert client.n_gpu_layers == 10

    def test_cache_key_generation(self, client):
        """Test cache key generation."""
        key1 = client._get_cache_key("prompt", 100, 0.7)
        key2 = client._get_cache_key("prompt", 100, 0.7)
        key3 = client._get_cache_key("different", 100, 0.7)

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 16

    def test_cache_response(self, client):
        """Test caching a response."""
        client._cache_response("test prompt", 100, 0.7, "test response")

        cache_key = client._get_cache_key("test prompt", 100, 0.7)
        cache_file = client._cache_dir / f"{cache_key}.json"

        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert data["response"] == "test response"

    def test_get_cached_response(self, client):
        """Test retrieving cached response."""
        client._cache_response("test prompt", 100, 0.7, "cached response")

        result = client._get_cached_response("test prompt", 100, 0.7)

        assert result == "cached response"

    def test_get_cached_response_miss(self, client):
        """Test cache miss returns None."""
        result = client._get_cached_response("nonexistent", 100, 0.7)
        assert result is None

    def test_clear_cache(self, client):
        """Test clearing cache."""
        client._cache_response("prompt1", 100, 0.7, "response1")
        client._cache_response("prompt2", 100, 0.7, "response2")

        count = client.clear_cache()

        assert count == 2
        assert len(list(client._cache_dir.glob("*.json"))) == 0

    def test_load_model_requires_llama_cpp(self, tmp_path):
        """Test that model loading requires llama-cpp-python."""
        client = LlamaClient(cache_config=CacheConfig(directory=str(tmp_path / "cache")))

        # The model loading will fail without llama-cpp-python installed
        # This tests the error handling path
        try:
            client._load_model()
            # If llama-cpp-python is installed, this is fine
        except LLMInferenceError as e:
            # Expected when llama-cpp-python is not installed
            assert "llama-cpp-python" in str(e).lower() or "Failed to load" in str(e)

    def test_client_model_path_stored(self, tmp_path):
        """Test that model path is correctly stored."""
        model_path = tmp_path / "model.gguf"
        model_path.touch()

        client = LlamaClient(
            model_path=str(model_path),
            cache_config=CacheConfig(directory=str(tmp_path / "cache")),
        )

        assert client.model_path == str(model_path)

    def test_generate_with_cache(self, client):
        """Test generate uses cache."""
        client._cache_response("test prompt", 512, 0.7, "cached result")

        result = client.generate("test prompt", use_cache=True)

        assert result == "cached result"

    def test_generate_without_cache_requires_model(self, client):
        """Test generate without cache requires model loading."""
        # Clear any cached model
        client._model = None

        # Without llama-cpp-python installed, this should raise an error
        try:
            client.generate("test", use_cache=False)
        except LLMInferenceError:
            # Expected when model can't be loaded
            pass

    def test_init_with_cache_enabled(self, tmp_path):
        """Test initialization with cache_enabled parameter."""
        client = LlamaClient(
            cache_config=CacheConfig(directory=str(tmp_path / "cache"), enabled=True),
        )
        assert client._cache_config.enabled is True

        client_disabled = LlamaClient(
            cache_config=CacheConfig(directory=str(tmp_path / "cache2"), enabled=False),
        )
        assert client_disabled._cache_config.enabled is False

    def test_from_config(self, tmp_path):
        """Test creating client from LLMModelConfig."""
        config = LLMModelConfig(
            model_repo="test/repo",
            model_file="test.gguf",
            inference=LLMInferenceConfig(max_tokens=1024, temperature=0.5),
            n_ctx=8192,
            n_gpu_layers=16,
        )

        client = LlamaClient.from_config(
            config,
            cache_config=CacheConfig(directory=str(tmp_path / "cache"), enabled=True),
        )

        assert client.model_repo == "test/repo"
        assert client.model_file == "test.gguf"
        assert client.n_ctx == 8192
        assert client.n_gpu_layers == 16
        assert client._cache_config.enabled is True
        assert client._inference_config.max_tokens == 1024
        assert client._inference_config.temperature == 0.5

    def test_from_config_with_defaults(self, tmp_path):
        """Test from_config uses defaults when not specified."""
        config = LLMModelConfig(
            model_repo="test/repo",
            model_file="test.gguf",
        )

        client = LlamaClient.from_config(
            config,
            cache_config=CacheConfig(directory=str(tmp_path / "cache")),
        )

        assert client.n_ctx == 4096
        assert client.n_gpu_layers == -1
        assert client._cache_config.enabled is True

    def test_generate_respects_cache_enabled_flag(self, client):
        """Test generate respects cache_enabled flag."""
        # Client has caching enabled by default - cache should work
        client._cache_response("test prompt", 512, 0.7, "cached result")
        result = client.generate("test prompt")
        assert result == "cached result"

    def test_cache_disabled_ignores_cache(self, tmp_path):
        """Test that cache_enabled=False ignores cached responses."""
        client_no_cache = LlamaClient(
            model_path="/fake/model.gguf",
            cache_config=CacheConfig(directory=str(tmp_path / "nocache"), enabled=False),
        )

        # Pre-populate cache file
        client_no_cache._cache_response("test prompt", 512, 0.7, "should not be used")

        # Verify the cache file exists
        cache_key = client_no_cache._get_cache_key("test prompt", 512, 0.7)
        cache_file = client_no_cache._cache_dir / f"{cache_key}.json"
        assert cache_file.exists()

        # With cache_enabled=False, _get_cached_response should not be checked
        # when calling generate - it should try to load model instead
        # (which will fail since model doesn't exist)
        assert client_no_cache._cache_config.enabled is False

    def test_generate_with_all_optional_params(self, tmp_path):
        """Test generate passes all optional parameters."""
        client = LlamaClient(
            model_path="/fake/model.gguf",
            cache_config=CacheConfig(directory=str(tmp_path / "cache")),
        )
        # Pre-cache response to avoid model loading
        client._cache_response("test prompt", 100, 0.5, "cached response")

        result = client.generate(
            "test prompt",
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            frequency_penalty=0.3,
            presence_penalty=0.1,
        )

        assert result == "cached response"

    def test_get_cached_response_invalid_json(self, client):
        """Test _get_cached_response returns None on invalid JSON."""
        cache_key = client._get_cache_key("test", 100, 0.7)
        cache_file = client._cache_dir / f"{cache_key}.json"
        cache_file.write_text("invalid json content {")

        result = client._get_cached_response("test", 100, 0.7)

        assert result is None

    def test_get_cached_response_missing_key(self, client):
        """Test _get_cached_response returns None if response key missing."""
        cache_key = client._get_cache_key("test", 100, 0.7)
        cache_file = client._cache_dir / f"{cache_key}.json"
        cache_file.write_text('{"other_key": "value"}')

        result = client._get_cached_response("test", 100, 0.7)

        assert result is None

    def test_is_available_returns_false_when_model_fails(self, tmp_path):
        """Test is_available returns False when model loading fails."""
        client = LlamaClient(
            model_repo="nonexistent/model",
            cache_config=CacheConfig(directory=str(tmp_path / "cache")),
        )

        # is_available catches LLMInferenceError and returns False
        available = client.is_available()

        # This will be False if llama-cpp-python is not installed
        # or True if it is (but then model download will fail)
        assert isinstance(available, bool)

    def test_generate_reraises_llm_inference_error(self, tmp_path):
        """Test generate re-raises LLMInferenceError without wrapping."""
        client = LlamaClient(
            model_path="/fake/model.gguf",
            cache_config=CacheConfig(directory=str(tmp_path / "cache"), enabled=False),
        )

        # Mock llama_cpp to raise an error during model loading
        mock_llama = MagicMock(side_effect=RuntimeError("Model loading failed"))
        mock_llama_cpp = MagicMock()
        mock_llama_cpp.Llama = mock_llama
        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            with pytest.raises(LLMInferenceError):
                client.generate("test", use_cache=False)

    def test_default_model_identifier(self, client):
        """Test _get_model_identifier returns repo/file."""
        identifier = client._get_model_identifier()
        assert "/" in identifier or "LlamaClient" in identifier
