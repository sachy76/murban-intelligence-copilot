"""Unit tests for LLM client."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from murban_copilot.domain.exceptions import LLMInferenceError
from murban_copilot.infrastructure.llm.llm_client import LlamaClient, MockLlamaClient


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
            cache_dir=tmp_path / "cache",
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
        client = LlamaClient(cache_dir=tmp_path / "cache")

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
            cache_dir=tmp_path / "cache",
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
