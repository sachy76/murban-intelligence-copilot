"""Contract tests for LLM inference protocol."""

import pytest

from murban_copilot.infrastructure.llm.llm_client import LlamaClient
from murban_copilot.infrastructure.llm.mock_client import MockLlamaClient
from murban_copilot.infrastructure.llm import LLMInference


@pytest.mark.contract
class TestLLMInferenceContract:
    """Contract tests verifying LLMInference implementations."""

    @pytest.fixture(params=["mock", "llama"])
    def llm_client(self, request, tmp_path) -> LLMInference:
        """Return implementations of LLMInference."""
        if request.param == "mock":
            return MockLlamaClient()
        elif request.param == "llama":
            return LlamaClient(
                model_path="/fake/model.gguf",
                cache_dir=tmp_path / "cache",
            )
        raise ValueError(f"Unknown client: {request.param}")

    def test_implements_protocol(self, llm_client):
        """Test that implementation has required methods."""
        assert hasattr(llm_client, "generate")
        assert hasattr(llm_client, "is_available")

    def test_generate_signature(self, llm_client):
        """Test generate method signature."""
        import inspect

        sig = inspect.signature(llm_client.generate)
        params = list(sig.parameters.keys())

        assert "prompt" in params
        assert "max_tokens" in params
        assert "temperature" in params

    def test_generate_returns_string(self, llm_client):
        """Test that generate returns a string."""
        if isinstance(llm_client, LlamaClient):
            # For LlamaClient, use cached response
            llm_client._cache_response("test", 512, 0.7, "cached response")

        result = llm_client.generate("Test prompt")
        assert isinstance(result, str)

    def test_is_available_returns_bool(self, llm_client):
        """Test that is_available returns a boolean."""
        if isinstance(llm_client, MockLlamaClient):
            result = llm_client.is_available()
            assert isinstance(result, bool)


@pytest.mark.contract
class TestMockLlamaClientContract:
    """Contract tests specific to MockLlamaClient."""

    def test_mock_client_always_available(self):
        """Test mock client is always available."""
        client = MockLlamaClient()
        assert client.is_available() is True

    def test_mock_client_tracks_calls(self):
        """Test mock client tracks call information."""
        client = MockLlamaClient()

        client.generate("First")
        assert client.call_count == 1
        assert client.last_prompt == "First"

        client.generate("Second")
        assert client.call_count == 2
        assert client.last_prompt == "Second"

    def test_mock_client_customizable(self):
        """Test mock client response is customizable."""
        response = "Custom response"
        client = MockLlamaClient(default_response=response)

        assert client.generate("Any prompt") == response


@pytest.mark.contract
class TestLlamaClientContract:
    """Contract tests specific to LlamaClient."""

    def test_client_has_defaults(self):
        """Test client has sensible defaults."""
        client = LlamaClient()

        assert client.model_repo is not None
        assert client.model_file is not None
        assert client.n_ctx > 0
        assert client.n_gpu_layers != 0

    def test_client_accepts_custom_config(self, tmp_path):
        """Test client accepts custom configuration."""
        client = LlamaClient(
            model_path="/custom/model.gguf",
            n_ctx=2048,
            n_gpu_layers=10,
            cache_dir=tmp_path / "cache",
        )

        assert client.model_path == "/custom/model.gguf"
        assert client.n_ctx == 2048
        assert client.n_gpu_layers == 10

    def test_client_supports_caching(self, tmp_path):
        """Test client supports response caching."""
        client = LlamaClient(cache_dir=tmp_path / "cache")

        # Cache a response
        client._cache_response("prompt", 100, 0.7, "response")

        # Should retrieve from cache
        cached = client._get_cached_response("prompt", 100, 0.7)
        assert cached == "response"
