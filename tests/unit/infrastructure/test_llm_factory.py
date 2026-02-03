"""Unit tests for LLM factory."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from murban_copilot.domain.config import LLMModelConfig, ModelType
from murban_copilot.infrastructure.llm.factory import (
    create_llm_client,
    get_client_type_name,
)
from murban_copilot.infrastructure.llm.llm_client import LlamaClient
from murban_copilot.infrastructure.llm.transformers_client import TransformersClient


class TestCreateLLMClient:
    """Tests for create_llm_client factory function."""

    def test_create_transformers_client(self, tmp_path):
        """Test creating a TransformersClient."""
        config = LLMModelConfig(
            model_repo="test/transformers-model",
            model_type=ModelType.TRANSFORMERS,
            task="sentiment-analysis",
            device="cpu",
        )

        client = create_llm_client(
            config,
            cache_dir=tmp_path / "cache",
            cache_enabled=True,
        )

        assert isinstance(client, TransformersClient)
        assert client.model_repo == "test/transformers-model"
        assert client.task == "sentiment-analysis"
        assert client.device == "cpu"

    def test_create_llama_client(self, tmp_path):
        """Test creating a LlamaClient."""
        config = LLMModelConfig(
            model_repo="test/llama-model",
            model_file="model.gguf",
            model_type=ModelType.LLAMA,
            n_ctx=2048,
            n_gpu_layers=10,
        )

        client = create_llm_client(
            config,
            cache_dir=tmp_path / "cache",
            cache_enabled=True,
            verbose=True,
        )

        assert isinstance(client, LlamaClient)
        assert client.model_repo == "test/llama-model"
        assert client.model_file == "model.gguf"
        assert client.n_ctx == 2048
        assert client.n_gpu_layers == 10

    def test_create_llama_client_default_type(self, tmp_path):
        """Test that LLAMA type is default."""
        config = LLMModelConfig(
            model_repo="test/model",
            model_file="model.gguf",
            model_type=ModelType.LLAMA,
        )

        client = create_llm_client(config, cache_dir=tmp_path / "cache")

        assert isinstance(client, LlamaClient)

    def test_create_client_with_cache_disabled(self, tmp_path):
        """Test creating client with cache disabled."""
        config = LLMModelConfig(
            model_repo="test/model",
            model_type=ModelType.TRANSFORMERS,
        )

        client = create_llm_client(
            config,
            cache_dir=tmp_path / "cache",
            cache_enabled=False,
        )

        assert client.cache_enabled is False


class TestGetClientTypeName:
    """Tests for get_client_type_name function."""

    def test_llama_client_name(self, tmp_path):
        """Test name for LlamaClient."""
        client = LlamaClient(
            model_repo="test/model",
            model_file="model.gguf",
            cache_dir=tmp_path / "cache",
        )

        name = get_client_type_name(client)

        assert name == "Llama (GGUF)"

    def test_transformers_client_name(self, tmp_path):
        """Test name for TransformersClient."""
        client = TransformersClient(
            model_repo="test/model",
            cache_dir=tmp_path / "cache",
        )

        name = get_client_type_name(client)

        assert name == "Transformers (HuggingFace)"

    def test_unknown_client_name(self):
        """Test name for unknown client type."""
        # Create a mock client that isn't LlamaClient or TransformersClient
        mock_client = MagicMock()
        mock_client.__class__ = type("UnknownClient", (), {})

        name = get_client_type_name(mock_client)

        assert name == "Unknown"