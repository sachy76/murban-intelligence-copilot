"""Unit tests for configuration domain entities."""

import pytest

from murban_copilot.domain.config import (
    LLMInferenceConfig,
    LLMModelConfig,
    LLMDefaultsConfig,
    LLMConfig,
    CacheConfig,
)


class TestLLMInferenceConfig:
    """Tests for LLMInferenceConfig."""

    def test_create_with_defaults(self):
        """Test creating config with default values."""
        config = LLMInferenceConfig()

        assert config.max_tokens == 512
        assert config.temperature == 0.7

    def test_create_with_custom_values(self):
        """Test creating config with custom values."""
        config = LLMInferenceConfig(max_tokens=1024, temperature=0.3)

        assert config.max_tokens == 1024
        assert config.temperature == 0.3

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {"max_tokens": 2048, "temperature": 0.5}
        config = LLMInferenceConfig.from_dict(data)

        assert config.max_tokens == 2048
        assert config.temperature == 0.5

    def test_from_dict_with_defaults(self):
        """Test creating config from partial dictionary."""
        config = LLMInferenceConfig.from_dict({})

        assert config.max_tokens == 512
        assert config.temperature == 0.7


class TestLLMModelConfig:
    """Tests for LLMModelConfig."""

    def test_create_with_required_fields(self):
        """Test creating config with required fields."""
        config = LLMModelConfig(
            model_repo="owner/repo",
            model_file="model.gguf",
        )

        assert config.model_repo == "owner/repo"
        assert config.model_file == "model.gguf"
        assert config.inference is not None
        assert config.n_ctx == 4096
        assert config.n_gpu_layers == -1

    def test_create_with_all_fields(self):
        """Test creating config with all fields."""
        inference = LLMInferenceConfig(max_tokens=1024, temperature=0.5)
        config = LLMModelConfig(
            model_repo="owner/repo",
            model_file="model.gguf",
            inference=inference,
            n_ctx=8192,
            n_gpu_layers=32,
        )

        assert config.n_ctx == 8192
        assert config.n_gpu_layers == 32
        assert config.inference.max_tokens == 1024

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "model_repo": "owner/repo",
            "model_file": "model.gguf",
            "inference": {"max_tokens": 2048, "temperature": 0.3},
            "n_ctx": 4096,
        }
        config = LLMModelConfig.from_dict(data)

        assert config.model_repo == "owner/repo"
        assert config.model_file == "model.gguf"
        assert config.inference.max_tokens == 2048
        assert config.inference.temperature == 0.3

    def test_from_dict_with_defaults(self):
        """Test creating config from dict with defaults applied."""
        defaults = LLMDefaultsConfig(n_ctx=8192, n_gpu_layers=16)
        data = {
            "model_repo": "owner/repo",
            "model_file": "model.gguf",
        }
        config = LLMModelConfig.from_dict(data, defaults=defaults)

        assert config.n_ctx == 8192
        assert config.n_gpu_layers == 16


class TestLLMDefaultsConfig:
    """Tests for LLMDefaultsConfig."""

    def test_create_with_defaults(self):
        """Test creating config with default values."""
        config = LLMDefaultsConfig()

        assert config.n_ctx == 4096
        assert config.n_gpu_layers == -1
        assert config.verbose is False

    def test_create_with_custom_values(self):
        """Test creating config with custom values."""
        config = LLMDefaultsConfig(
            n_ctx=8192,
            n_gpu_layers=32,
            verbose=True,
        )

        assert config.n_ctx == 8192
        assert config.n_gpu_layers == 32
        assert config.verbose is True

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {"n_ctx": 2048, "verbose": True}
        config = LLMDefaultsConfig.from_dict(data)

        assert config.n_ctx == 2048
        assert config.verbose is True


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_create_with_defaults(self):
        """Test creating config with default values."""
        config = CacheConfig()

        assert config.directory == ".llm_cache"
        assert config.enabled is True

    def test_create_with_custom_values(self):
        """Test creating config with custom values."""
        config = CacheConfig(directory="/tmp/cache", enabled=False)

        assert config.directory == "/tmp/cache"
        assert config.enabled is False

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {"directory": "my_cache", "enabled": False}
        config = CacheConfig.from_dict(data)

        assert config.directory == "my_cache"
        assert config.enabled is False


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_create_with_defaults(self):
        """Test creating config with default values."""
        config = LLMConfig()

        assert config.defaults is not None
        assert config.analysis is None
        assert config.extraction is None
        assert config.cache is not None

    def test_create_with_model_configs(self):
        """Test creating config with model configurations."""
        analysis = LLMModelConfig(
            model_repo="analysis/repo",
            model_file="analysis.gguf",
        )
        extraction = LLMModelConfig(
            model_repo="extraction/repo",
            model_file="extraction.gguf",
        )
        config = LLMConfig(analysis=analysis, extraction=extraction)

        assert config.analysis.model_repo == "analysis/repo"
        assert config.extraction.model_repo == "extraction/repo"

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "defaults": {"n_ctx": 8192, "verbose": True},
            "analysis": {
                "model_repo": "analysis/repo",
                "model_file": "analysis.gguf",
                "inference": {"max_tokens": 2048},
            },
            "extraction": {
                "model_repo": "extraction/repo",
                "model_file": "extraction.gguf",
                "inference": {"max_tokens": 1024, "temperature": 0.3},
            },
            "cache": {"directory": "cache_dir", "enabled": True},
        }
        config = LLMConfig.from_dict(data)

        assert config.defaults.n_ctx == 8192
        assert config.defaults.verbose is True
        assert config.analysis.model_repo == "analysis/repo"
        assert config.analysis.inference.max_tokens == 2048
        # Analysis should inherit defaults
        assert config.analysis.n_ctx == 8192
        assert config.extraction.model_repo == "extraction/repo"
        assert config.extraction.inference.temperature == 0.3
        assert config.cache.directory == "cache_dir"

    def test_from_dict_empty(self):
        """Test creating config from empty dictionary."""
        config = LLMConfig.from_dict({})

        assert config.defaults is not None
        assert config.analysis is None
        assert config.extraction is None
        assert config.cache is not None

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = LLMConfig.get_default()

        assert config.defaults.n_ctx == 4096
        assert config.analysis is not None
        assert config.analysis.model_repo == "MaziyarPanahi/gemma-3-12b-it-GGUF"
        assert config.extraction is not None
        assert config.extraction.model_repo == "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
