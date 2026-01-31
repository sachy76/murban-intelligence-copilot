"""Integration tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from murban_copilot.domain.config import LLMConfig
from murban_copilot.infrastructure.config import ConfigLoader
from murban_copilot.infrastructure.llm.llm_client import LlamaClient


class TestConfigIntegration:
    """Integration tests for configuration system."""

    @pytest.fixture
    def full_config_data(self):
        """Return full configuration data."""
        return {
            "llm": {
                "defaults": {
                    "n_ctx": 4096,
                    "n_gpu_layers": -1,
                    "cache_enabled": True,
                    "verbose": False,
                },
                "analysis": {
                    "model_repo": "MaziyarPanahi/gemma-3-12b-it-GGUF",
                    "model_file": "gemma-3-12b-it.Q6_K.gguf",
                    "inference": {
                        "max_tokens": 2048,
                        "temperature": 0.7,
                    },
                },
                "extraction": {
                    "model_repo": "bartowski/gemma-2-9b-it-GGUF",
                    "model_file": "gemma-2-9b-it-Q4_K_M.gguf",
                    "inference": {
                        "max_tokens": 1024,
                        "temperature": 0.3,
                    },
                },
            },
            "cache": {
                "directory": ".llm_cache",
                "enabled": True,
            },
        }

    def test_load_config_and_create_clients(self, full_config_data, tmp_path):
        """Test loading config and creating LLM clients from it."""
        config_path = tmp_path / "llm_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(full_config_data, f)

        # Load configuration
        loader = ConfigLoader(config_path=str(config_path))
        config = loader.load()

        # Create clients from config
        cache_dir = tmp_path / config.cache.directory

        analysis_client = LlamaClient.from_config(
            config.analysis,
            cache_dir=cache_dir,
            cache_enabled=config.cache.enabled,
        )

        extraction_client = LlamaClient.from_config(
            config.extraction,
            cache_dir=cache_dir,
            cache_enabled=config.cache.enabled,
        )

        # Verify analysis client
        assert analysis_client.model_repo == "MaziyarPanahi/gemma-3-12b-it-GGUF"
        assert analysis_client.model_file == "gemma-3-12b-it.Q6_K.gguf"
        assert analysis_client.n_ctx == 4096
        assert analysis_client.cache_enabled is True

        # Verify extraction client
        assert extraction_client.model_repo == "bartowski/gemma-2-9b-it-GGUF"
        assert extraction_client.model_file == "gemma-2-9b-it-Q4_K_M.gguf"
        assert extraction_client.n_ctx == 4096

    def test_environment_variable_override(self, full_config_data, tmp_path, monkeypatch):
        """Test that environment variable can override config path."""
        config_path = tmp_path / "custom_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(full_config_data, f)

        monkeypatch.setenv("MURBAN_LLM_CONFIG", str(config_path))

        loader = ConfigLoader()
        config = loader.load()

        assert config.analysis is not None
        assert config.analysis.model_repo == "MaziyarPanahi/gemma-3-12b-it-GGUF"

    def test_fallback_to_defaults(self):
        """Test fallback to default configuration when no file exists."""
        loader = ConfigLoader(config_path="/nonexistent/path.yaml")
        config = loader.load()

        # Should get default configuration
        assert config.analysis is not None
        assert config.extraction is not None
        assert config.defaults.n_ctx == 4096

    def test_partial_config_with_defaults(self, tmp_path):
        """Test partial config is merged with defaults."""
        partial_config = {
            "llm": {
                "defaults": {
                    "n_ctx": 8192,  # Override default
                },
                "analysis": {
                    "model_repo": "custom/analysis",
                    "model_file": "analysis.gguf",
                },
            },
        }

        config_path = tmp_path / "partial.yaml"
        with open(config_path, "w") as f:
            yaml.dump(partial_config, f)

        loader = ConfigLoader(config_path=str(config_path))
        config = loader.load()

        # Check overridden defaults
        assert config.defaults.n_ctx == 8192
        assert config.defaults.n_gpu_layers == -1  # default value

        # Check analysis config inherits custom defaults
        assert config.analysis.n_ctx == 8192
        assert config.analysis.model_repo == "custom/analysis"

        # Extraction should be None since not specified
        assert config.extraction is None

    def test_project_config_file_discovery(self, full_config_data, tmp_path, monkeypatch):
        """Test config discovery in project directories."""
        # Create config/llm_config.yaml
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_path = config_dir / "llm_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(full_config_data, f)

        # Change working directory to tmp_path
        monkeypatch.chdir(tmp_path)

        loader = ConfigLoader()
        config = loader.load()

        assert config.analysis.model_repo == "MaziyarPanahi/gemma-3-12b-it-GGUF"
