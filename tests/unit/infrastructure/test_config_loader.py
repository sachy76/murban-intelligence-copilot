"""Unit tests for configuration loader."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from murban_copilot.domain.config import LLMConfig
from murban_copilot.domain.exceptions import ConfigurationError
from murban_copilot.infrastructure.config.config_loader import ConfigLoader


class TestConfigLoader:
    """Tests for ConfigLoader."""

    @pytest.fixture
    def sample_config_data(self):
        """Return sample configuration data."""
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

    @pytest.fixture
    def config_file(self, sample_config_data, tmp_path):
        """Create a temporary config file."""
        config_path = tmp_path / "llm_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_data, f)
        return config_path

    def test_load_from_file(self, config_file):
        """Test loading config from a file."""
        loader = ConfigLoader(config_path=str(config_file))
        config = loader.load()

        assert isinstance(config, LLMConfig)
        assert config.analysis is not None
        assert config.analysis.model_repo == "MaziyarPanahi/gemma-3-12b-it-GGUF"
        assert config.extraction is not None
        assert config.extraction.model_repo == "bartowski/gemma-2-9b-it-GGUF"

    def test_load_returns_default_when_file_not_found(self):
        """Test loading returns default config when file not found."""
        loader = ConfigLoader(config_path="/nonexistent/path/config.yaml")
        config = loader.load()

        assert isinstance(config, LLMConfig)
        assert config.analysis is not None
        assert config.extraction is not None

    def test_load_from_environment_variable(self, config_file, monkeypatch):
        """Test loading config from environment variable path."""
        monkeypatch.setenv("MURBAN_LLM_CONFIG", str(config_file))
        loader = ConfigLoader()
        config = loader.load()

        assert isinstance(config, LLMConfig)
        assert config.analysis.model_repo == "MaziyarPanahi/gemma-3-12b-it-GGUF"

    def test_load_searches_default_paths(self, sample_config_data, tmp_path, monkeypatch):
        """Test loading searches default paths."""
        # Create config in current directory
        config_path = tmp_path / "llm_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_data, f)

        # Change to temp directory and load
        monkeypatch.chdir(tmp_path)
        loader = ConfigLoader()
        config = loader.load()

        assert config.analysis.model_repo == "MaziyarPanahi/gemma-3-12b-it-GGUF"

    def test_load_raises_on_invalid_yaml(self, tmp_path):
        """Test loading raises ConfigurationError on invalid YAML."""
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("invalid: yaml: content: [")

        loader = ConfigLoader(config_path=str(config_path))
        with pytest.raises(ConfigurationError):
            loader.load()

    def test_load_handles_partial_config(self, tmp_path):
        """Test loading handles partial configuration."""
        partial_config = {
            "llm": {
                "analysis": {
                    "model_repo": "custom/repo",
                    "model_file": "custom.gguf",
                },
            },
        }
        config_path = tmp_path / "partial.yaml"
        with open(config_path, "w") as f:
            yaml.dump(partial_config, f)

        loader = ConfigLoader(config_path=str(config_path))
        config = loader.load()

        assert config.analysis.model_repo == "custom/repo"
        assert config.extraction is None
        assert config.defaults.n_ctx == 4096  # default value

    def test_explicit_path_overrides_environment(self, config_file, tmp_path, monkeypatch):
        """Test explicit path overrides environment variable."""
        # Create a different config
        other_config = {
            "llm": {
                "analysis": {
                    "model_repo": "other/repo",
                    "model_file": "other.gguf",
                },
            },
        }
        other_path = tmp_path / "other.yaml"
        with open(other_path, "w") as f:
            yaml.dump(other_config, f)

        monkeypatch.setenv("MURBAN_LLM_CONFIG", str(other_path))

        # Explicit path should take precedence
        loader = ConfigLoader(config_path=str(config_file))
        config = loader.load()

        assert config.analysis.model_repo == "MaziyarPanahi/gemma-3-12b-it-GGUF"

    def test_config_directory_path_search(self, sample_config_data, tmp_path, monkeypatch):
        """Test config is found in config/ directory."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_path = config_dir / "llm_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_data, f)

        monkeypatch.chdir(tmp_path)
        loader = ConfigLoader()
        config = loader.load()

        assert config.analysis.model_repo == "MaziyarPanahi/gemma-3-12b-it-GGUF"

    def test_get_config_path_returns_explicit_path(self, config_file):
        """Test get_config_path returns explicit path when set."""
        loader = ConfigLoader(config_path=str(config_file))
        assert loader.get_config_path() == str(config_file)

    def test_get_config_path_returns_none_when_not_found(self):
        """Test get_config_path returns None when no config found."""
        loader = ConfigLoader(config_path="/nonexistent/path.yaml")
        assert loader.get_config_path() is None

    def test_load_empty_yaml_file(self, tmp_path):
        """Test loading an empty YAML file returns defaults."""
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        loader = ConfigLoader(config_path=str(config_path))
        config = loader.load()

        assert isinstance(config, LLMConfig)
        assert config.defaults is not None

    def test_load_file_read_error(self, tmp_path, monkeypatch):
        """Test loading raises ConfigurationError on file read error."""
        config_path = tmp_path / "unreadable.yaml"
        config_path.write_text("llm: {}")

        loader = ConfigLoader(config_path=str(config_path))

        # Mock open to raise an error
        def mock_open(*args, **kwargs):
            raise PermissionError("Permission denied")

        monkeypatch.setattr("builtins.open", mock_open)

        with pytest.raises(ConfigurationError):
            loader.load()


class TestConfigLoaderAppConfig:
    """Tests for ConfigLoader.load_app_config method."""

    @pytest.fixture
    def full_app_config_data(self):
        """Return full application configuration data."""
        return {
            "llm": {
                "defaults": {
                    "n_ctx": 4096,
                    "n_gpu_layers": -1,
                },
                "analysis": {
                    "model_repo": "test/analysis-model",
                    "model_file": "analysis.gguf",
                },
            },
            "market_data": {
                "wti_ticker": "CL=F",
                "brent_ticker": "BZ=F",
                "timeout": 30,
            },
            "analysis": {
                "short_ma_window": 5,
                "long_ma_window": 20,
            },
            "signal": {
                "default_signal": "neutral",
                "default_confidence": 0.5,
            },
        }

    def test_load_app_config_from_file(self, full_app_config_data, tmp_path):
        """Test loading full app config from file."""
        from murban_copilot.domain.config import AppConfig

        config_path = tmp_path / "app_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(full_app_config_data, f)

        loader = ConfigLoader(config_path=str(config_path))
        config = loader.load_app_config()

        assert isinstance(config, AppConfig)
        assert config.market_data is not None
        assert config.market_data.wti_ticker == "CL=F"

    def test_load_app_config_returns_default_when_not_found(self):
        """Test load_app_config returns default when file not found."""
        from murban_copilot.domain.config import AppConfig

        loader = ConfigLoader(config_path="/nonexistent/path.yaml")
        config = loader.load_app_config()

        assert isinstance(config, AppConfig)

    def test_load_app_config_raises_on_invalid_yaml(self, tmp_path):
        """Test load_app_config raises ConfigurationError on invalid YAML."""
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("invalid: yaml: [unclosed")

        loader = ConfigLoader(config_path=str(config_path))
        with pytest.raises(ConfigurationError):
            loader.load_app_config()

    def test_load_app_config_empty_file(self, tmp_path):
        """Test load_app_config with empty YAML file."""
        from murban_copilot.domain.config import AppConfig

        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")

        loader = ConfigLoader(config_path=str(config_path))
        config = loader.load_app_config()

        assert isinstance(config, AppConfig)

    def test_load_app_config_file_read_error(self, tmp_path, monkeypatch):
        """Test load_app_config raises on file read error."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("llm: {}")

        loader = ConfigLoader(config_path=str(config_path))
        # Force resolved path
        loader._resolved_path = None

        # Mock open to raise an error after path resolution
        original_open = open

        call_count = [0]

        def mock_open(path, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1 and "config.yaml" in str(path):
                raise IOError("Read error")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)

        with pytest.raises(ConfigurationError):
            loader.load_app_config()
