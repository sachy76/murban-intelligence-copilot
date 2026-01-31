"""Configuration loader for application settings."""

import os
from pathlib import Path
from typing import Optional

import yaml

from murban_copilot.domain.config import LLMConfig, AppConfig
from murban_copilot.domain.exceptions import ConfigurationError
from murban_copilot.infrastructure.logging import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """Loader for application configuration files."""

    ENV_VAR = "MURBAN_LLM_CONFIG"
    APP_CONFIG_ENV_VAR = "MURBAN_CONFIG"
    DEFAULT_PATHS = [
        "config/llm_config.yaml",
        "config/app_config.yaml",
        "llm_config.yaml",
        "app_config.yaml",
    ]

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the config loader.

        Args:
            config_path: Explicit path to config file. If not provided,
                         searches environment variable and default paths.
        """
        self._explicit_path = config_path
        self._resolved_path: Optional[str] = None

    def load(self) -> LLMConfig:
        """
        Load configuration from file.

        Returns:
            LLMConfig with loaded or default values.

        Raises:
            ConfigurationError: If config file exists but is invalid.
        """
        config_path = self._find_config_path()

        if config_path is None:
            logger.info("No config file found, using default configuration")
            return LLMConfig.get_default()

        self._resolved_path = config_path
        logger.info(f"Loading configuration from {config_path}")

        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in config file: {config_path}",
                config_path=config_path,
                original_error=e,
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to read config file: {config_path}",
                config_path=config_path,
                original_error=e,
            )

        if data is None:
            data = {}

        return self._parse_config(data)

    def get_config_path(self) -> Optional[str]:
        """
        Get the resolved config path.

        Returns:
            Path to config file if found, None otherwise.
        """
        if self._resolved_path:
            return self._resolved_path

        config_path = self._find_config_path()
        return config_path

    def _find_config_path(self) -> Optional[str]:
        """Find the configuration file path."""
        # 1. Explicit path takes precedence
        if self._explicit_path:
            if Path(self._explicit_path).exists():
                return self._explicit_path
            return None

        # 2. Check environment variable
        env_path = os.environ.get(self.ENV_VAR)
        if env_path and Path(env_path).exists():
            return env_path

        # 3. Search default paths
        for path in self.DEFAULT_PATHS:
            if Path(path).exists():
                return path

        return None

    def _parse_config(self, data: dict) -> LLMConfig:
        """Parse configuration data into LLMConfig."""
        llm_data = data.get("llm", {})
        cache_data = data.get("cache", {})

        # Merge cache config into llm data structure
        config_data = {
            "defaults": llm_data.get("defaults", {}),
            "analysis": llm_data.get("analysis"),
            "extraction": llm_data.get("extraction"),
            "cache": cache_data,
        }

        # Remove None values to allow defaults to apply
        config_data = {k: v for k, v in config_data.items() if v is not None}

        return LLMConfig.from_dict(config_data)

    def load_app_config(self) -> AppConfig:
        """
        Load full application configuration from file.

        Returns:
            AppConfig with loaded or default values.

        Raises:
            ConfigurationError: If config file exists but is invalid.
        """
        config_path = self._find_config_path()

        if config_path is None:
            logger.info("No config file found, using default application configuration")
            return AppConfig.get_default()

        self._resolved_path = config_path
        logger.info(f"Loading application configuration from {config_path}")

        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in config file: {config_path}",
                config_path=config_path,
                original_error=e,
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to read config file: {config_path}",
                config_path=config_path,
                original_error=e,
            )

        if data is None:
            data = {}

        return AppConfig.from_dict(data)
