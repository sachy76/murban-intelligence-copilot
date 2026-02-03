"""Base class for LLM clients with shared caching functionality."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from murban_copilot.domain.exceptions import LLMInferenceError
from murban_copilot.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from murban_copilot.domain.config import LLMModelConfig

logger = get_logger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients with shared caching functionality."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = True,
    ) -> None:
        """
        Initialize base client with caching support.

        Args:
            cache_dir: Directory for response caching
            cache_enabled: Whether to enable response caching
        """
        self.cache_enabled = cache_enabled
        self._cache_dir = cache_dir or Path.cwd() / ".llm_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: "LLMModelConfig",
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = True,
        **kwargs,
    ) -> "BaseLLMClient":
        """Create a client from configuration."""
        ...

    @abstractmethod
    def _load_model(self) -> None:
        """Load the underlying model. Called lazily on first generation."""
        ...

    @abstractmethod
    def _do_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Perform the actual generation (without caching).

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        ...

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        use_cache: bool = True,
    ) -> str:
        """
        Generate text from the given prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            use_cache: Whether to use response caching

        Returns:
            Generated text

        Raises:
            LLMInferenceError: If generation fails
        """
        should_use_cache = self.cache_enabled and use_cache

        if should_use_cache:
            cached = self._get_cached_response(prompt, max_tokens, temperature)
            if cached is not None:
                logger.debug("Using cached response")
                return cached

        self._load_model()

        try:
            result = self._do_generate(prompt, max_tokens, temperature)

            if should_use_cache:
                self._cache_response(prompt, max_tokens, temperature, result)

            return result

        except LLMInferenceError:
            raise
        except Exception as e:
            raise LLMInferenceError(
                f"Generation failed: {str(e)}",
                original_error=e,
            )

    def is_available(self) -> bool:
        """
        Check if the LLM is available and can be loaded.

        Returns:
            True if the model is ready or can be loaded
        """
        try:
            self._load_model()
            return True
        except LLMInferenceError:
            return False

    def _get_cache_key(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate a cache key for the given parameters."""
        model_id = self._get_model_identifier()
        content = f"{model_id}|{prompt}|{max_tokens}|{temperature}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_model_identifier(self) -> str:
        """
        Get a unique identifier for the model.

        Override in subclasses to include model-specific info in cache keys.
        """
        return self.__class__.__name__

    def _get_cached_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        """Get a cached response if available."""
        cache_key = self._get_cache_key(prompt, max_tokens, temperature)
        cache_file = self._cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text())
            return data.get("response")
        except (json.JSONDecodeError, KeyError):
            return None

    def _cache_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        response: str,
    ) -> None:
        """Cache a response for future use."""
        cache_key = self._get_cache_key(prompt, max_tokens, temperature)
        cache_file = self._cache_dir / f"{cache_key}.json"

        data = {
            "model_id": self._get_model_identifier(),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response": response,
            "cached_at": datetime.utcnow().isoformat(),
        }

        cache_file.write_text(json.dumps(data, indent=2))

    def clear_cache(self) -> int:
        """
        Clear all cached responses.

        Returns:
            Number of cache entries cleared
        """
        count = 0
        for cache_file in self._cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count
