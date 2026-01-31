"""LLM client using HuggingFace transformers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Any

from murban_copilot.domain.exceptions import LLMInferenceError
from murban_copilot.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from murban_copilot.domain.config import LLMModelConfig

logger = get_logger(__name__)


class TransformersClient:
    """Client for HuggingFace transformers models."""

    # Mapping from sentiment labels to trading signals
    SENTIMENT_TO_SIGNAL = {
        "positive": "bullish",
        "negative": "bearish",
        "neutral": "neutral",
        # Some models use different labels
        "POSITIVE": "bullish",
        "NEGATIVE": "bearish",
        "NEUTRAL": "neutral",
        "POS": "bullish",
        "NEG": "bearish",
        "NEU": "neutral",
    }

    def __init__(
        self,
        model_repo: str,
        task: str = "sentiment-analysis",
        device: str = "auto",
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = True,
    ) -> None:
        """
        Initialize the Transformers client.

        Args:
            model_repo: HuggingFace model repository ID
            task: Pipeline task type (e.g., "sentiment-analysis", "text-generation")
            device: Device to run on ("auto", "cpu", "cuda", "mps")
            cache_dir: Directory for response caching
            cache_enabled: Whether to enable response caching
        """
        self.model_repo = model_repo
        self.task = task
        self.device = device
        self.cache_enabled = cache_enabled

        self._pipeline = None
        self._cache_dir = cache_dir or Path.cwd() / ".llm_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        config: "LLMModelConfig",
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = True,
    ) -> "TransformersClient":
        """
        Create a TransformersClient from an LLMModelConfig.

        Args:
            config: Model configuration
            cache_dir: Directory for response caching
            cache_enabled: Whether to enable response caching

        Returns:
            Configured TransformersClient instance
        """
        return cls(
            model_repo=config.model_repo,
            task=config.task,
            device=config.device,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )

    def _load_pipeline(self) -> None:
        """Load the transformers pipeline."""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
            import torch

            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    device = 0  # First CUDA device
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = -1  # CPU
            elif self.device == "cpu":
                device = -1
            elif self.device == "cuda":
                device = 0
            elif self.device == "mps":
                device = "mps"
            else:
                device = -1

            logger.info(f"Loading transformers pipeline: {self.model_repo} (task: {self.task})")
            self._pipeline = pipeline(
                self.task,
                model=self.model_repo,
                device=device,
            )
            logger.info("Transformers pipeline loaded successfully")

        except ImportError as e:
            raise LLMInferenceError(
                "transformers is not installed. Install with: pip install transformers torch",
                original_error=e,
            )
        except Exception as e:
            raise LLMInferenceError(
                f"Failed to load transformers pipeline: {str(e)}",
                original_error=e,
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        use_cache: bool = True,
    ) -> str:
        """
        Generate text or classification from the given prompt.

        For sentiment-analysis tasks, returns formatted signal output.
        For text-generation tasks, returns generated text.

        Args:
            prompt: The input text
            max_tokens: Maximum tokens to generate (for text-generation)
            temperature: Sampling temperature (for text-generation)
            use_cache: Whether to use response caching

        Returns:
            Generated or classified text

        Raises:
            LLMInferenceError: If inference fails
        """
        should_use_cache = self.cache_enabled and use_cache

        if should_use_cache:
            cached = self._get_cached_response(prompt, max_tokens, temperature)
            if cached is not None:
                logger.debug("Using cached response")
                return cached

        self._load_pipeline()

        try:
            if self.task in ("sentiment-analysis", "text-classification"):
                result = self._run_classification(prompt)
            else:
                result = self._run_generation(prompt, max_tokens, temperature)

            if should_use_cache:
                self._cache_response(prompt, max_tokens, temperature, result)

            return result

        except Exception as e:
            raise LLMInferenceError(
                f"Inference failed: {str(e)}",
                original_error=e,
            )

    def _run_classification(self, text: str) -> str:
        """
        Run sentiment/text classification and format as signal output.

        Args:
            text: Input text to classify

        Returns:
            Formatted signal string (SIGNAL, CONFIDENCE, SUMMARY)
        """
        # Truncate long text for classification models (usually max 512 tokens)
        max_chars = 2000  # Approximate limit
        if len(text) > max_chars:
            text = text[:max_chars]

        results = self._pipeline(text)

        # Handle both single result and list of results
        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        label = result.get("label", "neutral")
        score = result.get("score", 0.5)

        # Map sentiment label to trading signal
        signal = self.SENTIMENT_TO_SIGNAL.get(label, "neutral")

        # Format output in expected extraction format
        return f"""SIGNAL: {signal}
CONFIDENCE: {score:.2f}
SUMMARY: Financial sentiment analysis indicates {signal} outlook based on the provided market analysis."""

    def _run_generation(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Run text generation.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        results = self._pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_full_text=False,
        )

        if isinstance(results, list) and len(results) > 0:
            return results[0].get("generated_text", "").strip()
        return ""

    def is_available(self) -> bool:
        """
        Check if the model is available and can be loaded.

        Returns:
            True if the model is ready or can be loaded
        """
        try:
            self._load_pipeline()
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
        content = f"{self.model_repo}|{prompt}|{max_tokens}|{temperature}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

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
            "model_repo": self.model_repo,
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
