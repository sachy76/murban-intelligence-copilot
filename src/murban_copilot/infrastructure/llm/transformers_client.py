"""LLM client using HuggingFace transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

from murban_copilot.domain.exceptions import LLMInferenceError
from murban_copilot.infrastructure.logging import get_logger
from .base_client import BaseLLMClient

if TYPE_CHECKING:
    from murban_copilot.domain.config import LLMModelConfig

logger = get_logger(__name__)


def detect_torch_device(preference: str = "auto") -> Union[int, str]:
    """
    Detect the best available device for torch.

    Args:
        preference: Device preference ("auto", "cpu", "cuda", "mps")

    Returns:
        Device identifier for transformers pipeline
    """
    import torch

    if preference == "auto":
        if torch.cuda.is_available():
            return 0  # First CUDA device
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return -1  # CPU
    elif preference == "cuda":
        return 0
    elif preference == "mps":
        return "mps"
    return -1  # CPU


class TransformersClient(BaseLLMClient):
    """Client for HuggingFace transformers models."""

    SENTIMENT_TO_SIGNAL = {
        "positive": "bullish",
        "negative": "bearish",
        "neutral": "neutral",
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
        super().__init__(cache_dir=cache_dir, cache_enabled=cache_enabled)

        self.model_repo = model_repo
        self.task = task
        self.device = device

        self._pipeline = None

    @classmethod
    def from_config(
        cls,
        config: "LLMModelConfig",
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = True,
        **kwargs,
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

    def _get_model_identifier(self) -> str:
        """Get unique identifier for cache keys."""
        return self.model_repo

    def _load_model(self) -> None:
        """Load the transformers pipeline."""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline

            device = detect_torch_device(self.device)

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

    def _do_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Perform generation or classification."""
        if self.task in ("sentiment-analysis", "text-classification"):
            return self._run_classification(prompt)
        return self._run_generation(prompt, max_tokens, temperature)

    def _run_classification(self, text: str) -> str:
        """
        Run sentiment/text classification and format as signal output.

        Args:
            text: Input text to classify

        Returns:
            Formatted signal string (SIGNAL, CONFIDENCE, SUMMARY)
        """
        max_chars = 2000
        if len(text) > max_chars:
            text = text[:max_chars]

        results = self._pipeline(text)

        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        label = result.get("label", "neutral")
        score = result.get("score", 0.5)

        signal = self.SENTIMENT_TO_SIGNAL.get(label, "neutral")

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
