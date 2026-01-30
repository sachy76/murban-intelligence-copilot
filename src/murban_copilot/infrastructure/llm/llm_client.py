"""LLM client using llama-cpp-python with GGUF models."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from murban_copilot.domain.exceptions import LLMInferenceError
from murban_copilot.infrastructure.logging import get_logger

logger = get_logger(__name__)


class LlamaClient:
    """Client for local LLM inference using llama-cpp-python."""

    DEFAULT_MODEL_REPO = "bartowski/gemma-2-9b-it-GGUF"
    DEFAULT_MODEL_FILE = "gemma-2-9b-it-Q4_K_M.gguf"

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_repo: Optional[str] = None,
        model_file: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 means use all available GPU layers
        cache_dir: Optional[Path] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the Llama client.

        Args:
            model_path: Direct path to GGUF model file
            model_repo: HuggingFace repo ID (if model_path not provided)
            model_file: Model filename in repo (if model_path not provided)
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            cache_dir: Directory for response caching
            verbose: Whether to enable verbose output
        """
        self.model_path = model_path
        self.model_repo = model_repo or self.DEFAULT_MODEL_REPO
        self.model_file = model_file or self.DEFAULT_MODEL_FILE
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose

        self._model = None
        self._cache_dir = cache_dir or Path.cwd() / ".llm_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> None:
        """Load the LLM model."""
        if self._model is not None:
            return

        try:
            from llama_cpp import Llama

            if self.model_path and Path(self.model_path).exists():
                model_file = self.model_path
                logger.info(f"Loading model from local path: {model_file}")
            else:
                from huggingface_hub import hf_hub_download

                logger.info(f"Downloading model from {self.model_repo}/{self.model_file}")
                model_file = hf_hub_download(
                    repo_id=self.model_repo,
                    filename=self.model_file,
                )

            self._model = Llama(
                model_path=model_file,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
            )
            logger.info("Model loaded successfully")

        except ImportError as e:
            raise LLMInferenceError(
                "llama-cpp-python is not installed. Install with: pip install llama-cpp-python",
                original_error=e,
            )
        except Exception as e:
            raise LLMInferenceError(
                f"Failed to load LLM model: {str(e)}",
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
        if use_cache:
            cached = self._get_cached_response(prompt, max_tokens, temperature)
            if cached is not None:
                logger.debug("Using cached response")
                return cached

        self._load_model()

        try:
            response = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "<|end|>", "<|eot_id|>"],
                echo=False,
            )

            generated_text = response["choices"][0]["text"].strip()

            if use_cache:
                self._cache_response(prompt, max_tokens, temperature, generated_text)

            return generated_text

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
        content = f"{prompt}|{max_tokens}|{temperature}"
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


class MockLlamaClient:
    """Mock LLM client for testing without a real model."""

    def __init__(self, default_response: Optional[str] = None) -> None:
        """
        Initialize the mock client.

        Args:
            default_response: Default response to return from generate()
        """
        self.default_response = default_response or self._get_default_response()
        self.call_count = 0
        self.last_prompt: Optional[str] = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        use_cache: bool = True,
    ) -> str:
        """Generate a mock response."""
        self.call_count += 1
        self.last_prompt = prompt
        return self.default_response

    def is_available(self) -> bool:
        """Always returns True for mock client."""
        return True

    @staticmethod
    def _get_default_response() -> str:
        return """Based on the current Murban-Brent spread data, the market shows a neutral to slightly bullish bias.

The spread has remained relatively stable over the past week, with the 5-day moving average hovering around the 20-day MA. This suggests consolidation rather than a strong directional move.

Key observations:
- Spread compression indicates potential mean reversion
- Recent price action shows balanced buyer/seller activity
- Volume patterns support the consolidation thesis

SIGNAL: neutral
CONFIDENCE: 0.6
SUMMARY: Market in consolidation with slight bullish undertone; await breakout confirmation."""
