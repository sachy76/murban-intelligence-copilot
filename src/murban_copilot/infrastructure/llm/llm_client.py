"""LLM client using llama-cpp-python with GGUF models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from murban_copilot.domain.exceptions import LLMInferenceError
from murban_copilot.infrastructure.logging import get_logger
from .base_client import BaseLLMClient

if TYPE_CHECKING:
    from murban_copilot.domain.config import LLMModelConfig

logger = get_logger(__name__)


class LlamaClient(BaseLLMClient):
    """Client for local LLM inference using llama-cpp-python."""

    DEFAULT_MODEL_REPO = "MaziyarPanahi/gemma-3-12b-it-GGUF"
    DEFAULT_MODEL_FILE = "gemma-3-12b-it.Q6_K.gguf"

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_repo: Optional[str] = None,
        model_file: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = True,
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
            cache_enabled: Whether to enable response caching
            verbose: Whether to enable verbose output
        """
        super().__init__(cache_dir=cache_dir, cache_enabled=cache_enabled)

        self.model_path = model_path
        self.model_repo = model_repo or self.DEFAULT_MODEL_REPO
        self.model_file = model_file or self.DEFAULT_MODEL_FILE
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose

        self._model = None

    @classmethod
    def from_config(
        cls,
        config: "LLMModelConfig",
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = True,
        verbose: bool = False,
    ) -> "LlamaClient":
        """
        Create a LlamaClient from an LLMModelConfig.

        Args:
            config: Model configuration
            cache_dir: Directory for response caching
            cache_enabled: Whether to enable response caching
            verbose: Whether to enable verbose output

        Returns:
            Configured LlamaClient instance
        """
        return cls(
            model_repo=config.model_repo,
            model_file=config.model_file,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            verbose=verbose,
        )

    def _get_model_identifier(self) -> str:
        """Get unique identifier for cache keys."""
        return f"{self.model_repo}/{self.model_file}"

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

    def _do_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> str:
        """Perform generation using llama-cpp."""
        # Build kwargs with only non-None values
        kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["</s>", "<|end|>", "<|eot_id|>"],
            "echo": False,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty

        response = self._model(prompt, **kwargs)
        return response["choices"][0]["text"].strip()
