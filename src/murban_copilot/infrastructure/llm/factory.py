"""Factory for creating LLM clients based on configuration."""

from pathlib import Path
from typing import Optional, Union

from murban_copilot.domain.config import LLMModelConfig, ModelType
from murban_copilot.infrastructure.logging import get_logger
from .llm_client import LlamaClient
from .transformers_client import TransformersClient
from .base_client import LLMInference

logger = get_logger(__name__)

# Type alias for any LLM client
LLMClient = Union[LlamaClient, TransformersClient]


def create_llm_client(
    config: LLMModelConfig,
    cache_dir: Optional[Path] = None,
    cache_enabled: bool = True,
    verbose: bool = False,
) -> LLMClient:
    """
    Create an LLM client based on the model configuration.

    This factory function examines the model_type in the configuration
    and returns the appropriate client implementation:
    - ModelType.LLAMA: Returns LlamaClient for GGUF models via llama-cpp-python
    - ModelType.TRANSFORMERS: Returns TransformersClient for HuggingFace models

    Args:
        config: Model configuration specifying repo, file, and type
        cache_dir: Directory for response caching
        cache_enabled: Whether to enable response caching
        verbose: Whether to enable verbose output (llama only)

    Returns:
        An LLM client instance implementing the LLMInference protocol

    Examples:
        >>> from murban_copilot.domain.config import LLMModelConfig, ModelType
        >>> config = LLMModelConfig(
        ...     model_repo="bartowski/gemma-2-9b-it-GGUF",
        ...     model_file="gemma-2-9b-it-Q4_K_M.gguf",
        ...     model_type=ModelType.LLAMA,
        ... )
        >>> client = create_llm_client(config)
        >>> isinstance(client, LlamaClient)
        True

        >>> config = LLMModelConfig(
        ...     model_repo="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        ...     model_type=ModelType.TRANSFORMERS,
        ...     task="sentiment-analysis",
        ... )
        >>> client = create_llm_client(config)
        >>> isinstance(client, TransformersClient)
        True
    """
    if config.model_type == ModelType.TRANSFORMERS:
        logger.info(
            f"Creating TransformersClient for {config.model_repo} "
            f"(task: {config.task}, device: {config.device})"
        )
        return TransformersClient.from_config(
            config=config,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
        )
    else:
        # Default to LlamaClient for LLAMA type or unknown types
        logger.info(
            f"Creating LlamaClient for {config.model_repo}/{config.model_file} "
            f"(n_ctx: {config.n_ctx}, n_gpu_layers: {config.n_gpu_layers})"
        )
        return LlamaClient.from_config(
            config=config,
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            verbose=verbose,
        )


def get_client_type_name(client: LLMClient) -> str:
    """
    Get a human-readable name for the client type.

    Args:
        client: An LLM client instance

    Returns:
        String describing the client type
    """
    if isinstance(client, LlamaClient):
        return "Llama (GGUF)"
    elif isinstance(client, TransformersClient):
        return "Transformers (HuggingFace)"
    else:
        return "Unknown"
