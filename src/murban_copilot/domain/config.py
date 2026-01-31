"""Domain configuration entities for LLM settings."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMInferenceConfig:
    """Configuration for LLM inference parameters."""

    max_tokens: int = 512
    temperature: float = 0.7

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMInferenceConfig":
        """Create config from dictionary."""
        return cls(
            max_tokens=data.get("max_tokens", 512),
            temperature=data.get("temperature", 0.7),
        )


@dataclass
class LLMDefaultsConfig:
    """Default configuration for all LLM models."""

    n_ctx: int = 4096
    n_gpu_layers: int = -1
    cache_enabled: bool = True
    verbose: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMDefaultsConfig":
        """Create config from dictionary."""
        return cls(
            n_ctx=data.get("n_ctx", 4096),
            n_gpu_layers=data.get("n_gpu_layers", -1),
            cache_enabled=data.get("cache_enabled", True),
            verbose=data.get("verbose", False),
        )


@dataclass
class LLMModelConfig:
    """Configuration for a specific LLM model."""

    model_repo: str
    model_file: str
    inference: LLMInferenceConfig = field(default_factory=LLMInferenceConfig)
    n_ctx: int = 4096
    n_gpu_layers: int = -1

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        defaults: Optional[LLMDefaultsConfig] = None,
    ) -> "LLMModelConfig":
        """Create config from dictionary, applying defaults."""
        if defaults is None:
            defaults = LLMDefaultsConfig()

        inference_data = data.get("inference", {})
        inference = LLMInferenceConfig.from_dict(inference_data)

        return cls(
            model_repo=data.get("model_repo", ""),
            model_file=data.get("model_file", ""),
            inference=inference,
            n_ctx=data.get("n_ctx", defaults.n_ctx),
            n_gpu_layers=data.get("n_gpu_layers", defaults.n_gpu_layers),
        )


@dataclass
class CacheConfig:
    """Configuration for LLM response caching."""

    directory: str = ".llm_cache"
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheConfig":
        """Create config from dictionary."""
        return cls(
            directory=data.get("directory", ".llm_cache"),
            enabled=data.get("enabled", True),
        )


@dataclass
class LLMConfig:
    """Complete LLM configuration."""

    defaults: LLMDefaultsConfig = field(default_factory=LLMDefaultsConfig)
    analysis: Optional[LLMModelConfig] = None
    extraction: Optional[LLMModelConfig] = None
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMConfig":
        """Create config from dictionary."""
        # Parse defaults first
        defaults_data = data.get("defaults", {})
        defaults = LLMDefaultsConfig.from_dict(defaults_data)

        # Parse model configs with defaults applied
        analysis = None
        if "analysis" in data:
            analysis = LLMModelConfig.from_dict(data["analysis"], defaults)

        extraction = None
        if "extraction" in data:
            extraction = LLMModelConfig.from_dict(data["extraction"], defaults)

        # Parse cache config
        cache_data = data.get("cache", {})
        cache = CacheConfig.from_dict(cache_data)

        return cls(
            defaults=defaults,
            analysis=analysis,
            extraction=extraction,
            cache=cache,
        )

    @classmethod
    def get_default(cls) -> "LLMConfig":
        """Get the default configuration matching hardcoded defaults."""
        defaults = LLMDefaultsConfig(
            n_ctx=4096,
            n_gpu_layers=-1,
            cache_enabled=True,
            verbose=False,
        )

        analysis = LLMModelConfig(
            model_repo="MaziyarPanahi/gemma-3-12b-it-GGUF",
            model_file="gemma-3-12b-it.Q6_K.gguf",
            inference=LLMInferenceConfig(max_tokens=2048, temperature=0.7),
            n_ctx=defaults.n_ctx,
            n_gpu_layers=defaults.n_gpu_layers,
        )

        extraction = LLMModelConfig(
            model_repo="bartowski/gemma-2-9b-it-GGUF",
            model_file="gemma-2-9b-it-Q4_K_M.gguf",
            inference=LLMInferenceConfig(max_tokens=1024, temperature=0.3),
            n_ctx=defaults.n_ctx,
            n_gpu_layers=defaults.n_gpu_layers,
        )

        cache = CacheConfig(directory=".llm_cache", enabled=True)

        return cls(
            defaults=defaults,
            analysis=analysis,
            extraction=extraction,
            cache=cache,
        )
