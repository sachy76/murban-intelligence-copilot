"""Domain configuration entities for application settings."""

from dataclasses import dataclass, field, fields, MISSING
from enum import Enum
from typing import Any, Optional, TypeVar, Type

T = TypeVar("T")


def config_from_dict(
    cls: Type[T],
    data: dict[str, Any],
    nested: Optional[dict[str, tuple]] = None,
) -> T:
    """
    Generic factory to create a config dataclass from a dictionary.

    Uses dataclass field introspection to automatically map dict keys to fields,
    applying default values when keys are missing.

    Args:
        cls: The dataclass type to instantiate
        data: Dictionary with config values
        nested: Optional mapping of field names to (config_class, extra_args) tuples
                for nested config objects. Use empty dict {} for extra_args if none needed.

    Returns:
        Instance of cls populated from data
    """
    kwargs = {}
    for f in fields(cls):
        if nested and f.name in nested:
            nested_cls, extra_args = nested[f.name]
            nested_data = data.get(f.name, {})
            if extra_args:
                kwargs[f.name] = nested_cls.from_dict(nested_data, **extra_args)
            else:
                kwargs[f.name] = nested_cls.from_dict(nested_data)
        elif f.name in data:
            kwargs[f.name] = data[f.name]
        elif f.default is not MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not MISSING:
            kwargs[f.name] = f.default_factory()
        # else: required field, let dataclass raise error

    return cls(**kwargs)


class ModelType(str, Enum):
    """Type of model backend to use."""

    LLAMA = "llama"  # llama-cpp-python for GGUF models
    TRANSFORMERS = "transformers"  # HuggingFace transformers


# =============================================================================
# Market Data Configuration
# =============================================================================


@dataclass
class MarketDataConfig:
    """Configuration for market data fetching.

    Ticker symbols should be provided via YAML config.
    """

    # Ticker symbols (from YAML - no Python defaults)
    wti_ticker: str = field(default="")
    brent_ticker: str = field(default="")

    # Request settings
    timeout: int = 60
    max_retries: int = 3
    min_retry_wait: int = 1
    max_retry_wait: int = 10

    # Data fetching
    buffer_days: int = 10
    latest_price_lookback_days: int = 7

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketDataConfig":
        """Create config from dictionary (typically from YAML)."""
        return config_from_dict(cls, data)

    def __post_init__(self):
        """Validate required fields are provided."""
        if not self.wti_ticker:
            raise ValueError("wti_ticker must be provided in config")
        if not self.brent_ticker:
            raise ValueError("brent_ticker must be provided in config")


# =============================================================================
# Analysis Configuration
# =============================================================================


@dataclass
class AnalysisConfig:
    """Configuration for spread analysis calculations."""

    # Moving average windows
    short_ma_window: int = 5
    long_ma_window: int = 20

    # Outlier detection
    outlier_threshold: float = 3.0

    # Data processing
    gap_fill_threshold: int = 2
    min_data_points: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisConfig":
        """Create config from dictionary."""
        return config_from_dict(cls, data)


# =============================================================================
# Signal Configuration
# =============================================================================


@dataclass
class SignalConfig:
    """Configuration for signal generation."""

    # Default values when extraction fails
    default_signal: str = "neutral"
    default_confidence: float = 0.5

    # Keywords for fallback classification
    bullish_keywords: list[str] = field(
        default_factory=lambda: ["bullish", "upward", "positive", "buy"]
    )
    bearish_keywords: list[str] = field(
        default_factory=lambda: ["bearish", "downward", "negative", "sell"]
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignalConfig":
        """Create config from dictionary."""
        return config_from_dict(cls, data)


# =============================================================================
# UI Configuration
# =============================================================================


@dataclass
class UIConfig:
    """Configuration for Streamlit UI."""

    # Historical days slider
    min_historical_days: int = 7
    max_historical_days: int = 90
    default_historical_days: int = 30

    # Sample data for demo mode
    sample_wti_base_price: float = 85.0
    sample_brent_base_price: float = 82.0
    sample_price_variation: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UIConfig":
        """Create config from dictionary."""
        return config_from_dict(cls, data)


# =============================================================================
# LLM Configuration
# =============================================================================


@dataclass
class LLMInferenceConfig:
    """Configuration for LLM inference parameters."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: Optional[float] = None  # Nucleus sampling threshold
    top_k: Optional[int] = None  # Top-k sampling
    frequency_penalty: Optional[float] = None  # Penalize frequent tokens
    presence_penalty: Optional[float] = None  # Penalize repeated tokens

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMInferenceConfig":
        """Create config from dictionary."""
        return config_from_dict(cls, data)


@dataclass
class LLMDefaultsConfig:
    """Default configuration for all LLM models."""

    n_ctx: int = 4096
    n_gpu_layers: int = -1
    verbose: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMDefaultsConfig":
        """Create config from dictionary."""
        return config_from_dict(cls, data)


@dataclass
class LLMModelConfig:
    """Configuration for a specific LLM model."""

    model_repo: str
    model_file: str = ""  # Optional for transformers models
    model_type: ModelType = ModelType.LLAMA
    inference: LLMInferenceConfig = field(default_factory=LLMInferenceConfig)
    # Llama-specific settings
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    # Transformers-specific settings
    task: str = "text-generation"  # or "sentiment-analysis", "text-classification"
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        defaults: Optional[LLMDefaultsConfig] = None,
    ) -> "LLMModelConfig":
        """Create config from dictionary, applying defaults."""
        if defaults is None:
            defaults = LLMDefaultsConfig()

        # Parse model_type enum
        model_type_str = data.get("model_type", "llama")
        try:
            model_type = ModelType(model_type_str.lower())
        except ValueError:
            model_type = ModelType.LLAMA

        # Apply defaults for n_ctx and n_gpu_layers
        merged_data = {
            "n_ctx": defaults.n_ctx,
            "n_gpu_layers": defaults.n_gpu_layers,
            **data,
            "model_type": model_type,  # Use parsed enum
        }

        return config_from_dict(cls, merged_data, nested={
            "inference": (LLMInferenceConfig, {}),
        })


@dataclass
class CacheConfig:
    """Configuration for LLM response caching."""

    directory: str = ".llm_cache"
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheConfig":
        """Create config from dictionary."""
        return config_from_dict(cls, data)


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
        """Get the default configuration using centralized defaults."""
        defaults = LLMDefaultsConfig()

        analysis = LLMModelConfig(
            model_repo="MaziyarPanahi/gemma-3-12b-it-GGUF",
            model_file="gemma-3-12b-it.Q6_K.gguf",
            model_type=ModelType.LLAMA,
            inference=LLMInferenceConfig(
                max_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                frequency_penalty=0.3,
                presence_penalty=0.1,
            ),
            n_ctx=defaults.n_ctx,
            n_gpu_layers=defaults.n_gpu_layers,
        )

        extraction = LLMModelConfig(
            model_repo="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            model_type=ModelType.TRANSFORMERS,
            task="sentiment-analysis",
            device="cpu",
            inference=LLMInferenceConfig(
                max_tokens=150,
                temperature=0.1,
                top_p=0.65,
                top_k=25,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            ),
        )

        cache = CacheConfig()

        return cls(
            defaults=defaults,
            analysis=analysis,
            extraction=extraction,
            cache=cache,
        )


# =============================================================================
# Application Configuration (Top-Level)
# =============================================================================


@dataclass
class AppConfig:
    """Complete application configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    market_data: MarketDataConfig = field(
        default_factory=lambda: MarketDataConfig(wti_ticker="CL=F", brent_ticker="BZ=F")
    )
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Create config from dictionary."""
        # Parse LLM config (handles nested 'llm' and 'cache' keys)
        llm_data = data.get("llm", {})
        cache_data = data.get("cache", {})
        llm_combined = {**llm_data, "cache": cache_data} if cache_data else llm_data
        llm = LLMConfig.from_dict(llm_combined)

        # Parse other configs
        market_data = MarketDataConfig.from_dict(data.get("market_data", {}))
        analysis = AnalysisConfig.from_dict(data.get("analysis", {}))
        signal = SignalConfig.from_dict(data.get("signal", {}))
        ui = UIConfig.from_dict(data.get("ui", {}))

        return cls(
            llm=llm,
            market_data=market_data,
            analysis=analysis,
            signal=signal,
            ui=ui,
        )

    @classmethod
    def get_default(cls) -> "AppConfig":
        """Get the default application configuration."""
        return cls(
            llm=LLMConfig.get_default(),
            market_data=MarketDataConfig(wti_ticker="CL=F", brent_ticker="BZ=F"),
            analysis=AnalysisConfig(),
            signal=SignalConfig(),
            ui=UIConfig(),
        )
