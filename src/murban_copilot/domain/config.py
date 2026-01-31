"""Domain configuration entities for application settings."""

from dataclasses import dataclass, field
from typing import Any, Optional


# =============================================================================
# Market Data Configuration
# =============================================================================


@dataclass
class MarketDataConfig:
    """Configuration for market data fetching."""

    # Ticker symbols
    wti_ticker: str = "CL=F"
    brent_ticker: str = "BZ=F"

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
        """Create config from dictionary."""
        return cls(
            wti_ticker=data.get("wti_ticker", "CL=F"),
            brent_ticker=data.get("brent_ticker", "BZ=F"),
            timeout=data.get("timeout", 60),
            max_retries=data.get("max_retries", 3),
            min_retry_wait=data.get("min_retry_wait", 1),
            max_retry_wait=data.get("max_retry_wait", 10),
            buffer_days=data.get("buffer_days", 10),
            latest_price_lookback_days=data.get("latest_price_lookback_days", 7),
        )


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
        return cls(
            short_ma_window=data.get("short_ma_window", 5),
            long_ma_window=data.get("long_ma_window", 20),
            outlier_threshold=data.get("outlier_threshold", 3.0),
            gap_fill_threshold=data.get("gap_fill_threshold", 2),
            min_data_points=data.get("min_data_points", 5),
        )


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
        return cls(
            default_signal=data.get("default_signal", "neutral"),
            default_confidence=data.get("default_confidence", 0.5),
            bullish_keywords=data.get(
                "bullish_keywords", ["bullish", "upward", "positive", "buy"]
            ),
            bearish_keywords=data.get(
                "bearish_keywords", ["bearish", "downward", "negative", "sell"]
            ),
        )


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
        return cls(
            min_historical_days=data.get("min_historical_days", 7),
            max_historical_days=data.get("max_historical_days", 90),
            default_historical_days=data.get("default_historical_days", 30),
            sample_wti_base_price=data.get("sample_wti_base_price", 85.0),
            sample_brent_base_price=data.get("sample_brent_base_price", 82.0),
            sample_price_variation=data.get("sample_price_variation", 0.5),
        )


# =============================================================================
# LLM Configuration
# =============================================================================


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


# =============================================================================
# Application Configuration (Top-Level)
# =============================================================================


@dataclass
class AppConfig:
    """Complete application configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
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
            market_data=MarketDataConfig(),
            analysis=AnalysisConfig(),
            signal=SignalConfig(),
            ui=UIConfig(),
        )
