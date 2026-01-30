"""Domain exceptions for the Murban Copilot application."""


class MurbanCopilotError(Exception):
    """Base exception for all Murban Copilot errors."""
    pass


class InsufficientDataError(MurbanCopilotError):
    """Raised when there is not enough data to perform calculations."""

    def __init__(self, message: str, required: int = 0, available: int = 0) -> None:
        self.required = required
        self.available = available
        super().__init__(message)


class ValidationError(MurbanCopilotError):
    """Raised when data validation fails."""

    def __init__(self, message: str, field: str = "", value: object = None) -> None:
        self.field = field
        self.value = value
        super().__init__(message)


class SpreadCalculationError(MurbanCopilotError):
    """Raised when spread calculation fails."""

    def __init__(self, message: str, murban_data: object = None, brent_data: object = None) -> None:
        self.murban_data = murban_data
        self.brent_data = brent_data
        super().__init__(message)


class MarketDataFetchError(MurbanCopilotError):
    """Raised when fetching market data fails."""

    def __init__(self, message: str, ticker: str = "", original_error: Exception | None = None) -> None:
        self.ticker = ticker
        self.original_error = original_error
        super().__init__(message)


class LLMInferenceError(MurbanCopilotError):
    """Raised when LLM inference fails."""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        self.original_error = original_error
        super().__init__(message)
