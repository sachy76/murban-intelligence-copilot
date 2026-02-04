"""Mock LLM client for testing and development."""

from __future__ import annotations

from typing import Optional


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
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: Optional[bool] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
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
        return """Based on the current WTI-Brent spread data, the market shows a neutral to slightly bullish bias.

The spread has remained relatively stable over the past week, with the 5-day moving average hovering around the 20-day MA. This suggests consolidation rather than a strong directional move.

Key observations:
- Spread compression indicates potential mean reversion
- Recent price action shows balanced buyer/seller activity
- Volume patterns support the consolidation thesis

SIGNAL: neutral
CONFIDENCE: 0.6
SUMMARY: Market in consolidation with slight bullish undertone; await breakout confirmation."""
