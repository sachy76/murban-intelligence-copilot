"""Protocols for LLM inference."""

from typing import Protocol


class LLMInference(Protocol):
    """Protocol for LLM inference implementations."""

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
        ...

    def is_available(self) -> bool:
        """
        Check if the LLM is available and loaded.

        Returns:
            True if the model is ready for inference
        """
        ...
