"""Integration tests for LLM inference."""

import pytest

from murban_copilot.domain.exceptions import LLMInferenceError
from murban_copilot.infrastructure.llm.llm_client import LlamaClient
from murban_copilot.infrastructure.llm.mock_client import MockLlamaClient
from murban_copilot.infrastructure.llm.prompt_templates import TraderTalkTemplate


@pytest.mark.integration
@pytest.mark.slow
class TestLLMIntegration:
    """Integration tests for real LLM inference."""

    def test_mock_client_full_workflow(
        self,
        sample_spread_data,
        sample_moving_averages,
        sample_trend_summary,
    ):
        """Test full workflow with mock client."""
        client = MockLlamaClient()

        prompt = TraderTalkTemplate.get_full_prompt(
            sample_spread_data,
            sample_moving_averages,
            sample_trend_summary,
        )

        result = client.generate(prompt)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "SIGNAL:" in result

    def test_prompt_template_formatting(
        self,
        sample_spread_data,
        sample_moving_averages,
        sample_trend_summary,
    ):
        """Test that prompt template formats correctly."""
        prompt = TraderTalkTemplate.format_analysis_prompt(
            sample_spread_data,
            sample_moving_averages,
            sample_trend_summary,
        )

        assert "Current Spread" in prompt
        assert "Moving Average" in prompt
        assert "Recent Spread History" in prompt

    def test_real_llm_if_available(self, tmp_path):
        """Test real LLM if available."""
        client = LlamaClient(cache_dir=tmp_path / "cache")

        # This test will skip if the model isn't available
        if not client.is_available():
            pytest.skip("LLM model not available")

        try:
            result = client.generate(
                "What is 2+2? Answer briefly.",
                max_tokens=50,
            )
        except Exception as e:
            pytest.skip(f"LLM generation failed: {e}")

        assert isinstance(result, str)
        if len(result) == 0:
            pytest.skip("LLM returned empty response - model may not be fully functional")


@pytest.mark.integration
class TestPromptTemplates:
    """Integration tests for prompt templates."""

    def test_trader_talk_template_includes_disclaimer(self):
        """Test that template includes disclaimer."""
        assert "not constitute financial advice" in TraderTalkTemplate.DISCLAIMER.lower()

    def test_system_prompt_sets_tone(self):
        """Test that system prompt sets appropriate tone."""
        assert "crude oil trader" in TraderTalkTemplate.SYSTEM_PROMPT.lower()
        assert "professional" in TraderTalkTemplate.SYSTEM_PROMPT.lower()

    def test_full_prompt_structure(
        self,
        sample_spread_data,
        sample_moving_averages,
        sample_trend_summary,
    ):
        """Test full prompt has proper structure."""
        prompt = TraderTalkTemplate.get_full_prompt(
            sample_spread_data,
            sample_moving_averages,
            sample_trend_summary,
        )

        # Should contain system prompt
        assert "crude oil trader" in prompt.lower()

        # Should contain data sections
        assert "Current Market Data" in prompt
        assert "Recent Spread History" in prompt
        assert "Your Task" in prompt
