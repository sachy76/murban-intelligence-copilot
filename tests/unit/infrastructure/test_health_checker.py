"""Unit tests for health checker."""

import pytest

from murban_copilot.infrastructure.health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
)


class MockMarketClient:
    """Mock market client for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    def get_latest_price(self, ticker: str) -> float:
        if self.should_fail:
            raise Exception("Market data unavailable")
        return 85.50


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, available: bool = True):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        """Test health status values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestComponentHealth:
    """Tests for ComponentHealth."""

    def test_create_component_health(self):
        """Test creating component health."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="All good",
            latency_ms=10.5,
        )

        assert health.name == "test"
        assert health.status == HealthStatus.HEALTHY
        assert health.message == "All good"
        assert health.latency_ms == 10.5

    def test_to_dict(self):
        """Test converting to dictionary."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK",
        )

        result = health.to_dict()

        assert result["name"] == "test"
        assert result["status"] == "healthy"
        assert result["message"] == "OK"
        assert "checked_at" in result


class TestHealthChecker:
    """Tests for HealthChecker."""

    def test_check_all_no_components(self):
        """Test check_all with no components configured."""
        checker = HealthChecker()
        result = checker.check_all()

        assert result.status == HealthStatus.HEALTHY
        assert len(result.components) == 0

    def test_check_all_healthy_components(self):
        """Test check_all with healthy components."""
        checker = HealthChecker(
            market_client=MockMarketClient(),
            llm_client=MockLLMClient(),
        )
        result = checker.check_all()

        assert result.status == HealthStatus.HEALTHY
        assert len(result.components) == 2

    def test_check_all_unhealthy_market(self):
        """Test check_all with unhealthy market client."""
        checker = HealthChecker(
            market_client=MockMarketClient(should_fail=True),
            llm_client=MockLLMClient(),
        )
        result = checker.check_all()

        assert result.status == HealthStatus.UNHEALTHY
        market_health = next(c for c in result.components if c.name == "market_data")
        assert market_health.status == HealthStatus.UNHEALTHY

    def test_check_market_data(self):
        """Test checking market data health."""
        checker = HealthChecker(market_client=MockMarketClient())
        health = checker.check_market_data()

        assert health.name == "market_data"
        assert health.status == HealthStatus.HEALTHY
        assert health.latency_ms is not None

    def test_check_market_data_not_configured(self):
        """Test checking market data when not configured."""
        checker = HealthChecker()
        health = checker.check_market_data()

        assert health.status == HealthStatus.UNHEALTHY
        assert "not configured" in health.message

    def test_check_llm(self):
        """Test checking LLM health."""
        checker = HealthChecker(llm_client=MockLLMClient())
        health = checker.check_llm()

        assert health.name == "llm"
        assert health.status == HealthStatus.HEALTHY

    def test_check_llm_not_configured(self):
        """Test checking LLM when not configured."""
        checker = HealthChecker()
        health = checker.check_llm()

        assert health.status == HealthStatus.UNHEALTHY
        assert "not configured" in health.message

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        checker = HealthChecker(
            market_client=MockMarketClient(),
            llm_client=MockLLMClient(),
        )
        result = checker.check_all()
        result_dict = result.to_dict()

        assert result_dict["status"] == "healthy"
        assert len(result_dict["components"]) == 2
        assert "checked_at" in result_dict
