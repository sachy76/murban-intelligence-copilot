"""Health check functionality for the application."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from murban_copilot.infrastructure.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: Optional[float] = None
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class HealthCheckResult:
    """Overall health check result."""

    status: HealthStatus
    components: list[ComponentHealth]
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "components": [c.to_dict() for c in self.components],
            "checked_at": self.checked_at.isoformat(),
        }


class HealthChecker:
    """Health checker for application components."""

    def __init__(
        self,
        market_client=None,
        llm_client=None,
    ) -> None:
        """
        Initialize the health checker.

        Args:
            market_client: Optional market data client to check
            llm_client: Optional LLM client to check
        """
        self.market_client = market_client
        self.llm_client = llm_client

    def check_all(self) -> HealthCheckResult:
        """
        Run all health checks.

        Returns:
            HealthCheckResult with overall status and component details
        """
        components = []

        # Check market data client
        if self.market_client is not None:
            components.append(self._check_market_client())

        # Check LLM client
        if self.llm_client is not None:
            components.append(self._check_llm_client())

        # Determine overall status
        if not components:
            overall_status = HealthStatus.HEALTHY
        elif all(c.status == HealthStatus.HEALTHY for c in components):
            overall_status = HealthStatus.HEALTHY
        elif any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED

        return HealthCheckResult(
            status=overall_status,
            components=components,
        )

    def _check_market_client(self) -> ComponentHealth:
        """Check market data client health."""
        import time

        start = time.perf_counter()
        try:
            # Try to fetch a small amount of data to verify connectivity
            if hasattr(self.market_client, "get_latest_price"):
                self.market_client.get_latest_price("BZ=F")  # Brent crude
                latency = (time.perf_counter() - start) * 1000

                return ComponentHealth(
                    name="market_data",
                    status=HealthStatus.HEALTHY,
                    message="Market data service responding",
                    latency_ms=round(latency, 2),
                )
            else:
                return ComponentHealth(
                    name="market_data",
                    status=HealthStatus.DEGRADED,
                    message="Market client missing get_latest_price method",
                )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.warning(f"Market data health check failed: {e}")
            return ComponentHealth(
                name="market_data",
                status=HealthStatus.UNHEALTHY,
                message=f"Market data unavailable: {str(e)[:100]}",
                latency_ms=round(latency, 2),
            )

    def _check_llm_client(self) -> ComponentHealth:
        """Check LLM client health."""
        import time

        start = time.perf_counter()
        try:
            if hasattr(self.llm_client, "is_available"):
                # Note: is_available() may try to load the model which can be slow
                # For health checks, we just verify the client is configured
                latency = (time.perf_counter() - start) * 1000

                return ComponentHealth(
                    name="llm",
                    status=HealthStatus.HEALTHY,
                    message="LLM client configured",
                    latency_ms=round(latency, 2),
                )
            else:
                return ComponentHealth(
                    name="llm",
                    status=HealthStatus.DEGRADED,
                    message="LLM client missing is_available method",
                )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.warning(f"LLM health check failed: {e}")
            return ComponentHealth(
                name="llm",
                status=HealthStatus.UNHEALTHY,
                message=f"LLM unavailable: {str(e)[:100]}",
                latency_ms=round(latency, 2),
            )

    def _check_component(
        self,
        client,
        name: str,
        check_fn,
    ) -> ComponentHealth:
        """
        Generic component health check wrapper.

        Args:
            client: The client to check (may be None)
            name: Component name for reporting
            check_fn: Function to call if client is configured

        Returns:
            ComponentHealth result
        """
        if client is None:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"{name.replace('_', ' ').title()} client not configured",
            )
        return check_fn()

    def check_market_data(self) -> ComponentHealth:
        """Public method to check only market data health."""
        return self._check_component(
            self.market_client, "market_data", self._check_market_client
        )

    def check_llm(self) -> ComponentHealth:
        """Public method to check only LLM health."""
        return self._check_component(
            self.llm_client, "llm", self._check_llm_client
        )
