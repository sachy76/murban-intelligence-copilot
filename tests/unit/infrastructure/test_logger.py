"""Unit tests for logging infrastructure."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from murban_copilot.infrastructure.logging.logger import (
    LogContext,
    StructuredFormatter,
    get_logger,
    setup_logging,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    @pytest.fixture(autouse=True)
    def reset_logging(self):
        """Reset logging configuration before each test."""
        import murban_copilot.infrastructure.logging.logger as logger_module
        logger_module._CONFIGURED = False

        # Clear handlers
        root_logger = logging.getLogger("murban_copilot")
        root_logger.handlers.clear()

        yield

        logger_module._CONFIGURED = False
        root_logger.handlers.clear()

    def test_setup_creates_log_directory(self, tmp_path):
        """Test that setup creates log directory."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=log_dir, console_output=False)

        assert log_dir.exists()

    def test_setup_creates_log_file(self, tmp_path):
        """Test that setup creates log file."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=log_dir, console_output=False)

        logger = get_logger("test")
        logger.info("Test message")

        log_file = log_dir / "murban_copilot.log"
        assert log_file.exists()

    def test_setup_idempotent(self, tmp_path):
        """Test that setup only runs once."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=log_dir, console_output=False)
        setup_logging(log_dir=log_dir, console_output=False)

        logger = logging.getLogger("murban_copilot")
        # Should only have one file handler
        file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1

    def test_setup_with_console_output(self, tmp_path):
        """Test setup with console output enabled."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir=log_dir, console_output=True)

        logger = logging.getLogger("murban_copilot")
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(stream_handlers) == 1


class TestGetLogger:
    """Tests for get_logger function."""

    @pytest.fixture(autouse=True)
    def reset_logging(self):
        """Reset logging configuration before each test."""
        import murban_copilot.infrastructure.logging.logger as logger_module
        logger_module._CONFIGURED = False
        yield
        logger_module._CONFIGURED = False

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_prefixes_name(self):
        """Test that get_logger prefixes name with murban_copilot."""
        logger = get_logger("test_module")
        assert logger.name == "murban_copilot.test_module"

    def test_get_logger_preserves_prefix(self):
        """Test that existing prefix is preserved."""
        logger = get_logger("murban_copilot.existing")
        assert logger.name == "murban_copilot.existing"


class TestStructuredFormatter:
    """Tests for StructuredFormatter class."""

    def test_format_includes_timestamp(self):
        """Test that format includes ISO timestamp."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "Z" in result  # ISO format with Z suffix
        assert "INFO" in result
        assert "Test message" in result

    def test_format_includes_exception(self):
        """Test that format includes exception info."""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)

        assert "ValueError" in result
        assert "Test error" in result


class TestLogContext:
    """Tests for LogContext class."""

    def test_context_logs_info(self, caplog):
        """Test that context logs info messages."""
        logger = logging.getLogger("test_context")
        logger.setLevel(logging.DEBUG)

        with LogContext(logger, operation="test") as ctx:
            ctx.info("Test info message")

        assert "Test info message" in caplog.text
        assert "operation=test" in caplog.text

    def test_context_logs_error(self, caplog):
        """Test that context logs error messages."""
        logger = logging.getLogger("test_context")
        logger.setLevel(logging.DEBUG)

        with LogContext(logger, operation="test") as ctx:
            ctx.error("Test error message")

        assert "Test error message" in caplog.text

    def test_context_logs_exception_on_error(self, caplog):
        """Test that context logs exception when exiting with error."""
        logger = logging.getLogger("test_context")
        logger.setLevel(logging.DEBUG)

        try:
            with LogContext(logger, operation="test"):
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert "Test exception" in caplog.text

    def test_context_extra_fields(self, caplog):
        """Test that context includes extra fields."""
        logger = logging.getLogger("test_context")
        logger.setLevel(logging.DEBUG)

        with LogContext(logger, ticker="BZ=F", days=30) as ctx:
            ctx.info("Fetching data", extra_field="value")

        assert "ticker=BZ=F" in caplog.text
        assert "days=30" in caplog.text
