"""Structured logging with rotation for the Murban Copilot application."""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console_output: bool = True,
) -> None:
    """
    Configure structured logging with file rotation.

    Args:
        log_dir: Directory for log files (default: ./logs)
        log_level: Minimum log level (default: INFO)
        max_bytes: Maximum size of each log file (default: 10 MB)
        backup_count: Number of backup files to keep (default: 5)
        console_output: Whether to also log to console (default: True)
    """
    global _CONFIGURED

    if _CONFIGURED:
        return

    if log_dir is None:
        log_dir = Path.cwd() / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("murban_copilot")
    root_logger.setLevel(log_level)

    root_logger.handlers.clear()

    formatter = StructuredFormatter()

    log_file = log_dir / "murban_copilot.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Name of the logger (usually __name__)

    Returns:
        Configured logger instance
    """
    if not _CONFIGURED:
        setup_logging()

    if not name.startswith("murban_copilot"):
        name = f"murban_copilot.{name}"

    return logging.getLogger(name)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured log output."""

    def __init__(self) -> None:
        super().__init__()
        self.default_format = (
            "%(timestamp)s | %(levelname)-8s | %(name)s | %(message)s"
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with structured fields."""
        record.timestamp = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

        if not hasattr(record, "extra_fields"):
            record.extra_fields = {}

        # Ensure message attribute is set
        record.message = record.getMessage()

        formatted = self.default_format % record.__dict__

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            formatted = f"{formatted}\n{exc_text}"

        return formatted


class LogContext:
    """Context manager for adding structured fields to log messages."""

    def __init__(self, logger: logging.Logger, **kwargs: object) -> None:
        """
        Initialize with context fields.

        Args:
            logger: Logger instance to use
            **kwargs: Fields to add to log messages
        """
        self.logger = logger
        self.fields = kwargs

    def __enter__(self) -> "LogContext":
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        if exc_val is not None:
            self.logger.exception(
                "Error in context",
                exc_info=(exc_type, exc_val, exc_tb),
                extra={"extra_fields": self.fields},
            )

    def info(self, message: str, **kwargs: object) -> None:
        """Log an info message with context."""
        extra = {**self.fields, **kwargs}
        self.logger.info(f"{message} | {self._format_extra(extra)}")

    def warning(self, message: str, **kwargs: object) -> None:
        """Log a warning message with context."""
        extra = {**self.fields, **kwargs}
        self.logger.warning(f"{message} | {self._format_extra(extra)}")

    def error(self, message: str, **kwargs: object) -> None:
        """Log an error message with context."""
        extra = {**self.fields, **kwargs}
        self.logger.error(f"{message} | {self._format_extra(extra)}")

    def debug(self, message: str, **kwargs: object) -> None:
        """Log a debug message with context."""
        extra = {**self.fields, **kwargs}
        self.logger.debug(f"{message} | {self._format_extra(extra)}")

    @staticmethod
    def _format_extra(fields: dict[str, object]) -> str:
        """Format extra fields as key=value pairs."""
        return " ".join(f"{k}={v}" for k, v in fields.items())
