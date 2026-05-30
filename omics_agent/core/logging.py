"""
Structured logging setup for OmicsAgent.

Uses structlog with ConsoleRenderer for development and JSONRenderer for production.
Integrates with LangSmith via the config.configure_tracing() call.
"""

from __future__ import annotations

import logging
import sys
import structlog


def configure_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=shared_processors + [renderer],
        )
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())
    root_logger.handlers = [handler]


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
