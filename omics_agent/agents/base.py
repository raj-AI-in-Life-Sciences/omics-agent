"""AgentBase — shared config access and LangSmith tracing helper."""

from __future__ import annotations

import functools
import uuid
from typing import Any, Callable

from ..core.config import get_config


class AgentBase:
    """Mixin providing config access and optional LangSmith tracing."""

    @property
    def config(self):
        return get_config()


def traced_node(name: str):
    """
    Decorator that wraps a node function with LangSmith run tracking.
    Falls back silently if LANGCHAIN_TRACING_V2 is not set.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: dict, *args, **kwargs) -> Any:
            try:
                from langsmith import traceable
                return traceable(name=name)(fn)(state, *args, **kwargs)
            except Exception:
                return fn(state, *args, **kwargs)
        return wrapper
    return decorator
