"""Utilities for running async coroutines synchronously.

Handles the Jupyter/IPython case where an event loop is already running
by using ``nest_asyncio`` if available.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run *coro* synchronously and return its result.

    * If no event loop is running → uses ``asyncio.run()``.
    * If an event loop *is* running (e.g. Jupyter) and ``nest_asyncio``
      is installed → patches the loop and runs the coroutine.
    * If an event loop is running and ``nest_asyncio`` is **not**
      installed → raises ``RuntimeError`` with install instructions.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running — simple case
        return asyncio.run(coro)

    # Loop is already running (Jupyter, IPython, etc.)
    try:
        import nest_asyncio

        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)
    except ImportError:
        raise RuntimeError(
            "An asyncio event loop is already running (e.g. Jupyter). "
            "Install nest_asyncio to use sync wrappers in this environment: "
            "pip install nest_asyncio"
        ) from None
