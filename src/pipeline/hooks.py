"""Lifecycle hooks framework for pipeline stages.

Defines type aliases for pre-stage and post-stage hook callbacks.
Hooks are registered on PipelineBase via add_pre_hook() and
add_post_hook(). Multiple hooks per stage execute in registration
order. Hook exceptions are logged but never abort the pipeline.
"""

from typing import Awaitable, Callable

from pipeline.pipeline_context import PipelineContext, StageResult

PreHook = Callable[[PipelineContext], Awaitable[None]]
"""Called before a stage executes. Receives the current pipeline context."""

PostHook = Callable[[PipelineContext, StageResult], Awaitable[None]]
"""Called after a stage executes. Receives the context and the stage result."""
