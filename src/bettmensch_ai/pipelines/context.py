"""`PipelineAssemblyContext`: the recording surface a `Pipeline`'s function
traces into.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Dict, List, Optional

from .io_binding import IOBinding, PipelineInput, PipelineOutput

if TYPE_CHECKING:
    from .task.assembled_task import AssembledTask

_current_context: "contextvars.ContextVar[Optional[PipelineAssemblyContext]]" = (
    contextvars.ContextVar("bettmensch_ai_pipeline_assembly_context", default=None)
)


class PipelineAssemblyError(Exception):
    """Raised when a `Task` or a pipeline output is assembled incorrectly,
    e.g. outside of an active `PipelineAssemblyContext` or with a name that
    collides with an already assembled task.
    """


class PipelineAssemblyContext:
    """Records the `AssembledTask`s and `IOBinding`s produced while a single
    `Pipeline`'s function is traced.

    Exactly one context is active at a time (re-entrant tracing of nested
    pipelines is out of scope for this first draft).
    """

    def __init__(self) -> None:
        """Initializes an empty context, with no tasks, bindings, inputs,
        or output recorded yet.
        """

        self.pipeline_inputs: Dict[str, PipelineInput] = {}
        self.assembled_tasks: Dict[str, "AssembledTask"] = {}
        self.io_bindings: List[IOBinding] = []
        self.pipeline_output: Optional[PipelineOutput] = None
        self._token: Optional[contextvars.Token] = None

    def generate_unique_name(self, base_name: str) -> str:
        """Generates a name unique to this context by appending a counter
        suffix to `base_name` if it has already been used.

        Args:
            base_name: The name a new `AssembledTask` would like to use.

        Returns:
            `base_name` itself if unused so far, otherwise `base_name`
            suffixed with the number of tasks already using it (e.g.
            `"add"`, then `"add-1"`, `"add-2"`, ...).
        """

        count = sum(
            1
            for name in self.assembled_tasks
            if name == base_name or name.startswith(f"{base_name}-")
        )

        return base_name if count == 0 else f"{base_name}-{count}"

    def register_task(self, assembled_task: "AssembledTask") -> None:
        """Records a newly assembled task.

        Args:
            assembled_task: The task to record.

        Raises:
            PipelineAssemblyError: If a task with the same name is already
                registered.
        """

        if assembled_task.name in self.assembled_tasks:
            raise PipelineAssemblyError(
                f"An assembled task named {assembled_task.name!r} has "
                "already been registered in this pipeline."
            )

        self.assembled_tasks[assembled_task.name] = assembled_task

    def register_binding(self, binding: IOBinding) -> None:
        """Records a newly created `IOBinding`.

        Args:
            binding: The binding to record.
        """

        self.io_bindings.append(binding)

    def __enter__(self) -> "PipelineAssemblyContext":
        """Activates this context as the currently active one.

        Returns:
            This context.

        Raises:
            PipelineAssemblyError: If another context is already active.
        """

        if _current_context.get() is not None:
            raise PipelineAssemblyError(
                "A pipeline is already being assembled; nested pipeline "
                "tracing is not supported."
            )

        self._token = _current_context.set(self)

        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        """Deactivates this context."""

        _current_context.reset(self._token)
        self._token = None


def get_active_context() -> Optional[PipelineAssemblyContext]:
    """Returns the `PipelineAssemblyContext` currently being traced, or
    `None` if no pipeline is being assembled.
    """

    return _current_context.get()
