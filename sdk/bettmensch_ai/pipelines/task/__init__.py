"""User-facing task definition: `Task`/`@task`, `@resource`/`@uv`, and the
`AssembledTask` a `Task` call is traced into.
"""

from ..io_binding import TaskInput, TaskOutput
from .assembled_task import AssembledTask
from .decorators import ResourceRequirements, UvRequirements, resource, uv
from .task import Task, task

__all__ = [
    "AssembledTask",
    "ResourceRequirements",
    "UvRequirements",
    "resource",
    "uv",
    "TaskInput",
    "TaskOutput",
    "Task",
    "task",
]
