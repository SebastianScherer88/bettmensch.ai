"""Symbolic references and bindings recorded while a `Pipeline` is traced:
`TaskInput`, `TaskOutput`, `PipelineInput`, `PipelineOutput`, `IOBinding`.
"""

from dataclasses import dataclass, field
from typing import Any, Union

# TaskInput, TaskOutput, PipelineInput and PipelineOutput live here alongside
# IOBinding, rather than under task/ and pipeline/ respectively, because they
# reference each other (an IOBinding/PipelineOutput's source can be either a
# TaskOutput or a PipelineInput): splitting them across the task and pipeline
# packages would make those two packages depend on each other for these types
# alone. This module has no dependencies on task/ or pipeline/ so that
# neither of them needs to reach into the other.
#
# All four reference types share one shape: a task-scoped reference names its
# owning (assembled) task plus a slot on it (`task_name` + `input_name` /
# `output_name`); a pipeline-scoped reference just names the slot, since
# there is only ever one, implicit, pipeline (`name`).


class _NoDefault:
    """Sentinel distinguishing "no default value" from a real default value
    of `None` on a `PipelineInput`.
    """

    def __repr__(self) -> str:
        return "NO_DEFAULT"


NO_DEFAULT = _NoDefault()

# A `Task`/`Pipeline` produces exactly one output - see design-decisions.md
# ("each task and pipeline can only return one object") - so its name is a
# fixed constant, not something that varies per task/pipeline, shared here
# so `task/task.py` and `pipeline/pipeline.py` don't each hardcode their own
# copy of the same string.
DEFAULT_OUTPUT_NAME = "result"


@dataclass(frozen=True)
class TaskInput:
    """A reference to a single input slot of an `AssembledTask`."""

    task_name: str
    input_name: str


@dataclass(frozen=True)
class TaskOutput:
    """A reference to a single output slot of an `AssembledTask`.

    Produced when a `Task` is invoked while a `Pipeline` is being traced.
    Passing a `TaskOutput` as an argument to another task call records an
    `IOBinding` between the two tasks.
    """

    task_name: str
    output_name: str


@dataclass(frozen=True)
class PipelineInput:
    """A reference to one of the enclosing `Pipeline`'s input slots.

    Substituted for the real argument while the pipeline function is traced,
    so its `default` (if any) is carried here rather than being resolved
    immediately, the way a `Task`'s own defaults are.
    """

    name: str
    default: Any = field(default=NO_DEFAULT)

    @property
    def required(self) -> bool:
        """Whether this pipeline input has no default value.

        Returns:
            `True` if `default` is `NO_DEFAULT`, `False` otherwise.
        """

        return self.default is NO_DEFAULT


@dataclass(frozen=True)
class PipelineOutput:
    """Binds a name exposed on the compiled pipeline to the `TaskOutput` (or
    passed-through `PipelineInput`) that produces it.
    """

    name: str
    source: Union[TaskOutput, PipelineInput]


@dataclass(frozen=True)
class IOBinding:
    """Describes a connection from the pipeline input or upstream task
    output that feeds a task input (`source`) to that input (`target`).
    """

    target: TaskInput
    source: Union[TaskOutput, PipelineInput]
