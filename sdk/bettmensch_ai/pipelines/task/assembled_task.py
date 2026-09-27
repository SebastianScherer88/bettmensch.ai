"""`AssembledTask`: the resolved, executable representation of a `Task`
call recorded during pipeline tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .decorators import ResourceRequirements, UvRequirements

if TYPE_CHECKING:
    from ..materializers.base_materializer import BaseMaterializer
    from .task import Task


@dataclass
class AssembledTask:
    """Resolved, executable representation of a `Task` being referenced in a
    pipeline definition.

    Instances are produced by calling a `Task` while a `Pipeline` is being
    traced; they are not meant to be constructed directly by users. `name`
    is this task's stable identity within the pipeline (unique among all
    tasks in the same `AssembledPipeline` - see
    `PipelineAssemblyContext.generate_unique_name`), used e.g. to address
    its artifacts in a `BaseArtifactStore`.

    Attributes:
        name: This task's unique name within the pipeline.
        task: The `Task` this was assembled from.
        static_inputs: This task's literal/constant input values - the
            ones not sourced from an `IOBinding`.
        output_names: The names of this task's output(s) - a single
            `(io_binding.DEFAULT_OUTPUT_NAME,)` for an ordinary task, or one
            name per field/key for a `NamedTuple`/`TypedDict`-returning
            (multi-output) one. See `Task.output_names`.
        materializers: Maps every input name and every name in
            `output_names` to the `BaseMaterializer` resolved for it.
            `None` until the `Assembler` resolves them.
    """

    name: str
    task: "Task"
    static_inputs: Dict[str, Any]
    output_names: Tuple[str, ...]
    materializers: Optional[Dict[str, "BaseMaterializer"]] = None

    @property
    def func(self):
        """The underlying, undecorated python function this task wraps."""

        return self.task.func

    @property
    def resource_requirements(self) -> ResourceRequirements:
        """The `@resource` requirements declared on this task's function."""

        return self.task.resource_requirements

    @property
    def uv_requirements(self) -> UvRequirements:
        """The `@uv` dependencies declared on this task's function."""

        return self.task.uv_requirements
