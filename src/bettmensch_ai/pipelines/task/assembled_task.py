"""`AssembledTask`: the resolved, executable representation of a `Task`
call recorded during pipeline tracing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .decorators import ResourceRequirements, UvRequirements

if TYPE_CHECKING:
    from ..compute.base_compute_backend import BaseComputeBackend
    from ..materializers.base_materializer import BaseMaterializer
    from .task import Task


def _default_compute_backend() -> "BaseComputeBackend":
    """Builds the default `LocalComputeBackend` a fresh `AssembledTask` is
    placed on.

    A lazy, function-local import - not a module-level one - because
    `compute/local_compute_backend.py` sits downstream of `runner/` (via
    `execute_task`), and `runner/local_runner.py` imports `AssembledTask`
    directly: a module-level import here would form a real cycle
    (`assembled_task.py` -> `compute/local_compute_backend.py` ->
    `runner/__init__.py` -> `local_runner.py` -> back into
    `assembled_task.py`, still mid-import). By the time an `AssembledTask`
    is actually constructed (during pipeline tracing), every module has
    long since finished loading, so the import is safe here - the same
    pattern `Pipeline.assemble()` already uses for `Assembler`.
    """

    from ..compute.local_compute_backend import LocalComputeBackend

    return LocalComputeBackend()


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
        resource_override: This specific node's `ResourceRequirements`,
            given as a `resources=` keyword argument on the task call that
            produced it, overriding `task.resource_requirements` for this
            node only - every other call of the same `Task` is unaffected.
            `None` if no override was given.
        uv_override: The `uv=`-keyword-argument equivalent of
            `resource_override`, overriding `task.uv_requirements` for this
            node only.
        compute_backend: Which `BaseComputeBackend` this node runs on -
            `LocalComputeBackend` (in-process, the default) unless placed on
            `aws_batch(...)`/`aws_lambda(...)` at the pipeline's call site.
    """

    name: str
    task: "Task"
    static_inputs: Dict[str, Any]
    output_names: Tuple[str, ...]
    materializers: Optional[Dict[str, "BaseMaterializer"]] = None
    resource_override: Optional[ResourceRequirements] = None
    uv_override: Optional[UvRequirements] = None
    compute_backend: "BaseComputeBackend" = field(default_factory=_default_compute_backend)

    @property
    def func(self):
        """The underlying, undecorated python function this task wraps."""

        return self.task.func

    @property
    def resource_requirements(self) -> ResourceRequirements:
        """This node's effective `ResourceRequirements`: `resource_override`
        if this call site set one, otherwise the task's own decorator-level
        default.
        """

        return self.resource_override or self.task.resource_requirements

    @property
    def uv_requirements(self) -> UvRequirements:
        """This node's effective `UvRequirements`: `uv_override` if this
        call site set one, otherwise the task's own decorator-level
        default.
        """

        return self.uv_override or self.task.uv_requirements
