"""`AssembledPipeline`: the `Assembler`'s validated output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type

from ..io_binding import IOBinding, PipelineInput, PipelineOutput
from ..task.assembled_task import AssembledTask

if TYPE_CHECKING:
    from ..materializers.base_materializer import BaseMaterializer


@dataclass(frozen=True)
class AssembledPipeline:
    """Validated execution plan for a `Pipeline`: its `AssembledTask`s
    grouped into topologically ordered ranks, plus the `IOBinding`s
    connecting them, resolved by the `Assembler`.

    `task_ranks` is a tuple of ranks (tuples of `AssembledTask`) rather than
    a flat tuple: every task within a rank depends only on tasks in earlier
    ranks, never on another task in the same rank, so a runtime that
    supports parallelism can execute an entire rank concurrently. A runtime
    that doesn't care about that can just iterate `tasks` instead.

    `inputs` and `outputs` are both tuples (not one tuple and one dict) so
    the two are structurally consistent; `outputs` holds at most one
    `PipelineOutput`, since a pipeline - like a `Task` - produces exactly
    one output. `get_input`/`output` are the ergonomic ways to look either
    up.

    This is the backend-agnostic artifact that Layer 2's backend-specific
    compilers turn into a `CompiledPipeline`.

    Attributes:
        name: This pipeline's name.
        task_ranks: The pipeline's tasks, grouped into topologically
            ordered ranks.
        io_bindings: Every connection between a task input and the
            pipeline input or upstream task output that feeds it.
        outputs: This pipeline's output, wrapped in a tuple: empty if it
            declares none, one `PipelineOutput` otherwise.
        inputs: This pipeline's declared inputs.
        input_materializers: Maps each declared input's name to the
            `BaseMaterializer` resolved for it, from the pipeline
            function's own type hints - the same static, assembly-time
            resolution a `Task`'s inputs/output get, since a pipeline input
            isn't scoped to any single task's type hints and so can't
            reuse theirs.
        default_materializer: The `Pipeline`'s own `default_materializer`,
            carried onto the assembled plan so a runner can re-resolve a
            materializer from an actual value (see `LocalRunner`'s
            reconciliation against a declared type hint that turns out not
            to match what a task actually produced) using the same
            fallback the `Assembler` used, rather than always falling back
            to the plain `DefaultMaterializer`.
    """

    name: str
    task_ranks: Tuple[Tuple[AssembledTask, ...], ...]
    io_bindings: Tuple[IOBinding, ...]
    outputs: Tuple[PipelineOutput, ...]
    inputs: Tuple[PipelineInput, ...]
    input_materializers: Dict[str, "BaseMaterializer"]
    default_materializer: Type["BaseMaterializer"]

    @property
    def tasks(self) -> Tuple[AssembledTask, ...]:
        """All `AssembledTask`s, flattened across ranks, in topological
        (but not necessarily parallelism-preserving) order.
        """

        return tuple(
            assembled_task
            for rank in self.task_ranks
            for assembled_task in rank
        )

    @property
    def output(self) -> Optional[PipelineOutput]:
        """This pipeline's one output, or `None` if it declares none."""

        return self.outputs[0] if self.outputs else None

    def get_task(self, name: str) -> AssembledTask:
        """Looks up one of this pipeline's tasks by name.

        Args:
            name: The task's (unique, assembled) name.

        Returns:
            The matching `AssembledTask`.

        Raises:
            KeyError: If no task named `name` exists in this pipeline.
        """

        for assembled_task in self.tasks:
            if assembled_task.name == name:
                return assembled_task

        raise KeyError(f"No assembled task named {name!r} in this pipeline.")

    def get_input(self, name: str) -> PipelineInput:
        """Looks up one of this pipeline's declared inputs by name.

        Args:
            name: The input's name.

        Returns:
            The matching `PipelineInput`.

        Raises:
            KeyError: If no input named `name` is declared.
        """

        for pipeline_input in self.inputs:
            if pipeline_input.name == name:
                return pipeline_input

        raise KeyError(f"No pipeline input named {name!r} in this pipeline.")

    def bindings_for(self, task_name: str) -> Tuple[IOBinding, ...]:
        """Returns the `IOBinding`s that feed the inputs of the task named
        `task_name`.

        Args:
            task_name: The task's (unique, assembled) name.

        Returns:
            That task's `IOBinding`s, in no particular order.
        """

        return tuple(
            binding
            for binding in self.io_bindings
            if binding.target.task_name == task_name
        )
