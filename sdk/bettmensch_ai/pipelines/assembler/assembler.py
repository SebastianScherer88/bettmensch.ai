"""`Assembler`: validates a traced pipeline and turns it into an
`AssembledPipeline`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Set, Type

from ..exceptions import CyclicGraphError, IOBindingError
from ..io_binding import PipelineInput, TaskOutput
from ..materializers import resolve_materializer_for_type
from ..pipeline.assembled_pipeline import AssembledPipeline

if TYPE_CHECKING:
    from ..context import PipelineAssemblyContext
    from ..materializers.base_materializer import BaseMaterializer
    from ..pipeline.pipeline import Pipeline
    from ..task.assembled_task import AssembledTask


class Assembler:
    """Validates a traced pipeline's graph and assembles it into an
    `AssembledPipeline`.

    Responsible for resolving task references, validating `IOBinding`s,
    resolving materializers, detecting cycles, and constructing a
    topological execution plan (grouped into parallelism-eligible ranks).
    Deliberately independent of any runtime scheduling concerns, and
    distinct from Layer 2's backend-specific compilers: it only ever
    produces a backend-agnostic `AssembledPipeline`, never a backend-
    specific `CompiledPipeline`.
    """

    def assemble(
        self, pipeline: "Pipeline", context: "PipelineAssemblyContext"
    ) -> AssembledPipeline:
        """Validates a traced pipeline and assembles it.

        Args:
            pipeline: The `Pipeline` `context` was traced from.
            context: The `PipelineAssemblyContext` produced by
                `pipeline.trace()`.

        Returns:
            The validated `AssembledPipeline`.

        Raises:
            IOBindingError: If an `IOBinding` or the pipeline output
                references a task, output, or pipeline input that doesn't
                exist.
            CyclicGraphError: If the task dependency graph contains a
                cycle.
        """

        self._validate_bindings(context)
        self._validate_output(context)
        ranks = self._topological_order(context)
        self._resolve_materializers(context, pipeline.default_materializer)
        input_materializers = self._resolve_pipeline_input_materializers(
            pipeline, context
        )

        task_ranks = tuple(
            tuple(context.assembled_tasks[name] for name in rank)
            for rank in ranks
        )
        outputs = (
            (context.pipeline_output,) if context.pipeline_output is not None else ()
        )

        return AssembledPipeline(
            name=pipeline.name,
            task_ranks=task_ranks,
            io_bindings=tuple(context.io_bindings),
            outputs=outputs,
            inputs=tuple(context.pipeline_inputs.values()),
            input_materializers=input_materializers,
            default_materializer=pipeline.default_materializer,
        )

    def _validate_bindings(self, context: "PipelineAssemblyContext") -> None:
        """Validates every recorded `IOBinding`.

        Args:
            context: The traced pipeline's context.

        Raises:
            IOBindingError: If a binding targets a task that was never
                assembled, or its source is invalid (see
                `_validate_source`).
        """

        for binding in context.io_bindings:
            if binding.target.task_name not in context.assembled_tasks:
                raise IOBindingError(
                    "IOBinding targets task "
                    f"{binding.target.task_name!r}, which was never "
                    "assembled."
                )

            self._validate_source(context, binding.source, str(binding))

    def _validate_output(self, context: "PipelineAssemblyContext") -> None:
        """Validates the pipeline's output, if it declares one.

        Args:
            context: The traced pipeline's context.

        Raises:
            IOBindingError: If the output's source is invalid (see
                `_validate_source`).
        """

        if context.pipeline_output is not None:
            self._validate_source(
                context,
                context.pipeline_output.source,
                f"pipeline output {context.pipeline_output.name!r}",
            )

    def _validate_source(
        self, context: "PipelineAssemblyContext", source: Any, described_by: str
    ) -> None:
        """Validates one `IOBinding`/pipeline output's source reference.

        Args:
            context: The traced pipeline's context.
            source: The `TaskOutput` or `PipelineInput` to validate.
            described_by: A human-readable description of what `source`
                belongs to, used in error messages.

        Raises:
            IOBindingError: If `source` is a `TaskOutput` referencing a
                task that was never assembled or an output name that task
                doesn't declare; a `PipelineInput` referencing a name that
                isn't a declared pipeline input; or neither type at all.
        """

        if isinstance(source, TaskOutput):
            source_task = context.assembled_tasks.get(source.task_name)
            if source_task is None:
                raise IOBindingError(
                    f"{described_by} references task "
                    f"{source.task_name!r}, which was never assembled."
                )
            if source.output_name not in source_task.output_names:
                raise IOBindingError(
                    f"{described_by} references output "
                    f"{source.output_name!r} of task {source_task.name!r}, "
                    f"which declares output(s) {source_task.output_names!r}."
                )
        elif isinstance(source, PipelineInput):
            if source.name not in context.pipeline_inputs:
                raise IOBindingError(
                    f"{described_by} references pipeline input "
                    f"{source.name!r}, which is not a declared pipeline "
                    "input."
                )
        else:
            raise IOBindingError(
                f"{described_by} has an unsupported source type: "
                f"{type(source)!r}."
            )

    def _topological_order(
        self, context: "PipelineAssemblyContext"
    ) -> List[List[str]]:
        """Groups tasks into topologically ordered ranks: every task in a
        rank depends only on tasks in earlier ranks, never on another task
        in the same rank, so a parallelism-capable runtime can run an
        entire rank concurrently.

        Args:
            context: The traced pipeline's context.

        Returns:
            The task names, grouped into ranks in dependency order.

        Raises:
            CyclicGraphError: If the task dependency graph contains a
                cycle (an iteration finds no task whose dependencies are
                all already resolved, while tasks remain).
        """

        dependencies: Dict[str, Set[str]] = {
            name: set() for name in context.assembled_tasks
        }
        for binding in context.io_bindings:
            if isinstance(binding.source, TaskOutput):
                dependencies[binding.target.task_name].add(
                    binding.source.task_name
                )

        ranks: List[List[str]] = []
        resolved: Set[str] = set()
        remaining = set(context.assembled_tasks)

        while remaining:
            rank = [
                name
                for name in context.assembled_tasks
                if name in remaining and dependencies[name] <= resolved
            ]
            if not rank:
                raise CyclicGraphError(
                    "Cycle detected in the pipeline's task graph among "
                    f"tasks: {sorted(remaining)!r}."
                )

            ranks.append(rank)
            resolved.update(rank)
            remaining.difference_update(rank)

        return ranks

    def _resolve_materializers(
        self,
        context: "PipelineAssemblyContext",
        default_materializer_cls: Type["BaseMaterializer"],
    ) -> None:
        """Resolves and assigns materializers for every assembled task's
        inputs and output, mutating each `AssembledTask` in place.

        Args:
            context: The traced pipeline's context.
            default_materializer_cls: The pipeline's `default_materializer`,
                used for any input/output type none of the specialised
                materializers support.
        """

        for assembled_task in context.assembled_tasks.values():
            assembled_task.materializers = self._resolve_task_materializers(
                assembled_task, default_materializer_cls
            )

    def _resolve_task_materializers(
        self,
        assembled_task: "AssembledTask",
        default_materializer_cls: Type["BaseMaterializer"],
    ) -> Dict[str, Any]:
        """Resolves the materializers for one task's inputs and output(s).

        Args:
            assembled_task: The task to resolve materializers for.
            default_materializer_cls: The pipeline's `default_materializer`,
                used for any input/output type none of the specialised
                materializers support.

        Returns:
            A mapping from every input name and every name in the task's
            `output_names` to its resolved `BaseMaterializer` - each output
            resolved from its own type hint (`Task.output_type_hints`),
            which for a multi-output task is that field's/key's own
            annotation, not the outer `NamedTuple`/`TypedDict` return type
            itself.
        """

        type_hints = assembled_task.task.type_hints
        materializers = {
            input_name: resolve_materializer_for_type(
                type_hints.get(input_name), default_materializer_cls
            )
            for input_name in assembled_task.task.signature.parameters
        }

        output_type_hints = assembled_task.task.output_type_hints
        materializers.update(
            {
                output_name: resolve_materializer_for_type(
                    output_type_hints.get(output_name), default_materializer_cls
                )
                for output_name in assembled_task.output_names
            }
        )

        return materializers

    def _resolve_pipeline_input_materializers(
        self, pipeline: "Pipeline", context: "PipelineAssemblyContext"
    ) -> Dict[str, Any]:
        """Resolves the materializers for the pipeline's own declared
        inputs, from the pipeline function's own type hints - the same
        static, assembly-time resolution a `Task`'s inputs/output get.

        A pipeline input isn't scoped to any single task's type hints (it
        may feed several tasks, or none), so it can't reuse
        `_resolve_task_materializers`'s per-task resolution; it needs its
        own, resolved once here and persisted on the `AssembledPipeline`
        rather than re-derived at run time.

        Args:
            pipeline: The `Pipeline` `context` was traced from.
            context: The traced pipeline's context.

        Returns:
            A mapping from every declared pipeline input's name to its
            resolved `BaseMaterializer`.
        """

        return {
            name: resolve_materializer_for_type(
                pipeline.type_hints.get(name), pipeline.default_materializer
            )
            for name in context.pipeline_inputs
        }
