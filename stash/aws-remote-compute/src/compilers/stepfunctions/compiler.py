"""`StepFunctionsCompiler`: turns an `AssembledPipeline` into a
`CompiledPipeline` (a Step Functions state machine definition)."""

from __future__ import annotations

from ...compute.local_compute_backend import LocalComputeBackend
from ...pipeline.assembled_pipeline import AssembledPipeline
from ..exceptions import UnsupportedComputeBackendError
from .asl import build_state_machine_definition
from .compiled_pipeline import CompiledPipeline


class StepFunctionsCompiler:
    """Compiles an `AssembledPipeline` into a `CompiledPipeline` targeting
    AWS Step Functions.

    Every task must already be placed on `AwsBatchComputeBackend`/
    `AwsLambdaComputeBackend` (via `aws_batch(...)`/`aws_lambda(...)` at the
    pipeline's own definition) - Step Functions itself doesn't run Python,
    only orchestrates calls to services that do.
    """

    def compile(self, assembled_pipeline: AssembledPipeline) -> CompiledPipeline:
        """Compiles `assembled_pipeline`.

        Args:
            assembled_pipeline: The pipeline to compile.

        Returns:
            The compiled pipeline, ready to
            `.register(metadata_store, state_machine_role_arn=...)`.

        Raises:
            UnsupportedComputeBackendError: If one or more tasks are still
                on `LocalComputeBackend`.
        """

        local_tasks = [
            task.name
            for task in assembled_pipeline.tasks
            if isinstance(task.compute_backend, LocalComputeBackend)
        ]
        if local_tasks:
            raise UnsupportedComputeBackendError(
                f"Pipeline {assembled_pipeline.name!r} has task(s) still on "
                f"local compute: {local_tasks!r}. Every task must be placed "
                "on aws_batch(...)/aws_lambda(...) before compiling to "
                "Step Functions."
            )

        definition = build_state_machine_definition(assembled_pipeline)

        return CompiledPipeline(assembled_pipeline=assembled_pipeline, definition=definition)
