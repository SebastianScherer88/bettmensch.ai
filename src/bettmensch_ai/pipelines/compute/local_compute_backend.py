"""`LocalComputeBackend`: runs a task in-process - the default every
`AssembledTask` is placed on.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Dict, Optional, Type

from ..runner.task_execution import execute_task
from .base_compute_backend import BaseComputeBackend

if TYPE_CHECKING:
    from ..artifact_store import BaseArtifactStore
    from ..materializers.base_materializer import BaseMaterializer
    from ..task.assembled_task import AssembledTask


class LocalComputeBackend(BaseComputeBackend):
    """Runs a task in-process, in whatever process is orchestrating the
    pipeline (typically `LocalRunner`) - the default `compute_backend`
    every `AssembledTask` has unless placed on a remote one at its pipeline
    definition's call site.
    """

    name = "local"
    requires_code_bundle = False

    def run(
        self,
        assembled_task: "AssembledTask",
        artifact_store: "BaseArtifactStore",
        code_bundle_key: Optional[str],
        input_keys: Dict[str, str],
        output_keys: Dict[str, str],
        pipeline_name: str,
        pipeline_run_id: uuid.UUID,
        default_materializer_cls: Type["BaseMaterializer"],
    ) -> Optional[str]:
        return execute_task(
            assembled_task.task,
            artifact_store,
            assembled_task.static_inputs,
            input_keys,
            output_keys,
            default_materializer_cls,
        )
