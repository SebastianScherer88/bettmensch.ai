"""`BaseComputeBackend`: where an `AssembledTask` actually runs."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Type

if TYPE_CHECKING:
    from ..artifact_store import BaseArtifactStore
    from ..materializers.base_materializer import BaseMaterializer
    from ..task.assembled_task import AssembledTask


class BaseComputeBackend(ABC):
    """Where one `AssembledTask` runs, independent of the pipeline it
    belongs to or how that pipeline itself is orchestrated.

    `LocalRunner` dispatches every task in a pipeline through its own
    `compute_backend.run(...)` unconditionally, whether that's
    `LocalComputeBackend` (in-process, the default) or a remote one
    (`AwsBatchComputeBackend`/`AwsLambdaComputeBackend`) - it never branches
    on backend type itself. A pipeline can freely mix backends across its
    tasks; nothing here assumes every task in a run shares one.
    """

    #: This backend's name, as serialized into `dag_structure` (see
    #: `assembler.recording.serialize_assembled_pipeline`) - write-only,
    #: for display; nothing reconstructs a live backend from a stored name.
    name: ClassVar[str]

    #: Whether running this task needs the pipeline's code bundle uploaded
    #: first (see `CodeBundler`) - `True` for every remote backend, `False`
    #: for `LocalComputeBackend` (already running in the same process that
    #: defined the task, so there's nothing to ship anywhere).
    requires_code_bundle: ClassVar[bool] = True

    @abstractmethod
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
        """Runs `assembled_task` and materializes each of its outputs.

        Args:
            assembled_task: The task to run.
            artifact_store: The store bound inputs are loaded from and
                outputs are saved to - by the same deterministic keys
                regardless of where the task actually executes.
            code_bundle_key: The key this pipeline run's code bundle was
                uploaded under, or `None` if `requires_code_bundle` is
                `False` and none was uploaded.
            input_keys: Maps each of `assembled_task`'s *bound* input names
                to the artifact key to load its value from.
            output_keys: Maps each of `assembled_task`'s declared output
                names to the artifact key to save it under.
            pipeline_name: The name of the pipeline this task belongs to.
            pipeline_run_id: The id of the run this task belongs to.
            default_materializer_cls: The pipeline's own
                `default_materializer`.

        Returns:
            Whatever was captured on stdout/stderr while the task ran, or
            `None` if nothing was.

        Raises:
            Exception: Whatever the task itself raised, or a backend-
                specific execution error if it failed on the remote
                compute itself rather than in the task's own code.
        """

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this backend's configuration for display (e.g. into
        `dag_structure`).

        Write-only: there is no `from_dict` counterpart, since nothing
        reconstructs a live `BaseComputeBackend` from stored JSON - a
        pipeline is always re-assembled from real code to get one, or
        placed on `LocalComputeBackend` by default.

        Returns:
            A plain, JSON-able dict. The base implementation returns `{}`;
            a backend with configuration (e.g. `AwsBatchComputeBackend`)
            overrides this.
        """

        return {}
