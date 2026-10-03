"""`RegisteredPipeline`: a lightweight reference to a pipeline registered
with a remote backend orchestrator - what `RemoteRunner` runs, the way
`LocalRunner` runs an `AssembledPipeline`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict

from ..metadata_store import BaseMetadataStore


@dataclass(frozen=True)
class RegisteredPipeline:
    """A thin, read-only projection of a `PipelineRegistrationRecord` -
    deliberately not a re-hydrated `AssembledPipeline`/`CompiledPipeline`:
    running an already-registered pipeline needs none of the Python code
    that originally assembled/compiled it, only a reference to the
    backend-specific resource that registration created (e.g. a Step
    Functions state machine ARN, in `backend_metadata`).

    Attributes:
        pipeline_registration_id: This registration's own id.
        pipeline_name: The name of the registered pipeline.
        backend: Which backend orchestrator it's registered with (e.g.
            `"aws_stepfunctions"`) - `RemoteRunner` dispatches on this.
        dag_structure: The registered DAG's structure.
        pipeline_inputs: The pipeline's declared inputs.
        backend_metadata: Backend-specific resource references (e.g. a
            state machine ARN) created at registration time.
    """

    pipeline_registration_id: uuid.UUID
    pipeline_name: str
    backend: str
    dag_structure: Dict[str, Any]
    pipeline_inputs: Dict[str, Any]
    backend_metadata: Dict[str, Any]

    @classmethod
    def from_registration(
        cls, metadata_store: BaseMetadataStore, pipeline_registration_id: uuid.UUID
    ) -> "RegisteredPipeline":
        """Re-creates a `RegisteredPipeline` by looking up its registration
        in `metadata_store`.

        Args:
            metadata_store: The store the registration was recorded in.
            pipeline_registration_id: The registration to look up.

        Returns:
            The corresponding `RegisteredPipeline`.

        Raises:
            KeyError: If no such registration exists.
        """

        record = metadata_store.get_pipeline_registration(pipeline_registration_id)

        return cls(
            pipeline_registration_id=record.pipeline_registration_id,
            pipeline_name=record.pipeline_name,
            backend=record.backend,
            dag_structure=record.dag_structure,
            pipeline_inputs=record.pipeline_inputs,
            backend_metadata=record.backend_metadata,
        )
