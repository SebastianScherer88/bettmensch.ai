"""`MetadataClient`: the local Python access route every runtime (the
developer's own session, or a remote compute task) holds for metadata
bookkeeping - never talks to a database (or a metadata service) directly,
only ever through the `BaseMetadataStore` it holds.

Deliberately not a `BaseMetadataStore` subclass - it's its own type, so a
client and a store can never be confused for one another even though
every method here has the same name/signature as the store method it
forwards to. In `local` mode that store is a `LocalMetadataStore`; in
`remote` mode it's a `RemoteMetadataStore` talking to the metadata service
(see `docker/metadata-service/`) - either way, `MetadataClient` itself
never knows or cares which.
"""

import uuid
from typing import Any, Dict, List, Optional

from ..metadata_store import (
    BaseMetadataStore,
    LocalMetadataStore,
    PipelineAssemblyRecord,
    PipelineRegistrationRecord,
    PipelineRunRecord,
    RemoteMetadataStore,
    RemoteMetadataStoreConfig,
    RunStatus,
    TaskOutputRecord,
    TaskRunRecord,
    TriggerRecord,
)
from .config import MetadataClientConfig


def _build_default_store(config: MetadataClientConfig) -> BaseMetadataStore:
    if config.backend == "remote":
        return RemoteMetadataStore(RemoteMetadataStoreConfig())
    return LocalMetadataStore()


class MetadataClient:
    """Forwards every call to whichever `BaseMetadataStore` it holds -
    either one given explicitly, or one built from `MetadataClientConfig`
    (environment-driven: `local` or `remote`) when none is given.
    """

    def __init__(self, store: Optional[BaseMetadataStore] = None):
        self.store = store or _build_default_store(MetadataClientConfig())

    # --- Pipeline assemblies -----------------------------------------

    def record_pipeline_assembly(
        self,
        pipeline_name: str,
        dag_structure: Dict[str, Any],
        pipeline_inputs: Dict[str, Any],
    ) -> uuid.UUID:
        return self.store.record_pipeline_assembly(
            pipeline_name, dag_structure, pipeline_inputs
        )

    def get_pipeline_assembly(
        self, pipeline_assembly_id: uuid.UUID
    ) -> PipelineAssemblyRecord:
        return self.store.get_pipeline_assembly(pipeline_assembly_id)

    def list_pipeline_assemblies(
        self, pipeline_name: Optional[str] = None
    ) -> List[PipelineAssemblyRecord]:
        return self.store.list_pipeline_assemblies(pipeline_name=pipeline_name)

    # --- Pipeline runs -------------------------------------------------

    def start_pipeline_run(
        self,
        pipeline_name: str,
        pipeline_run_id: uuid.UUID,
        pipeline_assembly_id: Optional[uuid.UUID] = None,
    ) -> None:
        self.store.start_pipeline_run(
            pipeline_name, pipeline_run_id, pipeline_assembly_id
        )

    def finish_pipeline_run(self, pipeline_run_id: uuid.UUID, status: RunStatus) -> None:
        self.store.finish_pipeline_run(pipeline_run_id, status)

    def get_pipeline_run(self, pipeline_run_id: uuid.UUID) -> PipelineRunRecord:
        return self.store.get_pipeline_run(pipeline_run_id)

    def list_pipeline_runs(
        self, pipeline_name: Optional[str] = None
    ) -> List[PipelineRunRecord]:
        return self.store.list_pipeline_runs(pipeline_name=pipeline_name)

    # --- Task runs ----------------------------------------------------

    def start_task_run(self, pipeline_run_id: uuid.UUID, task_name: str) -> None:
        self.store.start_task_run(pipeline_run_id, task_name)

    def finish_task_run(
        self,
        pipeline_run_id: uuid.UUID,
        task_name: str,
        status: RunStatus,
        logs: Optional[str] = None,
    ) -> None:
        self.store.finish_task_run(pipeline_run_id, task_name, status, logs=logs)

    def record_task_output(
        self,
        pipeline_run_id: uuid.UUID,
        task_name: str,
        output_name: str,
        artifact_key: str,
    ) -> None:
        self.store.record_task_output(
            pipeline_run_id, task_name, output_name, artifact_key
        )

    def get_task_run(self, pipeline_run_id: uuid.UUID, task_name: str) -> TaskRunRecord:
        return self.store.get_task_run(pipeline_run_id, task_name)

    def list_task_runs(self, pipeline_run_id: uuid.UUID) -> List[TaskRunRecord]:
        return self.store.list_task_runs(pipeline_run_id)

    def list_task_outputs(
        self, pipeline_run_id: uuid.UUID, task_name: str
    ) -> List[TaskOutputRecord]:
        return self.store.list_task_outputs(pipeline_run_id, task_name)

    # --- Pipeline registrations ----------------------------------------

    def register_pipeline(
        self,
        pipeline_name: str,
        backend: str,
        dag_structure: Dict[str, Any],
        pipeline_inputs: Dict[str, Any],
        backend_metadata: Optional[Dict[str, Any]] = None,
    ) -> uuid.UUID:
        return self.store.register_pipeline(
            pipeline_name,
            backend,
            dag_structure,
            pipeline_inputs,
            backend_metadata=backend_metadata,
        )

    def deregister_pipeline(self, pipeline_registration_id: uuid.UUID) -> None:
        self.store.deregister_pipeline(pipeline_registration_id)

    def get_pipeline_registration(
        self, pipeline_registration_id: uuid.UUID
    ) -> PipelineRegistrationRecord:
        return self.store.get_pipeline_registration(pipeline_registration_id)

    def list_pipeline_registrations(
        self, pipeline_name: Optional[str] = None, active_only: bool = False
    ) -> List[PipelineRegistrationRecord]:
        return self.store.list_pipeline_registrations(
            pipeline_name=pipeline_name, active_only=active_only
        )

    # --- Triggers -------------------------------------------------------

    def register_trigger(
        self,
        pipeline_registration_id: uuid.UUID,
        trigger_type: str,
        trigger_config: Dict[str, Any],
    ) -> uuid.UUID:
        return self.store.register_trigger(
            pipeline_registration_id, trigger_type, trigger_config
        )

    def deregister_trigger(self, trigger_id: uuid.UUID) -> None:
        self.store.deregister_trigger(trigger_id)

    def list_triggers(
        self, pipeline_registration_id: uuid.UUID, active_only: bool = False
    ) -> List[TriggerRecord]:
        return self.store.list_triggers(
            pipeline_registration_id, active_only=active_only
        )
