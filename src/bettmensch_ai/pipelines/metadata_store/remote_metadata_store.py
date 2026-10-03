"""`RemoteMetadataStore`: a `BaseMetadataStore` implementation that talks
HTTP to a metadata service instead of SQL to a database directly - Layer
2's "behind a service" flavour, alongside `PostgresMetadataStore`'s own
"direct connection" one. See `docker/metadata-service/` for the service
this talks to; that service is the only process actually holding a
Postgres DSN.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from .base_metadata_store import (
    BaseMetadataStore,
    PipelineAssemblyRecord,
    PipelineRegistrationRecord,
    PipelineRunRecord,
    RunStatus,
    TaskOutputRecord,
    TaskRunRecord,
    TriggerRecord,
)


class RemoteMetadataStoreConfig(BaseSettings):
    """Configuration for a `RemoteMetadataStore`.

    Attributes:
        base_url: The metadata service's own base URL (e.g.
            `"http://metadata-service:8081/api"` inside the compose
            network, or `"http://localhost:8081/api"` from the host). No
            default - there is no meaningful "local" fallback for a remote
            service. Can also be set via the
            `bettmensch_ai_metadata_service_base_url` environment variable.
    """

    base_url: str

    model_config = SettingsConfigDict(env_prefix="bettmensch_ai_metadata_service_")


def _raise_for_status(response: httpx.Response) -> None:
    """Raises `KeyError` for a 404, matching `LocalMetadataStore`/
    `PostgresMetadataStore`'s own "no such record" contract - any other
    error status still raises `httpx`'s own exception.
    """

    if response.status_code == 404:
        raise KeyError(response.text)
    response.raise_for_status()


class RemoteMetadataStore(BaseMetadataStore):
    """Remote, service-backed `BaseMetadataStore` - every call is one HTTP
    request to the metadata service named in `config.base_url`, which is
    the only thing that actually talks to Postgres. Opens a fresh
    connection per call (via a short-lived `httpx.Client`), mirroring
    `PostgresMetadataStore`'s own "no long-lived connection" choice, for
    the same reason: this is not a hot path.
    """

    def __init__(self, config: Optional[RemoteMetadataStoreConfig] = None):
        self.config = config or RemoteMetadataStoreConfig()

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.config.base_url, timeout=30.0)

    # --- Pipeline assemblies -----------------------------------------

    def record_pipeline_assembly(
        self,
        pipeline_name: str,
        dag_structure: Dict[str, Any],
        pipeline_inputs: Dict[str, Any],
    ) -> uuid.UUID:
        with self._client() as client:
            response = client.post(
                "/pipeline-assemblies",
                json={
                    "pipeline_name": pipeline_name,
                    "dag_structure": dag_structure,
                    "pipeline_inputs": pipeline_inputs,
                },
            )
            _raise_for_status(response)
            return uuid.UUID(response.json()["id"])

    def get_pipeline_assembly(
        self, pipeline_assembly_id: uuid.UUID
    ) -> PipelineAssemblyRecord:
        with self._client() as client:
            response = client.get(f"/pipeline-assemblies/{pipeline_assembly_id}")
            _raise_for_status(response)
            body = response.json()
            return PipelineAssemblyRecord(
                pipeline_assembly_id=uuid.UUID(body["pipeline_assembly_id"]),
                pipeline_name=body["pipeline_name"],
                dag_structure=body["dag_structure"],
                pipeline_inputs=body["pipeline_inputs"],
                assembled_at=datetime.fromisoformat(body["assembled_at"]),
            )

    def list_pipeline_assemblies(
        self, pipeline_name: Optional[str] = None
    ) -> List[PipelineAssemblyRecord]:
        with self._client() as client:
            response = client.get(
                "/pipeline-assemblies",
                params={"pipeline_name": pipeline_name} if pipeline_name else None,
            )
            _raise_for_status(response)
            return [
                PipelineAssemblyRecord(
                    pipeline_assembly_id=uuid.UUID(body["pipeline_assembly_id"]),
                    pipeline_name=body["pipeline_name"],
                    dag_structure=body["dag_structure"],
                    pipeline_inputs=body["pipeline_inputs"],
                    assembled_at=datetime.fromisoformat(body["assembled_at"]),
                )
                for body in response.json()
            ]

    # --- Pipeline runs -------------------------------------------------

    def start_pipeline_run(
        self,
        pipeline_name: str,
        pipeline_run_id: uuid.UUID,
        pipeline_assembly_id: Optional[uuid.UUID] = None,
    ) -> None:
        with self._client() as client:
            response = client.post(
                "/pipeline-runs",
                json={
                    "pipeline_name": pipeline_name,
                    "pipeline_run_id": str(pipeline_run_id),
                    "pipeline_assembly_id": (
                        str(pipeline_assembly_id) if pipeline_assembly_id else None
                    ),
                },
            )
            _raise_for_status(response)

    def finish_pipeline_run(
        self, pipeline_run_id: uuid.UUID, status: RunStatus
    ) -> None:
        with self._client() as client:
            response = client.patch(
                f"/pipeline-runs/{pipeline_run_id}", json={"status": status.value}
            )
            _raise_for_status(response)

    def get_pipeline_run(self, pipeline_run_id: uuid.UUID) -> PipelineRunRecord:
        with self._client() as client:
            response = client.get(f"/pipeline-runs/{pipeline_run_id}")
            _raise_for_status(response)
            return _pipeline_run_record(response.json())

    def list_pipeline_runs(
        self, pipeline_name: Optional[str] = None
    ) -> List[PipelineRunRecord]:
        with self._client() as client:
            response = client.get(
                "/pipeline-runs",
                params={"pipeline_name": pipeline_name} if pipeline_name else None,
            )
            _raise_for_status(response)
            return [_pipeline_run_record(body) for body in response.json()]

    # --- Task runs ----------------------------------------------------

    def start_task_run(self, pipeline_run_id: uuid.UUID, task_name: str) -> None:
        with self._client() as client:
            response = client.post(
                f"/pipeline-runs/{pipeline_run_id}/task-runs/{task_name}"
            )
            _raise_for_status(response)

    def finish_task_run(
        self,
        pipeline_run_id: uuid.UUID,
        task_name: str,
        status: RunStatus,
        logs: Optional[str] = None,
    ) -> None:
        with self._client() as client:
            response = client.patch(
                f"/pipeline-runs/{pipeline_run_id}/task-runs/{task_name}",
                json={"status": status.value, "logs": logs},
            )
            _raise_for_status(response)

    def record_task_output(
        self,
        pipeline_run_id: uuid.UUID,
        task_name: str,
        output_name: str,
        artifact_key: str,
    ) -> None:
        with self._client() as client:
            response = client.post(
                f"/pipeline-runs/{pipeline_run_id}/task-runs/{task_name}"
                f"/outputs/{output_name}",
                json={"artifact_key": artifact_key},
            )
            _raise_for_status(response)

    def get_task_run(
        self, pipeline_run_id: uuid.UUID, task_name: str
    ) -> TaskRunRecord:
        with self._client() as client:
            response = client.get(
                f"/pipeline-runs/{pipeline_run_id}/task-runs/{task_name}"
            )
            _raise_for_status(response)
            return _task_run_record(response.json())

    def list_task_runs(self, pipeline_run_id: uuid.UUID) -> List[TaskRunRecord]:
        with self._client() as client:
            response = client.get(f"/pipeline-runs/{pipeline_run_id}/task-runs")
            _raise_for_status(response)
            return [_task_run_record(body) for body in response.json()]

    def list_task_outputs(
        self, pipeline_run_id: uuid.UUID, task_name: str
    ) -> List[TaskOutputRecord]:
        with self._client() as client:
            response = client.get(
                f"/pipeline-runs/{pipeline_run_id}/task-runs/{task_name}/outputs"
            )
            _raise_for_status(response)
            return [
                TaskOutputRecord(
                    pipeline_run_id=uuid.UUID(body["pipeline_run_id"]),
                    task_name=body["task_name"],
                    output_name=body["output_name"],
                    artifact_key=body["artifact_key"],
                )
                for body in response.json()
            ]

    # --- Pipeline registrations ----------------------------------------

    def register_pipeline(
        self,
        pipeline_name: str,
        backend: str,
        dag_structure: Dict[str, Any],
        pipeline_inputs: Dict[str, Any],
        backend_metadata: Optional[Dict[str, Any]] = None,
    ) -> uuid.UUID:
        with self._client() as client:
            response = client.post(
                "/pipeline-registrations",
                json={
                    "pipeline_name": pipeline_name,
                    "backend": backend,
                    "dag_structure": dag_structure,
                    "pipeline_inputs": pipeline_inputs,
                    "backend_metadata": backend_metadata,
                },
            )
            _raise_for_status(response)
            return uuid.UUID(response.json()["id"])

    def deregister_pipeline(self, pipeline_registration_id: uuid.UUID) -> None:
        with self._client() as client:
            response = client.post(
                f"/pipeline-registrations/{pipeline_registration_id}/deregister"
            )
            _raise_for_status(response)

    def get_pipeline_registration(
        self, pipeline_registration_id: uuid.UUID
    ) -> PipelineRegistrationRecord:
        with self._client() as client:
            response = client.get(
                f"/pipeline-registrations/{pipeline_registration_id}"
            )
            _raise_for_status(response)
            return _pipeline_registration_record(response.json())

    def list_pipeline_registrations(
        self, pipeline_name: Optional[str] = None, active_only: bool = False
    ) -> List[PipelineRegistrationRecord]:
        with self._client() as client:
            response = client.get(
                "/pipeline-registrations",
                params={
                    **({"pipeline_name": pipeline_name} if pipeline_name else {}),
                    "active_only": active_only,
                },
            )
            _raise_for_status(response)
            return [_pipeline_registration_record(body) for body in response.json()]

    # --- Triggers -------------------------------------------------------

    def register_trigger(
        self,
        pipeline_registration_id: uuid.UUID,
        trigger_type: str,
        trigger_config: Dict[str, Any],
    ) -> uuid.UUID:
        with self._client() as client:
            response = client.post(
                f"/pipeline-registrations/{pipeline_registration_id}/triggers",
                json={"trigger_type": trigger_type, "trigger_config": trigger_config},
            )
            _raise_for_status(response)
            return uuid.UUID(response.json()["id"])

    def deregister_trigger(self, trigger_id: uuid.UUID) -> None:
        with self._client() as client:
            response = client.post(f"/triggers/{trigger_id}/deregister")
            _raise_for_status(response)

    def list_triggers(
        self, pipeline_registration_id: uuid.UUID, active_only: bool = False
    ) -> List[TriggerRecord]:
        with self._client() as client:
            response = client.get(
                f"/pipeline-registrations/{pipeline_registration_id}/triggers",
                params={"active_only": active_only},
            )
            _raise_for_status(response)
            return [
                TriggerRecord(
                    trigger_id=uuid.UUID(body["trigger_id"]),
                    pipeline_registration_id=uuid.UUID(
                        body["pipeline_registration_id"]
                    ),
                    trigger_type=body["trigger_type"],
                    trigger_config=body["trigger_config"],
                    registered_at=datetime.fromisoformat(body["registered_at"]),
                    deregistered_at=(
                        datetime.fromisoformat(body["deregistered_at"])
                        if body["deregistered_at"]
                        else None
                    ),
                )
                for body in response.json()
            ]


def _pipeline_run_record(body: Dict[str, Any]) -> PipelineRunRecord:
    return PipelineRunRecord(
        pipeline_run_id=uuid.UUID(body["pipeline_run_id"]),
        pipeline_name=body["pipeline_name"],
        status=RunStatus(body["status"]),
        started_at=datetime.fromisoformat(body["started_at"]),
        ended_at=datetime.fromisoformat(body["ended_at"]) if body["ended_at"] else None,
        pipeline_assembly_id=(
            uuid.UUID(body["pipeline_assembly_id"])
            if body.get("pipeline_assembly_id")
            else None
        ),
    )


def _task_run_record(body: Dict[str, Any]) -> TaskRunRecord:
    return TaskRunRecord(
        pipeline_run_id=uuid.UUID(body["pipeline_run_id"]),
        task_name=body["task_name"],
        status=RunStatus(body["status"]),
        started_at=datetime.fromisoformat(body["started_at"]),
        ended_at=datetime.fromisoformat(body["ended_at"]) if body["ended_at"] else None,
        logs=body.get("logs"),
    )


def _pipeline_registration_record(body: Dict[str, Any]) -> PipelineRegistrationRecord:
    return PipelineRegistrationRecord(
        pipeline_registration_id=uuid.UUID(body["pipeline_registration_id"]),
        pipeline_name=body["pipeline_name"],
        backend=body["backend"],
        dag_structure=body["dag_structure"],
        pipeline_inputs=body["pipeline_inputs"],
        backend_metadata=body.get("backend_metadata") or {},
        registered_at=datetime.fromisoformat(body["registered_at"]),
        deregistered_at=(
            datetime.fromisoformat(body["deregistered_at"])
            if body["deregistered_at"]
            else None
        ),
    )
