"""JSON-serializable response models for the frontend API.

These mirror the dataclasses in `bettmensch_ai.pipelines.metadata_store`
(`PipelineRunRecord`, `TaskRunRecord`, ...) but as Pydantic models, since
FastAPI needs those for automatic response serialization/typing - the SDK's
own record types stay plain dataclasses on purpose (no FastAPI/Pydantic
dependency in the core library).
"""

import uuid
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel

from bettmensch_ai.pipelines.metadata_store import (
    PipelineAssemblyRecord,
    PipelineRegistrationRecord,
    PipelineRunRecord,
    TaskOutputRecord,
    TaskRunRecord,
    TriggerRecord,
)


class PipelineAssembly(BaseModel):
    pipeline_assembly_id: uuid.UUID
    pipeline_name: str
    dag_structure: dict
    pipeline_inputs: dict
    assembled_at: datetime

    @classmethod
    def from_record(cls, record: PipelineAssemblyRecord) -> "PipelineAssembly":
        return cls(
            pipeline_assembly_id=record.pipeline_assembly_id,
            pipeline_name=record.pipeline_name,
            dag_structure=record.dag_structure,
            pipeline_inputs=record.pipeline_inputs,
            assembled_at=record.assembled_at,
        )


class PipelineRun(BaseModel):
    pipeline_run_id: uuid.UUID
    pipeline_name: str
    status: str
    started_at: datetime
    ended_at: Optional[datetime]
    pipeline_assembly_id: Optional[uuid.UUID] = None

    @classmethod
    def from_record(cls, record: PipelineRunRecord) -> "PipelineRun":
        return cls(
            pipeline_run_id=record.pipeline_run_id,
            pipeline_name=record.pipeline_name,
            status=record.status.value,
            started_at=record.started_at,
            ended_at=record.ended_at,
            pipeline_assembly_id=record.pipeline_assembly_id,
        )


class TaskRun(BaseModel):
    pipeline_run_id: uuid.UUID
    task_name: str
    status: str
    started_at: datetime
    ended_at: Optional[datetime]
    logs: Optional[str] = None

    @classmethod
    def from_record(cls, record: TaskRunRecord) -> "TaskRun":
        return cls(
            pipeline_run_id=record.pipeline_run_id,
            task_name=record.task_name,
            status=record.status.value,
            started_at=record.started_at,
            ended_at=record.ended_at,
            logs=record.logs,
        )


class TaskOutput(BaseModel):
    pipeline_run_id: uuid.UUID
    task_name: str
    output_name: str
    artifact_key: str

    @classmethod
    def from_record(cls, record: TaskOutputRecord) -> "TaskOutput":
        return cls(
            pipeline_run_id=record.pipeline_run_id,
            task_name=record.task_name,
            output_name=record.output_name,
            artifact_key=record.artifact_key,
        )


class ArtifactPreview(BaseModel):
    artifact_key: str
    materializer: Optional[str] = None
    value_type: Optional[str] = None
    previewable: bool
    value: Optional[Any] = None
    error: Optional[str] = None


class ArtifactSummary(BaseModel):
    pipeline_name: str
    pipeline_run_id: uuid.UUID
    run_started_at: datetime
    run_status: str
    task_name: str
    output_name: str
    artifact_key: str
    materializer: Optional[str] = None
    value_type: Optional[str] = None


class PipelineRegistration(BaseModel):
    pipeline_registration_id: uuid.UUID
    pipeline_name: str
    backend: str
    dag_structure: dict
    pipeline_inputs: dict
    backend_metadata: dict
    registered_at: datetime
    deregistered_at: Optional[datetime]
    is_active: bool

    @classmethod
    def from_record(cls, record: PipelineRegistrationRecord) -> "PipelineRegistration":
        return cls(
            pipeline_registration_id=record.pipeline_registration_id,
            pipeline_name=record.pipeline_name,
            backend=record.backend,
            dag_structure=record.dag_structure,
            pipeline_inputs=record.pipeline_inputs,
            backend_metadata=record.backend_metadata,
            registered_at=record.registered_at,
            deregistered_at=record.deregistered_at,
            is_active=record.is_active,
        )


class Trigger(BaseModel):
    trigger_id: uuid.UUID
    pipeline_registration_id: uuid.UUID
    trigger_type: str
    trigger_config: dict
    registered_at: datetime
    deregistered_at: Optional[datetime]
    is_active: bool

    @classmethod
    def from_record(cls, record: TriggerRecord) -> "Trigger":
        return cls(
            trigger_id=record.trigger_id,
            pipeline_registration_id=record.pipeline_registration_id,
            trigger_type=record.trigger_type,
            trigger_config=record.trigger_config,
            registered_at=record.registered_at,
            deregistered_at=record.deregistered_at,
            is_active=record.is_active,
        )


class PipelineSummary(BaseModel):
    """One row of the aggregate Pipelines view: a pipeline's name plus its
    most recent assembly/registration/run state, gathered by grouping
    `list_pipeline_assemblies`/`list_pipeline_registrations`/
    `list_pipeline_runs` by `pipeline_name` - there's no single store method
    for this since a "pipeline" isn't its own record, just a name shared
    across the three.
    """

    pipeline_name: str
    is_assembled: bool
    last_assembled_at: Optional[datetime] = None
    is_registered: bool
    backend: Optional[str] = None
    last_registered_at: Optional[datetime] = None
    run_count: int
    last_run_status: Optional[str] = None
    last_run_at: Optional[datetime] = None


class PipelineDetail(BaseModel):
    """A pipeline's full detail view: its latest assembly (backend-agnostic
    DAG), every registration recorded for it (each carrying its own,
    possibly backend-specific DAG + `backend_metadata`), and its run count.
    """

    pipeline_name: str
    latest_assembly: Optional[PipelineAssembly] = None
    registrations: List[PipelineRegistration]
    run_count: int
    last_run: Optional[PipelineRun] = None
