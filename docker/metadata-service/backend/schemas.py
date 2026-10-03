"""JSON-serializable request/response models for the metadata service.

The response models mirror `docker/frontend/backend/schemas.py`'s own
read-side models - deliberately duplicated, not imported, so the two
Docker images stay independently buildable/deployable (see this
directory's own README/the design-decisions entry for why). The request
models are new: this service is the one place that needs to accept
writes, not just serve reads.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from bettmensch_ai.pipelines.metadata_store import (
    PipelineAssemblyRecord,
    PipelineRegistrationRecord,
    PipelineRunRecord,
    TaskOutputRecord,
    TaskRunRecord,
    TriggerRecord,
)

# --- Response models (mirrors docker/frontend/backend/schemas.py) ---------


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


# --- Request bodies (write side - this service's own addition) -----------


class RecordPipelineAssemblyRequest(BaseModel):
    pipeline_name: str
    dag_structure: Dict[str, Any]
    pipeline_inputs: Dict[str, Any]


class IdResponse(BaseModel):
    id: uuid.UUID


class StartPipelineRunRequest(BaseModel):
    pipeline_name: str
    pipeline_run_id: uuid.UUID
    pipeline_assembly_id: Optional[uuid.UUID] = None


class FinishPipelineRunRequest(BaseModel):
    status: str


class FinishTaskRunRequest(BaseModel):
    status: str
    logs: Optional[str] = None


class RecordTaskOutputRequest(BaseModel):
    artifact_key: str


class RegisterPipelineRequest(BaseModel):
    pipeline_name: str
    backend: str
    dag_structure: Dict[str, Any]
    pipeline_inputs: Dict[str, Any]
    backend_metadata: Optional[Dict[str, Any]] = None


class RegisterTriggerRequest(BaseModel):
    trigger_type: str
    trigger_config: Dict[str, Any]
