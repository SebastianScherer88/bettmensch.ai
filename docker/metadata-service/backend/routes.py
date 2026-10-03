"""The metadata service's REST API: every `BaseMetadataStore` method, read
and write, backed by the real `PostgresMetadataStore` this service holds.

This is the one process in the stack that talks to Postgres directly -
every other process (a `LocalRunner` script, the frontend, later a remote
compute task) goes through `RemoteMetadataStore`, which calls these same
routes over HTTP instead. Each route is a thin, 1:1 wrapper around one
`BaseMetadataStore` method; a `KeyError` from the store becomes a 404,
matching `RemoteMetadataStore`'s own expectation on the client side.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from bettmensch_ai.pipelines.metadata_store import BaseMetadataStore, RunStatus

from . import schemas
from .stores import get_metadata_store

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


# --- Pipeline assemblies ---------------------------------------------------


@router.post("/pipeline-assemblies", response_model=schemas.IdResponse)
def record_pipeline_assembly(
    body: schemas.RecordPipelineAssemblyRequest,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    assembly_id = metadata_store.record_pipeline_assembly(
        body.pipeline_name, body.dag_structure, body.pipeline_inputs
    )
    return schemas.IdResponse(id=assembly_id)


@router.get(
    "/pipeline-assemblies/{pipeline_assembly_id}",
    response_model=schemas.PipelineAssembly,
)
def get_pipeline_assembly(
    pipeline_assembly_id: uuid.UUID,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    try:
        record = metadata_store.get_pipeline_assembly(pipeline_assembly_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Pipeline assembly not found")
    return schemas.PipelineAssembly.from_record(record)


@router.get("/pipeline-assemblies", response_model=list[schemas.PipelineAssembly])
def list_pipeline_assemblies(
    pipeline_name: Optional[str] = None,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    records = metadata_store.list_pipeline_assemblies(pipeline_name=pipeline_name)
    return [schemas.PipelineAssembly.from_record(r) for r in records]


# --- Pipeline runs ----------------------------------------------------------


@router.post("/pipeline-runs", status_code=status.HTTP_204_NO_CONTENT)
def start_pipeline_run(
    body: schemas.StartPipelineRunRequest,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    metadata_store.start_pipeline_run(
        body.pipeline_name, body.pipeline_run_id, body.pipeline_assembly_id
    )


@router.patch("/pipeline-runs/{pipeline_run_id}", status_code=status.HTTP_204_NO_CONTENT)
def finish_pipeline_run(
    pipeline_run_id: uuid.UUID,
    body: schemas.FinishPipelineRunRequest,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    metadata_store.finish_pipeline_run(pipeline_run_id, RunStatus(body.status))


@router.get("/pipeline-runs/{pipeline_run_id}", response_model=schemas.PipelineRun)
def get_pipeline_run(
    pipeline_run_id: uuid.UUID,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    try:
        record = metadata_store.get_pipeline_run(pipeline_run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return schemas.PipelineRun.from_record(record)


@router.get("/pipeline-runs", response_model=list[schemas.PipelineRun])
def list_pipeline_runs(
    pipeline_name: Optional[str] = None,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    records = metadata_store.list_pipeline_runs(pipeline_name=pipeline_name)
    return [schemas.PipelineRun.from_record(r) for r in records]


# --- Task runs ---------------------------------------------------------


@router.post(
    "/pipeline-runs/{pipeline_run_id}/task-runs/{task_name}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def start_task_run(
    pipeline_run_id: uuid.UUID,
    task_name: str,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    metadata_store.start_task_run(pipeline_run_id, task_name)


@router.patch(
    "/pipeline-runs/{pipeline_run_id}/task-runs/{task_name}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def finish_task_run(
    pipeline_run_id: uuid.UUID,
    task_name: str,
    body: schemas.FinishTaskRunRequest,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    metadata_store.finish_task_run(
        pipeline_run_id, task_name, RunStatus(body.status), logs=body.logs
    )


@router.get(
    "/pipeline-runs/{pipeline_run_id}/task-runs/{task_name}",
    response_model=schemas.TaskRun,
)
def get_task_run(
    pipeline_run_id: uuid.UUID,
    task_name: str,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    try:
        record = metadata_store.get_task_run(pipeline_run_id, task_name)
    except KeyError:
        raise HTTPException(status_code=404, detail="Task run not found")
    return schemas.TaskRun.from_record(record)


@router.get(
    "/pipeline-runs/{pipeline_run_id}/task-runs", response_model=list[schemas.TaskRun]
)
def list_task_runs(
    pipeline_run_id: uuid.UUID,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    records = metadata_store.list_task_runs(pipeline_run_id)
    return [schemas.TaskRun.from_record(r) for r in records]


# --- Task outputs -------------------------------------------------------


@router.post(
    "/pipeline-runs/{pipeline_run_id}/task-runs/{task_name}/outputs/{output_name}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def record_task_output(
    pipeline_run_id: uuid.UUID,
    task_name: str,
    output_name: str,
    body: schemas.RecordTaskOutputRequest,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    metadata_store.record_task_output(
        pipeline_run_id, task_name, output_name, body.artifact_key
    )


@router.get(
    "/pipeline-runs/{pipeline_run_id}/task-runs/{task_name}/outputs",
    response_model=list[schemas.TaskOutput],
)
def list_task_outputs(
    pipeline_run_id: uuid.UUID,
    task_name: str,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    records = metadata_store.list_task_outputs(pipeline_run_id, task_name)
    return [schemas.TaskOutput.from_record(r) for r in records]


# --- Pipeline registrations ----------------------------------------------


@router.post("/pipeline-registrations", response_model=schemas.IdResponse)
def register_pipeline(
    body: schemas.RegisterPipelineRequest,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    registration_id = metadata_store.register_pipeline(
        body.pipeline_name,
        body.backend,
        body.dag_structure,
        body.pipeline_inputs,
        backend_metadata=body.backend_metadata,
    )
    return schemas.IdResponse(id=registration_id)


@router.post(
    "/pipeline-registrations/{pipeline_registration_id}/deregister",
    status_code=status.HTTP_204_NO_CONTENT,
)
def deregister_pipeline(
    pipeline_registration_id: uuid.UUID,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    metadata_store.deregister_pipeline(pipeline_registration_id)


@router.get(
    "/pipeline-registrations/{pipeline_registration_id}",
    response_model=schemas.PipelineRegistration,
)
def get_pipeline_registration(
    pipeline_registration_id: uuid.UUID,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    try:
        record = metadata_store.get_pipeline_registration(pipeline_registration_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Pipeline registration not found")
    return schemas.PipelineRegistration.from_record(record)


@router.get(
    "/pipeline-registrations", response_model=list[schemas.PipelineRegistration]
)
def list_pipeline_registrations(
    pipeline_name: Optional[str] = None,
    active_only: bool = False,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    records = metadata_store.list_pipeline_registrations(
        pipeline_name=pipeline_name, active_only=active_only
    )
    return [schemas.PipelineRegistration.from_record(r) for r in records]


# --- Triggers ------------------------------------------------------------


@router.post(
    "/pipeline-registrations/{pipeline_registration_id}/triggers",
    response_model=schemas.IdResponse,
)
def register_trigger(
    pipeline_registration_id: uuid.UUID,
    body: schemas.RegisterTriggerRequest,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    trigger_id = metadata_store.register_trigger(
        pipeline_registration_id, body.trigger_type, body.trigger_config
    )
    return schemas.IdResponse(id=trigger_id)


@router.post(
    "/triggers/{trigger_id}/deregister", status_code=status.HTTP_204_NO_CONTENT
)
def deregister_trigger(
    trigger_id: uuid.UUID,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    metadata_store.deregister_trigger(trigger_id)


@router.get(
    "/pipeline-registrations/{pipeline_registration_id}/triggers",
    response_model=list[schemas.Trigger],
)
def list_triggers(
    pipeline_registration_id: uuid.UUID,
    active_only: bool = False,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    records = metadata_store.list_triggers(
        pipeline_registration_id, active_only=active_only
    )
    return [schemas.Trigger.from_record(r) for r in records]
