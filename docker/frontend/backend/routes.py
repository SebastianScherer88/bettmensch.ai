"""Read-only REST API over a PostgresMetadataStore/S3ArtifactStore pair.

This purely reads what a `LocalRunner` (or a real backend orchestrator, for
the registration endpoints) has already written - it doesn't run, register,
or delete anything itself, matching the frontend's own read-only remit.
"""

import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException

from bettmensch_ai.pipelines.artifact_metadata import metadata_key
from bettmensch_ai.pipelines.artifact_store import BaseArtifactStore
from bettmensch_ai.pipelines.materializers import resolve_materializer_from_artifact
from bettmensch_ai.pipelines.metadata_store import BaseMetadataStore
from pydantic import BaseModel

from . import schemas
from .stores import get_artifact_store, get_metadata_store

router = APIRouter()

# Materializers whose stored format is already (or trivially) JSON - anything
# else (e.g. polars_parquet) is shown as metadata only, not auto-rendered.
JSON_PREVIEWABLE_MATERIALIZERS = {"json", "pydantic_json"}


def _read_artifact_metadata(artifact_store: BaseArtifactStore, key: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = str(Path(tmp_dir) / "metadata.json")
        artifact_store.download(metadata_key(key), local_path)
        with open(local_path) as f:
            return json.load(f)


def _to_jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/pipelines", response_model=list[schemas.PipelineSummary])
def list_pipelines(metadata_store: BaseMetadataStore = Depends(get_metadata_store)):
    """Aggregates assemblies/registrations/runs by pipeline name.

    There's no single store method for "list pipelines" - a pipeline isn't
    its own record, just a name shared across three independent kinds of
    bookkeeping - so this groups all three in memory. Each list call is
    already sorted most-recent-first by the store, so the first record seen
    per name while iterating is that pipeline's latest one.
    """

    assemblies = metadata_store.list_pipeline_assemblies()
    registrations = metadata_store.list_pipeline_registrations()
    runs = metadata_store.list_pipeline_runs()

    names = set()
    latest_assembly_by_name = {}
    for assembly in assemblies:
        names.add(assembly.pipeline_name)
        latest_assembly_by_name.setdefault(assembly.pipeline_name, assembly)

    latest_registration_by_name = {}
    active_registration_by_name = {}
    for registration in registrations:
        names.add(registration.pipeline_name)
        latest_registration_by_name.setdefault(registration.pipeline_name, registration)
        if registration.is_active:
            active_registration_by_name.setdefault(registration.pipeline_name, registration)

    run_count_by_name: Dict[str, int] = {}
    latest_run_by_name = {}
    for run in runs:
        names.add(run.pipeline_name)
        run_count_by_name[run.pipeline_name] = run_count_by_name.get(run.pipeline_name, 0) + 1
        latest_run_by_name.setdefault(run.pipeline_name, run)

    summaries = []
    for name in sorted(names):
        assembly = latest_assembly_by_name.get(name)
        registration = active_registration_by_name.get(name) or latest_registration_by_name.get(
            name
        )
        run = latest_run_by_name.get(name)
        summaries.append(
            schemas.PipelineSummary(
                pipeline_name=name,
                is_assembled=assembly is not None,
                last_assembled_at=assembly.assembled_at if assembly else None,
                is_registered=name in active_registration_by_name,
                backend=registration.backend if registration else None,
                last_registered_at=registration.registered_at if registration else None,
                run_count=run_count_by_name.get(name, 0),
                last_run_status=run.status.value if run else None,
                last_run_at=run.started_at if run else None,
            )
        )

    return summaries


@router.get("/pipelines/{pipeline_name}", response_model=schemas.PipelineDetail)
def get_pipeline(
    pipeline_name: str, metadata_store: BaseMetadataStore = Depends(get_metadata_store)
):
    assemblies = metadata_store.list_pipeline_assemblies(pipeline_name)
    registrations = metadata_store.list_pipeline_registrations(pipeline_name)
    runs = metadata_store.list_pipeline_runs(pipeline_name)

    if not assemblies and not registrations and not runs:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return schemas.PipelineDetail(
        pipeline_name=pipeline_name,
        latest_assembly=schemas.PipelineAssembly.from_record(assemblies[0])
        if assemblies
        else None,
        registrations=[schemas.PipelineRegistration.from_record(r) for r in registrations],
        run_count=len(runs),
        last_run=schemas.PipelineRun.from_record(runs[0]) if runs else None,
    )


@router.get(
    "/pipeline-assemblies", response_model=list[schemas.PipelineAssembly]
)
def list_pipeline_assemblies(
    pipeline_name: Optional[str] = None,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    assemblies = metadata_store.list_pipeline_assemblies(pipeline_name=pipeline_name)
    return [schemas.PipelineAssembly.from_record(a) for a in assemblies]


@router.get(
    "/pipeline-assemblies/{pipeline_assembly_id}",
    response_model=schemas.PipelineAssembly,
)
def get_pipeline_assembly(
    pipeline_assembly_id: uuid.UUID,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    try:
        assembly = metadata_store.get_pipeline_assembly(pipeline_assembly_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Pipeline assembly not found")
    return schemas.PipelineAssembly.from_record(assembly)


@router.get("/pipeline-runs", response_model=list[schemas.PipelineRun])
def list_pipeline_runs(
    pipeline_name: Optional[str] = None,
    status: Optional[str] = None,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    runs = metadata_store.list_pipeline_runs(pipeline_name=pipeline_name)
    if status:
        runs = [r for r in runs if r.status.value == status]
    return [schemas.PipelineRun.from_record(r) for r in runs]


@router.get("/pipeline-runs/{pipeline_run_id}", response_model=schemas.PipelineRun)
def get_pipeline_run(
    pipeline_run_id: uuid.UUID,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    try:
        run = metadata_store.get_pipeline_run(pipeline_run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return schemas.PipelineRun.from_record(run)


@router.get(
    "/pipeline-runs/{pipeline_run_id}/task-runs",
    response_model=list[schemas.TaskRun],
)
def list_task_runs(
    pipeline_run_id: uuid.UUID,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    task_runs = metadata_store.list_task_runs(pipeline_run_id)
    return [schemas.TaskRun.from_record(t) for t in task_runs]


@router.get(
    "/pipeline-runs/{pipeline_run_id}/task-runs/{task_name}/outputs",
    response_model=list[schemas.TaskOutput],
)
def list_task_outputs(
    pipeline_run_id: uuid.UUID,
    task_name: str,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    outputs = metadata_store.list_task_outputs(pipeline_run_id, task_name)
    return [schemas.TaskOutput.from_record(o) for o in outputs]


@router.get("/artifacts", response_model=list[schemas.ArtifactSummary])
def list_artifacts(
    pipeline_name: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    include_metadata: bool = True,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
    artifact_store: BaseArtifactStore = Depends(get_artifact_store),
):
    """Searches artifacts by pipeline and/or run date range.

    There's no flat "list all outputs" query on `BaseMetadataStore` (by
    design - see its docstring), so this fans out: every matching run, every
    task run within it, every output of that task run. Fine for a local dev
    tool's scale; a store method doing this as one query server-side would
    be the move if this ever needs to handle many thousands of runs.

    `include_metadata=True` (the default) additionally reads each artifact's
    metadata sidecar to show its materializer/type in the results table -
    one extra store round-trip per artifact, skippable via
    `include_metadata=False` for a faster, metadata-less listing.
    """

    runs = metadata_store.list_pipeline_runs(pipeline_name=pipeline_name)

    results = []
    for run in runs:
        if since is not None and run.started_at < since:
            continue
        if until is not None and run.started_at > until:
            continue

        for task_run in metadata_store.list_task_runs(run.pipeline_run_id):
            for output in metadata_store.list_task_outputs(
                run.pipeline_run_id, task_run.task_name
            ):
                materializer = None
                value_type = None
                if include_metadata:
                    try:
                        metadata = _read_artifact_metadata(
                            artifact_store, output.artifact_key
                        )
                        materializer = metadata.get("materializer")
                        value_type = metadata.get("value_type")
                    except Exception:
                        pass

                results.append(
                    schemas.ArtifactSummary(
                        pipeline_name=run.pipeline_name,
                        pipeline_run_id=run.pipeline_run_id,
                        run_started_at=run.started_at,
                        run_status=run.status.value,
                        task_name=task_run.task_name,
                        output_name=output.output_name,
                        artifact_key=output.artifact_key,
                        materializer=materializer,
                        value_type=value_type,
                    )
                )

    results.sort(key=lambda a: a.run_started_at, reverse=True)
    return results


@router.get("/artifacts/preview", response_model=schemas.ArtifactPreview)
def preview_artifact(
    key: str,
    artifact_store: BaseArtifactStore = Depends(get_artifact_store),
):
    try:
        metadata = _read_artifact_metadata(artifact_store, key)
    except Exception as exc:
        raise HTTPException(
            status_code=404, detail=f"No artifact metadata found for key {key!r}: {exc}"
        )

    materializer_name = metadata.get("materializer")
    value_type = metadata.get("value_type")
    previewable = materializer_name in JSON_PREVIEWABLE_MATERIALIZERS
    value = None
    error = None

    if previewable:
        try:
            materializer = resolve_materializer_from_artifact(artifact_store, key)
            value = _to_jsonable(artifact_store.load(materializer, key))
        except Exception as exc:
            previewable = False
            error = str(exc)

    return schemas.ArtifactPreview(
        artifact_key=key,
        materializer=materializer_name,
        value_type=value_type,
        previewable=previewable,
        value=value,
        error=error,
    )


@router.get(
    "/pipeline-registrations", response_model=list[schemas.PipelineRegistration]
)
def list_pipeline_registrations(
    pipeline_name: Optional[str] = None,
    active_only: bool = False,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    registrations = metadata_store.list_pipeline_registrations(
        pipeline_name=pipeline_name, active_only=active_only
    )
    return [schemas.PipelineRegistration.from_record(r) for r in registrations]


@router.get(
    "/pipeline-registrations/{pipeline_registration_id}",
    response_model=schemas.PipelineRegistration,
)
def get_pipeline_registration(
    pipeline_registration_id: uuid.UUID,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    try:
        registration = metadata_store.get_pipeline_registration(pipeline_registration_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Pipeline registration not found")
    return schemas.PipelineRegistration.from_record(registration)


@router.get(
    "/pipeline-registrations/{pipeline_registration_id}/triggers",
    response_model=list[schemas.Trigger],
)
def list_triggers(
    pipeline_registration_id: uuid.UUID,
    active_only: bool = False,
    metadata_store: BaseMetadataStore = Depends(get_metadata_store),
):
    triggers = metadata_store.list_triggers(
        pipeline_registration_id, active_only=active_only
    )
    return [schemas.Trigger.from_record(t) for t in triggers]
