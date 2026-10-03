"""Integration tests for `PostgresMetadataStore` against a real PostgreSQL
server.

Uses the `postgres_metadata_store` fixture from tests/conftest.py, which
skips these tests unless both `psycopg` is installed and a reachable test
server is available (see that fixture, and
docker-compose/pipelines.docker-compose.yaml).

These exercise the same contract already covered against `LocalMetadataStore`
in tests/unit/pipelines/test_metadata_store.py - both implement
`BaseMetadataStore`, so callers should be able to treat them
interchangeably; this file exists to prove that's actually true against a
real database, not just SQLite.
"""

import uuid

import pytest
from bettmensch_ai.pipelines.metadata_store import BaseMetadataStore, RunStatus

pytestmark = pytest.mark.integration


def test_postgres_metadata_store_is_a_base_metadata_store(postgres_metadata_store):
    assert isinstance(postgres_metadata_store, BaseMetadataStore)


def test_pipeline_run_lifecycle_is_recorded(postgres_metadata_store):
    pipeline_run_id = uuid.uuid4()

    postgres_metadata_store.start_pipeline_run("my-pipeline", pipeline_run_id)
    running = postgres_metadata_store.get_pipeline_run(pipeline_run_id)
    assert running.status == RunStatus.RUNNING
    assert running.ended_at is None

    postgres_metadata_store.finish_pipeline_run(pipeline_run_id, RunStatus.SUCCEEDED)
    finished = postgres_metadata_store.get_pipeline_run(pipeline_run_id)
    assert finished.status == RunStatus.SUCCEEDED
    assert finished.ended_at is not None


def test_get_pipeline_run_raises_for_unknown_id(postgres_metadata_store):
    with pytest.raises(KeyError):
        postgres_metadata_store.get_pipeline_run(uuid.uuid4())


def test_task_run_and_output_bookkeeping(postgres_metadata_store):
    pipeline_run_id = uuid.uuid4()
    postgres_metadata_store.start_pipeline_run("my-pipeline", pipeline_run_id)
    postgres_metadata_store.start_task_run(pipeline_run_id, "divmod-task")

    postgres_metadata_store.record_task_output(
        pipeline_run_id, "divmod-task", "quotient", "k1"
    )
    postgres_metadata_store.record_task_output(
        pipeline_run_id, "divmod-task", "remainder", "k2"
    )
    postgres_metadata_store.finish_task_run(
        pipeline_run_id, "divmod-task", RunStatus.SUCCEEDED, logs="6\n"
    )
    task_run = postgres_metadata_store.get_task_run(pipeline_run_id, "divmod-task")
    assert task_run.status == RunStatus.SUCCEEDED
    assert task_run.logs == "6\n"

    outputs = {
        o.output_name: o.artifact_key
        for o in postgres_metadata_store.list_task_outputs(
            pipeline_run_id, "divmod-task"
        )
    }
    assert outputs == {"quotient": "k1", "remainder": "k2"}


def test_pipeline_assembly_lifecycle_is_recorded(postgres_metadata_store):
    dag_structure = {"tasks": [{"name": "add", "rank": 0}], "edges": []}
    pipeline_inputs = {"a": {"required": True, "default": None}}

    assembly_id = postgres_metadata_store.record_pipeline_assembly(
        "my-pipeline", dag_structure, pipeline_inputs
    )
    record = postgres_metadata_store.get_pipeline_assembly(assembly_id)
    assert record.dag_structure == dag_structure
    assert record.pipeline_inputs == pipeline_inputs

    assemblies = postgres_metadata_store.list_pipeline_assemblies("my-pipeline")
    assert assembly_id in {a.pipeline_assembly_id for a in assemblies}


def test_pipeline_run_can_reference_its_pipeline_assembly(postgres_metadata_store):
    assembly_id = postgres_metadata_store.record_pipeline_assembly(
        "my-pipeline", {}, {}
    )
    pipeline_run_id = uuid.uuid4()

    postgres_metadata_store.start_pipeline_run(
        "my-pipeline", pipeline_run_id, assembly_id
    )
    record = postgres_metadata_store.get_pipeline_run(pipeline_run_id)

    assert record.pipeline_assembly_id == assembly_id


def test_pipeline_registration_and_trigger_lifecycle(postgres_metadata_store):
    dag_structure = {"tasks": ["add"], "edges": []}
    pipeline_inputs = {"a": {"type": "int"}}
    backend_metadata = {"state_machine_arn": "arn:aws:states:::my-pipeline"}

    registration_id = postgres_metadata_store.register_pipeline(
        "my-pipeline",
        "aws_stepfunctions",
        dag_structure,
        pipeline_inputs,
        backend_metadata,
    )
    record = postgres_metadata_store.get_pipeline_registration(registration_id)
    assert record.dag_structure == dag_structure
    assert record.pipeline_inputs == pipeline_inputs
    assert record.backend_metadata == backend_metadata
    assert record.is_active is True

    trigger_id = postgres_metadata_store.register_trigger(
        registration_id, "cron", {"schedule": "0 0 * * *"}
    )
    triggers = postgres_metadata_store.list_triggers(registration_id)
    assert len(triggers) == 1
    assert triggers[0].trigger_id == trigger_id

    postgres_metadata_store.deregister_trigger(trigger_id)
    assert (
        postgres_metadata_store.list_triggers(registration_id, active_only=True) == []
    )

    postgres_metadata_store.deregister_pipeline(registration_id)
    assert (
        postgres_metadata_store.get_pipeline_registration(registration_id).is_active
        is False
    )
