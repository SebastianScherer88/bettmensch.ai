"""Integration tests for `RemoteMetadataStore` against a real instance of
the metadata service (`docker/metadata-service/`), run in-process in a
background thread (not a separate container/subprocess) - backed by the
real test PostgreSQL server (the `postgres_dsn` fixture from
tests/conftest.py).

Note on "in-process": `RemoteMetadataStore` opens a fresh sync
`httpx.Client` per call (see its own docstring), and httpx's ASGI-direct
transport (`httpx.ASGITransport`) only implements the *async* request
path - it can't back a sync `httpx.Client` the way it could an
`httpx.AsyncClient`. A real (but background-thread, same-process) uvicorn
server sidesteps that mismatch entirely while still needing no separate
container.

This proves the service correctly proxies `RemoteMetadataStore`'s calls
through to Postgres - the second of the two client/store touch points (the
other being `test_artifact_client_integration.py`). Exercises a
representative subset of the contract already covered against
`PostgresMetadataStore` directly in
tests/integration/pipelines/test_postgres_metadata_store_integration.py -
same "smaller independent subset, not a shared base class" convention
that file already uses relative to tests/unit/pipelines/test_metadata_store.py.
"""

import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
import uvicorn
from bettmensch_ai.pipelines.metadata_store import (
    BaseMetadataStore,
    PostgresMetadataStore,
    PostgresMetadataStoreConfig,
    RemoteMetadataStore,
    RemoteMetadataStoreConfig,
    RunStatus,
)

pytestmark = pytest.mark.integration

_METADATA_SERVICE_DIR = Path(__file__).resolve().parents[3] / "docker" / "metadata-service"


@pytest.fixture
def remote_metadata_store(postgres_dsn):
    """A `RemoteMetadataStore` talking, over a real (loopback) HTTP
    connection, to a real instance of the metadata service's FastAPI app
    running in a background thread of this same test process - itself
    backed by the real test Postgres server.

    Imports `docker/metadata-service/backend` fresh for each test and tears
    it back out of `sys.modules`/`sys.path` afterwards: nothing else in
    this test suite imports a top-level `backend` package, but this avoids
    permanently claiming that name for the rest of the session regardless.
    """

    sys.path.insert(0, str(_METADATA_SERVICE_DIR))
    for name in list(sys.modules):
        if name == "backend" or name.startswith("backend."):
            del sys.modules[name]

    try:
        from backend.main import app
        from backend.stores import get_metadata_store

        app.dependency_overrides[get_metadata_store] = lambda: PostgresMetadataStore(
            PostgresMetadataStoreConfig(dsn=postgres_dsn)
        )

        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("Metadata service did not start in time.")

        port = server.servers[0].sockets[0].getsockname()[1]

        try:
            yield RemoteMetadataStore(
                RemoteMetadataStoreConfig(base_url=f"http://127.0.0.1:{port}/api")
            )
        finally:
            server.should_exit = True
            thread.join(timeout=5)
    finally:
        sys.path.remove(str(_METADATA_SERVICE_DIR))
        for name in list(sys.modules):
            if name == "backend" or name.startswith("backend."):
                del sys.modules[name]


def test_remote_metadata_store_is_a_base_metadata_store(remote_metadata_store):
    assert isinstance(remote_metadata_store, BaseMetadataStore)


def test_pipeline_run_lifecycle_is_recorded(remote_metadata_store):
    pipeline_run_id = uuid.uuid4()

    remote_metadata_store.start_pipeline_run("my-pipeline", pipeline_run_id)
    running = remote_metadata_store.get_pipeline_run(pipeline_run_id)
    assert running.status == RunStatus.RUNNING
    assert running.ended_at is None

    remote_metadata_store.finish_pipeline_run(pipeline_run_id, RunStatus.SUCCEEDED)
    finished = remote_metadata_store.get_pipeline_run(pipeline_run_id)
    assert finished.status == RunStatus.SUCCEEDED
    assert finished.ended_at is not None


def test_get_pipeline_run_raises_for_unknown_id(remote_metadata_store):
    with pytest.raises(KeyError):
        remote_metadata_store.get_pipeline_run(uuid.uuid4())


def test_task_run_and_output_bookkeeping(remote_metadata_store):
    pipeline_run_id = uuid.uuid4()
    remote_metadata_store.start_pipeline_run("my-pipeline", pipeline_run_id)
    remote_metadata_store.start_task_run(pipeline_run_id, "divmod-task")

    remote_metadata_store.record_task_output(
        pipeline_run_id, "divmod-task", "quotient", "k1"
    )
    remote_metadata_store.record_task_output(
        pipeline_run_id, "divmod-task", "remainder", "k2"
    )
    remote_metadata_store.finish_task_run(
        pipeline_run_id, "divmod-task", RunStatus.SUCCEEDED, logs="6\n"
    )
    task_run = remote_metadata_store.get_task_run(pipeline_run_id, "divmod-task")
    assert task_run.status == RunStatus.SUCCEEDED
    assert task_run.logs == "6\n"

    outputs = {
        o.output_name: o.artifact_key
        for o in remote_metadata_store.list_task_outputs(pipeline_run_id, "divmod-task")
    }
    assert outputs == {"quotient": "k1", "remainder": "k2"}


def test_pipeline_assembly_lifecycle_is_recorded(remote_metadata_store):
    dag_structure = {"tasks": [{"name": "add", "rank": 0}], "edges": []}
    pipeline_inputs = {"a": {"required": True, "default": None}}

    assembly_id = remote_metadata_store.record_pipeline_assembly(
        "my-pipeline", dag_structure, pipeline_inputs
    )
    record = remote_metadata_store.get_pipeline_assembly(assembly_id)
    assert record.dag_structure == dag_structure
    assert record.pipeline_inputs == pipeline_inputs

    assemblies = remote_metadata_store.list_pipeline_assemblies("my-pipeline")
    assert assembly_id in {a.pipeline_assembly_id for a in assemblies}


def test_pipeline_registration_and_trigger_lifecycle(remote_metadata_store):
    dag_structure = {"tasks": ["add"], "edges": []}
    pipeline_inputs = {"a": {"type": "int"}}
    backend_metadata = {"state_machine_arn": "arn:aws:states:::my-pipeline"}

    registration_id = remote_metadata_store.register_pipeline(
        "my-pipeline", "aws_stepfunctions", dag_structure, pipeline_inputs, backend_metadata
    )
    record = remote_metadata_store.get_pipeline_registration(registration_id)
    assert record.dag_structure == dag_structure
    assert record.backend_metadata == backend_metadata
    assert record.is_active is True

    trigger_id = remote_metadata_store.register_trigger(
        registration_id, "cron", {"schedule": "0 0 * * *"}
    )
    triggers = remote_metadata_store.list_triggers(registration_id)
    assert len(triggers) == 1
    assert triggers[0].trigger_id == trigger_id

    remote_metadata_store.deregister_trigger(trigger_id)
    assert (
        remote_metadata_store.list_triggers(registration_id, active_only=True) == []
    )

    remote_metadata_store.deregister_pipeline(registration_id)
    assert (
        remote_metadata_store.get_pipeline_registration(registration_id).is_active
        is False
    )
