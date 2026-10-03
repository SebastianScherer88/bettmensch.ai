"""Unit tests for `RemoteMetadataStore`: pure request/response wiring,
mocked `httpx` - no real metadata service/Postgres. See
tests/integration/pipelines/test_remote_metadata_store_integration.py for
tests against the real service (in-process) backed by a real Postgres.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from bettmensch_ai.pipelines.metadata_store import (
    BaseMetadataStore,
    RemoteMetadataStore,
    RemoteMetadataStoreConfig,
    RunStatus,
)


def make_mock_client():
    """A `MagicMock` standing in for an `httpx.Client`, usable as the
    context manager `RemoteMetadataStore` opens one as (`with self._client()
    as client:`).
    """

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    return mock_client


def make_response(status_code=200, json_data=None, text=""):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    response.text = text
    if status_code >= 400:
        import httpx

        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status.return_value = None
    return response


@patch("bettmensch_ai.pipelines.metadata_store.remote_metadata_store.httpx.Client")
def test_remote_metadata_store_is_a_base_metadata_store(mock_client_cls):
    store = RemoteMetadataStore(
        RemoteMetadataStoreConfig(base_url="http://localhost:8081/api")
    )

    assert isinstance(store, BaseMetadataStore)


@patch("bettmensch_ai.pipelines.metadata_store.remote_metadata_store.httpx.Client")
def test_record_pipeline_assembly_posts_and_parses_id(mock_client_cls):
    mock_client = make_mock_client()
    mock_client_cls.return_value = mock_client
    assembly_id = uuid.uuid4()
    mock_client.post.return_value = make_response(json_data={"id": str(assembly_id)})

    store = RemoteMetadataStore(
        RemoteMetadataStoreConfig(base_url="http://localhost:8081/api")
    )
    result = store.record_pipeline_assembly("my-pipeline", {"tasks": []}, {})

    mock_client.post.assert_called_once_with(
        "/pipeline-assemblies",
        json={
            "pipeline_name": "my-pipeline",
            "dag_structure": {"tasks": []},
            "pipeline_inputs": {},
        },
    )
    assert result == assembly_id


@patch("bettmensch_ai.pipelines.metadata_store.remote_metadata_store.httpx.Client")
def test_get_pipeline_run_raises_keyerror_on_404(mock_client_cls):
    mock_client = make_mock_client()
    mock_client_cls.return_value = mock_client
    mock_client.get.return_value = make_response(
        status_code=404, text='{"detail": "Pipeline run not found"}'
    )

    store = RemoteMetadataStore(
        RemoteMetadataStoreConfig(base_url="http://localhost:8081/api")
    )

    with pytest.raises(KeyError):
        store.get_pipeline_run(uuid.uuid4())


@patch("bettmensch_ai.pipelines.metadata_store.remote_metadata_store.httpx.Client")
def test_list_pipeline_runs_builds_records_from_response_body(mock_client_cls):
    mock_client = make_mock_client()
    mock_client_cls.return_value = mock_client
    pipeline_run_id = uuid.uuid4()
    mock_client.get.return_value = make_response(
        json_data=[
            {
                "pipeline_run_id": str(pipeline_run_id),
                "pipeline_name": "my-pipeline",
                "status": "succeeded",
                "started_at": "2026-01-01T00:00:00+00:00",
                "ended_at": "2026-01-01T00:00:05+00:00",
                "pipeline_assembly_id": None,
            }
        ]
    )

    store = RemoteMetadataStore(
        RemoteMetadataStoreConfig(base_url="http://localhost:8081/api")
    )
    runs = store.list_pipeline_runs(pipeline_name="my-pipeline")

    mock_client.get.assert_called_once_with(
        "/pipeline-runs", params={"pipeline_name": "my-pipeline"}
    )
    assert len(runs) == 1
    assert runs[0].pipeline_run_id == pipeline_run_id
    assert runs[0].status == RunStatus.SUCCEEDED
    assert runs[0].pipeline_assembly_id is None


@patch("bettmensch_ai.pipelines.metadata_store.remote_metadata_store.httpx.Client")
def test_start_task_run_posts_to_the_expected_url(mock_client_cls):
    mock_client = make_mock_client()
    mock_client_cls.return_value = mock_client
    mock_client.post.return_value = make_response()
    pipeline_run_id = uuid.uuid4()

    store = RemoteMetadataStore(
        RemoteMetadataStoreConfig(base_url="http://localhost:8081/api")
    )
    store.start_task_run(pipeline_run_id, "add")

    mock_client.post.assert_called_once_with(
        f"/pipeline-runs/{pipeline_run_id}/task-runs/add"
    )


@patch("bettmensch_ai.pipelines.metadata_store.remote_metadata_store.httpx.Client")
def test_finish_task_run_patches_status_and_logs(mock_client_cls):
    mock_client = make_mock_client()
    mock_client_cls.return_value = mock_client
    mock_client.patch.return_value = make_response()
    pipeline_run_id = uuid.uuid4()

    store = RemoteMetadataStore(
        RemoteMetadataStoreConfig(base_url="http://localhost:8081/api")
    )
    store.finish_task_run(pipeline_run_id, "add", RunStatus.FAILED, logs="boom")

    mock_client.patch.assert_called_once_with(
        f"/pipeline-runs/{pipeline_run_id}/task-runs/add",
        json={"status": "failed", "logs": "boom"},
    )
