"""Unit tests for `S3ArtifactStore`: pure wiring/logic, no real S3/MinIO -
see sdk/test/integration/pipelines/test_s3_artifact_store_integration.py
for tests against a real (MinIO-backed) object store.
"""

import uuid
from unittest.mock import MagicMock, patch

from bettmensch_ai.pipelines.artifact_store import (
    BaseArtifactStore,
    S3ArtifactStore,
    S3ArtifactStoreConfig,
)


def make_store(mock_boto3_client):
    mock_boto3_client.return_value = MagicMock()
    return S3ArtifactStore(S3ArtifactStoreConfig(bucket="my-bucket"))


@patch("bettmensch_ai.pipelines.artifact_store.s3_artifact_store.boto3.client")
def test_s3_artifact_store_is_a_base_artifact_store(mock_client):
    store = make_store(mock_client)

    assert isinstance(store, BaseArtifactStore)


@patch("bettmensch_ai.pipelines.artifact_store.s3_artifact_store.boto3.client")
def test_init_builds_a_boto3_client_from_config(mock_client):
    mock_client.return_value = MagicMock()

    S3ArtifactStore(
        S3ArtifactStoreConfig(
            bucket="my-bucket",
            endpoint_url="http://localhost:9000",
            region_name="eu-west-1",
            aws_access_key_id="key",
            aws_secret_access_key="secret",
        )
    )

    mock_client.assert_called_once_with(
        "s3",
        endpoint_url="http://localhost:9000",
        region_name="eu-west-1",
        aws_access_key_id="key",
        aws_secret_access_key="secret",
    )


@patch("bettmensch_ai.pipelines.artifact_store.s3_artifact_store.boto3.client")
def test_key_joins_segments_with_forward_slashes(mock_client):
    store = make_store(mock_client)
    pipeline_run_id = uuid.uuid4()

    key = store.key("my-pipeline", pipeline_run_id, "my-task", "my-artifact")

    assert key == f"my-pipeline/{pipeline_run_id}/my-task/my-artifact"


@patch("bettmensch_ai.pipelines.artifact_store.s3_artifact_store.boto3.client")
def test_uri_is_an_s3_uri(mock_client):
    store = make_store(mock_client)
    pipeline_run_id = uuid.uuid4()

    uri = store.uri("my-pipeline", pipeline_run_id, "my-task", "my-artifact")

    assert uri == f"s3://my-bucket/my-pipeline/{pipeline_run_id}/my-task/my-artifact"


@patch("bettmensch_ai.pipelines.artifact_store.s3_artifact_store.boto3.client")
def test_upload_calls_boto3_upload_file(mock_client):
    store = make_store(mock_client)

    store.upload("/local/path", "some/key")

    store.client.upload_file.assert_called_once_with(
        "/local/path", "my-bucket", "some/key"
    )


@patch("bettmensch_ai.pipelines.artifact_store.s3_artifact_store.boto3.client")
def test_download_creates_parent_dir_and_calls_boto3_download_file(
    mock_client, tmp_path
):
    store = make_store(mock_client)
    destination = tmp_path / "nested" / "dir" / "value"

    store.download("some/key", str(destination))

    assert destination.parent.exists()
    store.client.download_file.assert_called_once_with(
        "my-bucket", "some/key", str(destination)
    )
