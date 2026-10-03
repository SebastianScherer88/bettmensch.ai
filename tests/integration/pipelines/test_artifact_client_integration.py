"""Integration tests for `ArtifactClient` configured for the `s3` backend,
against a real S3-compatible object store (MinIO) - proving the client
correctly forwards to the real store it builds, not just to a mock. The
artifact-side counterpart to
tests/integration/pipelines/test_remote_metadata_store_integration.py.

Uses the `s3_config`/`unique_key_prefix` fixtures from tests/conftest.py,
which skip these tests unless a reachable test MinIO is available (see
that fixture, and docker-compose/pipelines.docker-compose.yaml).
"""

import uuid

import pytest
from bettmensch_ai.pipelines.client import ArtifactClient
from bettmensch_ai.pipelines.materializers import (
    JsonMaterializer,
    resolve_materializer_from_artifact,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def artifact_client(s3_config, monkeypatch):
    """An `ArtifactClient` built from `ArtifactClientConfig(backend="s3")`,
    pointed at the real test MinIO via `s3_config`'s own
    `S3ArtifactStoreConfig` values (set as env vars, since that's how
    `ArtifactClient`'s default construction reads a backend's own config).
    """

    monkeypatch.setenv("BETTMENSCH_AI_ARTIFACT_CLIENT_BACKEND", "s3")
    monkeypatch.setenv("BETTMENSCH_AI_S3_ARTIFACT_STORE_BUCKET", s3_config.bucket)
    monkeypatch.setenv(
        "BETTMENSCH_AI_S3_ARTIFACT_STORE_ENDPOINT_URL", s3_config.endpoint_url
    )
    monkeypatch.setenv(
        "BETTMENSCH_AI_S3_ARTIFACT_STORE_AWS_ACCESS_KEY_ID", s3_config.aws_access_key_id
    )
    monkeypatch.setenv(
        "BETTMENSCH_AI_S3_ARTIFACT_STORE_AWS_SECRET_ACCESS_KEY",
        s3_config.aws_secret_access_key,
    )

    return ArtifactClient()


def test_artifact_client_is_backed_by_a_real_s3_artifact_store(artifact_client):
    from bettmensch_ai.pipelines.artifact_store import S3ArtifactStore

    assert isinstance(artifact_client.store, S3ArtifactStore)


def test_upload_then_download_roundtrips_file_contents(
    artifact_client, unique_key_prefix, tmp_path
):
    source = tmp_path / "source.txt"
    source.write_text("hello from a real object store, via ArtifactClient")
    key = f"{unique_key_prefix}/some/nested/key"

    artifact_client.upload(str(source), key)

    destination = tmp_path / "downloaded.txt"
    artifact_client.download(key, str(destination))

    assert destination.read_text() == "hello from a real object store, via ArtifactClient"


def test_save_and_load_round_trip_a_value_via_materializer(
    artifact_client, unique_key_prefix
):
    key = artifact_client.key(
        "my-pipeline", uuid.uuid4(), "add", f"{unique_key_prefix}-result"
    )

    artifact_client.save(JsonMaterializer(), {"a": 1, "b": [1, 2, 3]}, key)
    loaded = artifact_client.load(JsonMaterializer(), key)

    assert loaded == {"a": 1, "b": [1, 2, 3]}


def test_resolve_materializer_from_artifact_accepts_an_artifact_client(
    artifact_client, unique_key_prefix
):
    key = artifact_client.key(
        "my-pipeline", uuid.uuid4(), "add", f"{unique_key_prefix}-result"
    )
    artifact_client.save(JsonMaterializer(), 42, key)

    resolved = resolve_materializer_from_artifact(artifact_client, key)

    assert type(resolved).__name__ == "JsonMaterializer"
    assert artifact_client.load(resolved, key) == 42


def test_uri_is_an_s3_uri_pointing_at_the_test_bucket(artifact_client, s3_config):
    uri = artifact_client.uri("my-pipeline", uuid.uuid4(), "add", "result")

    assert uri.startswith(f"s3://{s3_config.bucket}/")
