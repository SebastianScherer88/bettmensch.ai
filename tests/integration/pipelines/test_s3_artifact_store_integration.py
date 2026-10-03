"""Integration tests for `S3ArtifactStore` against a real S3-compatible
object store (MinIO).

Uses the `s3_artifact_store`/`unique_key_prefix` fixtures from
tests/conftest.py, which skip these tests unless a reachable test MinIO
is available (see that fixture, and
docker-compose/pipelines.docker-compose.yaml).
"""

import uuid

import pytest
from bettmensch_ai.pipelines.artifact_store import BaseArtifactStore
from bettmensch_ai.pipelines.materializers import JsonMaterializer

pytestmark = pytest.mark.integration


def test_s3_artifact_store_is_a_base_artifact_store(s3_artifact_store):
    assert isinstance(s3_artifact_store, BaseArtifactStore)


def test_upload_then_download_roundtrips_file_contents(
    s3_artifact_store, unique_key_prefix, tmp_path
):
    source = tmp_path / "source.txt"
    source.write_text("hello from a real object store")
    key = f"{unique_key_prefix}/some/nested/key"

    s3_artifact_store.upload(str(source), key)

    destination = tmp_path / "downloaded.txt"
    s3_artifact_store.download(key, str(destination))

    assert destination.read_text() == "hello from a real object store"


def test_save_and_load_round_trip_a_value_via_materializer(
    s3_artifact_store, unique_key_prefix
):
    key = s3_artifact_store.key(
        "my-pipeline", uuid.uuid4(), "add", f"{unique_key_prefix}-result"
    )

    s3_artifact_store.save(JsonMaterializer(), {"a": 1, "b": [1, 2, 3]}, key)
    loaded = s3_artifact_store.load(JsonMaterializer(), key)

    assert loaded == {"a": 1, "b": [1, 2, 3]}


def test_save_also_uploads_a_retrievable_metadata_sidecar(
    s3_artifact_store, unique_key_prefix
):
    from bettmensch_ai.pipelines.artifact_metadata import metadata_key
    from bettmensch_ai.pipelines.materializers import (
        resolve_materializer_from_artifact,
    )

    key = s3_artifact_store.key(
        "my-pipeline", uuid.uuid4(), "add", f"{unique_key_prefix}-result"
    )

    s3_artifact_store.save(JsonMaterializer(), 42, key)

    resolved = resolve_materializer_from_artifact(s3_artifact_store, key)

    assert type(resolved).__name__ == "JsonMaterializer"
    assert s3_artifact_store.load(resolved, key) == 42
    # The metadata sidecar is a real, separate object under its own key.
    assert metadata_key(key) != key


def test_uri_is_an_s3_uri_pointing_at_the_test_bucket(s3_artifact_store):
    uri = s3_artifact_store.uri("my-pipeline", uuid.uuid4(), "add", "result")

    assert uri.startswith(f"s3://{s3_artifact_store.bucket}/")
