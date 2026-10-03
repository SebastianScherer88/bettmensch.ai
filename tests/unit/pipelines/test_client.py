"""Unit tests for the `client` package: `ArtifactClient`/`MetadataClient`
build the right default store from config, and forward every call to
whichever store - explicit or default - they hold. `Client` just bundles
both.

Neither `ArtifactClient` nor `MetadataClient` is a `BaseArtifactStore`/
`BaseMetadataStore` subclass (by design - see their own docstrings), so
these tests assert the opposite of that too: a client is never mistaken
for the store it forwards to.
"""

import uuid
from unittest.mock import MagicMock, patch

from bettmensch_ai.pipelines.artifact_store import BaseArtifactStore, LocalArtifactStore
from bettmensch_ai.pipelines.client import (
    ArtifactClient,
    ArtifactClientConfig,
    Client,
    MetadataClient,
    MetadataClientConfig,
)
from bettmensch_ai.pipelines.metadata_store import (
    BaseMetadataStore,
    LocalMetadataStore,
    RunStatus,
)

# --- ArtifactClient --------------------------------------------------------


def test_artifact_client_defaults_to_a_local_artifact_store(monkeypatch):
    monkeypatch.delenv("BETTMENSCH_AI_ARTIFACT_CLIENT_BACKEND", raising=False)

    client = ArtifactClient()

    assert isinstance(client.store, LocalArtifactStore)
    assert not isinstance(client, BaseArtifactStore)


@patch("bettmensch_ai.pipelines.client.artifact_client.S3ArtifactStore")
def test_artifact_client_builds_an_s3_artifact_store_when_configured(
    mock_s3_cls, monkeypatch
):
    monkeypatch.setenv("BETTMENSCH_AI_S3_ARTIFACT_STORE_BUCKET", "my-bucket")
    mock_s3_cls.return_value = MagicMock()

    import bettmensch_ai.pipelines.client.artifact_client as module

    client = module.ArtifactClient(
        store=module._build_default_store(ArtifactClientConfig(backend="s3"))
    )

    mock_s3_cls.assert_called_once()
    assert client.store is mock_s3_cls.return_value


def test_artifact_client_forwards_every_call_to_its_store():
    mock_store = MagicMock(spec=BaseArtifactStore)
    client = ArtifactClient(store=mock_store)
    pipeline_run_id = uuid.uuid4()

    client.key("my-pipeline", pipeline_run_id, "add", "result")
    client.uri("my-pipeline", pipeline_run_id, "add", "result")
    client.upload("/local/path", "some/key")
    client.download("some/key", "/local/path")
    client.save("materializer", "value", "some/key")
    client.load("materializer", "some/key")

    mock_store.key.assert_called_once_with(
        "my-pipeline", pipeline_run_id, "add", "result"
    )
    mock_store.uri.assert_called_once_with(
        "my-pipeline", pipeline_run_id, "add", "result"
    )
    mock_store.upload.assert_called_once_with("/local/path", "some/key")
    mock_store.download.assert_called_once_with("some/key", "/local/path")
    mock_store.save.assert_called_once_with("materializer", "value", "some/key")
    mock_store.load.assert_called_once_with("materializer", "some/key")


# --- MetadataClient ----------------------------------------------------


def test_metadata_client_defaults_to_a_local_metadata_store(monkeypatch):
    monkeypatch.delenv("BETTMENSCH_AI_METADATA_CLIENT_BACKEND", raising=False)

    client = MetadataClient()

    assert isinstance(client.store, LocalMetadataStore)
    assert not isinstance(client, BaseMetadataStore)


@patch("bettmensch_ai.pipelines.client.metadata_client.RemoteMetadataStore")
def test_metadata_client_builds_a_remote_metadata_store_when_configured(
    mock_remote_cls, monkeypatch
):
    monkeypatch.setenv(
        "BETTMENSCH_AI_METADATA_SERVICE_BASE_URL", "http://localhost:8081/api"
    )
    mock_remote_cls.return_value = MagicMock()

    import bettmensch_ai.pipelines.client.metadata_client as module

    client = module.MetadataClient(
        store=module._build_default_store(MetadataClientConfig(backend="remote"))
    )

    mock_remote_cls.assert_called_once()
    assert client.store is mock_remote_cls.return_value
    assert isinstance(client, MetadataClient)


def test_metadata_client_forwards_calls_to_its_store():
    mock_store = MagicMock(spec=BaseMetadataStore)
    client = MetadataClient(store=mock_store)
    pipeline_run_id = uuid.uuid4()

    client.start_pipeline_run("my-pipeline", pipeline_run_id)
    client.finish_pipeline_run(pipeline_run_id, RunStatus.SUCCEEDED)
    client.record_task_output(pipeline_run_id, "add", "result", "some/key")

    mock_store.start_pipeline_run.assert_called_once_with(
        "my-pipeline", pipeline_run_id, None
    )
    mock_store.finish_pipeline_run.assert_called_once_with(
        pipeline_run_id, RunStatus.SUCCEEDED
    )
    mock_store.record_task_output.assert_called_once_with(
        pipeline_run_id, "add", "result", "some/key"
    )


# --- Client --------------------------------------------------------------


def test_client_bundles_an_artifact_client_and_a_metadata_client():
    client = Client()

    assert isinstance(client.artifact_storage_client, ArtifactClient)
    assert isinstance(client.metadata_client, MetadataClient)


def test_client_accepts_explicit_sub_clients():
    artifact_client = ArtifactClient(store=MagicMock(spec=BaseArtifactStore))
    metadata_client = MetadataClient(store=MagicMock(spec=BaseMetadataStore))

    client = Client(artifact_client=artifact_client, metadata_client=metadata_client)

    assert client.artifact_storage_client is artifact_client
    assert client.metadata_client is metadata_client
