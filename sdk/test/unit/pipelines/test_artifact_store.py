import json
import os
import uuid

import pytest
from bettmensch_ai.pipelines.artifact_metadata import metadata_key
from bettmensch_ai.pipelines.artifact_store import (
    BaseArtifactStore,
    LocalArtifactStore,
    LocalArtifactStoreConfig,
)
from bettmensch_ai.pipelines.materializers import JsonMaterializer


def test_base_artifact_store_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseArtifactStore()


def test_local_artifact_store_is_a_base_artifact_store(tmp_path):
    assert isinstance(make_store(tmp_path), BaseArtifactStore)


def make_store(tmp_path):
    config = LocalArtifactStoreConfig(root_dir=str(tmp_path))
    return LocalArtifactStore(config)


def test_key_joins_segments_with_os_path_join(tmp_path):
    store = make_store(tmp_path)
    pipeline_run_id = uuid.uuid4()

    key = store.key("my-pipeline", pipeline_run_id, "my-task", "my-artifact")

    assert key == os.path.join(
        "my-pipeline", str(pipeline_run_id), "my-task", "my-artifact"
    )


def test_key_is_deterministic_given_the_same_arguments(tmp_path):
    store = make_store(tmp_path)
    pipeline_run_id = uuid.uuid4()

    key_one = store.key("my-pipeline", pipeline_run_id, "my-task", "my-artifact")
    key_two = store.key("my-pipeline", pipeline_run_id, "my-task", "my-artifact")

    assert key_one == key_two


def test_uri_is_a_file_uri_under_root_dir(tmp_path):
    store = make_store(tmp_path)
    pipeline_run_id = uuid.uuid4()

    uri = store.uri("my-pipeline", pipeline_run_id, "my-task", "my-artifact")

    assert uri.startswith("file://")
    assert tmp_path.resolve().as_posix() in uri


def test_upload_then_download_roundtrips_file_contents(tmp_path):
    store = make_store(tmp_path / "store")
    source = tmp_path / "source.txt"
    source.write_text("hello artifact")

    key = "some/nested/key"
    store.upload(str(source), key)

    assert (tmp_path / "store" / "some" / "nested" / "key").read_text() == (
        "hello artifact"
    )

    destination = tmp_path / "downloaded.txt"
    store.download(key, str(destination))

    assert destination.read_text() == "hello artifact"


def test_upload_creates_missing_parent_directories(tmp_path):
    store = make_store(tmp_path / "store")
    source = tmp_path / "source.txt"
    source.write_text("data")

    store.upload(str(source), "a/b/c/d")

    assert (tmp_path / "store" / "a" / "b" / "c" / "d").exists()


def test_default_root_dir_is_under_system_temp_dir():
    config = LocalArtifactStoreConfig()

    assert "bettmensch_ai" in config.root_dir
    assert "artifacts" in config.root_dir


def test_save_and_load_round_trip_a_value(tmp_path):
    store = make_store(tmp_path)
    key = store.key("my-pipeline", uuid.uuid4(), "my-task", "my-artifact")

    store.save(JsonMaterializer(), {"a": 1}, key)

    assert store.load(JsonMaterializer(), key) == {"a": 1}


def test_save_also_uploads_a_metadata_sidecar(tmp_path):
    store = make_store(tmp_path)
    key = store.key("my-pipeline", uuid.uuid4(), "my-task", "my-artifact")

    store.save(JsonMaterializer(), {"a": 1}, key)

    with open(tmp_path / metadata_key(key)) as metadata_file:
        metadata = json.load(metadata_file)

    assert metadata["materializer"] == "json"
