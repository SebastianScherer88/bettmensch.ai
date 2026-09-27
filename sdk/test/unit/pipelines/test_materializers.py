import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

import polars
import pytest
from bettmensch_ai.pipelines.artifact_store import (
    LocalArtifactStore,
    LocalArtifactStoreConfig,
)
from bettmensch_ai.pipelines.exceptions import MaterializerResolutionError
from bettmensch_ai.pipelines.materializers import (
    DefaultMaterializer,
    JsonMaterializer,
    PolarsParquetMaterializer,
    PydanticJsonMaterializer,
    resolve_materializer_for_type,
    resolve_materializer_for_value,
    resolve_materializer_from_artifact,
)
from bettmensch_ai.pipelines.artifact_metadata import metadata_key
from pydantic import BaseModel


class Point(BaseModel):
    x: int
    y: int


class Unsupported:
    pass


@pytest.mark.parametrize(
    "type_hint,expected_cls",
    [
        (int, JsonMaterializer),
        (str, JsonMaterializer),
        (dict, JsonMaterializer),
        (Dict[str, Any], JsonMaterializer),
        (List[int], JsonMaterializer),
        (polars.DataFrame, PolarsParquetMaterializer),
        (Point, PydanticJsonMaterializer),
        (Unsupported, DefaultMaterializer),
        (None, DefaultMaterializer),
    ],
)
def test_resolve_materializer_for_type(type_hint, expected_cls):
    assert isinstance(resolve_materializer_for_type(type_hint), expected_cls)


@pytest.mark.parametrize(
    "value,expected_cls",
    [
        (1, JsonMaterializer),
        ("a", JsonMaterializer),
        (polars.DataFrame({"a": [1]}), PolarsParquetMaterializer),
        (Point(x=1, y=2), PydanticJsonMaterializer),
        (Unsupported(), DefaultMaterializer),
    ],
)
def test_resolve_materializer_for_value(value, expected_cls):
    assert isinstance(resolve_materializer_for_value(value), expected_cls)


def test_default_materializer_refuses_to_save_or_load():
    materializer = DefaultMaterializer()

    with pytest.raises(NotImplementedError):
        materializer.save(Unsupported(), "some/path")

    with pytest.raises(NotImplementedError):
        materializer.load("some/path")


def test_json_materializer_roundtrip(tmp_path):
    materializer = JsonMaterializer()
    path = str(tmp_path / "value.json")

    materializer.save({"a": 1, "b": [1, 2, 3]}, path)

    assert materializer.load(path) == {"a": 1, "b": [1, 2, 3]}


def test_polars_materializer_roundtrip(tmp_path):
    materializer = PolarsParquetMaterializer()
    path = str(tmp_path / "value.parquet")
    df = polars.DataFrame({"a": [1, 2, 3]})

    materializer.save(df, path)
    loaded = materializer.load(path)

    assert loaded.equals(df)


def test_pydantic_materializer_roundtrip(tmp_path):
    materializer = PydanticJsonMaterializer(model=Point)
    path = str(tmp_path / "value.json")

    materializer.save(Point(x=1, y=2), path)
    loaded = materializer.load(path)

    assert loaded == Point(x=1, y=2)


def test_save_writes_a_metadata_sidecar_with_the_materializer_name(tmp_path):
    materializer = JsonMaterializer()
    path = str(tmp_path / "value.json")

    materializer.save(42, path)

    with open(metadata_key(path)) as metadata_file:
        metadata = json.load(metadata_file)

    assert metadata == {
        "schema_version": 1,
        "materializer": "json",
        "value_type": "builtins.int",
    }


def test_default_materializer_never_writes_metadata(tmp_path):
    materializer = DefaultMaterializer()
    path = str(tmp_path / "value")

    with pytest.raises(NotImplementedError):
        materializer.save(Unsupported(), path)

    assert not Path(metadata_key(path)).exists()


def test_resolve_materializer_from_artifact_uses_only_stored_metadata(tmp_path):
    store = LocalArtifactStore(
        LocalArtifactStoreConfig(root_dir=str(tmp_path / "store"))
    )
    key = store.key("my-pipeline", uuid.uuid4(), "my-task", "my-artifact")

    materializer = PolarsParquetMaterializer()
    local_path = str(tmp_path / "staging" / "value")
    Path(local_path).parent.mkdir(parents=True)
    materializer.save(polars.DataFrame({"a": [1]}), local_path)
    store.upload(local_path, key)
    store.upload(metadata_key(local_path), metadata_key(key))

    resolved = resolve_materializer_from_artifact(store, key)

    assert isinstance(resolved, PolarsParquetMaterializer)


def test_pydantic_materializer_records_model_module_and_qualname(tmp_path):
    materializer = PydanticJsonMaterializer()
    path = str(tmp_path / "value.json")

    materializer.save(Point(x=1, y=2), path)

    with open(metadata_key(path)) as metadata_file:
        metadata = json.load(metadata_file)

    assert metadata["model_module"] == Point.__module__
    assert metadata["model_qualname"] == Point.__qualname__


def test_resolve_materializer_from_artifact_reconstructs_pydantic_model(tmp_path):
    store = LocalArtifactStore(
        LocalArtifactStoreConfig(root_dir=str(tmp_path / "store"))
    )
    key = store.key("my-pipeline", uuid.uuid4(), "my-task", "my-artifact")

    # No `model` configured here on purpose: this is the constructor-less
    # path every automatic resolution path actually goes through.
    store.save(PydanticJsonMaterializer(), Point(x=1, y=2), key)

    resolved = resolve_materializer_from_artifact(store, key)
    loaded = store.load(resolved, key)

    assert loaded == Point(x=1, y=2)
    assert isinstance(loaded, Point)


def test_pydantic_materializer_from_metadata_falls_back_to_dict_for_unresolvable_class():
    materializer = PydanticJsonMaterializer.from_metadata(
        {
            "schema_version": 1,
            "materializer": "pydantic_json",
            "model_module": "no_such_module_at_all",
            "model_qualname": "NoSuchClass",
        }
    )

    assert materializer.model is None


def test_pydantic_materializer_from_metadata_falls_back_for_locally_defined_class(
    tmp_path,
):
    class LocalPoint(BaseModel):
        x: int

    materializer = PydanticJsonMaterializer.from_metadata(
        {
            "schema_version": 1,
            "materializer": "pydantic_json",
            "model_module": LocalPoint.__module__,
            "model_qualname": LocalPoint.__qualname__,
        }
    )

    assert materializer.model is None


def test_resolve_materializer_from_artifact_raises_for_unknown_name(tmp_path):
    store = LocalArtifactStore(
        LocalArtifactStoreConfig(root_dir=str(tmp_path / "store"))
    )
    key = store.key("my-pipeline", uuid.uuid4(), "my-task", "my-artifact")

    local_metadata_path = str(tmp_path / "staging" / "metadata.json")
    Path(local_metadata_path).parent.mkdir(parents=True)
    with open(local_metadata_path, "w") as metadata_file:
        json.dump({"schema_version": 1, "materializer": "bogus"}, metadata_file)
    store.upload(local_metadata_path, metadata_key(key))

    with pytest.raises(MaterializerResolutionError):
        resolve_materializer_from_artifact(store, key)
