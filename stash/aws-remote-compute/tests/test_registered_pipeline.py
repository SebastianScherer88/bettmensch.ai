from bettmensch_ai.pipelines.metadata_store import LocalMetadataStore, LocalMetadataStoreConfig
from bettmensch_ai.pipelines.runner.registered_pipeline import RegisteredPipeline


def make_metadata_store(tmp_path):
    return LocalMetadataStore(LocalMetadataStoreConfig(db_path=str(tmp_path / "metadata.db")))


def test_from_registration_projects_the_registration_record(tmp_path):
    metadata_store = make_metadata_store(tmp_path)
    dag_structure = {"tasks": [{"name": "add", "outputs": ["result"]}], "edges": [], "output": None}
    pipeline_inputs = {"a": {"required": True, "default": None, "materializer": "json"}}
    backend_metadata = {"state_machine_arn": "arn:...:sm", "batch_job_definitions": {}, "lambda_functions": {}}

    registration_id = metadata_store.register_pipeline(
        "my-pipeline",
        backend="aws_stepfunctions",
        dag_structure=dag_structure,
        pipeline_inputs=pipeline_inputs,
        backend_metadata=backend_metadata,
    )

    registered = RegisteredPipeline.from_registration(metadata_store, registration_id)

    assert registered.pipeline_registration_id == registration_id
    assert registered.pipeline_name == "my-pipeline"
    assert registered.backend == "aws_stepfunctions"
    assert registered.dag_structure == dag_structure
    assert registered.pipeline_inputs == pipeline_inputs
    assert registered.backend_metadata == backend_metadata


def test_from_registration_raises_for_an_unknown_id(tmp_path):
    import uuid

    import pytest

    metadata_store = make_metadata_store(tmp_path)

    with pytest.raises(KeyError):
        RegisteredPipeline.from_registration(metadata_store, uuid.uuid4())
