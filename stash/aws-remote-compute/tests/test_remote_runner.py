"""Unit tests for `RemoteRunner`: pure wiring/logic against a mocked
`boto3.client("stepfunctions")` and a mocked `CodeBundler`, driven against a
real `LocalArtifactStore`/`LocalMetadataStore` so per-task bookkeeping and
output resolution are exercised end to end - matching this project's
established "mock boto3 directly, use real local stores" testing
convention.
"""

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from bettmensch_ai.pipelines.artifact_store import LocalArtifactStore, LocalArtifactStoreConfig
from bettmensch_ai.pipelines.materializers import JsonMaterializer
from bettmensch_ai.pipelines.metadata_store import (
    LocalMetadataStore,
    LocalMetadataStoreConfig,
    RunStatus,
)
from bettmensch_ai.pipelines.runner.exceptions import (
    MissingPipelineInputError,
    RemoteTaskExecutionError,
    UnknownPipelineInputError,
    UnsupportedBackendError,
)
from bettmensch_ai.pipelines.runner.registered_pipeline import RegisteredPipeline
from bettmensch_ai.pipelines.runner.remote_runner import RemoteRunner


def make_stores(tmp_path):
    artifact_store = LocalArtifactStore(LocalArtifactStoreConfig(root_dir=str(tmp_path / "artifacts")))
    metadata_store = LocalMetadataStore(LocalMetadataStoreConfig(db_path=str(tmp_path / "metadata.db")))
    return artifact_store, metadata_store


def make_dag_structure():
    return {
        "tasks": [
            {"name": "add", "rank": 0, "outputs": ["result"]},
            {"name": "multiply", "rank": 1, "outputs": ["result"]},
        ],
        "edges": [["add", "multiply"]],
        "output": {
            "name": "output",
            "source": {"kind": "task_output", "task": "multiply", "output": "result"},
        },
    }


def make_pipeline_inputs():
    return {
        "a": {"required": True, "default": None, "materializer": "json"},
        "b": {"required": True, "default": None, "materializer": "json"},
    }


def register(metadata_store):
    return metadata_store.register_pipeline(
        "my-pipeline",
        backend="aws_stepfunctions",
        dag_structure=make_dag_structure(),
        pipeline_inputs=make_pipeline_inputs(),
        backend_metadata={"state_machine_arn": "arn:...:sm"},
    )


@patch("bettmensch_ai.pipelines.runner.remote_runner.time.sleep")
@patch("bettmensch_ai.pipelines.runner.remote_runner.CodeBundler")
@patch("bettmensch_ai.pipelines.runner.remote_runner.boto3.client")
def test_run_tracks_execution_to_success_and_resolves_output(
    mock_client, mock_bundler_cls, mock_sleep, tmp_path
):
    artifact_store, metadata_store = make_stores(tmp_path)
    registered = RegisteredPipeline.from_registration(metadata_store, register(metadata_store))

    mock_bundler_cls.return_value.bundle_and_upload.return_value = "code/bundle/key"

    # `run()` generates its own run id internally, so the fake `start_execution`
    # captures it from the real execution input (rather than patching
    # `uuid.uuid4`, which - being the shared stdlib `uuid` module - would also
    # break `LocalMetadataStore`'s own id generation used by `register()` above)
    # and, right there, simulates the remote execution having already written
    # "multiply"'s output by the time its `TaskStateExited` event fires -
    # matching Step Functions' own `.sync` service integration semantics.
    captured = {}

    def start_execution(**kwargs):
        sent_input = json.loads(kwargs["input"])
        run_id = uuid.UUID(sent_input["pipeline_run_id"])
        captured["run_id"] = run_id
        captured["sent_input"] = sent_input
        output_key = artifact_store.key("my-pipeline", run_id, "multiply", "result")
        artifact_store.save(JsonMaterializer(), 42, output_key)
        captured["output_key"] = output_key
        return {"executionArn": "arn:...:execution"}

    sfn_client = MagicMock()
    mock_client.return_value = sfn_client
    sfn_client.start_execution.side_effect = start_execution

    events_after_add = [
        {"id": 1, "type": "ExecutionStarted"},
        {"id": 2, "type": "TaskStateEntered", "stateEnteredEventDetails": {"name": "add"}},
    ]
    events_after_multiply = events_after_add + [
        {"id": 3, "type": "TaskStateExited", "stateExitedEventDetails": {"name": "add"}},
        {"id": 4, "type": "TaskStateEntered", "stateEnteredEventDetails": {"name": "multiply"}},
        {"id": 5, "type": "TaskStateExited", "stateExitedEventDetails": {"name": "multiply"}},
        {"id": 6, "type": "ExecutionSucceeded"},
    ]
    sfn_client.describe_execution.side_effect = [{"status": "RUNNING"}, {"status": "SUCCEEDED"}]
    sfn_client.get_execution_history.side_effect = [
        {"events": events_after_add},
        {"events": events_after_multiply},
    ]

    result = RemoteRunner(artifact_store, metadata_store).run(registered, a=1, b=2)

    assert result == 42
    run_id = captured["run_id"]
    output_key = captured["output_key"]

    assert sfn_client.start_execution.call_args.kwargs["stateMachineArn"] == "arn:...:sm"
    sent_input = captured["sent_input"]
    assert sent_input["pipeline_name"] == "my-pipeline"
    assert sent_input["pipeline_run_id"] == str(run_id)
    assert sent_input["code_bundle_key"] == "code/bundle/key"

    a_key = artifact_store.key("my-pipeline", run_id, "__pipeline_input__", "a")
    assert artifact_store.load(JsonMaterializer(), a_key) == 1

    assert metadata_store.get_task_run(run_id, "add").status == RunStatus.SUCCEEDED
    assert metadata_store.get_task_run(run_id, "multiply").status == RunStatus.SUCCEEDED
    outputs = metadata_store.list_task_outputs(run_id, "multiply")
    assert outputs[0].artifact_key == output_key

    pipeline_run = metadata_store.get_pipeline_run(run_id)
    assert pipeline_run.status == RunStatus.SUCCEEDED
    assert pipeline_run.pipeline_assembly_id is not None

    assemblies = metadata_store.list_pipeline_assemblies("my-pipeline")
    assert assemblies[0].pipeline_assembly_id == pipeline_run.pipeline_assembly_id
    assert assemblies[0].dag_structure == registered.dag_structure


@patch("bettmensch_ai.pipelines.runner.remote_runner.time.sleep")
@patch("bettmensch_ai.pipelines.runner.remote_runner.CodeBundler")
@patch("bettmensch_ai.pipelines.runner.remote_runner.boto3.client")
def test_run_raises_and_records_a_failure_partway_through(
    mock_client, mock_bundler_cls, mock_sleep, tmp_path
):
    artifact_store, metadata_store = make_stores(tmp_path)
    registered = RegisteredPipeline.from_registration(metadata_store, register(metadata_store))

    mock_bundler_cls.return_value.bundle_and_upload.return_value = "code/bundle/key"

    captured = {}

    def start_execution(**kwargs):
        sent_input = json.loads(kwargs["input"])
        captured["run_id"] = uuid.UUID(sent_input["pipeline_run_id"])
        return {"executionArn": "arn:...:execution"}

    sfn_client = MagicMock()
    mock_client.return_value = sfn_client
    sfn_client.start_execution.side_effect = start_execution

    events_after_add = [
        {"id": 1, "type": "ExecutionStarted"},
        {"id": 2, "type": "TaskStateEntered", "stateEnteredEventDetails": {"name": "add"}},
    ]
    events_after_failure = events_after_add + [
        {"id": 3, "type": "TaskStateExited", "stateExitedEventDetails": {"name": "add"}},
        {"id": 4, "type": "TaskStateEntered", "stateEnteredEventDetails": {"name": "multiply"}},
        {"id": 5, "type": "TaskFailed", "previousEventId": 4},
        {"id": 6, "type": "ExecutionFailed"},
    ]
    sfn_client.describe_execution.side_effect = [{"status": "RUNNING"}, {"status": "FAILED"}]
    sfn_client.get_execution_history.side_effect = [
        {"events": events_after_add},
        {"events": events_after_failure},
    ]

    with pytest.raises(RemoteTaskExecutionError):
        RemoteRunner(artifact_store, metadata_store).run(registered, a=1, b=2)

    run_id = captured["run_id"]
    assert metadata_store.get_task_run(run_id, "add").status == RunStatus.SUCCEEDED
    assert metadata_store.get_task_run(run_id, "multiply").status == RunStatus.FAILED
    assert metadata_store.get_pipeline_run(run_id).status == RunStatus.FAILED


def test_run_rejects_an_unsupported_backend(tmp_path):
    artifact_store, metadata_store = make_stores(tmp_path)
    registration_id = metadata_store.register_pipeline(
        "my-pipeline",
        backend="some_other_backend",
        dag_structure=make_dag_structure(),
        pipeline_inputs=make_pipeline_inputs(),
    )
    registered = RegisteredPipeline.from_registration(metadata_store, registration_id)

    with pytest.raises(UnsupportedBackendError):
        RemoteRunner(artifact_store, metadata_store).run(registered, a=1, b=2)


def test_run_rejects_unknown_pipeline_inputs(tmp_path):
    artifact_store, metadata_store = make_stores(tmp_path)
    registered = RegisteredPipeline.from_registration(metadata_store, register(metadata_store))

    with pytest.raises(UnknownPipelineInputError):
        RemoteRunner(artifact_store, metadata_store).run(registered, a=1, b=2, c=3)


def test_run_rejects_missing_required_pipeline_inputs(tmp_path):
    artifact_store, metadata_store = make_stores(tmp_path)
    registered = RegisteredPipeline.from_registration(metadata_store, register(metadata_store))

    with pytest.raises(MissingPipelineInputError):
        RemoteRunner(artifact_store, metadata_store).run(registered, a=1)
