"""Unit tests for `remote_entrypoint`: pure wiring/logic against a mocked
`S3ArtifactStore` (swapped for a real `LocalArtifactStore` under the hood)
and a mocked `CodeBundler` - no real S3, no real code bundle needed since
this test's own task is already importable on `sys.path`.
"""

import json
from unittest.mock import patch

from bettmensch_ai.pipelines.artifact_store import LocalArtifactStore, LocalArtifactStoreConfig
from bettmensch_ai.pipelines.materializers import (
    DefaultMaterializer,
    resolve_materializer_from_artifact,
)
from bettmensch_ai.pipelines.runner import remote_entrypoint
from bettmensch_ai.pipelines.task import task


@task
def add(a: int, b: int) -> int:
    return a + b


@task
def failing(a: int) -> int:
    raise ValueError("boom")


def _local_store(tmp_path):
    return LocalArtifactStore(LocalArtifactStoreConfig(root_dir=str(tmp_path / "artifacts")))


@patch("bettmensch_ai.pipelines.runner.remote_entrypoint.CodeBundler")
@patch("bettmensch_ai.pipelines.runner.remote_entrypoint.S3ArtifactStoreConfig")
@patch("bettmensch_ai.pipelines.runner.remote_entrypoint.S3ArtifactStore")
def test_main_runs_a_task_successfully(mock_s3_store, mock_s3_config, mock_bundler, tmp_path):
    store = _local_store(tmp_path)
    mock_s3_store.return_value = store

    exit_code = remote_entrypoint.main(
        [
            "--task-module",
            add.func.__module__,
            "--task-qualname",
            add.func.__qualname__,
            "--default-materializer-module",
            DefaultMaterializer.__module__,
            "--default-materializer-qualname",
            DefaultMaterializer.__qualname__,
            "--code-bundle-key",
            "bundle/key",
            "--static-inputs",
            json.dumps({"a": 1, "b": 2}),
            "--output-keys",
            json.dumps({"result": "k/result"}),
        ]
    )

    assert exit_code == 0
    materializer = resolve_materializer_from_artifact(store, "k/result")
    assert store.load(materializer, "k/result") == 3
    mock_bundler.download_and_extract.assert_called_once()


@patch("bettmensch_ai.pipelines.runner.remote_entrypoint.CodeBundler")
@patch("bettmensch_ai.pipelines.runner.remote_entrypoint.S3ArtifactStoreConfig")
@patch("bettmensch_ai.pipelines.runner.remote_entrypoint.S3ArtifactStore")
def test_main_returns_nonzero_on_task_failure(mock_s3_store, mock_s3_config, mock_bundler, tmp_path, capsys):
    store = _local_store(tmp_path)
    mock_s3_store.return_value = store

    exit_code = remote_entrypoint.main(
        [
            "--task-module",
            failing.func.__module__,
            "--task-qualname",
            failing.func.__qualname__,
            "--default-materializer-module",
            DefaultMaterializer.__module__,
            "--default-materializer-qualname",
            DefaultMaterializer.__qualname__,
            "--code-bundle-key",
            "bundle/key",
            "--static-inputs",
            json.dumps({"a": 1}),
            "--output-keys",
            json.dumps({"result": "k/result"}),
        ]
    )

    assert exit_code == 1
    assert "ValueError: boom" in capsys.readouterr().err


@patch("bettmensch_ai.pipelines.runner.remote_entrypoint.CodeBundler")
@patch("bettmensch_ai.pipelines.runner.remote_entrypoint.S3ArtifactStoreConfig")
@patch("bettmensch_ai.pipelines.runner.remote_entrypoint.S3ArtifactStore")
def test_lambda_handler_returns_succeeded_payload(mock_s3_store, mock_s3_config, mock_bundler, tmp_path):
    store = _local_store(tmp_path)
    mock_s3_store.return_value = store

    result = remote_entrypoint.lambda_handler(
        {
            "task_module": add.func.__module__,
            "task_qualname": add.func.__qualname__,
            "default_materializer_module": DefaultMaterializer.__module__,
            "default_materializer_qualname": DefaultMaterializer.__qualname__,
            "code_bundle_key": "bundle/key",
            "static_inputs": {"a": 5, "b": 6},
            "output_keys": {"result": "k/result"},
        },
        context=None,
    )

    assert result["status"] == "succeeded"
    materializer = resolve_materializer_from_artifact(store, "k/result")
    assert store.load(materializer, "k/result") == 11


@patch("bettmensch_ai.pipelines.runner.remote_entrypoint.CodeBundler")
@patch("bettmensch_ai.pipelines.runner.remote_entrypoint.S3ArtifactStoreConfig")
@patch("bettmensch_ai.pipelines.runner.remote_entrypoint.S3ArtifactStore")
def test_lambda_handler_returns_failed_payload_with_traceback(mock_s3_store, mock_s3_config, mock_bundler, tmp_path):
    store = _local_store(tmp_path)
    mock_s3_store.return_value = store

    result = remote_entrypoint.lambda_handler(
        {
            "task_module": failing.func.__module__,
            "task_qualname": failing.func.__qualname__,
            "default_materializer_module": DefaultMaterializer.__module__,
            "default_materializer_qualname": DefaultMaterializer.__qualname__,
            "code_bundle_key": "bundle/key",
            "static_inputs": {"a": 1},
            "output_keys": {"result": "k/result"},
        },
        context=None,
    )

    assert result["status"] == "failed"
    assert "ValueError: boom" in result["traceback"]
