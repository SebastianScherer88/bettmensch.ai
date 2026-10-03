"""Functional/e2e tests: run a real pipeline through `LocalRunner` across
all 4 valid `BaseArtifactStore` x `BaseMetadataStore` combinations, and
independently verify that both the run's metadata and its artifacts really
did land in the respective stores - not just that `LocalRunner.run()`
returned the right value.

+------------------------+-----------------------+-------------------------+
| Combination            | Artifact store        | Metadata store          |
+------------------------+-----------------------+-------------------------+
| local / local          | LocalArtifactStore     | LocalMetadataStore      |
| local / postgres       | LocalArtifactStore     | PostgresMetadataStore   |
| s3 / local             | S3ArtifactStore        | LocalMetadataStore      |
| s3 / postgres          | S3ArtifactStore        | PostgresMetadataStore   |
+------------------------+-----------------------+-------------------------+

The `local / local` combination needs no external infrastructure and always
runs; the other three use the `postgres_metadata_store`/`s3_artifact_store`
fixtures from tests/conftest.py, which skip independently if their own
backend isn't reachable (see
docker-compose/pipelines.docker-compose.yaml).
"""

import pytest
from bettmensch_ai.pipelines.artifact_store import (
    LocalArtifactStore,
    LocalArtifactStoreConfig,
)
from bettmensch_ai.pipelines.client import ArtifactClient, MetadataClient
from bettmensch_ai.pipelines.materializers import resolve_materializer_from_artifact
from bettmensch_ai.pipelines.metadata_store import (
    LocalMetadataStore,
    LocalMetadataStoreConfig,
    RunStatus,
)
from bettmensch_ai.pipelines.pipeline import pipeline
from bettmensch_ai.pipelines.runner import LocalRunner
from bettmensch_ai.pipelines.task import task

pytestmark = pytest.mark.functional


@task
def add(a: int, b: int) -> int:
    return a + b


@pipeline
def my_pipeline(a: int, b: int, c: int = 3):
    ab = add(a, b)
    return add(ab, c)


def _validate_run_end_to_end(runner, artifact_store, metadata_store):
    """Runs `my_pipeline`, then independently verifies (via the metadata
    store's own recorded records, not the runner's return value or any of
    its in-memory bookkeeping) that every task's output really was
    materialized in the artifact store with the content it should have.

    Args:
        runner: The `LocalRunner` to run the pipeline with.
        artifact_store: The same `BaseArtifactStore` `runner` was
            constructed with - used here only to independently re-load
            artifacts by the keys the metadata store recorded for them.
        metadata_store: The same `BaseMetadataStore` `runner` was
            constructed with - used here only to independently look up
            what got recorded, cross-checking it against the artifact
            store.

    Returns:
        The pipeline's result, for the caller's own additional assertions.
    """

    result = runner.run(my_pipeline, a=1, b=2)
    assert result == 6

    # `LocalRunner.run()` doesn't hand back the `pipeline_run_id` it
    # generated internally, and a shared, persistent backend (Postgres/
    # MinIO, unlike a fresh-per-test tmp_path) may already hold earlier
    # runs of "my-pipeline" from previous invocations of this same test -
    # so take the most recent one (`list_pipeline_runs` orders newest
    # first) rather than assuming there's exactly one.
    pipeline_runs = metadata_store.list_pipeline_runs("my-pipeline")
    assert len(pipeline_runs) >= 1
    pipeline_run = pipeline_runs[0]
    assert pipeline_run.status == RunStatus.SUCCEEDED
    assert pipeline_run.ended_at is not None

    task_runs = {r.task_name: r for r in metadata_store.list_task_runs(
        pipeline_run.pipeline_run_id
    )}
    assert set(task_runs) == {"add", "add-1"}
    assert all(r.status == RunStatus.SUCCEEDED for r in task_runs.values())

    # Cross-check: reload each task's output *from the artifact store*,
    # using only the key the metadata store recorded for it - proving the
    # two stores actually agree on what happened, not just that LocalRunner
    # claims they do.
    add_outputs = metadata_store.list_task_outputs(pipeline_run.pipeline_run_id, "add")
    assert len(add_outputs) == 1
    add_key = add_outputs[0].artifact_key
    add_materializer = resolve_materializer_from_artifact(artifact_store, add_key)
    assert artifact_store.load(add_materializer, add_key) == 3

    add1_outputs = metadata_store.list_task_outputs(
        pipeline_run.pipeline_run_id, "add-1"
    )
    assert len(add1_outputs) == 1
    add1_key = add1_outputs[0].artifact_key
    add1_materializer = resolve_materializer_from_artifact(artifact_store, add1_key)
    assert artifact_store.load(add1_materializer, add1_key) == 6

    return result


def test_local_artifact_store_and_local_metadata_store(tmp_path):
    artifact_store = LocalArtifactStore(
        LocalArtifactStoreConfig(root_dir=str(tmp_path / "artifacts"))
    )
    metadata_store = LocalMetadataStore(
        LocalMetadataStoreConfig(db_path=str(tmp_path / "metadata.db"))
    )
    runner = LocalRunner(ArtifactClient(artifact_store), MetadataClient(metadata_store))

    _validate_run_end_to_end(runner, artifact_store, metadata_store)


def test_local_artifact_store_and_postgres_metadata_store(
    tmp_path, postgres_metadata_store
):
    artifact_store = LocalArtifactStore(
        LocalArtifactStoreConfig(root_dir=str(tmp_path / "artifacts"))
    )
    runner = LocalRunner(
        ArtifactClient(artifact_store), MetadataClient(postgres_metadata_store)
    )

    _validate_run_end_to_end(runner, artifact_store, postgres_metadata_store)


def test_s3_artifact_store_and_local_metadata_store(tmp_path, s3_artifact_store):
    metadata_store = LocalMetadataStore(
        LocalMetadataStoreConfig(db_path=str(tmp_path / "metadata.db"))
    )
    runner = LocalRunner(ArtifactClient(s3_artifact_store), MetadataClient(metadata_store))

    _validate_run_end_to_end(runner, s3_artifact_store, metadata_store)


def test_s3_artifact_store_and_postgres_metadata_store(
    s3_artifact_store, postgres_metadata_store
):
    runner = LocalRunner(
        ArtifactClient(s3_artifact_store), MetadataClient(postgres_metadata_store)
    )

    _validate_run_end_to_end(runner, s3_artifact_store, postgres_metadata_store)
