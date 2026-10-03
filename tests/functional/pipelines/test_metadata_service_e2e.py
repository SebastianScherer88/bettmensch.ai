"""Functional/e2e test: runs a minimal pipeline through
`LocalRunner(ArtifactClient(...), MetadataClient(...))` against the real,
docker-compose-deployed metadata service (not an in-process app, unlike
tests/integration/pipelines/test_remote_metadata_store_integration.py) and
a real S3-compatible store (MinIO), then independently re-fetches what got
recorded - both via `MetadataClient` and via one raw `httpx` call directly
against the service's own API - and asserts they agree with what the run
actually produced.

Proves the full `client -> service -> Postgres` path end to end, the way
tests/functional/pipelines/test_store_combinations_e2e.py already proves
`LocalRunner -> store` directly for every `BaseArtifactStore`/
`BaseMetadataStore` combination.

Requires `docker compose -f docker-compose/pipelines.docker-compose.yaml
up -d` (the `metadata_service_url`/`s3_config` fixtures from
tests/conftest.py skip this test otherwise).
"""

import httpx
import pytest
from bettmensch_ai.pipelines.client import ArtifactClient, MetadataClient
from bettmensch_ai.pipelines.metadata_store import (
    RemoteMetadataStore,
    RemoteMetadataStoreConfig,
    RunStatus,
)
from bettmensch_ai.pipelines.pipeline import Pipeline, pipeline
from bettmensch_ai.pipelines.runner import LocalRunner
from bettmensch_ai.pipelines.task import task

pytestmark = pytest.mark.functional


@task
def add(a: int, b: int) -> int:
    return a + b


@pipeline(assemble=False)
def _metadata_service_e2e_pipeline(a: int, b: int, c: int = 3):
    ab = add(a, b)
    return add(ab, c)


def test_pipeline_run_via_metadata_service_is_visible_through_both_the_client_and_the_raw_api(
    metadata_service_url, s3_config, unique_key_prefix
):
    artifact_client = ArtifactClient(store=_s3_store(s3_config))
    metadata_client = MetadataClient(
        store=RemoteMetadataStore(
            RemoteMetadataStoreConfig(base_url=f"{metadata_service_url}/api")
        )
    )
    runner = LocalRunner(artifact_client, metadata_client)

    pipeline_name = f"metadata-service-e2e-{unique_key_prefix}"
    assembled_pipeline = Pipeline(
        _metadata_service_e2e_pipeline.func, name=pipeline_name
    ).assemble()

    result = runner.run(assembled_pipeline, a=2, b=3)
    assert result == 8

    pipeline_runs = metadata_client.list_pipeline_runs(pipeline_name)
    assert len(pipeline_runs) == 1
    pipeline_run = pipeline_runs[0]
    assert pipeline_run.status == RunStatus.SUCCEEDED

    task_runs = {
        r.task_name: r
        for r in metadata_client.list_task_runs(pipeline_run.pipeline_run_id)
    }
    assert set(task_runs) == {"add", "add-1"}
    assert all(r.status == RunStatus.SUCCEEDED for r in task_runs.values())

    add1_outputs = metadata_client.list_task_outputs(
        pipeline_run.pipeline_run_id, "add-1"
    )
    assert len(add1_outputs) == 1
    final_key = add1_outputs[0].artifact_key
    final_materializer = _resolve_materializer(artifact_client, final_key)
    assert artifact_client.load(final_materializer, final_key) == 8

    # Independently re-fetch the same run directly from the service's own
    # HTTP API - not through `MetadataClient`/`RemoteMetadataStore` at
    # all - proving the two routes to the same data agree.
    raw_response = httpx.get(
        f"{metadata_service_url}/api/pipeline-runs/{pipeline_run.pipeline_run_id}"
    )
    raw_response.raise_for_status()
    raw_run = raw_response.json()

    assert raw_run["pipeline_name"] == pipeline_name
    assert raw_run["status"] == RunStatus.SUCCEEDED.value

    raw_task_runs_response = httpx.get(
        f"{metadata_service_url}/api/pipeline-runs/"
        f"{pipeline_run.pipeline_run_id}/task-runs"
    )
    raw_task_runs_response.raise_for_status()
    raw_task_names = {t["task_name"] for t in raw_task_runs_response.json()}

    assert raw_task_names == set(task_runs)


def _s3_store(s3_config):
    from bettmensch_ai.pipelines.artifact_store import S3ArtifactStore

    return S3ArtifactStore(s3_config)


def _resolve_materializer(artifact_client, key):
    from bettmensch_ai.pipelines.materializers import resolve_materializer_from_artifact

    return resolve_materializer_from_artifact(artifact_client, key)
