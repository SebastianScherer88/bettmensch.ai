"""Shared fixtures for tests that need real infrastructure (a PostgreSQL
server, an S3-compatible object store) rather than Layer 1's local
defaults.

Backed by `sdk/test/docker-compose/pipelines.docker-compose.yaml` - start
it (`docker compose -f sdk/test/docker-compose/pipelines.docker-compose.yaml up -d`)
before running `sdk/test/integration/pipelines` or
`sdk/test/functional/pipelines`, or their tests skip. Each fixture here
skips independently rather than failing, so e.g. having only Postgres up
still lets Postgres-only tests run while S3-only ones skip.
"""

import os
import uuid

import pytest

_TEST_POSTGRES_DSN = os.environ.get(
    "BETTMENSCH_AI_TEST_POSTGRES_DSN",
    "postgresql://bettmensch_ai:bettmensch_ai@localhost:5433/bettmensch_ai_metadata",
)
_TEST_S3_ENDPOINT_URL = os.environ.get(
    "BETTMENSCH_AI_TEST_S3_ENDPOINT_URL", "http://localhost:9000"
)
_TEST_S3_ACCESS_KEY_ID = os.environ.get(
    "BETTMENSCH_AI_TEST_S3_ACCESS_KEY_ID", "bettmensch_ai"
)
_TEST_S3_SECRET_ACCESS_KEY = os.environ.get(
    "BETTMENSCH_AI_TEST_S3_SECRET_ACCESS_KEY", "bettmensch_ai_secret"
)
_TEST_S3_BUCKET = os.environ.get(
    "BETTMENSCH_AI_TEST_S3_BUCKET", "bettmensch-ai-test"
)


@pytest.fixture(scope="session")
def postgres_dsn() -> str:
    """The test PostgreSQL server's DSN, skipping if it isn't reachable.

    Defaults to the credentials/port `pipelines.docker-compose.yaml`
    itself uses, overridable via `BETTMENSCH_AI_TEST_POSTGRES_DSN` (e.g. to
    point at a CI-managed Postgres instead of the compose one).
    """

    pytest.importorskip("psycopg")
    import psycopg

    try:
        with psycopg.connect(_TEST_POSTGRES_DSN, connect_timeout=2) as connection:
            connection.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001 - any connection failure means skip
        pytest.skip(
            f"No reachable test PostgreSQL server at {_TEST_POSTGRES_DSN!r} "
            f"({exc}). Start it with `docker compose -f "
            "sdk/test/docker-compose/pipelines.docker-compose.yaml up -d`."
        )

    return _TEST_POSTGRES_DSN


@pytest.fixture(scope="session")
def postgres_metadata_store(postgres_dsn):
    """A `PostgresMetadataStore` pointed at the reachable test server.

    Session-scoped, like `postgres_dsn`: it's a thin, stateless client
    wrapper around the real database, so sharing one instance across every
    test that needs it just avoids reconnecting per test - individual tests
    stay isolated from each other via their own fresh `pipeline_run_id`s,
    not via a fresh store object.
    """

    from bettmensch_ai.pipelines.metadata_store import (
        PostgresMetadataStore,
        PostgresMetadataStoreConfig,
    )

    return PostgresMetadataStore(PostgresMetadataStoreConfig(dsn=postgres_dsn))


@pytest.fixture(scope="session")
def s3_config():
    """An `S3ArtifactStoreConfig` pointed at the reachable test MinIO,
    skipping if it isn't reachable, and ensuring the test bucket exists.

    Defaults to the credentials/port `pipelines.docker-compose.yaml`
    itself uses, overridable via the `BETTMENSCH_AI_TEST_S3_*` environment
    variables (e.g. to point at a CI-managed MinIO/S3 instead).
    """

    import boto3
    import botocore.exceptions
    from botocore.config import Config

    from bettmensch_ai.pipelines.artifact_store import S3ArtifactStoreConfig

    # Explicit, short timeouts and a single attempt: boto3's defaults (long
    # timeouts plus several retries with backoff) make each unreachable-
    # endpoint check slow, and slower still the more of them a test session
    # runs - fine for real requests, but this is purely a reachability probe
    # that should fail fast and consistently when nothing is listening.
    client = boto3.client(
        "s3",
        endpoint_url=_TEST_S3_ENDPOINT_URL,
        region_name="us-east-1",
        aws_access_key_id=_TEST_S3_ACCESS_KEY_ID,
        aws_secret_access_key=_TEST_S3_SECRET_ACCESS_KEY,
        config=Config(
            connect_timeout=2,
            read_timeout=2,
            retries={"max_attempts": 1},
        ),
    )

    try:
        client.head_bucket(Bucket=_TEST_S3_BUCKET)
    except botocore.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
            client.create_bucket(Bucket=_TEST_S3_BUCKET)
        else:
            pytest.skip(
                f"Test bucket {_TEST_S3_BUCKET!r} at {_TEST_S3_ENDPOINT_URL!r} "
                f"is not usable ({exc})."
            )
    except Exception as exc:  # noqa: BLE001 - any connection failure means skip
        pytest.skip(
            f"No reachable test S3/MinIO endpoint at {_TEST_S3_ENDPOINT_URL!r} "
            f"({exc}). Start it with `docker compose -f "
            "sdk/test/docker-compose/pipelines.docker-compose.yaml up -d`."
        )

    return S3ArtifactStoreConfig(
        bucket=_TEST_S3_BUCKET,
        endpoint_url=_TEST_S3_ENDPOINT_URL,
        aws_access_key_id=_TEST_S3_ACCESS_KEY_ID,
        aws_secret_access_key=_TEST_S3_SECRET_ACCESS_KEY,
    )


@pytest.fixture(scope="session")
def s3_artifact_store(s3_config):
    """An `S3ArtifactStore` pointed at the reachable test MinIO.

    Session-scoped for the same reason as `postgres_metadata_store`: tests
    isolate their own data via `unique_key_prefix`/a fresh `pipeline_run_id`
    in the key, not via a fresh store object per test.
    """

    from bettmensch_ai.pipelines.artifact_store import S3ArtifactStore

    return S3ArtifactStore(s3_config)


@pytest.fixture
def unique_key_prefix() -> str:
    """A fresh, unique string - used to namespace artifacts/keys a test
    writes, so repeated test runs against the same real bucket/database
    never collide with data a previous run left behind.
    """

    return str(uuid.uuid4())
