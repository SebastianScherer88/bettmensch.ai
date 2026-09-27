"""Cached connections to the stores this API reads from.

`PostgresMetadataStore`/`S3ArtifactStore` are cheap to construct (they don't
hold a persistent connection open - see their docstrings), but re-reading
config env vars and re-running schema DDL on every request is wasteful.
`lru_cache` gives us a process-wide singleton of each, created lazily on
first use.
"""

from functools import lru_cache

from bettmensch_ai.pipelines.artifact_store import S3ArtifactStore, S3ArtifactStoreConfig
from bettmensch_ai.pipelines.metadata_store import (
    PostgresMetadataStore,
    PostgresMetadataStoreConfig,
)


@lru_cache
def get_metadata_store() -> PostgresMetadataStore:
    return PostgresMetadataStore(PostgresMetadataStoreConfig())


@lru_cache
def get_artifact_store() -> S3ArtifactStore:
    return S3ArtifactStore(S3ArtifactStoreConfig())
