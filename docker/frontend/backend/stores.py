"""Cached connections to the stores/clients this API reads from.

`S3ArtifactStore` is cheap to construct (it doesn't hold a persistent
connection open - see its docstring), but re-reading config env vars on
every request is wasteful. `lru_cache` gives us a process-wide singleton,
created lazily on first use.

Metadata access goes through a `MetadataClient` rather than a
`PostgresMetadataStore` directly - this API never holds its own Postgres
connection; it talks to the metadata service (`docker/metadata-service/`)
over HTTP, the same way every other metadata consumer does.
"""

from functools import lru_cache

from bettmensch_ai.pipelines.artifact_store import S3ArtifactStore, S3ArtifactStoreConfig
from bettmensch_ai.pipelines.client import MetadataClient


@lru_cache
def get_metadata_client() -> MetadataClient:
    return MetadataClient()


@lru_cache
def get_artifact_store() -> S3ArtifactStore:
    return S3ArtifactStore(S3ArtifactStoreConfig())
