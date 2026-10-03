"""The real `PostgresMetadataStore` this service wraps - the only process
in the whole stack allowed to hold a Postgres connection/DSN.

`lru_cache` gives us a process-wide singleton, created lazily on first use
(same convention as `docker/frontend/backend/stores.py`).
"""

from functools import lru_cache

from bettmensch_ai.pipelines.metadata_store import (
    PostgresMetadataStore,
    PostgresMetadataStoreConfig,
)


@lru_cache
def get_metadata_store() -> PostgresMetadataStore:
    return PostgresMetadataStore(PostgresMetadataStoreConfig())
