"""Run and registration bookkeeping: `BaseMetadataStore` (the interface),
`LocalMetadataStore` (Layer 1's default, SQLite-backed implementation),
`PostgresMetadataStore` (a remote, shared, direct-connection, Layer
2-flavoured implementation), and `RemoteMetadataStore` (a remote, shared,
service-backed implementation - talks HTTP to a metadata service rather
than SQL to a database directly; see `docker/metadata-service/`).
"""

from .base_metadata_store import (
    BaseMetadataStore,
    PipelineAssemblyRecord,
    PipelineRegistrationRecord,
    PipelineRunRecord,
    RunStatus,
    TaskOutputRecord,
    TaskRunRecord,
    TriggerRecord,
)
from .local_metadata_store import LocalMetadataStore, LocalMetadataStoreConfig
from .postgres_metadata_store import (
    PostgresMetadataStore,
    PostgresMetadataStoreConfig,
)
from .remote_metadata_store import RemoteMetadataStore, RemoteMetadataStoreConfig

__all__ = [
    "BaseMetadataStore",
    "PipelineAssemblyRecord",
    "PipelineRegistrationRecord",
    "PipelineRunRecord",
    "RunStatus",
    "TaskOutputRecord",
    "TaskRunRecord",
    "TriggerRecord",
    "LocalMetadataStore",
    "LocalMetadataStoreConfig",
    "PostgresMetadataStore",
    "PostgresMetadataStoreConfig",
    "RemoteMetadataStore",
    "RemoteMetadataStoreConfig",
]
