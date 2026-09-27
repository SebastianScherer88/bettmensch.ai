"""Run and registration bookkeeping: `BaseMetadataStore` (the interface),
`LocalMetadataStore` (Layer 1's default, SQLite-backed implementation), and
`PostgresMetadataStore` (a remote, shared, Layer 2-flavoured implementation).
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
]
