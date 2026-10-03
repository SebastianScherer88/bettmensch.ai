"""Artifact persistence: `BaseArtifactStore` (the interface),
`LocalArtifactStore` (Layer 1's default, filesystem-backed implementation),
and `S3ArtifactStore` (a remote, shared, Layer 2-flavoured implementation).
"""

from .base_artifact_store import BaseArtifactStore
from .local_artifact_store import LocalArtifactStore, LocalArtifactStoreConfig
from .s3_artifact_store import S3ArtifactStore, S3ArtifactStoreConfig

__all__ = [
    "BaseArtifactStore",
    "LocalArtifactStore",
    "LocalArtifactStoreConfig",
    "S3ArtifactStore",
    "S3ArtifactStoreConfig",
]
