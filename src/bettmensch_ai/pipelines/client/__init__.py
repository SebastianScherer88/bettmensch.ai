"""The `Client` layer: `ArtifactClient`/`MetadataClient` (each a thin,
backend-agnostic forwarding layer in front of a `BaseArtifactStore`/
`BaseMetadataStore`), and `Client` - the "master client" bundling both,
that every runtime (a `LocalRunner` script, later a remote compute task)
holds instead of a raw store directly.
"""

from .artifact_client import ArtifactClient
from .client import Client
from .config import ArtifactClientConfig, MetadataClientConfig
from .metadata_client import MetadataClient

__all__ = [
    "ArtifactClient",
    "MetadataClient",
    "Client",
    "ArtifactClientConfig",
    "MetadataClientConfig",
]
