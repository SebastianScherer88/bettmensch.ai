"""`Client`: the "master client" every runtime (the developer's own
session, or a remote compute task) attaches - bundles one `ArtifactClient`
(`.artifact_storage_client`) and one `MetadataClient` (`.metadata_client`),
each independently configured/overridable.
"""

from typing import Optional

from .artifact_client import ArtifactClient
from .metadata_client import MetadataClient


class Client:
    """Bundles an `ArtifactClient` and a `MetadataClient` - the one object
    a runtime needs to hold for both artifact and metadata access.
    """

    def __init__(
        self,
        artifact_client: Optional[ArtifactClient] = None,
        metadata_client: Optional[MetadataClient] = None,
    ):
        self.artifact_storage_client = artifact_client or ArtifactClient()
        self.metadata_client = metadata_client or MetadataClient()
