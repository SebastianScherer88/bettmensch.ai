"""`ArtifactClient`: the local Python access route every runtime (the
developer's own session, or a remote compute task) holds for artifact
access - never talks to storage directly, only ever through the
`BaseArtifactStore` it holds.

Deliberately not a `BaseArtifactStore` subclass - it's its own type, so a
client and a store can never be confused for one another even though
every method here has the same name/signature as the store method it
forwards to.
"""

import uuid
from typing import Any, Optional

from ..artifact_store import (
    BaseArtifactStore,
    LocalArtifactStore,
    LocalArtifactStoreConfig,
    S3ArtifactStore,
    S3ArtifactStoreConfig,
)
from .config import ArtifactClientConfig


def _build_default_store(config: ArtifactClientConfig) -> BaseArtifactStore:
    if config.backend == "s3":
        return S3ArtifactStore(S3ArtifactStoreConfig())
    return LocalArtifactStore(LocalArtifactStoreConfig())


class ArtifactClient:
    """Forwards every call to whichever `BaseArtifactStore` it holds -
    either one given explicitly, or one built from `ArtifactClientConfig`
    (environment-driven: `local` or `s3`) when none is given.
    """

    def __init__(self, store: Optional[BaseArtifactStore] = None):
        self.store = store or _build_default_store(ArtifactClientConfig())

    def key(
        self, pipeline: str, pipeline_run_id: uuid.UUID, task: str, artifact_name: str
    ) -> str:
        return self.store.key(pipeline, pipeline_run_id, task, artifact_name)

    def uri(
        self, pipeline: str, pipeline_run_id: uuid.UUID, task: str, artifact_name: str
    ) -> str:
        return self.store.uri(pipeline, pipeline_run_id, task, artifact_name)

    def upload(self, path: str, key: str) -> None:
        self.store.upload(path, key)

    def download(self, key: str, path: str) -> None:
        self.store.download(key, path)

    def save(self, materializer: Any, value: Any, key: str) -> None:
        self.store.save(materializer, value, key)

    def load(self, materializer: Any, key: str) -> Any:
        return self.store.load(materializer, key)
