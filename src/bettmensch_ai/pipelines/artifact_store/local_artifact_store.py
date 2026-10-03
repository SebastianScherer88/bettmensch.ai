"""`LocalArtifactStore`: Layer 1's default, filesystem-backed artifact
store.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from .base_artifact_store import BaseArtifactStore

_DEFAULT_ROOT_DIR = str(Path(tempfile.gettempdir()) / "bettmensch_ai" / "artifacts")


class LocalArtifactStoreConfig(BaseSettings):
    """Configuration for a `LocalArtifactStore`.

    Attributes:
        root_dir: The local directory artifacts are persisted under.
            Defaults to a `bettmensch_ai/artifacts` folder under the
            system temp directory. Can also be set via the
            `bettmensch_ai_local_artifact_store_root_dir` environment
            variable.
    """

    root_dir: str = _DEFAULT_ROOT_DIR

    model_config = SettingsConfigDict(
        env_prefix="bettmensch_ai_local_artifact_store_"
    )


class LocalArtifactStore(BaseArtifactStore):
    """Default, backend-agnostic Layer 1 artifact store.

    Persists artifacts as plain files on the local filesystem, under
    `root_dir`. Remote, backend-specific flavours (e.g. an S3-backed store)
    belong to Layer 2, not here.
    """

    def __init__(self, config: Optional[LocalArtifactStoreConfig] = None):
        """Initializes the store.

        Args:
            config: The store's configuration. Defaults to
                `LocalArtifactStoreConfig()` (reading from the environment)
                if omitted.
        """

        self.config = config or LocalArtifactStoreConfig()
        self.root_dir = Path(self.config.root_dir)

    def _path(self, key: str) -> Path:
        """Resolves a key to its local filesystem path.

        Args:
            key: The artifact's logical key.

        Returns:
            The local path `key` maps to, under `root_dir`.
        """

        return self.root_dir / key

    def _join_key_parts(self, parts: List[str]) -> str:
        """Joins `key()`'s segments the way this OS natively joins path
        components (backslash-separated on Windows, forward-slash on
        POSIX) - `pathlib` reads either back correctly regardless, but this
        keeps a key's on-disk form idiomatic for whichever OS produced it.

        Args:
            parts: The key's segments, in order.

        Returns:
            The joined key.
        """

        return os.path.join(*parts)

    def uri(
        self,
        pipeline: str,
        pipeline_run_id: uuid.UUID,
        task: str,
        artifact_name: str,
    ) -> str:
        """Constructs a `file://` uri for the artifact.

        Args:
            pipeline: The name of the pipeline.
            pipeline_run_id: The id of this run of the pipeline.
            task: The name of the task within the pipeline (or a reserved
                sentinel - see `BaseArtifactStore.key`).
            artifact_name: The artifact's own name within `task`.

        Returns:
            The artifact's `file://` uri.
        """

        key = self.key(pipeline, pipeline_run_id, task, artifact_name)

        return self._path(key).resolve().as_uri()

    def upload(self, path: str, key: str) -> None:
        """Copies a local file into the store under `key`.

        Args:
            path: The local path to upload from.
            key: The logical key to store the artifact under.
        """

        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(path, destination)

    def download(self, key: str, path: str) -> None:
        """Copies the artifact stored under `key` to a local path.

        Args:
            key: The logical key to download the artifact from.
            path: The local path to save to.
        """

        source = self._path(key)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(source, path)
