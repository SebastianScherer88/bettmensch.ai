"""The `BaseArtifactStore` interface: addressing, transport, and the
materializer-aware `save`/`load` convenience built on top of it.
"""

import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

from ..artifact_metadata import metadata_key

if TYPE_CHECKING:
    from ..materializers.base_materializer import BaseMaterializer


class BaseArtifactStore(ABC):
    """A dedicated artifact persistence layer that pipeline executions and
    task invocations can submit inputs and outputs to and retrieve from.

    `key()` is shared across backends: it is the backend-agnostic logical
    address of an artifact, built from the pipeline's and task's own
    (stable, human-readable) names, the run's own fresh id, and the
    artifact's own (stable, human-readable) name within that task/pipeline.
    Backends only need to supply `_join_key_parts` (how those segments
    become one string - forward-slash-joined for an object store, an
    OS-native join for a local filesystem), plus `upload`/`download`/`uri`
    to map that key onto their own physical storage. `save`/`load` are
    concrete conveniences, built on those same primitives, that also know
    how to use a `BaseMaterializer` and its metadata sidecar - so every
    backend gets them for free.
    """

    def key(
        self,
        pipeline: str,
        pipeline_run_id: uuid.UUID,
        task: str,
        artifact_name: str,
    ) -> str:
        """Constructs the artifact's logical key according to the pattern
        pipeline/pipeline_run_id/task/artifact_name.

        `pipeline`, `task`, and `artifact_name` are all stable,
        human-readable names - an `AssembledPipeline`'s/`AssembledTask`'s
        own `name`, and a task output's/pipeline input's own name (unique
        within its task/pipeline) - rather than generated ids. Only
        `pipeline_run_id` needs to be fresh per invocation: a task within a
        given pipeline run currently always executes at most once, so its
        name plus its output's own name is already enough to address that
        output uniquely within the run, with no separate "which invocation"
        id needed. This also makes a key fully deterministic given these
        four values - useful for retrying/overwriting the same artifact,
        and for addressing one without having tracked it through a runner's
        own bookkeeping.

        Args:
            pipeline: The name of the pipeline.
            pipeline_run_id: The id of this run of the pipeline.
            task: The name of the task within the pipeline (or a reserved
                sentinel, for an artifact that isn't a task's output at
                all - e.g. a pipeline input, or the code bundle).
            artifact_name: The artifact's own name within `task` - a task
                output's name, a pipeline input's name, or a reserved
                sentinel, depending on what `task` itself denotes.

        Returns:
            The artifact's logical key.
        """

        return self._join_key_parts(
            [pipeline, str(pipeline_run_id), task, artifact_name]
        )

    @abstractmethod
    def _join_key_parts(self, parts: List[str]) -> str:
        """Joins `key()`'s ordered segments into this backend's native
        storage path/reference format (e.g. forward-slash-joined for an
        object store, `os.path.join` for a local filesystem).

        Args:
            parts: The key's segments, in order.

        Returns:
            The joined key.
        """

    @abstractmethod
    def uri(
        self,
        pipeline: str,
        pipeline_run_id: uuid.UUID,
        task: str,
        artifact_name: str,
    ) -> str:
        """Constructs a display/reference uri for the artifact, in whatever
        scheme this backend uses (e.g. `file://...`, `s3://...`).

        Args:
            pipeline: The name of the pipeline.
            pipeline_run_id: The id of this run of the pipeline.
            task: The name of the task within the pipeline (or a reserved
                sentinel - see `key`).
            artifact_name: The artifact's own name within `task`.

        Returns:
            The artifact's display/reference uri.
        """

    @abstractmethod
    def upload(self, path: str, key: str) -> None:
        """Uploads an artifact from a local path to the store, under `key`.

        Args:
            path: The local path to upload from.
            key: The logical key to store the artifact under.
        """

    @abstractmethod
    def download(self, key: str, path: str) -> None:
        """Downloads an artifact stored under `key` to a local path.

        Args:
            key: The logical key to download the artifact from.
            path: The local path to save to.
        """

    def save(self, materializer: "BaseMaterializer", value: Any, key: str) -> None:
        """Materializes `value` and uploads it (and its metadata sidecar)
        to the store under `key`.

        This is the operation every runner needs to persist a task
        input/output, regardless of backend - it is implemented once, here,
        on top of `upload` and a materializer's own `save`, rather than
        each runner re-implementing "stage to a temp file, then upload"
        itself.

        Args:
            materializer: The materializer to serialize `value` with.
            value: The value to save.
            key: The key to store `value` under.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = str(Path(tmp_dir) / "value")
            materializer.save(value, path)
            self.upload(path, key)
            self.upload(metadata_key(path), metadata_key(key))

    def load(self, materializer: "BaseMaterializer", key: str) -> Any:
        """Downloads the artifact stored under `key` and deserializes it.

        Args:
            materializer: The materializer to deserialize the artifact
                with.
            key: The key the artifact is stored under.

        Returns:
            The deserialized value.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = str(Path(tmp_dir) / "value")
            self.download(key, path)

            return materializer.load(path)
