"""`S3ArtifactStore`: a remote, shared artifact store backed by S3 (or an
S3-compatible service, e.g. MinIO) - Layer 2's flavour of
`BaseArtifactStore`.
"""

import uuid
from pathlib import Path
from typing import List, Optional

import boto3
from pydantic_settings import BaseSettings, SettingsConfigDict

from .base_artifact_store import BaseArtifactStore


class S3ArtifactStoreConfig(BaseSettings):
    """Configuration for an `S3ArtifactStore`.

    Attributes:
        bucket: The S3 bucket to persist artifacts under. Must already
            exist - this store never creates it (provisioning storage is a
            deployment concern, not a runtime one), the same way
            `LocalArtifactStore` never creates its own filesystem volume.
        endpoint_url: A non-AWS, S3-compatible endpoint (e.g.
            `"http://localhost:9000"` for a local MinIO). Leave unset to
            talk to AWS S3 itself.
        region_name: The AWS region to use. Ignored by most S3-compatible
            services (e.g. MinIO), but still required by boto3's client
            constructor either way.
        aws_access_key_id / aws_secret_access_key: Explicit credentials, for
            an S3-compatible service that doesn't participate in boto3's
            usual credential chain (e.g. MinIO's static root credentials).
            Left `None` to fall back to boto3's own default credential
            resolution (an IAM role, environment variables,
            `~/.aws/credentials`, etc.) - the right choice against real AWS
            S3, where hardcoding credentials here would be a step backwards.
    """

    bucket: str
    endpoint_url: Optional[str] = None
    region_name: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="bettmensch_ai_s3_artifact_store_")


class S3ArtifactStore(BaseArtifactStore):
    """Remote, shared Layer 2 artifact store, backed by S3 (or an
    S3-compatible service such as MinIO).

    The `BaseArtifactStore`/`LocalArtifactStore` split's Layer 2
    counterpart: where `LocalArtifactStore` persists to one local
    filesystem, this persists to an object store reachable by every worker
    in a genuinely distributed run, so a remote/distributed Layer 2
    orchestrator's workers can all read/write the *same* artifacts.
    `boto3` is already a hard dependency of this project (unlike
    `psycopg` for `PostgresMetadataStore`), so no lazy import is needed
    here - constructing an `S3ArtifactStore` never requires anything beyond
    what's already installed.

    `_join_key_parts` forward-slash-joins segments, matching S3's own key
    convention (object stores don't have a native "path separator" the way
    a filesystem does - forward slash is just the display/UI convention
    every S3-compatible service and console shares).
    """

    def __init__(self, config: Optional[S3ArtifactStoreConfig] = None):
        """Initializes the store.

        Args:
            config: The store's configuration. Defaults to
                `S3ArtifactStoreConfig()` (reading from the environment) if
                omitted - note `bucket` has no default, unlike
                `LocalArtifactStoreConfig.root_dir`, since there's no
                meaningful "local" fallback bucket to assume.
        """

        self.config = config or S3ArtifactStoreConfig()
        self.bucket = self.config.bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=self.config.endpoint_url,
            region_name=self.config.region_name,
            aws_access_key_id=self.config.aws_access_key_id,
            aws_secret_access_key=self.config.aws_secret_access_key,
        )

    def _join_key_parts(self, parts: List[str]) -> str:
        """Joins `key()`'s segments with forward slashes, S3's own
        convention.

        Args:
            parts: The key's segments, in order.

        Returns:
            The joined key.
        """

        return "/".join(parts)

    def uri(
        self,
        pipeline: str,
        pipeline_run_id: uuid.UUID,
        task: str,
        artifact_name: str,
    ) -> str:
        """Constructs an `s3://` uri for the artifact.

        Args:
            pipeline: The name of the pipeline.
            pipeline_run_id: The id of this run of the pipeline.
            task: The name of the task within the pipeline (or a reserved
                sentinel - see `BaseArtifactStore.key`).
            artifact_name: The artifact's own name within `task`.

        Returns:
            The artifact's `s3://` uri.
        """

        key = self.key(pipeline, pipeline_run_id, task, artifact_name)

        return f"s3://{self.bucket}/{key}"

    def upload(self, path: str, key: str) -> None:
        """Uploads a local file into the bucket under `key`.

        Args:
            path: The local path to upload from.
            key: The logical key to store the artifact under.
        """

        self.client.upload_file(path, self.bucket, key)

    def download(self, key: str, path: str) -> None:
        """Downloads the artifact stored under `key` to a local path.

        Args:
            key: The logical key to download the artifact from.
            path: The local path to save to.
        """

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, path)
