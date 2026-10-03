"""Config for `ArtifactClient`/`MetadataClient`: which backend each one
builds by default, when not given an already-constructed store directly.
Each backend's *own* config (`S3ArtifactStoreConfig`,
`RemoteMetadataStoreConfig`, ...) is still read independently, from its own
existing env vars, exactly as if it were constructed directly.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ArtifactClientConfig(BaseSettings):
    """Which `BaseArtifactStore` backend `ArtifactClient` builds by default.

    Attributes:
        backend: `"local"` (`LocalArtifactStore`) or `"s3"`
            (`S3ArtifactStore`). Defaults to `"local"`. Can also be set via
            the `bettmensch_ai_artifact_client_backend` environment
            variable.
    """

    backend: Literal["local", "s3"] = "local"

    model_config = SettingsConfigDict(env_prefix="bettmensch_ai_artifact_client_")


class MetadataClientConfig(BaseSettings):
    """Which `BaseMetadataStore` backend `MetadataClient` builds by
    default.

    Attributes:
        backend: `"local"` (`LocalMetadataStore`) or `"remote"`
            (`RemoteMetadataStore`, talking to the metadata service).
            Defaults to `"local"`. Can also be set via the
            `bettmensch_ai_metadata_client_backend` environment variable.
    """

    backend: Literal["local", "remote"] = "local"

    model_config = SettingsConfigDict(env_prefix="bettmensch_ai_metadata_client_")
