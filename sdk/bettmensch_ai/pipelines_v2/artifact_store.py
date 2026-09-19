import uuid

from boto3 import client
from pydantic_settings import BaseSettings, SettingsConfigDict


class ArtifactStoreConfig(BaseSettings):
    s3_bucket: str
    s3_prefix: str = ""

    model_config = SettingsConfigDict(
        env_prefix="bettmensch_ai_artifact_store_"
    )


class ArtifactStore:
    def __init__(self, config: ArtifactStoreConfig):
        """_summary_"""

        self.config = config
        self.client = client("s3")
        self.s3_bucket = config.s3_bucket
        self.s3_prefix = config.s3_prefix

    @property
    def s3_base_path(self) -> str:
        base_path = "s3://{self.s3_bucket}"

        if self.s3_prefix:
            base_path += f"/{self.s3_prefix}"

        return base_path

    def uri(
        self,
        pipeline_id: uuid.UUID,
        pipeline_run_id: uuid.UUID,
        component_id: uuid.UUID,
        artifact_id: uuid.UUID | None = None,
    ) -> str:
        """Constructs the uri according to the pattern
        base_path/pipeline_id/pipeline_run_id/component_id/artifact_id

        Args:
            pipeline_id (UUID): The id of the pipeline
            pipeline_run_id (UUID): The id of the pipeline run
            component_id (UUID): The id of the component
            artifact_id (UUID): The id of the artifact

        Returns:
            str: The remote uri
        """

        if artifact_id is None:
            artifact_id = uuid.uuid4()

        return f"{self.s3_base_path}/{pipeline_id}/{pipeline_run_id}/{component_id}/{artifact_id}"

    def upload(self, path: str, uri: str):
        """Uploads an artifact from a local path to a remote uri.

        Args:
            path (str): The local path to upload from.
            uri (str): The remote uri to upload to.
        """

        self.client.upload_file(path, self.s3_bucket, uri)

    def download(self, uri: str, path: str):
        """Downloads and artifact from the remote uri to the local path.

        Args:
            uri (str): The remote uri to download from.
            path (str): The local path to save to.
        """

        self.client.downlad_file(self.s3_bucket, uri, path)
