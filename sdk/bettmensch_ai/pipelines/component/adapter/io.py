import json
import os
from typing import Any, Dict, Optional, Tuple

from bettmensch_ai.pipelines.constants import (
    S3_ARTIFACT_REPOSITORY_BUCKET,
    S3_ARTIFACT_REPOSITORY_PREFIX,
)
from bettmensch_ai.pipelines.io import InputParameter


class AdapterIO(object):

    s3_client: Any
    s3_prefix: str
    external_input_parameters: str = "input_parameters.json"
    external_output_parameters: str = "output_parameters.json"

    def __init__(self, s3_prefix: Optional[str] = None):
        self.s3_client = self.get_s3_client()

        if s3_prefix is None:
            # get workflow id from argo context variable - see
            # https://argo-workflows.readthedocs.io/en/latest/variables/#global
            workflow_name = json.loads(r"""{{workflow.name}}""")
            # workflow_name = os.environ.get['ARGO_WORKFLOW_NAME']
            # get task name from argo context variable - see
            # https://argo-workflows.readthedocs.io/en/latest/variables/...
            # ...#dag-templates
            # task_node_id = json.loads(r"""{{task.name}}""")
            task_node_id = os.environ.get["ARGO_NODE_ID"]
            self.s3_prefix = f"{S3_ARTIFACT_REPOSITORY_PREFIX}/{workflow_name}/{task_node_id}"  # noqa: E501
        else:
            self.s3_prefix = s3_prefix

    @staticmethod
    def get_s3_client():
        """Returns the boto3 client for the artifact repository S3 bucket"""

        try:
            import boto3
        except ModuleNotFoundError as e:
            print(
                "Boto3 could not be found. Did you install the bettmensch.ai"
                "sdk with the `pipelines-adapter` extras?"
            )
            raise e

        client = boto3.client("s3")

        return client

    @staticmethod
    @property
    def hera_input_parameters() -> Dict[str, Any]:
        return json.loads(r"""{{inputs.parameters}}""")

    def upload_dict_to_s3(self, data: Dict, s3_prefix: str):
        """Utility function to upload a dictionary as a json file to the
        specified prefix in the artifact repository S3 bucket.

        Args:
            data (Dict): The dictionary to be uploaded
            s3_prefix (str): The upload target s3 prefix
        """
        data_file = bytes(json.dumps(data).encode("UTF-8"))

        self.s3_client.upload_fileobj(
            data_file, S3_ARTIFACT_REPOSITORY_BUCKET, s3_prefix
        )

    def download_dict_from_s3(self, s3_prefix: str) -> Dict:
        """Utility function to download a json as a dictionary from the
        specified prefix in the artifact repository S3 bucket.

        Args:
            s3_prefix (str): The s3 source prefix of the json file

        Returns:
            Dict: The content of the remote json file
        """

        s3_object = self.s3_client.get_object(
            Bucket=S3_ARTIFACT_REPOSITORY_BUCKET, Key=s3_prefix
        )
        data_json_str = s3_object["Body"].read().decode()
        data_dict = json.loads(data_json_str)

        return data_dict

    def get_artifact_repository_prefix(self, s3_file_name: str) -> str:
        """Returns the artifact repository S3 prefix for the given filename.

        Args:
            s3_file_name (str): The name of the S3 artifact to be created

        Raises:
            e: A ModuleNotFoundError if the boto3 module is missing

        Returns:
            (str): The s3 prefix
        """

        return f"{self.s3_prefix}/{s3_file_name}"  # noqa: E501

    def upload_artifact_to_s3(self, artifact_name: str):
        """Utility to upload a TorchDDPComponent's specified input artifact to
        S3. Useful for the IO pre adapter task that exposes argo inputs to the
        DDP jobset processes via S3

        Args:
            artifact_name (str): The name of the input artifact to be uploaded.
        """

        artifact_s3_prefix = self.get_artifact_repository_prefix(artifact_name)
        self.s3_client.upload_file(
            artifact_name, S3_ARTIFACT_REPOSITORY_BUCKET, artifact_s3_prefix
        )

    def download_artifact_from_s3(
        self,
        artifact_name: str,
    ):
        """Utility to download a TorchDDPComponent's specified output artifact
        from S3. Useful for the IO post adapter task that exposes the DDP
        jobset outputs to argo output artifacts

        Args:
            artifact_name (str): The name of the output artifact to be
                downloaded.
        """

        artifact_s3_prefix = self.get_artifact_repository_prefix(artifact_name)
        self.s3_client.download_file(
            S3_ARTIFACT_REPOSITORY_BUCKET, artifact_s3_prefix, artifact_name
        )

    def hera_input_parameters_to_s3(self):
        """Utility to upload a TorchDDPComponent's input parameters to S3.
        Useful for the IO pre adapter task that exposes argo inputs to the DDP
        jobset processes via S3

        Raises:
            e: A ModuleNotFoundError if the boto3 module is missing
        """

        # get input parameter values from argo context variable - see
        # https://argo-workflows.readthedocs.io/en/latest/variables/...
        # ...#all-templates
        s3_prefix = self.get_artifact_repository_prefix(
            self.external_input_parameters
        )
        self.upload_dict_to_s3(self.hera_input_parameters, s3_prefix)

    def download_container_inputs_from_s3(
        self, artifact_names: Tuple[str]
    ) -> Dict[str, Any]:
        """Utility to download input parameters and artifacts from S3. Useful
        for the TorchDDPComponent's main containers running outside the
        ArgoWorkflow context (requires that the IO pre adapter task has
        completed).

        Args:
            workflow_id (str): The unique id of the ArgoWorkflow that created
            the external container/K8s JobSet. Required to construct the S3
            prefix where inputs are located

        Returns:
            Dict[str,Any]: The input parameters
        """

        s3_prefix = self.get_artifact_repository_prefix(
            self.external_input_parameters
        )

        # downloading input artifacts to their expected paths on disk for the
        # external container process to pick up
        [
            self.download_artifact_from_s3(artifact_name, self.workflow_id)
            for artifact_name in artifact_names
        ]

        # download json holding all input parameters
        input_parameters_dict = self.download_dict_from_s3(s3_prefix)

        return input_parameters_dict

    def upload_container_outputs_to_s3(
        self,
        output_parameters: Dict[str, Any],
        artifact_names: Tuple[str],
    ):

        # upload all output parameters into one json file
        s3_prefix = self.get_artifact_repository_prefix(
            self.external_output_parameters
        )
        self.upload_dict_to_s3(output_parameters, s3_prefix)

        # upload all output artifacts into files named the same as the artifact
        [
            self.upload_artifact_to_s3(artifact_name)
            for artifact_name in artifact_names
        ]

    def s3_to_hera_output_parameters(self):
        """Utility to download a TorchDDPComponent's output parameters from S3.
        Useful for the IO post adapter task that exposes the DDP jobset outputs
        to argo output parameters
        """

        s3_prefix = self.get_artifact_repository_prefix(
            self.external_output_parameters
        )

        # downloading output artifacts to their expected paths on disk for the
        # hera IO post adapter Task to pick up
        input_parameters_dict = self.download_dict_from_s3(s3_prefix)

        # downloading output parameters to their expected paths on disk for the
        # hera IO post adapter Task to pick up
        for (
            input_parameter_name,
            input_parameter_value,
        ) in input_parameters_dict.items():
            InputParameter(
                name=input_parameter_name, value=input_parameter_value
            ).export()
