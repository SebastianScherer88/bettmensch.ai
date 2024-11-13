import json
import os
from typing import Any, Dict, Optional, Tuple, Union

from bettmensch_ai.constants import (
    COMPONENT_TYPE,
    DDP_TASK_ALIAS,
    INPUT_TYPE,
    OUTPUT_TYPE,
    PIPELINE_TYPE,
    S3_ARTIFACT_REPOSITORY_BUCKET,
    S3_ARTIFACT_REPOSITORY_PREFIX,
)
from hera.workflows import Artifact, Parameter, models


class IO(object):

    type = OUTPUT_TYPE

    def __init__(self, name: str, value: Any = None):
        self.name = name
        self.value = value

    def assign(self, value: Any):
        self.value = value

        self.export()

    @property
    def path(self):

        return os.path.join(self.name)

    def export(self):

        with open(self.path, "w") as output_file:
            output_file.write(str(self.value))

    def __eq__(self, other):

        return (
            self.name == other.name
            and self.value == other.value
            and self.source == other.source
            and self.owner == other.owner
        )

    def __repr__(self):

        return str(
            {
                "name": self.name,
                "value": self.value,
                "source": self.source,
                "owner": self.owner,
            }
        )


class Input(IO):

    type = INPUT_TYPE


class Output(IO):

    type = OUTPUT_TYPE


class OriginMixin(object):

    owner: Union["Component", "Pipeline"] = None  # noqa: F821
    source: Union["InputParameter", "OutputParameter", "OutputArtifact"] = None
    id: str = None

    def set_owner(self, owner: Union["Component", "Pipeline"]):  # noqa: F821
        if getattr(owner, "type", None) not in (PIPELINE_TYPE, COMPONENT_TYPE):
            raise TypeError(
                f"The specified parameter owner {owner} has to be "
                "either a Pipeline or Component type."
            )

        self.owner = owner

    def set_source(
        self,
        source: Union["InputParameter", "OutputParameter", "OutputArtifact"],
    ):
        if not isinstance(
            source, (InputParameter, OutputParameter, OutputArtifact)
        ):
            raise TypeError(
                f"The specified parameter source {source} has to be one of: "
                "(InputParameter, OutputParameter, OutputArtifact)."
            )

        self.source = source


class InputParameter(OriginMixin, Input):
    @property
    def id(self) -> str:
        """Utility method to generate a hera/ArgoWorkflow parameter reference
        to be used when constructing the hera DAG.

        Returns:
            str: The hera parameter reference expression.
        """

        hera_expression = (
            "{{" + f"{self.owner.io_owner_name}.parameters.{self.name}" + "}}"
        )

        return hera_expression

    def to_hera(self) -> Parameter:
        # PipelineInput annotated function arguments' default values are
        # retained by the Pipeline class. We only include a default value
        # if its non-trivial

        owner = getattr(self, "owner", None)
        owner_type = getattr(owner, "type", None)
        source_owner = getattr(self.source, "owner", None)

        if owner_type == PIPELINE_TYPE:
            # Pipeline owned InputParameters dont reference anything, so its
            # enough to pass name and value (which might be none)
            return Parameter(name=self.name, value=self.value)
        elif owner_type == COMPONENT_TYPE:
            # Component owned InputParameters are not always
            # referencing another InputParameter, so
            # we reference the source parameter's `id` '{{...}}' expression
            # only if the provided source has a non-trivial owner. In that
            # case, the value will be the hera expression referencing the
            # source argument. If the provided source has no owner, we are
            # dealing with a hardcoded template function argument spec for this
            # component, and retain the value (which could be None). This
            # allows us to hardcode an input to a Component in a Pipeline that
            # is different to the Component's template function's default value
            # for that argument, without having to create a PipelineInput.
            if source_owner is not None:
                return Parameter(name=self.name, value=self.source.id)
            else:
                return Parameter(name=self.name, value=self.value)


class OutputParameter(OriginMixin, Output):
    @property
    def id(self) -> str:
        """Utility method to generate a hera/ArgoWorkflow parameter reference
        to be used when constructing the hera DAG.

        Returns:
            str: The hera parameter reference expression.
        """

        hera_expression = (
            "{{"
            + f"{self.owner.io_owner_name}.{self.type}.parameters.{self.name}"
            + "}}"
        )

        return hera_expression

    def to_hera(self) -> Parameter:

        owner = getattr(self, "owner", None)
        owner_type = getattr(owner, "type", None)

        if owner_type == PIPELINE_TYPE:
            # Pipeline owned InputParameters dont reference anything, so its
            # enough to pass name and value (which might be none)
            raise NotImplementedError(
                "Pipeline owned OutputParameters are not" " supported yet."
            )
        elif owner_type == COMPONENT_TYPE:
            return Parameter(
                name=self.name, value_from=models.ValueFrom(path=self.path)
            )


class InputArtifact(OriginMixin, Input):
    def to_hera(self, template: bool = False) -> Artifact:
        # InputArtifacts need to be converted to hera inputs in two places:
        # - when defining the hera template
        # - when defining the input value/reference for the DAG's task
        if template:
            return Artifact(name=self.name, path=self.path)
        else:
            source_owner = getattr(self.source, "owner", None)
            if source_owner is not None:
                return Artifact(name=self.name, from_=self.source.id)
            else:
                raise ValueError(
                    "InputArtifact must be associated with a source that is "
                    "owned either by a Pipeline or another Component"
                )


class OutputArtifact(OriginMixin, Output):
    @property
    def id(self) -> str:
        """Utility method to generate a hera/ArgoWorkflow parameter reference
        to be used when constructing the hera DAG.

        Returns:
            str: The hera parameter reference expression.
        """

        hera_expression = (
            "{{"
            + f"{self.owner.io_owner_name}.{self.type}.artifacts.{self.name}"
            + "}}"
        )

        return hera_expression

    def to_hera(self) -> Artifact:

        owner = getattr(self, "owner", None)
        owner_type = getattr(owner, "type", None)

        if owner_type == PIPELINE_TYPE:
            # OutputArtifacts dont reference anything, so its
            # enough to pass name and path
            raise NotImplementedError(
                "Pipeline owned OutputArtifacts are not" " supported yet."
            )
        elif owner_type == COMPONENT_TYPE:
            return Artifact(name=self.name, path=self.path)


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


def upload_dict_to_s3(data: Dict, s3_prefix: str):
    """Utility function to upload a dictionary as a json file to the specified
    prefix in the artifact repository S3 bucket.

    Args:
        data (Dict): The dictionary to be uploaded
        s3_prefix (str): The upload target s3 prefix
    """
    data_file = bytes(json.dumps(data).encode("UTF-8"))

    s3_client = get_s3_client()
    s3_client.upload_fileobj(
        data_file, S3_ARTIFACT_REPOSITORY_BUCKET, s3_prefix
    )


def download_dict_from_s3(s3_prefix: str) -> Dict:
    """Utility function to download a json as a dictionary from the specified
    prefix in the artifact repository S3 bucket.

    Args:
        s3_prefix (str): The s3 source prefix of the json file

    Returns:
        Dict: The content of the remote json file
    """

    s3_client = get_s3_client()
    s3_object = s3_client.get_object(
        Bucket=S3_ARTIFACT_REPOSITORY_BUCKET, Key=s3_prefix
    )
    data_json_str = s3_object["Body"].read().decode()
    data_dict = json.loads(data_json_str)

    return data_dict


def get_artifact_repository_prefix(
    s3_file_name: str, workflow_id: Optional[str] = None
) -> str:
    """Returns the artifact repository S3 prefix for the given filename.

    Args:
        s3_file_name (str): The name of the S3 artifact to be created

    Raises:
        e: A ModuleNotFoundError if the boto3 module is missing

    Returns:
        (str): The s3 prefix
    """

    # get workflow id from argo context variable - see
    # https://argo-workflows.readthedocs.io/en/latest/variables/#global
    if workflow_id is None:
        workflow_id = json.loads(r"""{{workflow.uid}}""")

    return f"{S3_ARTIFACT_REPOSITORY_PREFIX}/{workflow_id}/{DDP_TASK_ALIAS}/{s3_file_name}"  # noqa: E501


def upload_artifact_to_s3(
    artifact_name: str, workflow_id: Optional[str] = None
):
    """Utility to upload a TorchDDPComponent's specified input artifact to S3.
    Useful for the IO pre adapter task that exposes argo inputs to the DDP
    jobset processes via S3

    Args:
        artifact_name (str): The name of the input artifact to be uploaded.
    """

    s3_client = get_s3_client()
    artifact_s3_prefix = get_artifact_repository_prefix(
        artifact_name, workflow_id
    )
    s3_client.upload_file(
        artifact_name, S3_ARTIFACT_REPOSITORY_BUCKET, artifact_s3_prefix
    )


def download_artifact_from_s3(
    artifact_name: str, workflow_id: Optional[str] = None
):
    """Utility to download a TorchDDPComponent's specified output artifact from
    S3. Useful for the IO post adapter task that exposes the DDP jobset outputs
    to argo output artifacts

    Args:
        artifact_name (str): The name of the output artifact to be downloaded.
    """

    s3_client = get_s3_client()
    artifact_s3_prefix = get_artifact_repository_prefix(
        artifact_name, workflow_id
    )
    s3_client.download_file(
        S3_ARTIFACT_REPOSITORY_BUCKET, artifact_s3_prefix, artifact_name
    )


def hera_input_parameters_to_s3():
    """Utility to upload a TorchDDPComponent's input parameters to S3. Useful
    for the IO pre adapter task that exposes argo inputs to the DDP jobset
    processes via S3

    Raises:
        e: A ModuleNotFoundError if the boto3 module is missing
    """

    # get input parameter values from argo context variable - see
    # https://argo-workflows.readthedocs.io/en/latest/variables/#all-templates
    input_parameters_dict = json.loads(r"""{{inputs.parameters}}""")
    s3_prefix = get_artifact_repository_prefix("input_parameters.json")
    upload_dict_to_s3(input_parameters_dict, s3_prefix)


def download_container_inputs_from_s3(
    workflow_id: str, artifact_names: Tuple[str]
) -> Dict[str, Any]:
    """Utility to download input parameters and artifacts from S3. Useful for
    the TorchDDPComponent's main containers running outside the ArgoWorkflow
    context (requires that the IO pre adapter task has completed).

    Args:
        workflow_id (str): The unique id of the ArgoWorkflow that created the
        external container/K8s JobSet

    Returns:
        Dict[str,Any]: The input parameters
    """

    s3_prefix = get_artifact_repository_prefix("input_parameters.json")
    input_parameters_dict = download_dict_from_s3(s3_prefix)

    # artifacts
    [
        download_artifact_from_s3(artifact_name, workflow_id)
        for artifact_name in artifact_names
    ]

    # parameters

    return input_parameters_dict


def upload_container_outputs_to_s3(
    workflow_id: str,
    output_parameters: Dict[str, Any],
    artifact_names: Tuple[str],
):

    # parameters
    s3_prefix = get_artifact_repository_prefix(
        "output_parameters.json", workflow_id
    )
    upload_dict_to_s3(output_parameters, s3_prefix)

    # artifacts
    [
        upload_artifact_to_s3(artifact_name, workflow_id)
        for artifact_name in artifact_names
    ]


def s3_to_hera_output_parameters():
    """Utility to download a TorchDDPComponent's output parameters from S3.
    Useful for the IO post adapter task that exposes the DDP jobset outputs to
    argo output parameters
    """

    s3_prefix = get_artifact_repository_prefix("output_parameters.json")
    input_parameters_dict = download_dict_from_s3(s3_prefix)

    for (
        input_parameter_name,
        input_parameter_value,
    ) in input_parameters_dict.items():
        InputParameter(
            name=input_parameter_name, value=input_parameter_value
        ).export()
