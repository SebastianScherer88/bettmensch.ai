from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from bettmensch_ai.server.pipeline import (
    NodeArtifactInput,
    NodeArtifactOutput,
    NodeParameterInput,
    NodeParameterOutput,
    Pipeline,
    PipelineParameterInput,
    ResourceTemplate,
    ScriptTemplate,
)
from bettmensch_ai.server.utils import copy_non_null_dict
from hera.workflows.models import NodeStatus as NodeStatusModel
from hera.workflows.models import Workflow as WorkflowModel
from hera.workflows.models import WorkflowSpec as WorkflowSpecModel
from hera.workflows.models import WorkflowStatus as WorkflowStatusModel
from pydantic import BaseModel


# --- FlowMetadata
class FlowMetadata(BaseModel):
    uid: str
    name: str
    namespace: str = "default"
    creation_timestamp: datetime
    labels: Dict[str, str]
    annotations: Optional[Dict[str, str]]


# --- FlowStatus
class FlowState(BaseModel):
    phase: Literal["Pending", "Running", "Succeeded", "Failed", "Error"]
    started_at: Optional[Union[datetime, None]] = None
    finshed_at: Optional[Union[datetime, None]] = None
    progress: str
    conditions: Optional[List[Dict[str, Optional[str]]]] = None
    resources_duration: Optional[Dict]
    task_results_completion_status: Optional[Dict[str, bool]] = None


# --- FlowNode
# inputs
class FlowNodeParameterInput(NodeParameterInput):
    pass


class FlowNodeArtifactInput(NodeArtifactInput):
    s3: Optional[ArtifactS3] = None


class FlowNodeInputs(BaseModel):
    parameters: Optional[List[FlowNodeParameterInput]] = None
    artifacts: Optional[List[FlowNodeArtifactInput]] = None


# outputs
class FlowNodeParameterOutput(NodeParameterOutput):
    value: Optional[str] = None


class ArtifactS3(BaseModel):
    bucket: Optional[str] = None
    key: Optional[str] = None


class FlowNodeArtifactOutput(NodeArtifactOutput):
    s3: Optional[ArtifactS3] = None


class FlowNodeOutputs(BaseModel):
    parameters: Optional[List[FlowNodeParameterOutput]] = None
    artifacts: Optional[List[FlowNodeArtifactOutput]] = None
    exit_code: Optional[int] = None


class FlowNode(BaseModel):
    id: Optional[str] = None
    name: str
    type: Optional[Literal["Pod", "Skipped", "Retry"]] = None
    pod_name: str  # this will match the PipelineNode.name, i.e the task name
    template: str
    phase: Literal[
        "Not Scheduled",
        "Pending",
        "Running",
        "Succeeded",
        "Failed",
        "Error",
        "Omitted",
    ]
    template: str
    inputs: Optional[FlowNodeInputs] = None
    outputs: Optional[FlowNodeOutputs] = None
    logs: Optional[Dict] = None
    depends: Optional[Union[str, List[str]]] = None
    dependants: Optional[Union[str, List[str]]] = None
    host_node_name: Optional[str] = None

    @property
    def ios(
        self,
    ) -> List[
        Union[
            FlowNodeParameterInput,
            FlowNodeParameterOutput,
            FlowNodeArtifactInput,
            FlowNodeArtifactOutput,
        ]
    ]:
        return (
            self.inputs.parameters
            + self.outputs.parameters
            + self.inputs.artifacts
            + self.outputs.artifacts
        )


# --- FlowArtifactConfiguration
class FlowArtifactConfiguration(BaseModel):
    repository_ref: Dict
    gc_status: Dict


# --- Flow
# inputs
class FlowParameterInput(PipelineParameterInput):
    pass


class FlowInputs(BaseModel):
    parameters: List[FlowParameterInput] = []


# outputs
class FlowParameterOutput(FlowNodeParameterInput):
    type: str = "outputs"


class FlowArtifactOutput(FlowNodeArtifactInput):
    type: str = "outputs"


class FlowOutputs(BaseModel):
    parameters: List[FlowParameterOutput] = []
    artifacts: List[FlowArtifactOutput] = []


class Flow(Pipeline):
    """A flow node is an instantiated pipeline node on Kubernetes."""

    metadata: FlowMetadata
    state: FlowState
    artifact_configuration: FlowArtifactConfiguration
    templates: List[Union[ScriptTemplate, ResourceTemplate]]
    inputs: Optional[FlowInputs] = None
    outputs: Optional[FlowOutputs] = None
    dag: List[FlowNode]

    @classmethod
    def get_node_status_by_display_name(
        cls, workflow_status: WorkflowStatusModel
    ) -> Dict[str, NodeStatusModel]:
        """Generates a display_name -> NodeStatus dictionary from the passed
        workflow_status' `nodes` attribute (holds a node_id -> NodeStatusModel
        dictionary).

        Args:
            workflow_status (WorkflowStatusModel): The workflow status
                instance.

        Returns:
            Dict[str,NodeStatusModel]: The display_name -> NodeStatusModel
                dictionary
        """

        return dict(
            [
                (node_status.display_name, node_status)
                for node_status in workflow_status.nodes.values()
            ]
        )

    @classmethod
    def build_io(
        cls,
        workflow_template_spec: WorkflowSpecModel,
        workflow_nodes_dict: Dict[str, NodeStatusModel],
    ) -> Union[FlowInputs, FlowOutputs]:
        """Build the io attributes of a Flow instance.

        Args:
            inner_dag_node (NodeStatusModel): The NodeStatus of the Flow's
                Workflow's inner dag.


        Returns:
            Union[FlowInputs,FlowOutputs]: The data for the Flow instance's
                `inputs` and `outputs` attributes
        """

        inner_dag_node = workflow_nodes_dict["bettmensch-ai-inner-dag"]

        # add flow inputs directly from inner dag node status
        if inner_dag_node.inputs is not None:
            if inner_dag_node.inputs.parameters is not None:
                flow_inputs = FlowInputs(
                    parameters=[
                        FlowParameterInput.model_validate(
                            dag_input_parameter.dict()
                        )
                        for dag_input_parameter in inner_dag_node.inputs.parameters  # noqa: E501
                    ]
                )
            else:
                flow_inputs = FlowInputs(parameters=[], artifacts=[])

        # add flow inputs directly from inner dag node status
        flow_outputs = {"parameters": [], "artifacts": []}
        _, pipeline_outputs = super().build_io(workflow_template_spec)

        if inner_dag_node.outputs is not None:
            # output parameters
            output_parameters = inner_dag_node.outputs.parameters
            if output_parameters is not None:
                for output_parameter in output_parameters:
                    # grab source attribute from pipeline output parameter
                    pipeline_output_parameter_source = [
                        pop
                        for pop in pipeline_outputs.parameters
                        if pop.name == output_parameter.name
                    ][0].source

                    flow_output_parameter = FlowParameterOutput(
                        source=pipeline_output_parameter_source,
                        **output_parameter.dict(),
                    )
                    flow_outputs["parameters"].append(flow_output_parameter)

            # output artifacts
            output_artifacts = inner_dag_node.outputs.artifacts
            if output_artifacts is not None:
                for output_artifact in output_artifacts:
                    # grab source attribute from pipeline output parameter
                    pipeline_output_artifact_source = [
                        poa
                        for poa in pipeline_outputs.artifacts
                        if poa.name == output_artifact.name
                    ][0].source

                    flow_output_artifact = FlowArtifactOutput(
                        source=pipeline_output_artifact_source,
                        **output_artifact.dict(),
                    )
                    flow_outputs["artifacts"].append(flow_output_artifact)

        flow_outputs = FlowOutputs.model_validate(flow_outputs)

        return flow_inputs, flow_outputs

    @classmethod
    def build_dag(
        cls,
        workflow_status: WorkflowStatusModel,
        workflow_nodes_dict: NodeStatusModel,
    ) -> List[FlowNode]:
        """Utility to build the Flow class' dag attribute. Identical to the
        Pipeline class' dag attribute, but with additional values resolved at
        runtime.

        Args:
            workflow_status (WorkflowStatusModel): The status field of a hera
                Workflow model instance.

        Returns:
            List[FlowNode]: The constructed dag attribute of a Flow instance.
        """

        # build pipeline dag
        pipeline_dag = super().build_dag(
            workflow_status.stored_workflow_template_spec
        )

        # add FlowNode specific values and available resolved input/output
        # values for each FlowNode
        flow_dag = []

        for pipeline_node in pipeline_dag:

            flow_node_dict = {
                "name": pipeline_node.name,
                "template": pipeline_node.template,
                "inputs": pipeline_node.inputs.model_dump(),
                "depends": pipeline_node.depends,
            }

            try:
                workflow_node_dict = workflow_nodes_dict[
                    pipeline_node.name
                ].dict()
                workflow_node_dict = copy_non_null_dict(workflow_node_dict)
            except KeyError:
                flow_node_dict["pod_name"] = pipeline_node.name
                flow_node_dict["phase"] = "Not Scheduled"
                flow_node_dict["outputs"] = FlowNodeOutputs(
                    **pipeline_node.outputs.model_dump(),
                ).model_dump()
                flow_node_dict["logs"] = None
            else:
                flow_node_dict["id"] = workflow_node_dict["id"]
                flow_node_dict["type"] = workflow_node_dict["type"]
                flow_node_dict["pod_name"] = workflow_node_dict["name"]
                flow_node_dict["phase"] = workflow_node_dict["phase"]
                flow_node_dict["outputs"] = FlowNodeOutputs(
                    exit_code=workflow_node_dict.get("exit_code", None),
                    **pipeline_node.outputs.model_dump(),
                ).model_dump()
                flow_node_dict["dependants"] = workflow_node_dict.get(
                    "children", None
                )
                flow_node_dict["host_node_name"] = workflow_node_dict.get(
                    "host_node_name", None
                )

                # inject resolved input/output values where possible
                for argument_io in ("inputs", "outputs"):
                    for argument_type in ("parameters", "artifacts"):
                        try:
                            workflow_node_arguments = workflow_node_dict[
                                argument_io
                            ][argument_type]
                            flow_node_arguments = flow_node_dict[argument_io][
                                argument_type
                            ]

                            if workflow_node_arguments is None:
                                continue
                            else:
                                for i, argument in enumerate(
                                    workflow_node_arguments
                                ):
                                    if i < len(flow_node_arguments):
                                        if (
                                            flow_node_arguments[i]["name"]
                                            == argument["name"]
                                        ):
                                            if argument_type == "parameters":
                                                flow_node_arguments[i][
                                                    "value"
                                                ] = argument["value"]
                                            elif argument_type == "artifacts":
                                                flow_node_arguments[i][
                                                    "s3"
                                                ] = {
                                                    "key": argument["s3"][
                                                        "key"
                                                    ],
                                                    "bucket": argument["s3"][
                                                        "bucket"
                                                    ],
                                                }
                                    elif argument["name"] == "main-logs":
                                        flow_node_dict["logs"] = argument
                                    else:
                                        pass
                        except KeyError:
                            pass
            finally:
                flow_node = FlowNode(**flow_node_dict)
                flow_dag.append(flow_node)

        return flow_dag

    @classmethod
    def from_hera_workflow_model(cls, workflow_model: WorkflowModel) -> Flow:
        """Utility to generate a Flow instance from a
        hera.workflows.models.Workflow instance.

        To be used to easily convert the API response data structure
        to the bettmensch.ai pipeline data structure optimized for visualizing
        the DAG.

        Args:
            workflow_model (hera.workflows.models.Workflow): Instance of hera's
                WorkflowService response model.

        Returns:
            Flow: A Flow class instance.
        """
        workflow_status = workflow_model.status

        # metadata
        metadata = FlowMetadata(**workflow_model.metadata.dict())

        # state
        state = FlowState(**workflow_status.dict())

        # artifact_configuration
        artifact_configuration = FlowArtifactConfiguration(
            repository_ref=workflow_status.artifact_gc_status.dict(),
            gc_status=workflow_status.artifact_repository_ref.dict(),
        )

        # templates
        templates = []
        for (
            template
        ) in workflow_status.stored_workflow_template_spec.templates:
            # we are only interested in Script and Resource type templates
            if template.script is not None:
                templates.append(
                    ScriptTemplate.model_validate(template.dict())
                )
            elif template.resource is not None:
                templates.append(
                    ResourceTemplate.model_validate(template.dict())
                )

        # io
        workflow_nodes_dict = cls.get_node_status_by_display_name(
            workflow_status
        )
        inputs, outputs = cls.build_io(
            workflow_status.stored_workflow_template_spec, workflow_nodes_dict
        )

        # dag
        dag = cls.build_dag(workflow_status, workflow_nodes_dict)

        return cls(
            metadata=metadata,
            state=state,
            artifact_configuration=artifact_configuration,
            templates=templates,
            inputs=inputs,
            outputs=outputs,
            dag=dag,
        )
