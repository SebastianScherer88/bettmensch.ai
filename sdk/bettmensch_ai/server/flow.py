from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from bettmensch_ai.server.pipeline import (
    NodeArtifactInput,
    NodeArtifactOutput,
    NodeParameterInput,
    NodeParameterOutput,
    Pipeline,
    PipelineNode,
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
        pipeline_inputs, pipeline_outputs = super().build_io(
            workflow_template_spec
        )

        # --- inputs
        flow_inputs = {
            "parameters": [],
        }

        # parameters
        for ppi in pipeline_inputs.parameters:
            flow_parameter_input_data = ppi.model_dump()
            try:
                fpi_value = [
                    fpi
                    for fpi in inner_dag_node.inputs.parameters
                    if fpi.name == ppi.name
                ][0].value
            except (AttributeError, IndexError):
                fpi_value = None
            finally:
                flow_parameter_input_data["value"] = fpi_value
                flow_parameter_input = FlowParameterInput.model_validate(
                    flow_parameter_input_data
                )
                flow_inputs["parameters"].append(flow_parameter_input)

        # --- outputs
        flow_outputs = {"parameters": [], "artifacts": []}

        # parameters
        for ppo in pipeline_outputs.parameters:
            flow_parameter_output_data = ppo.model_dump()
            try:
                fpo_value = [
                    fpo
                    for fpo in inner_dag_node.outputs.parameters
                    if fpo.name == ppo.name
                ][0].value
            except (AttributeError, IndexError):
                fpo_value = None
            finally:
                flow_parameter_output_data["value"] = fpo_value
                flow_parameter_output = FlowParameterOutput.model_validate(
                    flow_parameter_output_data
                )
                flow_outputs["parameters"].append(flow_parameter_output)

        # artifacts
        for pao in pipeline_outputs.artifacts:
            flow_artifact_output_data = pao.model_dump()
            try:
                fao_s3 = [
                    fao
                    for fao in inner_dag_node.outputs.artifacts
                    if fao.name == pao.name
                ][0].s3.dict()
            except (AttributeError, IndexError):
                fao_s3 = None
            finally:
                flow_artifact_output_data["s3"] = fao_s3
                flow_parameter_output = FlowArtifactOutput.model_validate(
                    flow_artifact_output_data
                )
                flow_outputs["artifacts"].append(flow_parameter_output)

        flow_outputs = FlowOutputs.model_validate(flow_outputs)

        return flow_inputs, flow_outputs

    @classmethod
    def build_flow_node_io_parameter(
        cls, pipeline_node_io: Union[NodeParameterInput, NodeParameterOutput]
    ):
        pass

    @classmethod
    def build_flow_node_io_artifact(
        cls, pipeline_node_io: Union[NodeArtifactInput, NodeArtifactOutput]
    ):
        pass

    @classmethod
    def build_flow_node(
        cls,
        pipeline_node: PipelineNode,
        workflow_nodes_dict: Dict[str, NodeStatusModel],
    ) -> FlowNode:
        """Builds a FlowNode

        Args:
            pipeline_node (PipelineNode): _description_
            workflow_nodes_dict (Dict[str,NodeStatusModel]): _description_

        Returns:
            FlowNode: _description_
        """

        flow_node_dict = {
            "name": pipeline_node.name,
            "template": pipeline_node.template,
            "inputs": pipeline_node.inputs.model_dump(),
            "depends": pipeline_node.depends,
        }

        try:
            workflow_node_dict = workflow_nodes_dict[pipeline_node.name].dict()
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
                                            flow_node_arguments[i]["s3"] = {
                                                "key": argument["s3"]["key"],
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

        return flow_node

    @classmethod
    def build_dag(
        cls,
        workflow_status: WorkflowStatusModel,
        workflow_nodes_dict: Dict[str, NodeStatusModel],
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
            flow_node = cls.build_flow_node(pipeline_node, workflow_nodes_dict)
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
