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
    s3_prefix: Optional[str] = None


class FlowNodeInputs(BaseModel):
    parameters: Optional[List[FlowNodeParameterInput]] = None
    artifacts: Optional[List[FlowNodeArtifactInput]] = None


# outputs
class FlowNodeParameterOutput(NodeParameterOutput):
    value: Optional[str] = None


class FlowNodeArtifactOutput(NodeArtifactOutput):
    s3_prefix: Optional[str] = None


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
    def build_dag(cls, workflow_status: WorkflowStatusModel) -> List[FlowNode]:
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
        workflow_nodes_dict = cls.get_node_status_by_display_name(
            workflow_status
        )
        # workflow_nodes = list(workflow_status.nodes.values())

        # print(
        #     f"Workflow node display names:{[wn['display_name'] for wn in workflow_nodes]}"  # noqa: E501
        # )

        for pipeline_node in pipeline_dag:
            # pipeline_node_dict = pipeline_node.model_dump()
            # print(f"Pipeline node name: {pipeline_node_dict['name']}")

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
                print("Exception")
                flow_node_dict["pod_name"] = pipeline_node.name
                flow_node_dict["phase"] = "Not Scheduled"
                flow_node_dict["outputs"] = FlowNodeOutputs(
                    parameters=pipeline_node.outputs.parameters,
                    artifacts=pipeline_node.outputs.artifacts,
                ).model_dump()
                flow_node_dict["logs"] = None
            else:
                print("No Exception")
                flow_node_dict["id"] = workflow_node_dict["id"]
                flow_node_dict["type"] = workflow_node_dict["type"]
                flow_node_dict["pod_name"] = workflow_node_dict["name"]
                flow_node_dict["phase"] = workflow_node_dict["phase"]
                flow_node_dict["outputs"] = FlowNodeOutputs(
                    parameters=pipeline_node.outputs.parameters,
                    artifacts=pipeline_node.outputs.artifacts,
                    exit_code=workflow_node_dict["exit_code"],
                ).model_dump()
                flow_node_dict["logs"] = None
                flow_node_dict["dependants"] = workflow_node_dict["children"]
                flow_node_dict["host_node_name"] = workflow_node_dict[
                    "host_node_name"
                ]

                # inject resolved input values where possible
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
                                                    "s3_prefix"
                                                ] = argument["s3"]["key"]
                                    elif argument["name"] == "main-logs":
                                        flow_node_dict["logs"] = argument
                                    else:
                                        pass
                        except KeyError:
                            pass
            finally:
                print(f"Flow node dict:{flow_node_dict}")
                flow_node = FlowNode(**flow_node_dict)
                print(f"Flow node:{flow_node}")
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
        inputs, outputs = cls.build_io(
            workflow_status.stored_workflow_template_spec
        )

        # dag
        dag = cls.build_dag(workflow_status.status)

        return cls(
            metadata=metadata,
            state=state,
            artifact_configuration=artifact_configuration,
            templates=templates,
            inputs=inputs,
            outputs=outputs,
            dag=dag,
        )
