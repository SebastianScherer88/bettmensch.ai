from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

from bettmensch_ai.server.pipeline import (
    NodeInput,
    NodeOutput,
    Pipeline,
    PipelineInputParameter,
    ResourceTemplate,
    ScriptTemplate,
)
from bettmensch_ai.utils import copy_non_null_dict
from hera.workflows.models import Workflow as WorkflowModel
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
    conditions: List[Dict[str, Optional[str]]]
    resources_duration: Optional[Dict]
    task_results_completion_status: Dict[str, bool]


# --- FlowInput
class FlowInputParameter(PipelineInputParameter):
    pass


# --- FlowNode
# inputs
class FlowNodeParameterInput(NodeInput):
    value: Optional[str] = None
    value_from: Optional[Union[str, Dict]] = None


class FlowNodeArtifactInput(NodeInput):
    s3_prefix: Optional[str] = None


class FlowNodeInputs(BaseModel):
    parameters: Optional[List[FlowNodeParameterInput]] = None
    artifacts: Optional[List[FlowNodeArtifactInput]] = None


# outputs
class FlowNodeParameterOutput(NodeOutput):
    value: Optional[str] = None
    value_from: Optional[Union[str, Dict]] = None


class FlowNodeArtifactOutput(NodeOutput):
    path: str
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
class Flow(Pipeline):
    """A flow node is an instantiated pipeline node on Kubernetes."""

    metadata: FlowMetadata
    state: FlowState
    artifact_configuration: FlowArtifactConfiguration
    templates: List[Union[ScriptTemplate, ResourceTemplate]]
    inputs: List[FlowInputParameter] = []
    dag: List[FlowNode]

    @classmethod
    def build_dag(cls, workflow_status: Dict) -> List[FlowNode]:
        """Utility to build the Flow class' dag attribute. Identical to the
        Pipeline class' dag attribute, but with additional values resolved at
        runtime.

        Args:
            workflow_status (_type_): The status field of a dict-ified
                IoArgoprojWorkflowV1alpha1Workflow class instance

        Returns:
            List[FlowNode]: The constructed dag attribute of a Flow instance.
        """

        # build pipeline dag
        workflow_template_spec = workflow_status[
            "stored_workflow_template_spec"
        ]
        pipeline_dag = super().build_dag(workflow_template_spec)

        # add FlowNode specific values and available resolved input/output
        # values for each FlowNode
        flow_dag = []
        workflow_nodes = list(workflow_status["nodes"].values())

        print(
            f"Workflow node display names:{[wn['display_name'] for wn in workflow_nodes]}"  # noqa: E501
        )

        for pipeline_node in pipeline_dag:
            pipeline_node_dict = pipeline_node.model_dump()
            print(f"Pipeline node name: {pipeline_node_dict['name']}")

            flow_node_dict = {
                "name": pipeline_node_dict["name"],
                "template": pipeline_node_dict["template"],
                "inputs": pipeline_node_dict["inputs"],
                "depends": pipeline_node_dict["depends"],
            }

            potential_workflow_node_dict = [
                workflow_node
                for workflow_node in workflow_nodes
                if workflow_node["display_name"] == pipeline_node_dict["name"]
            ]

            if potential_workflow_node_dict:
                workflow_node_dict = copy_non_null_dict(
                    potential_workflow_node_dict[0]
                )

                flow_node_dict["id"] = workflow_node_dict["id"]
                flow_node_dict["type"] = workflow_node_dict["type"]
                flow_node_dict["pod_name"] = workflow_node_dict["name"]
                flow_node_dict["phase"] = workflow_node_dict["phase"]
                flow_node_dict["outputs"] = dict(
                    **pipeline_node_dict["outputs"],
                    **{"exit_code": workflow_node_dict.get("exit_code")},
                )
                flow_node_dict["logs"] = None
                flow_node_dict["dependants"] = workflow_node_dict.get(
                    "children"
                )
                flow_node_dict["host_node_name"] = workflow_node_dict.get(
                    "host_node_name"
                )

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

            else:
                flow_node_dict["pod_name"] = pipeline_node_dict["name"]
                flow_node_dict["phase"] = "Not Scheduled"
                flow_node_dict["outputs"] = dict(
                    **pipeline_node_dict["outputs"],
                )
                flow_node_dict["logs"] = None
            try:
                flow_node = FlowNode(**flow_node_dict)
            except Exception as e:
                raise (e)

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

        workflow_dict = workflow_model.dict()
        workflow_spec = workflow_dict["spec"].copy()
        workflow_status = workflow_dict["status"].copy()
        workflow_template_spec = workflow_status[
            "stored_workflow_template_spec"
        ].copy()

        # metadata
        metadata = FlowMetadata(**workflow_dict["metadata"])

        # state
        state = FlowState(**workflow_status)

        # artifact_configuration
        artifact_configuration = FlowArtifactConfiguration(
            repository_ref=workflow_status["artifact_gc_status"],
            gc_status=workflow_status["artifact_repository_ref"],
        )

        # templates
        entrypoint_template = workflow_template_spec["entrypoint"]
        templates = []
        for template in workflow_template_spec["templates"]:
            # we are not interested in the entrypoint template
            if template["name"] == entrypoint_template:
                continue
            elif template["script"] is not None:
                templates.append(ScriptTemplate.model_validate(template))
            elif template["resource"] is not None:
                templates.append(ResourceTemplate.model_validate(template))

        # inputs
        inputs = [
            FlowInputParameter(**parameter)
            for parameter in workflow_spec["arguments"]["parameters"]
        ]

        # dag
        dag = cls.build_dag(workflow_status)

        return cls(
            metadata=metadata,
            state=state,
            artifact_configuration=artifact_configuration,
            templates=templates,
            inputs=inputs,
            dag=dag,
        )
