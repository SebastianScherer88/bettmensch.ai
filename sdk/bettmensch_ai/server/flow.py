from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Union

import argo_workflows
import yaml
from argo_workflows.api import workflow_service_api
from argo_workflows.model.io_argoproj_workflow_v1alpha1_workflow import (
    IoArgoprojWorkflowV1alpha1Workflow,
)
from bettmensch_ai.server.dag import (
    DagConnection,
    DagNode,
    DagVisualizationSchema,
)
from bettmensch_ai.server.pipeline import (
    NodeInput,
    NodeOutput,
    Pipeline,
    PipelineInputParameter,
    ScriptTemplate,
)
from bettmensch_ai.server.utils import PIPELINE_NODE_EMOJI_MAP
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
    phase: Literal["Succeeded", "Failed", "Pending", "Error"]
    started_at: Optional[Union[datetime, None]] = None
    finshed_at: Optional[Union[datetime, None]] = None
    progress: str
    conditions: List[Dict[str, str]]
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
    s3_prefix: str


class FlowNodeInputs(BaseModel):
    parameters: Optional[List[FlowNodeParameterInput]] = None
    artifacts: Optional[List[FlowNodeArtifactInput]] = None


# outputs
class FlowNodeParameterOutput(NodeOutput):
    value: Optional[str] = None
    value_from: Optional[Union[str, Dict]] = None


class FlowNodeArtifactOutput(NodeOutput):
    path: str
    s3_prefix: str


class FlowNodeOutputs(BaseModel):
    parameters: Optional[List[FlowNodeParameterOutput]] = None
    artifacts: Optional[List[FlowNodeArtifactOutput]] = None
    exit_code: Optional[int] = None


class FlowNode(BaseModel):
    id: str
    name: str
    type: Literal["Pod", "Skipped"]
    pod_name: str  # this will match the PipelineNode.name, i.e the task name
    template: str
    phase: Literal["Succeeded", "Failed", "Pending", "Error", "Omitted"]
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
class Flow(BaseModel):
    """A flow node is an instantiated pipeline node on Kubernetes."""

    metadata: FlowMetadata
    state: FlowState
    artifact_configuration: FlowArtifactConfiguration
    templates: List[ScriptTemplate]
    inputs: List[FlowInputParameter] = []
    dag: List[FlowNode]

    def get_template(self, name: str) -> ScriptTemplate:

        return [
            template for template in self.templates if template.name == name
        ][0]

    def get_dag_task(self, name: str) -> FlowNode:

        return [task for task in self.dag if task.name == name][0]

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
        pipeline_dag = Pipeline.build_dag(workflow_template_spec)

        # add FlowNode specific values and available resolved input/output
        # values for each FlowNode
        flow_dag = []
        workflow_nodes = list(workflow_status["nodes"].values())

        for pipeline_node in pipeline_dag:
            pipeline_node_dict = pipeline_node.model_dump()
            workflow_node_dict = [
                workflow_node
                for workflow_node in workflow_nodes
                if workflow_node["display_name"] == pipeline_node_dict["name"]
            ][0]

            flow_node_dict = {
                "id": workflow_node_dict["id"],
                "name": workflow_node_dict["display_name"],
                "type": workflow_node_dict["type"],
                "pod_name": workflow_node_dict["name"],
                "template": workflow_node_dict["template_name"],
                "phase": workflow_node_dict["phase"],
                "inputs": pipeline_node_dict["inputs"],
                "outputs": dict(
                    **pipeline_node_dict["outputs"],
                    **{"exit_code": workflow_node_dict.get("exit_code")},
                ),
                "logs": None,
                "depends": pipeline_node_dict["depends"],
                "dependants": workflow_node_dict.get("children"),
                "host_node_name": workflow_node_dict.get("host_node_name"),
            }
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

                        for i, argument in enumerate(workflow_node_arguments):
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
            try:
                flow_node = FlowNode(**flow_node_dict)
            except Exception as e:
                raise (e)

            flow_dag.append(flow_node)

        return flow_dag

    @classmethod
    def from_argo_workflow_cr(
        cls, workflow_resource: IoArgoprojWorkflowV1alpha1Workflow
    ) -> Flow:
        """Utility to generate a Flow instance from a
        IoArgoprojWorkflowV1alpha1Workflow instance.

        To be used to easily convert the API response data structure
        to the bettmensch.ai pipeline data structure optimized for visualizing
        the DAG.

        Args:
            workflow_resource (IoArgoprojWorkflowV1alpha1Workflow): The return
                of ArgoWorkflow's api/v1/workflow/{namespace}/{name} endpoint.

        Returns:
            Flow: A Flow class instance.
        """

        workflow_dict = workflow_resource.to_dict()
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
        templates = [
            ScriptTemplate.model_validate(template)
            for template in workflow_template_spec["templates"]
            if template["name"] != entrypoint_template
        ]

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

    def create_dag_visualization_schema(
        self, include_task_io: bool = True
    ) -> DagVisualizationSchema:
        """Utility method to generate the assets the barfi/baklavajs rendering
        engine uses to display the Pipeline's dag property on the frontend."""

        node_positions = Pipeline.create_dag_visualization_node_positions(
            self.inputs, self.dag, include_task_io
        )
        connections: List[Dict] = []
        nodes: List[Dict] = []

        for task_node in self.dag:

            task_node_name = task_node.name

            nodes.append(
                DagNode(
                    id=task_node_name,
                    pos=node_positions[task_node_name],
                    data={
                        "label": f"{PIPELINE_NODE_EMOJI_MAP['task']} {task_node_name}"
                    },
                )
            )

            # we only create task_node <-> task_node connections if we dont
            # display the tasks' IO specs
            if not include_task_io:
                if task_node.depends is not None:
                    for upstream_node_name in task_node.depends:
                        connections.append(
                            DagConnection(
                                id=f"{upstream_node_name}->{task_node_name}",
                                source=upstream_node_name,
                                target=task_node_name,
                                animated=True,
                                edge_type="smoothstep",
                            )
                        )
            # if we include the tasks' IO specs, we need to draw
            # - io nodes and
            # connections between
            # - inputs and outputs, and
            # - inputs/outputs and associated task_nodes
            else:
                for interface_type in ["inputs", "outputs"]:
                    interfaces = getattr(task_node, interface_type)

                    if interfaces is None:
                        continue

                    for argument_type in ["parameters", "artifacts"]:
                        arguments = getattr(interfaces, argument_type, None)

                        if arguments is None:
                            continue

                        for argument in arguments:
                            # add the task io node
                            task_io_node_name = f"{task_node_name}_{interface_type}_{argument_type}_{argument.name}"
                            nodes.append(
                                DagNode(
                                    id=task_io_node_name,
                                    pos=node_positions[task_io_node_name],
                                    data={
                                        "label": f"{PIPELINE_NODE_EMOJI_MAP[interface_type]['task']} {argument.name}",
                                        "value": getattr(
                                            argument, "value", None
                                        ),
                                    },
                                    style={"backgroundColor": "lightgrey"},
                                )
                            )

                            # connect that task io node with the task node
                            if interface_type == "inputs":
                                upstream_node_name = task_io_node_name
                                node_name = task_node_name
                            else:
                                upstream_node_name = task_node_name
                                node_name = task_io_node_name

                            connections.append(
                                DagConnection(
                                    id=f"{upstream_node_name}->{node_name}",
                                    source=upstream_node_name,
                                    target=node_name,
                                    animated=False,
                                    edge_type="smoothstep",
                                )
                            )

                            # connect the input type task io node with the
                            # upstream output type task io node - where
                            # appropriate
                            if (
                                interface_type == "inputs"
                                and getattr(argument, "source", None)
                                is not None
                            ):
                                task_io_source = argument.source
                                upstream_node_name = f"{task_io_source.node}_outputs_{task_io_source.output_type}_{task_io_source.output_name}"
                                connections.append(
                                    DagConnection(
                                        id=f"{upstream_node_name}->{task_io_node_name}",
                                        source=upstream_node_name,
                                        target=task_io_node_name,
                                        animated=True,
                                        edge_type="smoothstep",
                                    )
                                )

        if include_task_io:
            for input in self.inputs:
                node_name = f"pipeline_outputs_parameters_{input.name}"
                nodes.append(
                    DagNode(
                        id=node_name,
                        pos=node_positions[node_name],
                        data={
                            "label": f"{PIPELINE_NODE_EMOJI_MAP['inputs']['pipeline']} {input.name}",
                            "value": input.value,
                        },
                        style={"backgroundColor": "lightblue"},
                        node_type="input",
                    )
                )

        return DagVisualizationSchema(connections=connections, nodes=nodes)


def main_test():
    """Unit test the Pipeline class."""

    # get a sample pipeline from the ArgoWorkflow server
    configuration = argo_workflows.Configuration(host="https://127.0.0.1:2746")
    configuration.verify_ssl = False

    api_client = argo_workflows.ApiClient(configuration)
    api_instance = workflow_service_api.WorkflowServiceApi(api_client)

    workflows = api_instance.list_workflows(namespace="argo")["items"]
    print(
        f"Registered flows: {[workflow['metadata']['name'] for workflow in workflows]}"
    )
    workflow = workflows[0]
    print(f"Workflow: {workflow.to_dict()}")

    with open("flow_test_workflow.yaml", "w") as file:
        yaml.dump(workflow.to_dict(), file)

    # convert to pipeline and show results
    flow = Flow.from_argo_workflow_cr(workflow)
    print(f"Flow: {flow.model_dump()}")

    with open("flow_test_flow.yaml", "w") as file:
        yaml.dump(flow.model_dump(), file)


if __name__ == "__main__":
    main_test()
