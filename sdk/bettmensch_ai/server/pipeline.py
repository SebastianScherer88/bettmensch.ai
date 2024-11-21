from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from bettmensch_ai.server.dag import (
    DagConnection,
    DagPipelineIONode,
    DagTaskIONode,
    DagTaskNode,
    DagVisualizationItems,
)
from bettmensch_ai.server.utils import copy_non_null_dict
from hera.workflows.models import WorkflowTemplate as WorkflowTemplateModel
from pydantic import BaseModel

PIPELINE_NODE_EMOJI_MAP = {
    "task": "🔵",  # :large_blue_circle:
    "inputs": {
        "task": "⤵️",  # :arrow_heading_down:
        "pipeline": "⏬",  # :arrow_double_down:
    },
    "outputs": {"task": "↪️"},  # :arrow_right_hook:
    "parameters": "📃",  # :page_with_curl
    "artifacts": "📂",  # :open_file_folder:
}


# --- PipelineMetadata
class WorkflowTemplateMetadata(BaseModel):
    uid: str
    name: str
    namespace: str = "default"
    creation_timestamp: datetime
    labels: Dict[str, str]


class PipelineMetadata(BaseModel):
    pipeline: WorkflowTemplateMetadata
    flow: Optional[Dict] = None
    component: Optional[Dict] = None


# --- ScriptTemplate
# script
class Script(BaseModel):
    image: str
    source: str
    name: str
    command: List[str]
    resources: Optional[Dict] = None
    env: Optional[List[Dict]] = None
    ports: Optional[List[Dict]] = None


# inputs
class ScriptTemplateParameterInput(BaseModel):
    name: str
    value: Optional[str] = None
    value_from: Optional[Union[str, Dict]] = None
    default: Optional[Any] = None


class ScriptTemplateArtifactInput(BaseModel):
    name: str
    path: Optional[str] = None


class ScriptTemplateInputs(BaseModel):
    parameters: Optional[List[ScriptTemplateParameterInput]] = None
    artifacts: Optional[List[ScriptTemplateArtifactInput]] = None


# outputs
class ScriptTemplateParameterOutput(BaseModel):
    name: str
    value_from: Optional[Union[str, Dict]] = None


class ScriptTemplateArtifactOutput(BaseModel):
    name: str
    path: Optional[str] = None
    value_from: Optional[Union[str, Dict]] = None


class ScriptTemplateOutputs(BaseModel):
    parameters: Optional[List[ScriptTemplateParameterOutput]] = None
    artifacts: Optional[List[ScriptTemplateArtifactOutput]] = None


class ScriptTemplate(BaseModel):
    name: str
    inputs: ScriptTemplateInputs = {}
    outputs: ScriptTemplateOutputs = {}
    metadata: Dict = {}
    script: Script
    tolerations: Optional[List[Dict]] = None


class Resource(BaseModel):
    action: Literal["create", "delete"]
    manifest: Optional[str]
    flags: Optional[List[str]]


class ResourceTemplate(BaseModel):
    name: str
    resource: Resource


# --- PipelineInput
class PipelineInputParameter(BaseModel):
    name: str
    value: Optional[str] = None
    default: Optional[Any] = None


# --- PipelineNode
# inputs
class NodeInputSource(BaseModel):
    node: str
    output_name: str
    output_type: Literal["parameters", "artifacts"]


class NodeInput(BaseModel):
    name: str
    source: Optional[NodeInputSource] = None


class NodeParameterInput(NodeInput):
    value: Optional[str] = None
    value_from: Optional[Union[str, Dict]] = None


class NodeArtifactInput(NodeInput):
    pass


class NodeInputs(BaseModel):
    parameters: List[NodeParameterInput] = []
    artifacts: List[NodeArtifactInput] = []


# outputs
class NodeOutput(BaseModel):
    name: str


class NodeParameterOutput(NodeOutput):
    value_from: Optional[Union[str, Dict]] = None


class NodeArtifactOutput(NodeOutput):
    path: str


class NodeOutputs(BaseModel):
    parameters: List[NodeParameterOutput] = []
    artifacts: List[NodeArtifactOutput] = []


class PipelineNode(BaseModel):
    """A pipeline node is an ArgoWorkflow DAG type template's task."""

    name: str
    template: str
    inputs: NodeInputs
    outputs: NodeOutputs
    depends: List[str] = []


# --- Pipeline
class Pipeline(BaseModel):
    metadata: PipelineMetadata
    templates: List[Union[ScriptTemplate, ResourceTemplate]]
    inputs: List[PipelineInputParameter] = []
    dag: List[PipelineNode]

    def get_template(self, name: str) -> ScriptTemplate:

        return [
            template for template in self.templates if template.name == name
        ][0]

    def get_dag_task(self, name: str) -> PipelineNode:

        return [task for task in self.dag if task.name == name][0]

    @staticmethod
    def resolve_value_expression(expression: str) -> Tuple[str, str, str]:
        """Utility to resolve a node argument's value expression to the node
        and output references.

        Args:
            expression (str): A node argument value expression, e.g.
            - '{{workflow.parameters.coin}}' # references the workflow
                parameter type argument "coin"
            - '{{tasks.Set-a-coin.outputs.parameters.coin}}' # references the
                "Set-a-coin" node's parameter type argument "coin"

        Returns:
            Tuple[str, str,str]: The (upstream_task,output_type,output_name)
                expressed in the expression, e.g.
                - ('pipeline','parameters','coin')
                - ('Set-a-coin','parameters','coin')
        """
        # '{{workflow.parameters.coin}}' -> 'workflow.parameters.coin'
        # '{{tasks.Set-a-coin.outputs.parameters.coin}}' ->
        #   'tasks.Set-a-coin.outputs.parameters.coin'
        expression_content = expression.replace("{{", "").replace("}}", "")

        # 'workflow.parameters.coin' -> ['workflow','parameters','coin']
        # 'tasks.Set-a-coin.outputs.parameters.coin' ->
        #   ['tasks','Set-a-coin','outputs','parameters','coin']
        tokens = expression_content.split(".")

        if tokens[0] == "workflow":
            # ('pipeline','parameters','coin')
            return ("pipeline", tokens[1], tokens[2])
        elif tokens[0] == "tasks":
            # ('Set-a-coin','parameters','coin')
            return (tokens[1], tokens[3], tokens[4])
        else:
            raise ValueError(f"First token {tokens[0]} not supported.")

    @classmethod
    def build_dag(cls, workflow_template_spec: Dict) -> List[PipelineNode]:
        """Utility to build the Pipeline class' dag attribute.

        Args:
            workflow_template_spec (Dict): The spec field of a dict-ified
                hera.workflows.WorkflowTemplate class instance

        Returns:
            List[PipelineNode]: The constructed dag attribute of a Pipeline
                instance.
        """
        dag = []
        templates_dict = dict(
            [
                (template["name"], template)
                for template in workflow_template_spec["templates"]
            ]
        )
        entrypoint_template = templates_dict.pop(
            workflow_template_spec["entrypoint"]
        )
        tasks = entrypoint_template["dag"]["tasks"]

        for task in tasks:
            # assemble task data structure: name, template and depends can be
            # copied straight from the task entry of the dag template
            pipeline_node = {
                "name": task["name"],
                "template": task["template"],
                "depends": task["depends"].split(" && ")
                if task["depends"] is not None
                else [],
            }
            # the outputs can be obtained from the reference template's outputs
            pipeline_node["outputs"] = NodeOutputs(
                **copy_non_null_dict(
                    templates_dict[pipeline_node["template"]]["outputs"]
                )
            )

            # the inputs need to resolve the expressions to either the pipeline
            # or reference task in the expression if no expression is used, the
            # argument spec can be directly appended to the corresponding
            # parameters/artifacts list
            pipeline_node_inputs = {"parameters": [], "artifacts": []}
            # try:
            #     node_input_parameters = task["arguments"]["parameters"]
            # except KeyError:
            #     node_input_parameters = []
            if task["arguments"]["parameters"] is not None:
                node_input_parameters = task["arguments"]["parameters"]
            else:
                node_input_parameters = []

            # try:
            #     node_input_artifacts = task["arguments"]["artifacts"]
            # except KeyError:
            #     node_input_artifacts = []
            if task["arguments"]["artifacts"] is not None:
                node_input_artifacts = task["arguments"]["artifacts"]
            else:
                node_input_artifacts = []

            # build parameter inputs
            for node_argument in node_input_parameters:
                if "{{" not in node_argument["value"]:
                    pipeline_node_inputs["parameters"].append(
                        NodeParameterInput(**node_argument)
                    )
                elif node_argument.get("value") is not None:
                    (
                        upstream_node,
                        output_type,
                        output_name,
                    ) = cls.resolve_value_expression(node_argument["value"])
                    pipeline_node_inputs["parameters"].append(
                        NodeParameterInput(
                            name=node_argument["name"],
                            source=NodeInputSource(
                                node=upstream_node,
                                output_name=output_name,
                                output_type=output_type,
                            ),
                        )
                    )

            # build artifact inputs
            for node_argument in node_input_artifacts:
                (
                    upstream_node,
                    output_type,
                    output_name,
                ) = cls.resolve_value_expression(node_argument["from_"])
                pipeline_node_inputs["artifacts"].append(
                    NodeArtifactInput(
                        name=node_argument["name"],
                        source=NodeInputSource(
                            node=upstream_node,
                            output_name=output_name,
                            output_type=output_type,
                        ),
                    )
                )

            pipeline_node["inputs"] = pipeline_node_inputs
            dag.append(PipelineNode.model_validate(pipeline_node))

        return dag

    @classmethod
    def from_hera_workflow_template_model(
        cls,
        workflow_template_resource: WorkflowTemplateModel,
    ) -> Pipeline:
        """Utility to generate a Pipeline instance from a
        hera.workflows.models.WorkflowTemplate instance.

        To be used to easily convert the API response data structure
        to the bettmensch.ai pipeline data structure optimized for visualizing
        the DAG.

        Args:
            workflow_template_resource
            (hera.workflows.models.WorkflowTemplate): Instance of hera's
                WorkflowTemplateService response model.

        Returns:
            Pipeline: A Pipeline class instance.
        """

        workflow_template_dict = workflow_template_resource.dict()
        workflow_template_spec = workflow_template_dict["spec"].copy()

        # metadata
        metadata = PipelineMetadata(
            pipeline=workflow_template_dict["metadata"],
            flow=workflow_template_spec.get("workflow_metadata", None),
            component=workflow_template_spec.get("pod_metadata", None),
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
            PipelineInputParameter(**parameter)
            for parameter in workflow_template_spec["arguments"]["parameters"]
        ]

        # dag
        dag = cls.build_dag(workflow_template_spec)

        return cls(
            metadata=metadata, templates=templates, inputs=inputs, dag=dag
        )

    @classmethod
    def transform_dag_visualization_node_position(
        cls, x_y: Tuple[float, float]
    ) -> Tuple[float, float]:

        transformed_x_y = 350 * x_y[0], 150 * x_y[1]

        return transformed_x_y

    def create_dag_visualization_schema(
        self,
        include_task_io: bool = True,
    ) -> DagVisualizationItems:
        """Utility method to generate the assets the barfi/baklavajs rendering
        engine uses to display the Pipeline's dag property on the frontend.

        Args:
            include_task_io (bool): Whether to include the IO objects in the
                visualization, or just display task nodes.
        Returns:
            DagVisualizationItems: The schema containing all design specs for
                visualizing the DAG on the dashboard.
        """

        connections: List[Dict] = []
        nodes: List[Dict] = []

        for task_node in self.dag:

            task_node_name = task_node.name

            nodes.append(
                DagTaskNode(
                    id=task_node_name,
                    data={
                        "label": f"{PIPELINE_NODE_EMOJI_MAP['task']} {task_node_name}"  # noqa: E501
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

                    for argument_type in ["parameters", "artifacts"]:
                        arguments = getattr(
                            getattr(task_node, interface_type), argument_type
                        )
                        if not arguments:
                            continue

                        for argument in arguments:
                            # add the task io node
                            task_io_node_name = f"{task_node_name}_{interface_type}_{argument_type}_{argument.name}"  # noqa: E501
                            nodes.append(
                                DagTaskIONode(
                                    id=task_io_node_name,
                                    data={
                                        "label": f"{PIPELINE_NODE_EMOJI_MAP[interface_type]['task']} {PIPELINE_NODE_EMOJI_MAP[argument_type]} {argument.name}",  # noqa: E501
                                        "value": getattr(
                                            argument, "value", None
                                        ),
                                    },
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
                                upstream_node_name = f"{task_io_source.node}_outputs_{task_io_source.output_type}_{task_io_source.output_name}"  # noqa: E501
                                connections.append(
                                    DagConnection(
                                        id=f"{upstream_node_name}->{task_io_node_name}",  # noqa: E501
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
                    DagPipelineIONode(
                        id=node_name,
                        data={
                            "label": f"{PIPELINE_NODE_EMOJI_MAP['inputs']['pipeline']} {PIPELINE_NODE_EMOJI_MAP['parameters']} {input.name}",  # noqa: E501
                            "value": input.value,
                        },
                        node_type="input",
                    )
                )

        return DagVisualizationItems(connections=connections, nodes=nodes)
