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
from hera.workflows.models import Artifact as ArtifactModel
from hera.workflows.models import DAGTask as DAGTaskModel
from hera.workflows.models import Metadata as PodMetadataModel
from hera.workflows.models import Parameter as ParameterModel
from hera.workflows.models import Template as TemplateModel
from hera.workflows.models import WorkflowMetadata as WorkflowMetadataModel
from hera.workflows.models import WorkflowSpec as WorkflowSpecModel
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

INNER_PIPELINE_DAG = "bettmensch-ai-inner-dag"
OUTER_PIPELINE_DAG = "bettmensch-ai-outer-dag"


# --- PipelineMetadata
class WorkflowTemplateMetadata(BaseModel):
    uid: str
    name: str
    namespace: str = "default"
    creation_timestamp: datetime
    labels: Dict[str, str]


class PipelineMetadata(BaseModel):
    pipeline: WorkflowTemplateMetadata
    flow: Optional[WorkflowMetadataModel] = None
    component: Optional[PodMetadataModel] = None


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
# inputs
class PipelineParameterInput(BaseModel):
    name: str
    value: Optional[str] = None
    default: Optional[Any] = None


class PipelineInputs(BaseModel):
    parameters: List[PipelineParameterInput] = []


# outputs
# class NodeInputSource(BaseModel):
#     node: str
#     output_name: str
#     output_type: Literal["parameters", "artifacts"]


# class NodeInput(BaseModel):
#     name: str
#     source: Optional[NodeInputSource] = None


# class NodeParameterInput(NodeInput):
#     value: Optional[str] = None
#     value_from: Optional[Union[str, Dict]] = None
class PipelineParameterOutput(NodeParameterOutput):
    pass


class PipelineArtifactOutput(NodeArtifactInput):
    pass


class PipelineOutputs(BaseModel):
    parameters: List[PipelineParameterOutput] = []
    artifacts: List[PipelineArtifactOutput] = []


class Pipeline(BaseModel):
    metadata: PipelineMetadata
    templates: List[Union[ScriptTemplate, ResourceTemplate]]
    inputs: Optional[PipelineInputs] = None
    outputs: Optional[PipelineOutputs] = None
    dag: List[PipelineNode]

    def get_template(self, name: str) -> ScriptTemplate:

        return [
            template for template in self.templates if template.name == name
        ][0]

    def get_dag_task(self, name: str) -> PipelineNode:

        return [task for task in self.dag if task.name == name][0]

    @staticmethod
    def contains_parameter_reference(value: str) -> bool:
        return "{{" in value

    @staticmethod
    def resolve_parameter_reference(reference: str) -> Tuple[str, str, str]:
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
        expression_content = reference.replace("{{", "").replace("}}", "")

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
    def get_relevant_templates(
        cls, workflow_template_spec: WorkflowSpecModel
    ) -> Tuple[TemplateModel, Dict[str, TemplateModel]]:
        """Converts a workflow template spec instance into a tuple with the
        inner dag template in the first entry, and a template.name -> template
        dictionary in the second entry.

        Args:
            workflow_template_spec (WorkflowSpecModel): The workflow template
                spec.

        Returns:
            Tuple[TemplateModel, Dict[str,TemplateModel]]: The inner dag
                template and a dict with the remaining templates (excluding the
                outer dag template).
        """

        component_templates_dict = dict(
            [
                (template.name, template)
                for template in workflow_template_spec.templates
            ]
        )

        # remove outer dag template
        inner_dag_template = component_templates_dict.pop(INNER_PIPELINE_DAG)

        return inner_dag_template, component_templates_dict

    @classmethod
    def build_pipeline_output_parameter(
        cls, output_parameter: ParameterModel
    ) -> PipelineParameterOutput:
        """Builds and returns a PipelineParameterOutput instance from the
        provided ParameterModel instance. Checks for source parameter
        references and resolves them into a NodeInputSource attribute on the
        return.

        Raises a ValueError if no such reference can be found.

        Args:
            output_parameter (ParameterModel): The output parameter
        Returns:
            PipelineParameterOutput: The assembled pipeline parameter
                output.
        """

        output_parameter_value_from = getattr(
            output_parameter, "valueFrom", None
        )
        output_parameter_value_from_parameter = getattr(
            output_parameter_value_from, "parameter", None
        )

        if cls.contains_parameter_reference(
            output_parameter_value_from_parameter
        ):
            raise ValueError(
                "Pipeline parameter output must provide a"
                " source via its `valueFrom.parameter` attribute,"
                " but seems to be missing:"
                f"{output_parameter_value_from_parameter}"
            )
        (
            upstream_node,
            output_type,
            output_name,
        ) = cls.resolve_parameter_reference(
            output_parameter_value_from_parameter
        )
        return PipelineParameterOutput(
            name=output_parameter.name,
            source=NodeInputSource(
                node=upstream_node,
                output_name=output_name,
                output_type=output_type,
            ),
        )

    @classmethod
    def build_pipeline_node_output_artifact(
        cls, output_artifact: ArtifactModel
    ) -> PipelineArtifactOutput:
        """Builds and returns a PipelineArtifactOutput instance from the
        provided ArtifactModel instance. Checks for source artifact references
        and  resolves them into a NodeInputSource attribute on the return.

        Raises a ValueError if no such reference can be found.

        Args:
            output_artifact (ArtifactModel): The output artifact

        Returns:
            PipelineArtifactOutput: The assembled pipeline artifact output.
        """

        output_artifact_from_ = getattr(output_artifact, "from_", None)

        if cls.contains_parameter_reference(output_artifact_from_):
            raise ValueError(
                "Pipeline artifact input must provide a"
                " source via its `from_` attribute, but seems to"
                f" be missing: {output_artifact_from_}"
            )

        (
            upstream_node,
            output_type,
            output_name,
        ) = cls.resolve_parameter_reference(output_artifact_from_)
        return PipelineArtifactOutput(
            name=output_artifact.name,
            source=NodeInputSource(
                node=upstream_node,
                output_name=output_name,
                output_type=output_type,
            ),
        )

    @classmethod
    def build_io(
        cls, workflow_template_spec: WorkflowSpecModel
    ) -> Tuple[PipelineInputs, PipelineOutputs]:

        inner_dag_template, _ = cls.get_relevant_templates(
            workflow_template_spec
        )

        # add pipeline inputs directly from inner dag template
        pipeline_inputs = PipelineInputs(
            parameters=[
                PipelineParameterInput(dag_input_parameter)
                for dag_input_parameter in inner_dag_template.inputs.parameters
            ]
        )

        pipeline_outputs = {"parameters": [], "artifacts": []}

        output_parameters = inner_dag_template.outputs.parameters

        if output_parameters is not None:
            for output_parameter in output_parameters:
                pipeline_output_parameter = (
                    cls.build_pipeline_output_parameter(output_parameter)
                )
                pipeline_outputs["parameters"].append(
                    pipeline_output_parameter
                )

        output_artifacts = inner_dag_template.outputs.artifacts

        if output_artifacts is not None:
            for output_artifact in output_artifacts:
                pipeline_output_artifact = (
                    cls.build_pipeline_node_output_artifact(output_artifact)
                )
                pipeline_outputs["artifacts"].append(pipeline_output_artifact)

        pipeline_outputs = PipelineOutputs.model_validate(pipeline_outputs)

        return pipeline_inputs, pipeline_outputs

    @classmethod
    def build_pipeline_node_input_parameter(
        cls, input_parameter: ParameterModel
    ) -> NodeParameterInput:
        """Builds and returns a NodeParameterInput instance from the provided
        ParameterModel instance. Checks for source parameter references and
        resolves them into a NodeInputSource attribute on the return when
        needed.

        Args:
            input_parameter (ParameterModel): The input parameter

        Returns:
            NodeParameterInput: The assembled pipeline node parameter input.
        """

        input_parameter_value = getattr(input_parameter, "value", None)

        if cls.contains_parameter_reference(input_parameter_value):
            return NodeParameterInput(**input_parameter)
        else:
            (
                upstream_node,
                output_type,
                output_name,
            ) = cls.resolve_parameter_reference(input_parameter_value)
            return NodeParameterInput(
                name=input_parameter.name,
                source=NodeInputSource(
                    node=upstream_node,
                    output_name=output_name,
                    output_type=output_type,
                ),
            )

    @classmethod
    def build_pipeline_node_input_artifact(
        cls, input_artifact: ArtifactModel
    ) -> NodeArtifactInput:
        """Builds and returns a NodeArtifactInput instance from the provided
        ArtifactModel instance. Checks for source artifact references and
        resolves them into a NodeInputSource attribute on the return.

        Raises a ValueErro if no such reference could be found.

        Args:
            input_artifact (ArtifactModel): The input artifact

        Returns:
            NodeArtifactInput: The assembled pipeline node artifact input.
        """

        input_artifact_from_ = getattr(input_artifact, "from_", None)

        if cls.contains_parameter_reference(input_artifact_from_):
            raise ValueError(
                "Pipeline node artifact input must provide a"
                " source via its `from_` attribute, but seems to"
                f" be missing: {input_artifact_from_}"
            )

        (
            upstream_node,
            output_type,
            output_name,
        ) = cls.resolve_parameter_reference(input_artifact_from_)
        return NodeArtifactInput(
            name=input_artifact.name,
            source=NodeInputSource(
                node=upstream_node,
                output_name=output_name,
                output_type=output_type,
            ),
        )

    @classmethod
    def build_pipeline_node(
        cls,
        task: DAGTaskModel,
        component_templates_dict: Dict[str, TemplateModel],
    ) -> PipelineNode:

        # initialize pipeline node dict container with basic attributes
        pipeline_node = {
            "name": task.name,
            "template": task.template,
            "depends": task.depends.split(" && ")
            if task.depends is not None
            else [],
        }

        # add node outputs directly from template
        pipeline_node["outputs"] = NodeOutputs(
            **copy_non_null_dict(
                component_templates_dict[pipeline_node["template"]].outputs
            )
        )

        # add node inputs from task arguments; resolve any source references
        # into custom NodeInputSource
        pipeline_node["inputs"] = {"parameters": [], "artifacts": []}

        input_parameters = task.arguments.parameters

        if input_parameters is not None:
            for input_parameter in input_parameters:
                pipeline_node_input_parameter = (
                    cls.build_pipeline_node_input_parameter(input_parameter)
                )
                pipeline_node["inputs"]["parameters"].append(
                    pipeline_node_input_parameter
                )

        input_artifacts = task.arguments.artifacts
        if input_artifacts is not None:
            for input_artifact in input_artifacts:
                pipeline_node_input_artifact = (
                    cls.build_pipeline_node_input_artifact(input_artifact)
                )
                pipeline_node["inputs"]["artifacts"].append(
                    pipeline_node_input_artifact
                )

        return PipelineNode.model_validate(pipeline_node)

    @classmethod
    def build_dag(
        cls, workflow_template_spec: WorkflowSpecModel
    ) -> List[PipelineNode]:
        """Utility to build the Pipeline class' dag attribute.

        Args:
            workflow_template_spec (WorkflowSpecModel): The spec field of a
                hera.workflows.WorkflowTemplate class instance

        Returns:
            List[PipelineNode]: The constructed dag attribute of a Pipeline
                instance.
        """

        (
            inner_dag_template,
            component_templates_dict,
        ) = cls.get_relevant_templates(workflow_template_spec)

        dag = []

        for task in inner_dag_template.dag.tasks:
            pipeline_node = cls.build_pipeline_node(
                task, component_templates_dict
            )
            dag.append(pipeline_node)

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

        workflow_template_spec: WorkflowSpecModel = (
            workflow_template_resource.spec
        )

        # metadata
        metadata = PipelineMetadata(
            pipeline=workflow_template_resource.metadata.model_dump(),
            flow=getattr(workflow_template_spec, "workflow_metadata", None),
            component=getattr(workflow_template_spec, "pod_metadata", None),
        )

        # templates
        templates = []
        for template in workflow_template_spec.templates:
            # we are only interested in Script and Resource type templates
            if template.script is not None:
                templates.append(ScriptTemplate.model_validate(template))
            elif template.resource is not None:
                templates.append(ResourceTemplate.model_validate(template))

        # io
        inputs, outputs = cls.build_io(workflow_template_spec)

        # dag
        dag = cls.build_dag(workflow_template_spec)

        return cls(
            metadata=metadata,
            templates=templates,
            inputs=inputs,
            outputs=outputs,
            dag=dag,
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
