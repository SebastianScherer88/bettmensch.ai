from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from bettmensch_ai.server.pipeline import (
    NodeArtifactInput,
    NodeArtifactOutput,
    NodeInputs,
    NodeOutputs,
    NodeParameterInput,
    NodeParameterOutput,
    Pipeline,
    PipelineInputs,
    PipelineNode,
    PipelineOutputs,
    PipelineParameterInput,
    ResourceTemplate,
    ScriptTemplate,
)
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
    def _get_model_classes(
        cls,
        pipeline_io: Union[
            Type[PipelineInputs],
            Type[PipelineOutputs],
            Type[NodeInputs],
            Type[NodeOutputs],
        ],
    ) -> Tuple[
        Literal["inputs", "outputs"],
        Union[
            Type[FlowParameterInput],
            Type[FlowParameterOutput],
            Type[FlowNodeParameterInput],
            Type[FlowNodeParameterOutput],
        ],
        Union[
            Type[FlowArtifactOutput],
            Type[FlowNodeArtifactInput],
            Type[FlowNodeArtifactOutput],
        ],
        Union[
            Type[FlowInputs],
            Type[FlowOutputs],
            Type[FlowNodeInputs],
            Type[FlowNodeOutputs],
        ],
    ]:
        """_summary_

        Args:
            pipeline_io (Union[
                type[PipelineInputs],
                type[PipelineOutputs],
                type[NodeInputs],
                type[NodeOutputs]
            ]): _description_

        Returns:
            Tuple[
                Union[
                    type[FlowParameterInput],
                    type[FlowParameterOutput],
                    type[FlowNodeParameterInput],
                    type[FlowNodeParameterOutput]
                ],
                Union[
                    type[FlowArtifactOutput],
                    type[FlowNodeArtifactInput],
                    type[FlowNodeArtifactOutput]
                ],
                Union[
                    type[FlowInputs],
                    type[FlowOutputs],
                    type[FlowNodeInputs],
                    type[FlowNodeOutputs]
                ]
            ]: _description_
        """

        if pipeline_io == PipelineInputs:
            return "inputs", FlowParameterInput, None, FlowInputs
        elif pipeline_io == PipelineOutputs:
            return (
                "outputs",
                FlowParameterOutput,
                FlowArtifactOutput,
                FlowOutputs,
            )
        elif pipeline_io == NodeInputs:
            return (
                "inputs",
                FlowNodeParameterInput,
                FlowNodeArtifactInput,
                FlowNodeInputs,
            )
        elif pipeline_io == NodeOutputs:
            return (
                "outputs",
                FlowNodeParameterOutput,
                FlowNodeArtifactOutput,
                FlowNodeOutputs,
            )
        else:
            raise TypeError(
                f"Type {pipeline_io} not supported. Must be one of"
                "- PipelineInputs"
                "- PipelineOutputs"
                "- NodeInputs"
                "- NodeOutputs"
            )

    @classmethod
    def _build_generic_io(
        cls,
        pipeline_io: Union[
            PipelineInputs,
            PipelineOutputs,
            NodeInputs,
            NodeOutputs,
        ],
        flow_node: NodeStatusModel,
    ) -> Union[FlowInputs, FlowOutputs, FlowNodeInputs, FlowNodeOutputs]:
        """Parametrizable utility function to build either inputs or outputs
        for the Flow or one of its FlowNodes using Pipeline or Node I/Os.

        Args:
            pipeline_io (Union[
                PipelineInputs,
                PipelineOutputs,
                NodeInputs,
                NodeOutputs,
            ]): _description_
            flow_node (NodeStatusModel): A pipeline or pipeline node I/O
                instance

        Returns:
            Union[FlowInputs, FlowOutputs, FlowNodeInputs, FlowNodeOutputs]:
                A flow or flow node I/O instance
        """

        io = {"parameters": [], "artifacts": []}

        (
            io_type,
            parameter_io_class,
            artifact_io_class,
            io_class,
        ) = cls._get_model_classes(pipeline_io.__class__)

        # parameters
        for ppo in pipeline_io.parameters:
            flow_parameter_io_data = ppo.model_dump()
            try:
                fpo_value = [
                    fpo
                    for fpo in getattr(flow_node, io_type).parameters
                    if fpo.name == ppo.name
                ][0].value
            except (AttributeError, IndexError):
                fpo_value = None
            finally:
                flow_parameter_io_data["value"] = fpo_value
                flow_parameter_output = parameter_io_class.model_validate(
                    flow_parameter_io_data
                )
                io["parameters"].append(flow_parameter_output)

        # artifacts
        if artifact_io_class is not None:
            for pao in pipeline_io.artifacts:
                flow_artifact_io_data = pao.model_dump()
                try:
                    fao_s3 = [
                        fao
                        for fao in getattr(flow_node, io_type).artifacts
                        if fao.name == pao.name
                    ][0].s3.dict()
                except (AttributeError, IndexError):
                    fao_s3 = None
                finally:
                    flow_artifact_io_data["s3"] = fao_s3
                    flow_parameter_output = artifact_io_class.model_validate(
                        flow_artifact_io_data
                    )
                    io["artifacts"].append(flow_parameter_output)
        else:
            del io["artifacts"]

        return io_class.model_validate(io)

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
        flow_inputs = cls._build_generic_io(pipeline_inputs, inner_dag_node)

        # --- outputs
        flow_outputs = cls._build_generic_io(pipeline_outputs, inner_dag_node)

        return flow_inputs, flow_outputs

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

        flow_node_data = {
            "name": pipeline_node.name,
            "template": pipeline_node.template,
            "depends": pipeline_node.depends,
        }

        flow_node = workflow_nodes_dict.get(pipeline_node.name, None)

        if flow_node is None:
            flow_node_data["pod_name"] = pipeline_node.name
            flow_node_data["phase"] = "Not Scheduled"
            flow_node_inputs, flow_node_outputs = (
                pipeline_node.inputs.model_dump(),
                pipeline_node.outputs.model_dump(),
            )
        else:
            flow_node_data["id"] = flow_node.id
            flow_node_data["type"] = flow_node.type
            flow_node_data["pod_name"] = flow_node.name
            flow_node_data["phase"] = flow_node.phase
            flow_node_data["dependants"] = getattr(flow_node, "children", None)
            flow_node_data["host_node_name"] = getattr(
                flow_node, "host_node_name", None
            )
            flow_node_inputs = cls._build_generic_io(
                pipeline_node.inputs, flow_node
            )
            flow_node_outputs = cls._build_generic_io(
                pipeline_node.outputs, flow_node
            )

        flow_node_data["inputs"] = flow_node_inputs
        flow_node_data["outputs"] = flow_node_outputs

        flow_node = FlowNode.model_validate(flow_node_data)

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
