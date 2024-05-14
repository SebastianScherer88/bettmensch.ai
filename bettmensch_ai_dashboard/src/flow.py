from __future__ import annotations
from typing import Dict, Optional, List, Union, Literal, Tuple
import argo_workflows
from argo_workflows.model.io_argoproj_workflow_v1alpha1_workflow import IoArgoprojWorkflowV1alpha1Workflow
from argo_workflows.api import workflow_service_api
from pydantic import BaseModel
from datetime import datetime
from pipeline import ScriptTemplate, Pipeline, PipelineNode, PipelineInputParameter, NodeInputs, NodeOutput
import yaml
from dag import DagConnection, DagNode, DagVisualizationSchema

# --- FlowMetadata
class FlowMetadata(BaseModel):
    uid: str
    name: str
    namespace: str = "default"
    creation_timestamp: datetime
    labels: Dict[str,str]
    annotations: Optional[Dict[str,str]]
    
# --- FlowStatus
class FlowState(BaseModel):
    phase: Literal['Succeeded','Failed','Pending']
    started_at: Optional[datetime] = None
    finshed_at: Optional[datetime] = None
    progress: str
    conditions: List[Dict[str,str]]
    resources_duration: Optional[Dict]
    task_results_completion_status: Dict[str,bool]

# --- FlowInput
class FlowInputParameter(PipelineInputParameter):
    pass

# --- FlowNode
# outputs
class FlowNodeOutput(NodeOutput):
    value: Optional[str] = None
        
class FlowNodeParameterOutput(FlowNodeOutput):
    value_from: Optional[Union[str,Dict]] = None

class FlowNodeArtifactOutput(FlowNodeOutput):
    path: str
    value_from: Optional[Union[str,Dict]] = None
    
class FlowNodeOutputs(BaseModel):
    parameters: Optional[List[FlowNodeParameterOutput]] = None
    artifacts: Optional[List[FlowNodeArtifactOutput]] = None
    exit_code: Optional[int] = None

class FlowNode(BaseModel):
    id: str
    name: str
    type: Literal['Pod']
    pod_name: str # this will match the PipelineNode.name, i.e the task name
    template: str
    phase: Literal['Succeeded']
    template: str
    inputs: Optional[NodeInputs] = None
    outputs: Optional[FlowNodeOutputs] = None
    logs: Optional[FlowNodeArtifactOutput] = None
    depends: Optional[Union[str,List[str]]] = None
    dependants: Optional[Union[str,List[str]]] = None
    host_node_name: str
    
# --- FlowArtifactConfiguration
class FlowArtifactConfiguration(BaseModel):
    repository_ref: Dict
    gc_status: Dict

# --- Flow
class Flow(BaseModel):
    """A flow node is an instantiated pipeline node on Kubernetes.
    """
    metadata: FlowMetadata
    state: FlowState
    artifact_configuration: FlowArtifactConfiguration
    templates: List[ScriptTemplate]
    inputs: List[FlowInputParameter] = []
    dag: List[FlowNode]
        
    @classmethod
    def build_dag(cls, workflow_status: Dict) -> List[FlowNode]:
        """Utility to build the Flow class' dag attribute. Identical to the Pipeline class' dag attribute, but with additional
        values resolved at runtime. 

        Args:
            workflow_status (_type_): The status field of a dict-ified IoArgoprojWorkflowV1alpha1Workflow class instance

        Returns:
            List[FlowNode]: The constructed dag attribute of a Flow instance.
        """
        
        # build pipeline dag
        workflow_template_spec = workflow_status['stored_workflow_template_spec']
        pipeline_dag = Pipeline.build_dag(workflow_template_spec)
        
        # add FlowNode specific values and available resolved input/output values for each FlowNode
        flow_dag = []
        workflow_nodes = list(workflow_status['nodes'].values())
        
        for pipeline_node in pipeline_dag:
            pipeline_node_dict = pipeline_node.model_dump()
            workflow_node_dict = [workflow_node for workflow_node in workflow_nodes if workflow_node["display_name"] == pipeline_node_dict["name"]][0]
            
            # import pdb
            # pdb.set_trace()
            
            flow_node_dict = {
                "id": workflow_node_dict["id"],
                "name": workflow_node_dict["display_name"],
                "type": workflow_node_dict["type"],
                "pod_name": workflow_node_dict["name"],
                "template": workflow_node_dict["template_name"],
                "phase": workflow_node_dict["phase"],
                "inputs": pipeline_node_dict["inputs"],
                "outputs": dict(**pipeline_node_dict["outputs"],**{"exit_code":workflow_node_dict.get("exit_code")}),
                "logs": None,
                "depends": pipeline_node_dict["depends"],
                "dependants": workflow_node_dict.get("children"),
                "host_node_name": workflow_node_dict["host_node_name"],
            }
            # inject resolved input values where possible
            for argument_io in ("inputs","outputs"):
                for argument_type in ("parameters","artifacts"):
                    try:
                        workflow_node_arguments = workflow_node_dict[argument_io][argument_type]
                        for i, argument in enumerate(workflow_node_arguments):
                            if workflow_node_arguments[i]["name"] == argument["name"]:
                                flow_node_dict[argument_io][argument_type][i]["value"] = argument["value"]
                            elif workflow_node_arguments[i]["name"] == "main-logs":
                                flow_node_dict["logs"] = workflow_node_arguments[i]
                            else:
                                pass
                    except KeyError:
                        pass
            try:
                flow_node = FlowNode(**flow_node_dict)
            except:
                import pdb
                pdb.set_trace()
                
            flow_dag.append(flow_node)
            
        return flow_dag
            
    @classmethod
    def from_argo_workflow_cr(cls, workflow_resource: IoArgoprojWorkflowV1alpha1Workflow) -> Flow:
        """Utility to generate a Flow instance from a IoArgoprojWorkflowV1alpha1Workflow instance. 
        
        To be used to easily convert the API response data structure
        to the bettmensch.ai pipeline data structure optimized for visualizing the DAG.

        Args:
            workflow_resource (IoArgoprojWorkflowV1alpha1Workflow): The return of ArgoWorkflow's 
                api/v1/workflow/{namespace}/{name} endpoint.

        Returns:
            Flow: A Flow class instance.
        """
        
        workflow_dict = workflow_resource.to_dict()
        workflow_spec = workflow_dict['spec'].copy()
        workflow_status = workflow_dict['status'].copy()
        workflow_template_spec = workflow_status['stored_workflow_template_spec'].copy()
        
        # metadata
        metadata = FlowMetadata(**workflow_dict['metadata'])
        
        # state
        state = FlowState(**workflow_status)
        
        # artifact_configuration
        artifact_configuration = FlowArtifactConfiguration(
            repository_ref=workflow_status["artifact_gc_status"],
            gc_status=workflow_status["artifact_repository_ref"]
        )
        
        # templates
        entrypoint_template = workflow_template_spec['entrypoint']
        templates = [
            ScriptTemplate.model_validate(template) 
                for template in workflow_template_spec['templates']
                    if template['name'] != entrypoint_template
        ]
        
        # inputs
        inputs = [FlowInputParameter(**parameter) for parameter in workflow_spec['arguments']['parameters']]
        
        # dag
        dag = cls.build_dag(workflow_status)

        return cls(
            metadata=metadata,
            state=state,
            artifact_configuration=artifact_configuration,
            templates=templates,
            inputs=inputs,
            dag=dag
        )
        
    def create_dag_visualization_schema(self) -> DagVisualizationSchema:
        """Utility method to generate the assets the barfi/baklavajs rendering engine uses to display the Pipeline's dag property on the frontend.
        """
        
        node_positions = Pipeline.create_dag_visualization_node_positions(self.dag)
        connections: List[Dict] = []
        nodes: List[Dict] = []
        
        for pipeline_node in self.dag:
            
            pipeline_node_name = pipeline_node.name
            
            nodes.append(
                DagNode(
                    id=pipeline_node_name,
                    pos =node_positions[pipeline_node_name],
                    data = {"label": pipeline_node_name},
                    # node_type = "default"
                    # source_position = "top"
                    # target_position = "bottom"
                )
            )
            
            if pipeline_node.depends is not None:
                for upstream_node_name in pipeline_node.depends:
                    connections.append(
                        DagConnection(
                            id=f"{upstream_node_name}->{pipeline_node_name}",
                            source = upstream_node_name,
                            target = pipeline_node_name,
                            animated = True,
                            edge_type = "smoothstep",
                        )
                    )
            
        return DagVisualizationSchema(
            connections=connections,
            nodes=nodes
        )
        
def main_test():
    '''Unit test the Pipeline class.'''
    
    # get a sample pipeline from the ArgoWorkflow server
    configuration = argo_workflows.Configuration(host="https://127.0.0.1:2746")
    configuration.verify_ssl = False

    api_client = argo_workflows.ApiClient(configuration)
    api_instance = workflow_service_api.WorkflowServiceApi(api_client)

    workflows = api_instance.list_workflows(namespace="argo")['items']
    print(f"Registered flows: {[workflow['metadata']['name'] for workflow in workflows]}")
    workflow = workflows[0]
    print(f"Workflow: {workflow.to_dict()}")
    
    with open('flow_test_workflow.yaml','w') as file:
        yaml.dump(workflow.to_dict(),file)

    # convert to pipeline and show results
    flow = Flow.from_argo_workflow_cr(workflow)
    print(f"Flow: {flow.model_dump()}")
    
    with open('flow_test_flow.yaml','w') as file:
        yaml.dump(flow.model_dump(),file)
    
if __name__ == "__main__":
    main_test()