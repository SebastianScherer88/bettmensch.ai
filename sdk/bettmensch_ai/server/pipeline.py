from __future__ import annotations
from typing import Dict, Optional, List, Union, Tuple, Literal
import argo_workflows
from argo_workflows.model.io_argoproj_workflow_v1alpha1_workflow_template import IoArgoprojWorkflowV1alpha1WorkflowTemplate
from argo_workflows.api import workflow_template_service_api
from pydantic import BaseModel
from datetime import datetime
import yaml
import networkx as nx
from bettmensch_ai.server.utils import PIPELINE_NODE_EMOJI_MAP
from bettmensch_ai.server.dag import DagConnection, DagNode, DagVisualizationSchema

# --- PipelineMetadata
class WorkflowTemplateMetadata(BaseModel):
    uid: str
    name: str
    namespace: str = "default"
    creation_timestamp: datetime
    labels: Dict[str,str]

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
    resources: Dict = {}
# inputs    
class ScriptTemplateParameterInput(BaseModel):
    name : str
    value: Optional[str] = None
    value_from: Optional[Union[str,Dict]] = None
    
class ScriptTemplateArtifactInput(BaseModel):
    name: str
    path: Optional[str] = None
    
class ScriptTemplateInputs(BaseModel):
    parameters: Optional[List[ScriptTemplateParameterInput]] = None
    artifacts: Optional[List[ScriptTemplateArtifactInput]] = None

# outputs
class ScriptTemplateParameterOutput(BaseModel):
    name : str
    value_from: Optional[Union[str,Dict]] = None
    
class ScriptTemplateArtifactOutput(BaseModel):
    name: str
    path: Optional[str] = None
    value_from: Optional[Union[str,Dict]] = None
    
class ScriptTemplateOutputs(BaseModel):
    parameters: Optional[List[ScriptTemplateParameterOutput]] = None
    artifacts: Optional[List[ScriptTemplateArtifactOutput]] = None
    
class ScriptTemplate(BaseModel):
    name: str
    inputs: ScriptTemplateInputs = {}
    outputs: ScriptTemplateOutputs = {}
    metadata: Dict = {}
    script: Script

# --- PipelineInput
class PipelineInputParameter(BaseModel):
    name: str
    value: Optional[str] = None

# --- PipelineNode
# inputs
class NodeInputSource(BaseModel):
    node: str
    output_name: str
    output_type: Literal['parameters','artifacts']

class NodeInput(BaseModel):
    name: str
    value: Optional[str] = None
    value_from: Optional[Union[str,Dict]] = None
    source: Optional[NodeInputSource] = None

class NodeParameterInput(NodeInput):
    pass

class NodeArtifactInput(NodeInput):
    path: Optional[str] = None
    
class NodeInputs(BaseModel):
    parameters: List[NodeParameterInput] = []
    artifacts: List[NodeArtifactInput] = []

# outputs
class NodeOutput(BaseModel):
    name: str
        
class NodeParameterOutput(NodeOutput):
    value_from: Optional[Union[str,Dict]] = None

class NodeArtifactOutput(NodeOutput):
    path: str
    value_from: Optional[Union[str,Dict]] = None
    

class NodeOutputs(BaseModel):
    parameters: List[NodeParameterOutput] = []
    artifacts: List[NodeArtifactOutput] = []

class PipelineNode(BaseModel):
    """A pipeline node is an ArgoWorkflow DAG type template's task.
    """
    name: str
    template: str
    inputs: NodeInputs
    outputs: NodeOutputs
    depends: List[str] = []

# --- Pipeline
class Pipeline(BaseModel):
    metadata: PipelineMetadata
    templates: List[ScriptTemplate]
    inputs: List[PipelineInputParameter] = []
    dag: List[PipelineNode]
    
    def get_template(self, name: str) -> ScriptTemplate:
        
        return [template for template in self.templates if template.name == name][0]
    
    def get_dag_task(self, name: str) -> PipelineNode:
        
        return [task for task in self.dag if task.name == name][0]
    
    @staticmethod
    def resolve_value_expression(expression: str) -> Tuple[str, str,str]:
        """Utility to resolve a node argument's value expression to the node and output references.

        Args:
            expression (str): A node argument value expression, e.g. 
            - '{{workflow.parameters.coin}}' # references the workflow parameter type argument "coin"
            - '{{tasks.Set-a-coin.outputs.parameters.coin}}' # references the "Set-a-coin" node's parameter type argument "coin"

        Returns:
            Tuple[str, str,str]: The (upstream_task,output_type,output_name) expressed in the expression, e.g.
                - ('pipeline','parameters','coin')
                - ('Set-a-coin','parameters','coin')
        """
        # '{{workflow.parameters.coin}}' -> 'workflow.parameters.coin'
        # '{{tasks.Set-a-coin.outputs.parameters.coin}}' -> 'tasks.Set-a-coin.outputs.parameters.coin'
        expression_content = expression.replace('{{','').replace('}}','')
        
        # 'workflow.parameters.coin' -> ['workflow','parameters','coin']
        # 'tasks.Set-a-coin.outputs.parameters.coin' -> ['tasks','Set-a-coin','outputs','parameters','coin']
        tokens = expression_content.split('.')
        
        if tokens[0] == 'workflow':
            # ('pipeline','parameters','coin')
            return ('pipeline',tokens[1],tokens[2])
        elif tokens[0] == 'tasks':
            # ('Set-a-coin','parameters','coin')
            return (tokens[1],tokens[3],tokens[4])
        else:
            raise ValueError(f"First token {tokens[0]} not supported.")
        
    @classmethod
    def build_dag(cls, workflow_template_spec: Dict) -> List[PipelineNode]:
        """Utility to build the Pipeline class' dag attribute. 

        Args:
            workflow_template_spec (Dict): The spec field of a dict-ified IoArgoprojWorkflowV1alpha1WorkflowTemplate class instance

        Returns:
            List[PipelineNode]: The constructed dag attribute of a Pipeline instance.
        """
        dag = []
        templates_dict = dict([(template['name'], template) for template in workflow_template_spec['templates']])
        entrypoint_template = templates_dict.pop(workflow_template_spec['entrypoint'])
        tasks = entrypoint_template['dag']['tasks']
        
        for task in tasks:
            # import pdb
            # pdb.set_trace()
            # assemble task data structure: name, template and depends can be copied straight from the task entry of the dag template
            pipeline_node = {
                'name':task['name'],
                'template':task['template'],
                'depends':task["depends"].split(" && ") if task.get('depends') else []
            }
            # the outputs can be obtained from the reference template's outputs
            pipeline_node['outputs'] = NodeOutputs(**templates_dict[pipeline_node['template']]['outputs'])
            
            # the inputs need to resolve the expressions to either the pipeline or reference task in the expression
            # if no expression is used, the argument spec can be directly appended to the corresponding parameters/artifacts list
            pipeline_node_inputs = {'parameters':[],'artifacts':[]}
            try:
                node_input_parameters = task['arguments']['parameters']
            except KeyError:
                node_input_parameters = []
            
            try:
                node_input_artifacts = task['arguments']['artifacts']
            except KeyError:
                node_input_artifacts = []
                
            for (argument_type, NodeArgumentTypeInputClass, argument_node_inputs) in (
                ('parameters',NodeParameterInput,node_input_parameters),
                ('artifacts',NodeArtifactInput,node_input_artifacts)
            ):
                for node_argument in  argument_node_inputs:
                    if "{{" not in node_argument['value']:
                        pipeline_node_inputs[argument_type].append(NodeParameterInput(**node_argument))
                    elif node_argument.get('value') is not None:
                        (upstream_node, output_type, output_name) = cls.resolve_value_expression(node_argument['value'])
                        pipeline_node_inputs[argument_type].append(
                            NodeArgumentTypeInputClass(
                                name=node_argument["name"],
                                source=NodeInputSource(
                                    node=upstream_node,
                                    output_name=output_name,
                                    output_type=output_type
                                )
                            )
                        )
                    
            pipeline_node['inputs'] = pipeline_node_inputs
            dag.append(PipelineNode.model_validate(pipeline_node))
            
        return dag
            
    @classmethod
    def from_argo_workflow_cr(cls, workflow_template_resource: IoArgoprojWorkflowV1alpha1WorkflowTemplate) -> Pipeline:
        """Utility to generate a Pipeline instance from a IoArgoprojWorkflowV1alpha1WorkflowTemplate instance. 
        
        To be used to easily convert the API response data structure
        to the bettmensch.ai pipeline data structure optimized for visualizing the DAG.

        Args:
            workflow_template_resource (IoArgoprojWorkflowV1alpha1WorkflowTemplate): The return of ArgoWorkflow's 
                api/v1/workflow-templates/{namespace}/{name} endpoint.

        Returns:
            Pipeline: A Pipeline class instance.
        """
        
        workflow_template_dict = workflow_template_resource.to_dict()
        workflow_template_spec = workflow_template_dict['spec'].copy()
        
        # metadata
        metadata = PipelineMetadata(
            pipeline=workflow_template_dict['metadata'],
            flow=workflow_template_spec.get('workflow_metadata',None),
            component=workflow_template_spec.get('pod_metadata',None),
        )
        
        # templates
        entrypoint_template = workflow_template_spec['entrypoint']
        templates = [
            ScriptTemplate.model_validate(template) 
                for template in workflow_template_spec['templates']
                    if template['name'] != entrypoint_template
        ]
        
        # inputs
        inputs = [PipelineInputParameter(**parameter) for parameter in workflow_template_spec['arguments']['parameters']]
        
        # dag
        dag = cls.build_dag(workflow_template_spec)

        return cls(
            metadata=metadata,
            templates=templates,
            inputs=inputs,
            dag=dag
        )
        
    @classmethod
    def transform_dag_visualization_node_position(cls, x_y: Tuple[float,float]) -> Tuple[float,float]:
        
        transformed_x_y = 350 * x_y[0], 150 * x_y[1]
        
        return transformed_x_y
        
    @staticmethod
    def create_dag_visualization_node_positions(inputs: List[PipelineInputParameter], dag: List[PipelineNode],include_task_io: bool = True) -> Dict[str,Tuple[int,int]]:
        """Utility to generate positional (x,y) coordinate tuples for each node of the dag that is getting visualized.
        Uses the networkx library's utilities to arrange nodes in a dag-friendly directional manner for visualization.
        Based on the DAG example at https://networkx.org/documentation/stable/auto_examples/graph/plot_dag_layout.html.
        Args:
            dag (List[PipelineNode]): The dag object.

        Returns:
            Dict[str,Tuple[int,int]]: A node uid->positional (x,y) coordinate mapping.
        """
        
        node_positions = {}
        
        G = nx.DiGraph()
        
        # add directed edges (adds nodes automatically)
        for task_node in dag:
            G.add_node(task_node.name)
            
            if not include_task_io:
                if task_node.depends is not None:
                    for upstream_node_name in task_node.depends:
                        G.add_edge(upstream_node_name,task_node.name)
            else:
                for interface_type in ['inputs','outputs']:

                    for argument_type in ['parameters','artifacts']:
                        arguments = getattr(getattr(task_node,interface_type),argument_type)
                        if not arguments:
                            continue
                    
                        for argument in arguments:
                            # add the task io node
                            task_io_node_name = f"{task_node.name}_{interface_type}_{argument_type}_{argument.name}"
                            G.add_node(task_io_node_name)
                            
                            # connect that task io node with the task node
                            if interface_type == 'inputs':
                                upstream_node_name = task_io_node_name
                                node_name = task_node.name
                            else:
                                upstream_node_name = task_node.name
                                node_name = task_io_node_name
                            
                            G.add_edge(upstream_node_name,node_name)
                            
                            # connect the input type task io node with the upstream output type task io node - where appropriate
                            if interface_type == 'inputs' and getattr(argument,'source',None) is not None:
                                task_io_source = argument.source
                                upstream_node_name = f"{task_io_source.node}_outputs_{task_io_source.output_type}_{task_io_source.output_name}"
                                G.add_edge(upstream_node_name,task_io_node_name)
        
        if include_task_io:
            for input in inputs:
                G.add_node(f"pipeline_outputs_parameters_{input.name}")
        
        # add layer attribute - required for multipartite layout
        for layer, nodes in enumerate(nx.topological_generations(G)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                G.nodes[node]["layer"] = layer
        
        node_positions_ar = nx.multipartite_layout(G, subset_key="layer")
                
        node_positions = dict(
            [
                (
                    k,
                    Pipeline.transform_dag_visualization_node_position(list(v))
                ) for k,v in node_positions_ar.items()
            ]
        )
                    
        return node_positions
        
    def create_dag_visualization_schema(self,include_task_io: bool = True) -> DagVisualizationSchema:
        """Utility method to generate the assets the barfi/baklavajs rendering engine uses to display the Pipeline's dag property on the frontend.
        """
        
        node_positions = self.create_dag_visualization_node_positions(self.inputs,self.dag,include_task_io)
        connections: List[Dict] = []
        nodes: List[Dict] = []
        
        for task_node in self.dag:
            
            task_node_name = task_node.name
            
            nodes.append(
                DagNode(
                    id=task_node_name,
                    pos =node_positions[task_node_name],
                    data = {"label": f"{PIPELINE_NODE_EMOJI_MAP['task']} {task_node_name}"},
                )
            )
            
            # we only create task_node <-> task_node connections if we dont display the tasks' IO specs
            if not include_task_io:
                if task_node.depends is not None:
                    for upstream_node_name in task_node.depends:
                        connections.append(
                            DagConnection(
                                id=f"{upstream_node_name}->{task_node_name}",
                                source = upstream_node_name,
                                target = task_node_name,
                                animated = True,
                                edge_type = "smoothstep",
                            )
                        )
            # if we include the tasks' IO specs, we need to draw 
            # - io nodes and
            # connections between 
            # - inputs and outputs, and 
            # - inputs/outputs and associated task_nodes
            else:
                for interface_type in ['inputs','outputs']:

                    for argument_type in ['parameters','artifacts']:
                        arguments = getattr(getattr(task_node,interface_type),argument_type)
                        if not arguments:
                            continue
                    
                        for argument in arguments:
                            # add the task io node
                            task_io_node_name = f"{task_node_name}_{interface_type}_{argument_type}_{argument.name}"
                            nodes.append(
                                DagNode(
                                    id=task_io_node_name,
                                    pos = node_positions[task_io_node_name],
                                    data = {
                                        "label": f"{PIPELINE_NODE_EMOJI_MAP[interface_type]['task']} {argument.name}",
                                        "value":getattr(argument,'value',None)
                                    },
                                    style={
                                        'backgroundColor': 'lightgrey'
                                    },
                                )
                            )
                            
                            # connect that task io node with the task node
                            if interface_type == 'inputs':
                                upstream_node_name = task_io_node_name
                                node_name = task_node_name
                            else:
                                upstream_node_name = task_node_name
                                node_name = task_io_node_name
                            
                            connections.append(
                                DagConnection(
                                    id=f"{upstream_node_name}->{node_name}",
                                    source = upstream_node_name,
                                    target = node_name,
                                    animated = False,
                                    edge_type = "smoothstep",
                                )
                            )
                            
                            # connect the input type task io node with the upstream output type task io node - where appropriate
                            if interface_type == 'inputs' and getattr(argument,'source',None) is not None:
                                task_io_source = argument.source
                                upstream_node_name = f"{task_io_source.node}_outputs_{task_io_source.output_type}_{task_io_source.output_name}"
                                connections.append(
                                    DagConnection(
                                        id=f"{upstream_node_name}->{task_io_node_name}",
                                        source = upstream_node_name,
                                        target = task_io_node_name,
                                        animated = True,
                                        edge_type = "smoothstep",
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
                            "value": input.value
                        },
                        style={
                            'backgroundColor': 'lightblue'
                        },
                        node_type="input"
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
    api_instance = workflow_template_service_api.WorkflowTemplateServiceApi(api_client)

    workflow_templates = api_instance.list_workflow_templates(namespace="argo")['items']
    print(f"Registered pipelines: {[workflow_template['metadata']['name'] for workflow_template in workflow_templates]}")
    workflow_template = workflow_templates[0]
    print(f"Workflow template: {workflow_template.to_dict()}")
    
    with open('pipeline_test_workflow_template.yaml','w') as file:
        yaml.dump(workflow_template.to_dict(),file)

    # convert to pipeline and show results
    pipeline = Pipeline.from_argo_workflow_cr(workflow_template)
    print(f"Pipeline: {pipeline.model_dump()}")
    
    with open('pipeline_test_pipeline.yaml','w') as file:
        yaml.dump(pipeline.model_dump(),file)
        
    # create dag frontend visuzliation schema and show results
    render_schema, render_blocks = pipeline.create_dag_visualization_assets()
    
    import pdb
    pdb.set_trace()
    
    print(f"Pipeline visualization schema: {render_schema}")
    
    with open('pipeline_test_visualization_schema.yaml','w') as file:
        yaml.dump(render_schema,file)
    
if __name__ == "__main__":
    main_test()