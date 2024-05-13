from __future__ import annotations
from typing import Dict, Optional, List, Union, Tuple, Literal
import argo_workflows
from argo_workflows.model.io_argoproj_workflow_v1alpha1_workflow_template import IoArgoprojWorkflowV1alpha1WorkflowTemplate
from argo_workflows.api import workflow_template_service_api
from pydantic import BaseModel
from datetime import datetime
import yaml
from dag import DagConnection, DagNodePosition, DagNode, DagPanning, DagSchema
from dag_builder.block_builder import Block

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
    parameters: Optional[List[NodeParameterInput]] = None
    artifacts: Optional[List[NodeArtifactInput]] = None

# outputs
class NodeOutput(BaseModel):
    name: str
        
class NodeParameterOutput(NodeOutput):
    value_from: Optional[Union[str,Dict]] = None

class NodeArtifactOutput(NodeOutput):
    path: str
    value_from: Optional[Union[str,Dict]] = None
    

class NodeOutputs(BaseModel):
    parameters: Optional[List[NodeParameterOutput]] = None
    artifacts: Optional[List[NodeArtifactOutput]] = None

class PipelineNode(BaseModel):
    """A pipeline node is an ArgoWorkflow DAG type template's task.
    """
    name: str
    template: str
    inputs: Optional[NodeInputs] = None
    outputs: Optional[NodeOutputs] = None
    depends: Optional[Union[str,List[str]]] = None

# --- Pipeline
class Pipeline(BaseModel):
    metadata: PipelineMetadata
    templates: List[ScriptTemplate]
    inputs: List[PipelineInputParameter] = []
    dag: List[PipelineNode]
    
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
        #print(f"Trying to resolve value expression: {expression}")
        
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
                'depends':task["depends"].split(" && ") if task.get('depends') else None
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
        
    @staticmethod
    def create_dag_interface_name(interface_type: Literal['input','output'],argument_name: str) -> str:
        """Utility method to generate a baklavajs node interface name using the convention f"{interface_type[:-4].title()}:{argument_name}".

        Args:
            interface_type (Literal[&#39;input&#39;,&#39;output&#39;]): _description_
            argument_name (str): _description_

        Returns:
            str: The name of the node interface.
        """
        return f"{interface_type[:-4].title()}: {argument_name}"
        
    @staticmethod
    def create_dag_interface_id(interface_type: Literal['input','output'],pipeline_node_name: str, argument_type: Literal['parameters','artifacts'],argument_name: str) -> str:
        """Utility method to generate a baklavajs node interface id using the convention f"{interface_type}_{pipeline_node_name}_{argument_type}_{argument_name}".

        Args:
            interface_type (Literal[&#39;inputs&#39;,&#39;output&#39;]): _description_
            pipeline_node_name (str): _description_
            argument_type (Literal[&#39;parameters&#39;,&#39;artifacts&#39;]): _description_
            argument_name (str): _description_

        Returns:
            str: The id of the node interface
        """
        return f"{interface_type}_{pipeline_node_name}_{argument_type}_{argument_name}"
        
    def create_dag_visualization_assets(self) -> Tuple[DagSchema,List[type[Block]]]:
        """Utility method to generate the assets the barfi/baklavajs rendering engine uses to display the Pipeline's dag property on the frontend.
        """
                
        connections: List[Dict] = []
        nodes: List[Dict] = []
        panning = DagPanning().model_dump()
        scaling = 1
        
        node_classes = []
        
        for i, pipeline_node in enumerate(self.dag):

            schema_node_interfaces = []
            node_class = Block(name=pipeline_node.name)
            
            for interface_type in ['inputs','outputs']:
                interfaces = getattr(pipeline_node,interface_type)

                if interfaces is None:
                    continue

                for argument_type in ['parameters','artifacts']:
                    arguments = getattr(interfaces,argument_type,None)
                    
                    if arguments is None:
                        continue
                    
                    for argument in arguments:
                        interface_name = self.create_dag_interface_name(interface_type, argument.name)
                        interface_id = self.create_dag_interface_id(interface_type,pipeline_node.name,argument_type,argument.name)
                        interface_value = getattr(argument,"value",None)
                        
                        if interface_type == 'inputs':
                            node_class.add_input(interface_name)
                            
                            if getattr(argument,"source",None) is not None:
                                dag_connection_dict = {
                                    "from":self.create_dag_interface_id("outputs",argument.source.node,argument.source.output_type,argument.source.output_name),
                                    "to":interface_id
                                }
                                dag_connection_dict['id'] = f"{dag_connection_dict['from']}->{dag_connection_dict['to']}"
                                DagConnection(**dag_connection_dict).model_validate(dag_connection_dict)
                                connections.append(dag_connection_dict)
                        
                        elif interface_type == 'outputs':
                            node_class.add_output(interface_name)
                        
                        schema_node_interfaces.append([interface_name, {"id":interface_id, "value":interface_value}])
                        
            node_classes.append(node_class)

            dag_schema_node = DagNode(
                id = pipeline_node.name,
                interfaces=schema_node_interfaces,
                name = pipeline_node.name,
                type = pipeline_node.name,
                position = DagNodePosition(
                    x = i * 200 + 50,
                    y = 200
                )
            )
            nodes.append(dag_schema_node.model_dump())
        
        entrypoint_node_class = Block(name="pipeline")
        [entrypoint_node_class.add_output(name=self.create_dag_interface_name("inputs",input.name)) for input in self.inputs]
        node_classes.append(entrypoint_node_class)
        
        dag_schema_entrypoint_node = DagNode(
            id = "pipeline",
            interfaces=[
                [
                    self.create_dag_interface_name("inputs",input.name),
                    {
                        "id": self.create_dag_interface_id("outputs","pipeline","parameters",input.name),
                        "value": input.value
                    }
                 ] for input in self.inputs
            ],
            name = f"pipeline {self.metadata.pipeline.name}",
            type = "pipeline",
            position = DagNodePosition(
                x = -200 + 50,
                y = 200
            )
        )
        nodes.append(dag_schema_entrypoint_node.model_dump())
        
        return {
            "connections": connections,
            "nodes": nodes,
            "panning":panning,
            "scaling": scaling
        }, node_classes
        
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