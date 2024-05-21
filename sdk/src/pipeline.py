from typing import List, Dict, Any
from utils import get_func_args, validate_func_args
from arguments import PipelineInput, ComponentInput, ComponentOutput
from component import PipelineContext, _pipeline_context, component
from typing import Callable
from hera.workflows import WorkflowTemplate, DAG
import inspect
import yaml

class Pipeline(object):
    """Manages the PipelineContext meta data storage utility."""
    
    type: str = 'workflow'
    name: str = None
    namespace: str = None
    clear_context: bool = None
    inputs: Dict[str,PipelineInput] = None
    workflow_template: WorkflowTemplate = None
    
    def __init__(self, name: str, namespace: str, func: Callable, clear_context: bool = True):
        # pipelines dont have outputs        
        self.build(name, namespace, func, clear_context)
        
    def build(self, name: str, namespace: str, func: Callable, clear_context: bool):
        
        self.name = name
        self.namespace = namespace
        self.clear_context = clear_context
        self.func = func
        self.inputs = self.generate_inputs_from_func(func)
        self.workflow_template = self.build_workflow_template()
     
    @property
    def id(self):
        return f"{self.type}"
    
    @property
    def context(self) -> PipelineContext:
        """Utility handle for global pipeline context instance.

        Returns:
            PipelineContext: The global pipeline context.
        """
        
        return _pipeline_context
    
    @property
    def tasks(self) -> List:
        return self.context.components
    
    def generate_inputs_from_func(self,func: Callable) -> Dict[str,PipelineInput]:
        """Generates pipeline inputs from the underlying function. Also
        - checks for correct PipelineInput type annotations in the decorated original function
        - ensures all original function inputs without default values are being specified

        Args:
            func (Callable): The function the we want to wrap in a Component.
            
        Raises:
            Exception: Raised if the pipeline is not given an input for at least one of the underlying function's arguments 
                without default value.

        Returns:
            Dict[str,PipelineInput]: The pipeline's inputs.
        """
        
        validate_func_args(func,argument_types=[PipelineInput])
        func_inputs = get_func_args(func,'annotation',[PipelineInput])
        non_default_args = get_func_args(func,'default',[inspect._empty])
        required_func_inputs = dict([(k,v) for k,v in func_inputs.items() if k in non_default_args])
        
        result = {}
        
        for name in func_inputs:
            
            print(f"Pipeline input: {name}")
            
            # assemble component input
            pipeline_input = PipelineInput(name=name)
                
            pipeline_input.set_owner(self)
            
            # remove declared input from required inputs (if relevant)
            if name in required_func_inputs:
                del required_func_inputs[name]
            
            result[name] = pipeline_input
            
        # ensure no required inputs are left unspecified
        if required_func_inputs:
            raise Exception(f"Unspecified required input(s) left: {required_func_inputs}")
        
        return result
    
    def build_workflow_template(self) -> WorkflowTemplate:
        
        # add components to the global pipeline context
        with self:
            self.func(**self.inputs)
        
        # invoke all components' hera task generators from within a nested WorkflowTemplate & DAG context
        with WorkflowTemplate(
            name=self.name, 
            entrypoint='bettmensch_ai_dag',
            namespace=self.namespace,
            arguments=[input.to_hera_parameter() for input in self.inputs.values()],
        ) as wft:
        
            with DAG(name='bettmensch_ai_dag'):
                for component in self.context.components:
                    component.to_hera_task()
                    
        return wft
                
    def __enter__(self):
        # clear the global pipeline context when entering the pipeline instance's context, if specified
        if self.clear_context:
            _pipeline_context.clear()
            
        _pipeline_context.activate()
        
        return self
    
    def __exit__(self, *args, **kwargs):
        _pipeline_context.deactivate()
    
# def test_context_editing():
#     print(f"0 Pipeline context activated: {_pipeline_context.active}")
#     print(f"0 Pipeline context components: {_pipeline_context.components}")
    
#     with Pipeline('pipeline_1') as p1:
#         print(f"1 Pipeline context activated: {_pipeline_context.active}")
#         print(f"1 Pipeline context components: {_pipeline_context.components}")
    
#     print(f"2 Pipeline context activated: {_pipeline_context.active}")
#     print(f"2 Pipeline context components: {_pipeline_context.components}")
    
#     #c1 = Component('component_1') # pipeline context not active exception
        
#     with Pipeline('pipeline_2') as p2:
#         c1 = Component('component_1')
#         print(f"3 Pipeline context activated: {_pipeline_context.active}")
#         print(f"3 Pipeline context components: {[comp.id for comp in _pipeline_context.components]}")
        
#         c2 = Component('component_2')
#         print(f"4 Pipeline context activated: {_pipeline_context.active}")
#         print(f"4 Pipeline context components: {[comp.id for comp in _pipeline_context.components]}")
            
#     print(f"5 Pipeline context activated: {_pipeline_context.active}")
#     print(f"5 Pipeline context components: {[comp.id for comp in _pipeline_context.components]}")

# def test_parameter_attachment():
#     p1 = Pipeline('p1',inputs=[Input('param1',1),Input('param2',)]) # creates p1.inputs.param1 & p1.inputs.param2 handles
    
#     with p1:
                
#         c1 = Component('c1',inputs=[p1.inputs['param1'],p1.inputs['param2'],Input('param3','a')],outputs=[Output('out1')])
        
#         c2 = Component('c2',inputs=[c1.outputs['out1'],p1.inputs['param2']],outputs=[Output('out1')])
        
#     # p1
#     for p1_input_name, p1_input in p1.inputs.items():
#         print(f"Pipeline 1 input: {p1_input_name}: {p1_input.__dict__}")
#     print(f"Pipeline 1 tasks: {p1.tasks}")
        
#     # c1
#     for c1_input_name, c1_input in c1.inputs.items():
#         print(f"Component 1 input: {c1_input_name}: {c1_input.__dict__}")
        
#     for c1_output_name, c1_output in c1.outputs.items():
#         print(f"Component 1 output: {c1_output_name}: {c1_output.__dict__}")
        
#     print(f"Component 1 dependencies: {c1.depends_on}")
        
#     # c2
#     for c2_input_name, c2_input in c2.inputs.items():
#         print(f"Component 2 input: {c2_input_name}: {c2_input.__dict__}")
        
#     for c2_output_name, c2_output in c2.outputs.items():
#         print(f"Component 2 output: {c2_output_name}: {c2_output.__dict__}")
        
#     print(f"Component 2 dependencies: {c2.depends_on}")

# def decorator_func(x, y):
 
#     def Inner(func):
 
#         def wrapper(*args, **kwargs):
#             print("I like Geeksforgeeks")
#             print("Summation of values - {}".format(x+y) )
 
#             func(*args, **kwargs)
             
#         return wrapper
#     return Inner

def pipeline(name: str, namespace: str, clear_context: bool) -> Callable:
    """Takes a calleable and generates a configured Component factory that will
    generate a Component version of the callable if its __call__ is invokes inside an active PipelineContext.
    
    Usage:
    @bettmensch_ai.component #-> component factory
    def add(a: ComponentInput, b: ComponentInput, sum: ComponentOutput):
        sum.assign(a + b)
    
    Decorating the above `add` method should return a component factory that
    - generates a Component class instance when called from within an active PipelineContext
      1. add a post function output processing step that ensures hera-compatible writing of output parameters to sensible local file paths
      2. add the inputs parameter type inputs 'a' and 'b' to the component
      3. add the parameter type output 'sum' to the component
          3.1 this should facilitate the reference the file path from step 1. in the `from` argument further downstream at the stage of mapping to a ArgoWorkflowTemplate
    """
    
    def pipeline_factory(func: Callable) -> Pipeline:
        
        return Pipeline(name=name,namespace=namespace,clear_context=clear_context,func=func)
        
    return pipeline_factory

def test_pipeline():
    
    @component
    def add(a: ComponentInput, b: ComponentInput, sum: ComponentOutput) -> None:
        
        sum.assign(a + b)
        
    @pipeline('test_pipeline','argo',True)
    def a_plus_b_plus_c(a: PipelineInput,
                        b: PipelineInput,
                        c: PipelineInput = None):
        
        a_plus_b = add(a = a, b = b)        
        a_plus_b_plus_c = add(a = a_plus_b.outputs['sum'], b = c)
        
    print(f"Pipeline type: {type(a_plus_b_plus_c)}")
    print(f"Pipeline Workflow template: {a_plus_b_plus_c.workflow_template}")
    
    with open(f'{a_plus_b_plus_c.name}.yaml','w') as pipeline_file:
        yaml.dump(a_plus_b_plus_c.workflow_template.to_yaml(),pipeline_file)
    
    
if __name__ == "__main__":
    test_pipeline()
