from typing import List, Union, Any, Tuple, Dict, Optional

   
# pipeline input
# - when defined:
#       take name & value from [Input] 
# - when stored as attributes: [PipelineInput]
#       need to take name, value & the pipeline owner reference

# component input:
# - when defined: one of [Input], [PipelineInput], [ComponentOutput]
#       3 options:
#           1. take name & value if: [Input]
#           2. take name & pipeline input if: [PipelineInput]
#           3. take name & component output if: [ComponentOutput]
# - when stored as component attribute: [ComponentInput]
#       3 options:
#           1. take name, value & leave source empty
#           2. take name, leave value empty & take pipeline input reference
#           3. take name, leave value empmty & take component output reference

# component output:
# - when defined: [Output]
#       take name
# - when stored as attributes: [ComponentOutput]
#       need to take name & the component owner reference

class PipelineContext(object):
    
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.components: List = []
        
    @property
    def active(self):
        return self.pipeline is not None
    
    def activate(self,pipeline):
        self.pipeline = pipeline
        
    def deactivate(self):
        self.pipeline = None
    
    def add_component(self,component):
        if self.active:
            if component.id in set([el.id for el in self.components]):
                raise ValueError(f"No duplicate Pipeline or Components allowed in one context. Duplicate id: {component.id}")
            else:
                self.components.append(component)
        else:
            raise Exception(f"Unable to add component {component.id} - pipeline context is not active.")
            
    def clear(self):
        self.components = []
        
_pipeline_context = PipelineContext()

class Input(object):
    
    type = 'input'
    
    def __init__(self, name: str, value: Any = None):
        self.name = name
        self.value = value
    
class Output(object):
    
    type = 'output'
    
    def __init__(self, name: str):
        self.name = name

class ParameterMetaMixin(object):
    
    owner: str = None
    source: str = None
    id: str = None
    
    def set_owner(self, owner: Any):
        if not isinstance(owner, (Pipeline,Component)):
            raise TypeError(f"The specified parameter owner {owner} has to be either a Pipeline or Component type.")
        
        self.owner = owner.id # "pipeline-p1", "component-c1" etc.
        self.id = f"{self.owner}.{self.type}.{self.name}"
    
    def set_source(self, source: Any):
        if not isinstance(source, (ContainerInput,ComponentOutput)):
            raise TypeError(f"The specified parameter source {source} has to be either a PipelineInput or ComponentOutput type.")
        
        self.source = source.id # "pipeline-p1.input.input_1", "component-c1.output.output_1" etc. 
        
class ContainerInput(ParameterMetaMixin,Input):
    ...
    
class ComponentOutput(ParameterMetaMixin,Output):
    ...
    
class BaseContainerMixin(object):
    def __init__(self,name: str, inputs: List[Union[Input,ContainerInput]] = [], outputs: List[Output] = []):
        self.name = name
        self.type = 'container'
        
    @property
    def id(self):
        return f"container.{self.type}.{self.name}"
        
class Pipeline(BaseContainerMixin):
    def __init__(self,name: str, inputs: List[Input] = [], clear_context: bool = True):
        # pipelines dont have outputs
        super().__init__(name, inputs=inputs)
        self.type = 'pipeline'
        
        self.inputs: Dict = self.generate_inputs(inputs)
        self.clear_context = clear_context
        
    def generate_inputs(self,inputs:List[Input]) -> Dict[str,ContainerInput]:
        
        result = {}
        
        for i in inputs:
            pipeline_input = ContainerInput(name=i.name,value=i.value)
            pipeline_input.set_owner(self)

            result[i.name] = pipeline_input
        
        return result
    
    def __enter__(self):
        if self.clear_context:
            _pipeline_context.clear()
            
        _pipeline_context.activate(self)
        
        return self
    
    def __exit__(self, *args, **kwargs):
        _pipeline_context.deactivate()
    
class Component(BaseContainerMixin):
    def __init__(self,name: str, inputs: List[Union[Input,ContainerInput]] = [], outputs: List[Output] = []):
        super().__init__(name,inputs=inputs,outputs=outputs)
        self.type = 'component'
        
        self.inputs: Dict = self.generate_inputs(inputs)
        self.outputs: Dict = self.generate_outputs(outputs)
        
        _pipeline_context.add_component(self)
        
    def generate_inputs(self,inputs: List[Union[Input,ContainerInput,ComponentOutput]]) -> Dict[str,ContainerInput]:
        
        result = {}
        
        for i in inputs:
            if isinstance(i,(Input,ContainerInput)):
                component_input = ContainerInput(name=i.name,value=i.value)
            else:
                component_input = ContainerInput(name=i.name)
                
            if isinstance(i,(ContainerInput,ComponentOutput)):
                component_input.set_source(i)
                
            component_input.set_owner(self)

            result[i.name] = component_input
        
        return result
    
    def generate_outputs(self,outputs:List[Output]) -> Dict[str,ComponentOutput]:
        
        result = {}
        
        for output in outputs:
            component_output = ComponentOutput(name=output.name)
            component_output.set_owner(self)
            
            result[output.name] = component_output
    
        return result
    
def test_context_editing():
    print(f"0 Pipeline context activated: {_pipeline_context.active}")
    print(f"0 Pipeline context components: {_pipeline_context.components}")
    
    with Pipeline('pipeline_1') as p1:
        print(f"1 Pipeline context activated: {_pipeline_context.active}")
        print(f"1 Pipeline context components: {_pipeline_context.components}")
    
    print(f"2 Pipeline context activated: {_pipeline_context.active}")
    print(f"2 Pipeline context components: {_pipeline_context.components}")
    
    #c1 = Component('component_1') # pipeline context not active exception
        
    with Pipeline('pipeline_2') as p2:
        c1 = Component('component_1')
        print(f"3 Pipeline context activated: {_pipeline_context.active}")
        print(f"3 Pipeline context components: {[comp.id for comp in _pipeline_context.components]}")
        
        c2 = Component('component_2')
        print(f"4 Pipeline context activated: {_pipeline_context.active}")
        print(f"4 Pipeline context components: {[comp.id for comp in _pipeline_context.components]}")
            
    print(f"5 Pipeline context activated: {_pipeline_context.active}")
    print(f"5 Pipeline context components: {[comp.id for comp in _pipeline_context.components]}")

def test_parameter_attachment():
    p1 = Pipeline('p1',inputs=[Input('param1',1),Input('param2',)]) # creates p1.inputs.param1 & p1.inputs.param2 handles
    
    with p1:
                
        c1 = Component('c1',inputs=[p1.inputs['param1'],p1.inputs['param2'],Input('param3','a')],outputs=[Output('out1')])
        
        c2 = Component('c2',inputs=[c1.outputs['out1'],p1.inputs['param2']],outputs=[Output('out1')])
        
    # p1
    for p1_input_name, p1_input in p1.inputs.items():
        print(f"Pipeline 1 input: {p1_input_name}: {p1_input.__dict__}")
        
    # c1
    for c1_input_name, c1_input in c1.inputs.items():
        print(f"Component 1 input: {c1_input_name}: {c1_input.__dict__}")
        
    for c1_output_name, c1_output in c1.outputs.items():
        print(f"Component 1 output: {c1_output_name}: {c1_output.__dict__}")
        
    # c2
    for c2_input_name, c2_input in c2.inputs.items():
        print(f"Component 2 input: {c2_input_name}: {c2_input.__dict__}")
        
    for c2_output_name, c2_output in c2.outputs.items():
        print(f"Component 2 output: {c2_output_name}: {c2_output.__dict__}")
        
if __name__ == "__main__":
    test_context_editing()
    test_parameter_attachment()