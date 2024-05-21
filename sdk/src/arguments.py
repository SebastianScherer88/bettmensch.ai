from typing import Any
from hera.workflows import Parameter
import os

# --- type annotations
OUTPUT_BASE_PATH = './temp/outputs'

# class InputParameter(object):
#     ...
    
# class OutputParameter(object):
#     """Utility class for type annotation and retrieving function outputs from inside the function context."""
    
#     def __init__(self,name: str, type: type):
#         self.name = name
#         #self.type = type
#         self.value: type = None
    
#     def assign(self, value: Any):
#         # if not isinstance(value,self.type):
#         #     raise TypeError(f"Value {value} is not of OutputParameter type {self.type}.")
        
#         self.value = value
        
# class OutputPath(object):
#     """Utility class for type annotation and resolving making function file output locations available outside of function context."""
    
#     def __init__(self,name: str):
#         self.name = name
    
#     @property
#     def path(self):
#         """Resolves to local file path inside component container."""
        
#         return os.path.join(OUTPUT_BASE_PATH,self.name)    
    
# --- container interfaces
class Input(object):
    
    type = 'inputs'
    
    def __init__(self, name: str, value: Any = None):
        self.name = name
        self.value = value
    
class Output(object):
    
    type = 'outputs'
    
    def __init__(self, name: str):
        self.name = name
        self.value = None
        
    def assign(self, value: Any):
        self.value = value
        
        self.export()
        
    @property
    def path(self):
        
        return os.path.join(OUTPUT_BASE_PATH,self.name)
    
    def export(self):
        
        with open(self.path,'w') as output_file:
            output_file.write(self.value)

class ParameterMetaMixin(object):
    
    owner: str = None
    source: str = None
    id: str = None
    
    def set_owner(self, owner: Any):
        # if not isinstance(owner, BaseContainerMixin):
        #     raise TypeError(f"The specified parameter owner {owner} has to be either a Pipeline or Component type.")
        
        self.owner = owner.id # "workflow", "tasks.component-c1-0" etc.
        
    def set_source(self, source: Any):
        if not isinstance(source, (PipelineInput,ComponentOutput)):
            raise TypeError(f"The specified parameter source {source} has to be either a PipelineInput or ComponentOutput type.")
        
        self.source = "{{" + source.id + "}}" # "workflow.parameters.input_1", "tasks.component-c1-0.outputs.output_1" etc.
        
    @property
    def id(self) -> str:
        """Utility method to generate a hera/ArgoWorkflow parameter reference to be used when constructing the hera DAG.

        Returns:
            str: The hera parameter reference expression.
        """
        
        if self.owner == 'workflow':
            return f"workflow.parameters.{self.name}"
        else:
            return f"{self.owner}.{self.type}.parameters.{self.name}"
        
    def to_hera_parameter(self) -> Parameter:
        
        return Parameter(name=self.name,value=self.source)
        
class ContainerInput(ParameterMetaMixin,Input):
    ...
    
class PipelineInput(ContainerInput):
    ...
    
class ComponentInput(PipelineInput):
    ...
    
class ComponentOutput(ParameterMetaMixin,Output):
    ...