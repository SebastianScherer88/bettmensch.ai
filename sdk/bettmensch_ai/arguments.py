import os
from typing import Any, Union

from hera.workflows import Parameter

# --- type annotations
OUTPUT_BASE_PATH = "./temp/outputs"


class Input(object):

    type = "inputs"

    def __init__(self, name: str, value: Any = None):
        self.name = name
        self.value = value


class Output(object):

    type = "outputs"

    def __init__(self, name: str):
        self.name = name
        self.value = None

    def assign(self, value: Any):
        self.value = value

        self.export()

    @property
    def path(self):

        return os.path.join(OUTPUT_BASE_PATH, self.name)

    def export(self):

        with open(self.path, "w") as output_file:
            output_file.write(self.value)


class ParameterMetaMixin(object):

    owner: Union["Component", "Pipeline"] = None
    source: str = None
    id: str = None

    def set_owner(self, owner: Union["Component", "Pipeline"]):
        # if not isinstance(owner, BaseContainerMixin):
        #     raise TypeError(f"The specified parameter owner {owner} has to be "
        #                     "either a Pipeline or Component type.")

        self.owner = owner  # "workflow", "tasks.component-c1-0" etc.

    def set_source(self, source: Union["PipelineInput", "ComponentOutput"]):
        if not isinstance(source, (PipelineInput, ComponentOutput)):
            raise TypeError(
                f"The specified parameter source {source} has to be either a "
                "PipelineInput or ComponentOutput type."
            )

        self.source = "{{" + source.id + "}}"  # "workflow.parameters.input_1",
        # "tasks.component-c1-0.outputs.output_1" etc.


class ContainerInput(ParameterMetaMixin, Input):
    ...


class PipelineInput(ContainerInput):
    @property
    def id(self) -> str:
        """Utility method to generate a hera/ArgoWorkflow parameter reference
        to be used when constructing the hera DAG.

        Returns:
            str: The hera parameter reference expression.
        """
        return f"{self.owner.parameter_owner_name}.parameters.{self.name}"

    def to_hera_parameter(self) -> Parameter:
        # PipelineInput annotated function arguments' default values are
        # retained by the Pipeline class. We only include a default value
        # if its non-trivial
        if self.value is not None:
            return Parameter(name=self.name, value=self.value)
        else:
            return Parameter(name=self.name)


class ComponentInput(PipelineInput):
    @property
    def id(self) -> str:
        """Utility method to generate a hera/ArgoWorkflow parameter reference
        to be used when constructing the hera DAG.

        Returns:
            str: The hera parameter reference expression.
        """

        return f"{self.owner.parameter_owner_name}.{self.type}.parameters.{self.name}"

    def to_hera_parameter(self) -> Parameter:
        # ComponentInput annotated function arguments' are always referencing
        # another parameter (PipelineInput or ComponentOutput), so we reference
        # the source parameter '{{...}}' expression that was stored in the
        # ComponentInput's source attribute
        return Parameter(name=self.name, value=self.source)


class ComponentOutput(ParameterMetaMixin, Output):
    @property
    def id(self) -> str:
        """Utility method to generate a hera/ArgoWorkflow parameter reference
        to be used when constructing the hera DAG.

        Returns:
            str: The hera parameter reference expression.
        """

        return f"{self.owner.parameter_owner_name}.{self.type}.parameters.{self.name}"

    def to_hera_parameter(self) -> Parameter:
        # ComponentOutput annotated function arguments wont have a value
        # defined, and will export 'null' as a default value in the Script
        # template definition, allowing us to invoke it from a DAG without
        # specifying the inputs that are of type ComponentOutput
        return Parameter(name=self.name, value=self.value)
