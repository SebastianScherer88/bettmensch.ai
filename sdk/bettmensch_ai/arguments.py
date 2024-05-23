import os
from typing import Any, Union

from hera.workflows import Parameter

# --- type annotations
OUTPUT_BASE_PATH = os.path.join(".", "temp", "outputs")


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

        return os.path.join(self.name)

    def export(self):

        with open(self.path, "w") as output_file:
            output_file.write(str(self.value))


class ParameterMetaMixin(object):

    owner: Union["Component", "Pipeline"] = None
    source: Union["PipelineInput", "ComponentOutput"] = None
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

        self.source = (
            source  # "{{" + source.id + "}}"  # "workflow.parameters.input_1",
        )
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

        hera_expression = (
            "{{"
            + f"{self.owner.parameter_owner_name}.parameters.{self.name}"
            + "}}"
        )

        return hera_expression

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

        We add this for completeness' sake, even though ComponentInputs will
        typically not be referenced as parameter source by any container.

        Returns:
            str: The hera parameter reference expression.
        """

        hera_expression = (
            "{{"
            + f"{self.owner.parameter_owner_name}.{self.type}.parameters.{self.name}"
            + "}}"
        )

        return hera_expression

    def to_hera_parameter(self) -> Parameter:
        # ComponentInput annotated function arguments' are not always
        # referencing another parameter (PipelineInput or ComponentOutput), so
        # we reference the source parameter's `id` '{{...}}' expression only if
        # the provided source has a non-trivial owner. In that case, the value
        # will be the hera expression referencing the source argument.
        # If the provided source has no owner, we are dealing with a hardcoded
        # template function argument spec for this component, and retain the
        # value (which could be None). This allows us to hardcode an
        # input to a Component in a Pipeline that is different to the
        # Component's template function's default value for that argument,
        # without having to create a PipelineInput.
        if self.source.owner is not None:
            return Parameter(name=self.name, value=self.source.id)
        else:
            return Parameter(name=self.name, value=self.value)


class ComponentOutput(ParameterMetaMixin, Output):
    @property
    def id(self) -> str:
        """Utility method to generate a hera/ArgoWorkflow parameter reference
        to be used when constructing the hera DAG.

        Returns:
            str: The hera parameter reference expression.
        """

        hera_expression = (
            "{{"
            + f"{self.owner.parameter_owner_name}.{self.type}.parameters.{self.name}"
            + "}}"
        )

        return hera_expression

    def to_hera_parameter(self) -> Parameter:
        # ComponentOutput annotated function arguments wont have a value
        # defined, and will export 'null' as a default value in the Script
        # template definition, allowing us to invoke it from a DAG without
        # specifying the inputs that are of type ComponentOutput
        return Parameter(name=self.name, value=self.value)
