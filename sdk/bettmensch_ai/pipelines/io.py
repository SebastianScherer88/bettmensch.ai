import os
from typing import Any, Union

from hera.workflows import Artifact, Parameter, models

from .constants import ArgumentType, IOType, ResourceType


class IO(object):

    name: str
    value: Any

    def __init__(self, name: str, value: Any = None):
        self.name = name
        self.value = value

    def assign(self, value: Any):
        self.value = value

        self.export()

    @property
    def path(self):

        return os.path.join(self.name)

    def export(self):

        with open(self.path, "w") as output_file:
            output_file.write(str(self.value))

    def __eq__(self, other):

        return (
            self.name == other.name
            and self.value == other.value
            and self.source == other.source
            and self.owner == other.owner
        )

    def __repr__(self):

        return f"""Type: {type(self)}, {str(
            {
                "name": self.name,
                "value": self.value,
                "source": self.source,
                "owner": self.owner,
                "type": self.type,
                "argument_type": self.argument_type
            }
        )}"""


class Input(IO):

    type: str = IOType.inputs.value


class Output(IO):

    type: str = IOType.outputs.value


class OriginMixin(object):

    owner: Union["Component", "Pipeline"] = None  # noqa: F821
    source: Union["InputParameter", "OutputParameter", "OutputArtifact"] = None
    id: str = None

    def set_owner(self, owner: Union["Component", "Pipeline"]):  # noqa: F821
        try:
            assert owner.type in (
                ResourceType.component.value,
                ResourceType.pipeline.value,
            )
        except (AttributeError, AssertionError):
            raise TypeError(
                f"The specified parameter owner {owner} has to be either a "
                "Pipeline or Component type."
            )

        self.owner = owner

    def set_source(
        self,
        source: Union["InputParameter", "OutputParameter", "OutputArtifact"],
    ):
        if not isinstance(
            source, (InputParameter, OutputParameter, OutputArtifact)
        ):
            raise TypeError(
                f"The specified parameter source {source} has to be one of: "
                "(InputParameter, OutputParameter, OutputArtifact)."
            )

        self.source = source


class InputParameter(OriginMixin, Input):

    argument_type: str = ArgumentType.parameter.value

    @property
    def id(self) -> str:
        """Utility method to generate a hera/ArgoWorkflow parameter reference
        to be used when constructing the hera DAG.

        Returns:
            str: The hera parameter reference expression.
        """

        hera_expression = (
            "{{" + f"{self.owner.io_owner_name}.parameters.{self.name}" + "}}"
        )

        return hera_expression

    def to_hera(self) -> Parameter:
        # PipelineInput annotated function arguments' default values are
        # retained by the Pipeline class. We only include a default value
        # if its non-trivial

        owner = getattr(self, "owner", None)
        owner_type = getattr(owner, "type", None)
        source_owner = getattr(self.source, "owner", None)

        if owner_type == ResourceType.pipeline.value:
            # Pipeline owned InputParameters dont reference anything, so its
            # enough to pass name and value (which might be none)
            return Parameter(name=self.name, value=self.value)
        elif owner_type == ResourceType.component.value:
            # Component owned InputParameters are not always
            # referencing another InputParameter, so
            # we reference the source parameter's `id` '{{...}}' expression
            # only if the provided source has a non-trivial owner. In that
            # case, the value will be the hera expression referencing the
            # source argument. If the provided source has no owner, we are
            # dealing with a hardcoded template function argument spec for this
            # component, and retain the value (which could be None). This
            # allows us to hardcode an input to a Component in a Pipeline that
            # is different to the Component's template function's default value
            # for that argument, without having to create a PipelineInput.
            if source_owner is not None:
                return Parameter(name=self.name, value=self.source.id)
            else:
                return Parameter(name=self.name, value=self.value)


class OutputParameter(OriginMixin, Output):

    argument_type: str = ArgumentType.parameter.value

    @property
    def id(self) -> str:
        """Utility method to generate a hera/ArgoWorkflow parameter reference
        to be used when constructing the hera DAG.

        Returns:
            str: The hera parameter reference expression.
        """

        hera_expression = (
            "{{"
            + f"{self.owner.io_owner_name}.{self.type}.parameters.{self.name}"
            + "}}"
        )

        return hera_expression

    def to_hera(self) -> Parameter:

        owner = getattr(self, "owner", None)
        owner_type = getattr(owner, "type", None)

        if owner_type == ResourceType.pipeline.value:
            # Pipeline owned InputParameters dont reference anything, so its
            # enough to pass name and value (which might be none)
            raise NotImplementedError(
                "Pipeline owned OutputParameters are not" " supported yet."
            )
        elif owner_type == ResourceType.component.value:
            return Parameter(
                name=self.name, value_from=models.ValueFrom(path=self.path)
            )
        raise TypeError(
            f"The specified parameter owner {owner} has to be one of: "
            "(Component, TorchDDComponent)."
        )


class InputArtifact(OriginMixin, Input):

    argument_type: str = ArgumentType.artifact.value

    def to_hera(self, template: bool = False) -> Artifact:
        # InputArtifacts need to be converted to hera inputs in two places:
        # - when defining the hera template
        # - when defining the input value/reference for the DAG's task
        if template:
            return Artifact(name=self.name, path=self.path)
        else:
            source_owner = getattr(self.source, "owner", None)
            if source_owner is not None:
                return Artifact(name=self.name, from_=self.source.id)
            else:
                raise ValueError(
                    "InputArtifact must be associated with a source that is "
                    "owned either by a Pipeline or another Component"
                )


class OutputArtifact(OriginMixin, Output):

    argument_type: str = ArgumentType.artifact.value

    @property
    def id(self) -> str:
        """Utility method to generate a hera/ArgoWorkflow parameter reference
        to be used when constructing the hera DAG.

        Returns:
            str: The hera parameter reference expression.
        """

        hera_expression = (
            "{{"
            + f"{self.owner.io_owner_name}.{self.type}.artifacts.{self.name}"
            + "}}"
        )

        return hera_expression

    def to_hera(self) -> Artifact:

        owner = getattr(self, "owner", None)
        owner_type = getattr(owner, "type", None)

        if owner_type == ResourceType.pipeline.value:
            # OutputArtifacts dont reference anything, so its
            # enough to pass name and path
            raise NotImplementedError(
                "Pipeline owned OutputArtifacts are not" " supported yet."
            )
        elif owner_type == ResourceType.component.value:
            return Artifact(name=self.name, path=self.path)
