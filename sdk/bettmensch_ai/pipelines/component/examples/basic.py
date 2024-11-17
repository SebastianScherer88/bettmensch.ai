from bettmensch_ai.pipelines.component import (  # noqa: F401
    component,
    torch_ddp_component,
)
from bettmensch_ai.pipelines.io import (  # noqa: F401
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
)


def convert_to_artifact(
    a: InputParameter,
    a_art: OutputArtifact = None,
) -> None:
    """When decorated with the bettmensch_ai.components.component decorator,
    implements a bettmensch_ai.Component that converts its InputParameter into
    an OutputArtifact."""
    with open(a_art.path, "w") as a_art_file:
        a_art_file.write(str(a))


def show_artifact(a: InputArtifact) -> None:
    """When decorated with the bettmensch_ai.components.component decorator,
    implements a bettmensch_ai.Component that prints the values of its
    InputArtifact."""

    with open(a.path, "r") as a_art_file:
        a_content = a_art_file.read()

    print(f"Content of input artifact a: {a_content}")


def add_parameters(
    a: InputParameter = 1,
    b: InputParameter = 2,
    sum: OutputParameter = None,
) -> None:
    """When decorated with the bettmensch_ai.components.component decorator,
    implements a simple addition bettmensch_ai.Component."""

    sum.assign(a + b)


def show_parameter(a: InputParameter) -> None:
    """When decorated with the bettmensch_ai.components.component decorator,
    implements a bettmensch_ai.Component that prints the values of its
    InputParameter."""

    print(f"Content of input parameter a is: {a}")


convert_to_artifact_factory = component(convert_to_artifact)
convert_to_artifact_torch_ddp_factory = torch_ddp_component(
    convert_to_artifact
)  # noqa: E501
show_artifact_factory = component(show_artifact)
show_artifact_torch_ddp_factory = torch_ddp_component(show_artifact)
add_parameters_factory = component(add_parameters)
add_parameters_torch_ddp_factory = torch_ddp_component(add_parameters)
show_parameter_factory = component(show_parameter)
show_parameter_torch_ddp_factory = torch_ddp_component(show_parameter)
