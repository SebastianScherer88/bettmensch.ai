from bettmensch_ai.io import (
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
