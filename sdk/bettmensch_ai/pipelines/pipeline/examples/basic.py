from bettmensch_ai.pipelines import as_pipeline
from bettmensch_ai.pipelines.component.examples import (
    add_parameters_factory,
    convert_to_artifact_factory,
    show_artifact_factory,
)
from bettmensch_ai.pipelines.constants import ARGO_NAMESPACE
from bettmensch_ai.pipelines.io import (
    InputParameter,
    OutputArtifact,
    OutputParameter,
)


@as_pipeline("test-artifact-pipeline", ARGO_NAMESPACE, True)
def parameter_to_artifact_pipeline(
    a: InputParameter = "Param A",
    b: OutputArtifact = None,
) -> None:
    convert = convert_to_artifact_factory(
        "convert-to-artifact",
        a=a,
    )

    show_artifact = show_artifact_factory(
        "show-artifact",
        a=convert.outputs["a_art"],
    )

    # assign pipeline output
    b.set_source(show_artifact.outputs["b"])


@as_pipeline("test-parameter-pipeline", ARGO_NAMESPACE, True)
def adding_parameters_pipeline(
    a: InputParameter = 1, b: InputParameter = 2, sum: OutputParameter = None
) -> None:
    a_plus_b = add_parameters_factory(
        "a-plus-b",
        a=a,
        b=b,
    )

    a_plus_b_plus_2 = add_parameters_factory(
        "a-plus-b-plus-2",
        a=a_plus_b.outputs["sum"],
        b=InputParameter("two", 2),
    )

    # assign pipeline output
    sum.set_source(a_plus_b_plus_2.outputs["sum"])
