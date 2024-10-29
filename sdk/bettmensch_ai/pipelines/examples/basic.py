from bettmensch_ai.components.examples import (
    add_parameters_factory,
    convert_to_artifact_factory,
    show_artifact_factory,
)
from bettmensch_ai.constants import ARGO_NAMESPACE
from bettmensch_ai.io import InputParameter
from bettmensch_ai.pipelines import pipeline


@pipeline("test-artifact-pipeline", ARGO_NAMESPACE, True)
def parameter_to_artifact_pipeline(
    a: InputParameter = "Param A",
) -> None:
    convert = convert_to_artifact_factory(
        "convert-to-artifact",
        a=a,
    )

    show_artifact_factory(
        "show-artifact",
        a=convert.outputs["a_art"],
    )


@pipeline("test-parameter-pipeline", ARGO_NAMESPACE, True)
def adding_parameters_pipeline(
    a: InputParameter = 1, b: InputParameter = 2
) -> None:
    a_plus_b = add_parameters_factory(
        "a-plus-b",
        a=a,
        b=b,
    )

    add_parameters_factory(
        "a-plus-b-plus-2",
        a=a_plus_b.outputs["sum"],
        b=InputParameter("two", 2),
    )
