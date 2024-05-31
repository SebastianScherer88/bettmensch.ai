from bettmensch_ai import (
    PIPELINE_TYPE,
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
    component,
    pipeline,
)


def test_artifact_pipeline(test_output_dir):
    """Declaration of Pipeline using InputArtifact and OutputArtifact"""

    @component
    def convert_to_artifact(
        a_param: InputParameter,
        b_param: InputParameter,
        a_art: OutputArtifact = None,
        b_art: OutputArtifact = None,
    ) -> None:

        with open(a_art.path, "w") as a_art_file:
            a_art_file.write(str(a_param))

        with open(b_art.path, "w") as b_art_file:
            b_art_file.write(str(b_param))

    @component
    def show_artifact(a: InputArtifact, b: InputArtifact) -> None:

        with open(a.path, "r") as a_art_file:
            a_content = a_art_file.read()

        with open(b.path, "r") as b_art_file:
            b_content = b_art_file.read()

        print(f"Content of input artifact a: {a_content}")
        print(f"Content of input artifact b: {b_content}")

    @pipeline("test-artifact-pipeline", "argo", True)
    def parameter_to_artifact(
        a: InputParameter = "Param A",
        b: InputParameter = "Param B",
    ) -> None:
        convert = convert_to_artifact(
            "convert-to-artifact",
            a_param=a,
            b_param=b,
        )

        show = show_artifact(
            "show-artifact",
            a=convert.outputs["a_art"],
            b=convert.outputs["b_art"],
        )

    parameter_to_artifact.export(test_output_dir)
    # parameter_to_artifact.register()
    # parameter_to_artifact.run(a="Test value A", b="Test value b")


def test_parameter_pipeline(test_output_dir):
    """Declaration of Pipeline using InputParameter and OutputParameter"""

    @component
    def add(
        a: InputParameter = 1,
        b: InputParameter = 2,
        sum: OutputParameter = None,
    ) -> None:

        sum.assign(a + b)

    @pipeline("test-parameter-pipeline", "argo", True)
    def a_plus_b_plus_2(a: InputParameter = 1, b: InputParameter = 2) -> None:
        a_plus_b = add(
            "a-plus-b",
            a=a,
            b=b,
        )

        a_plus_b_plus_2 = add(
            "a-plus-b-plus-2",
            a=a_plus_b.outputs["sum"],
            b=InputParameter("two", 2),
        )

    a_plus_b_plus_2.export(test_output_dir)
    # a_plus_b_plus_2.register()
    # a_plus_b_plus_2.run(a=3, b=2)
