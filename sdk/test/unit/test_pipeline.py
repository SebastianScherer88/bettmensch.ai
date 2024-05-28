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
    """Declaration of Component's using InputArtifact and OutputArtifact"""

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
        # add components to pipeline context
        convert = convert_to_artifact(
            "convert_to_artifact",
            a_param=a,
            b_param=b,
        )

        show = show_artifact(
            "show_artifact",
            a=convert.outputs["a_art"],
            b=convert.outputs["b_art"],
        )

    parameter_to_artifact.export(test_output_dir)
    parameter_to_artifact.register()
    # parameter_to_artifact.run(a="Test value A", b="Test value b")


def test_parameter_pipeline(test_output_dir):
    """Declaration of Component's using InputParameter and OutputParameter"""

    @component
    def add(
        a: InputParameter = 1,
        b: InputParameter = 2,
        sum: OutputParameter = None,
    ) -> None:

        sum.assign(a + b)

    @pipeline("test-parameter-pipeline", "argo", True)
    def a_plus_b_plus_2(a: InputParameter = 1, b: InputParameter = 2) -> None:
        # add components to pipeline context
        a_plus_b = add(
            "a_plus_b",
            a=a,
            b=b,
        )

        a_plus_b_plus_2 = add(
            "a_plus_b_plus_2",
            a=a_plus_b.outputs["sum"],
            b=InputParameter("two", 2),
        )

    a_plus_b_plus_2.export(test_output_dir)
    a_plus_b_plus_2.register()
    # a_plus_b_plus_2.run(a=3, b=2)


# from bettmensch_ai import InputParameter, OutputParameter, component, pipeline


# def test_parameter_pipeline(test_output_dir):
#     @component
#     def add(
#         a: InputParameter, b: InputParameter, sum: OutputParameter = None
#     ) -> None:

#         sum.assign(a + b)

#     @component
#     def multiply(
#         a: InputParameter, b: InputParameter, product: OutputParameter = None
#     ) -> None:

#         product.assign(a * b)

#     @pipeline("test-parameter-pipeline", "argo", True)
#     def a_plus_bc_plus_2b(
#         a: InputParameter = 1, b: InputParameter = 2, c: InputParameter = 3
#     ):

#         b_c = multiply(
#             "bc",
#             a=b,
#             b=c,
#         )

#         two_b = multiply(
#             "b2",
#             a=b,
#             b=InputParameter(name="two", value=2),
#         )

#         a_plus_bc = add(
#             "a-plus-bc",
#             a=a,
#             b=b_c.outputs["product"],
#         )

#         result = add(
#             "a-plus-bc-plus-2b",
#             a=a_plus_bc.outputs["sum"],
#             b=two_b.outputs["product"],
#         )

#     a_plus_bc_plus_2b.export(test_output_dir)
#     # a_plus_bc_plus_2b.register()
#     # a_plus_bc_plus_2b.run(a=3, b=2, c=1)
