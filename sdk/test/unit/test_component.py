from bettmensch_ai import (
    PIPELINE_TYPE,
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
    _pipeline_context,
    component,
)
from hera.workflows import DAG, Parameter, WorkflowTemplate


def test_artifact_component(test_output_dir):
    """Declaration of Component using InputArtifact and OutputArtifact"""

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

    class MockPipeline:
        type: str = PIPELINE_TYPE
        io_owner_name: str = PIPELINE_TYPE

    # mock active pipeline with 3 inputs
    pipeline_input_a = InputParameter(name="a", value=1)
    pipeline_input_a.set_owner(MockPipeline())
    pipeline_input_b = InputParameter(name="b", value=2)
    pipeline_input_b.set_owner(MockPipeline())

    _pipeline_context.activate()
    _pipeline_context.clear()

    # add components to pipeline context
    convert = convert_to_artifact(
        "convert_to_artifact",
        a_param=pipeline_input_a,
        b_param=pipeline_input_b,
    )

    show = show_artifact(
        "show_artifact",
        a=convert.outputs["a_art"],
        b=convert.outputs["b_art"],
    )

    # close pipeline context
    _pipeline_context.deactivate()

    with WorkflowTemplate(
        name="test-parameter-component-workflow-template",
        entrypoint="test_dag",
        namespace="argo",
        arguments=[
            Parameter(name="a", value="Parameter A"),
            Parameter(name="b", value="Parameter B"),
        ],
    ) as wft:

        with DAG(name="test_dag"):
            convert.to_hera()
            show.to_hera()

    wft.to_file(test_output_dir)


def test_parameter_component(test_output_dir):
    """Declaration of Component using InputParameter and OutputParameter"""

    @component
    def add(
        a: InputParameter = 1,
        b: InputParameter = 2,
        sum: OutputParameter = None,
    ) -> None:

        sum.assign(a + b)

    class MockPipeline:
        type: str = PIPELINE_TYPE
        io_owner_name: str = PIPELINE_TYPE

    # mock active pipeline with 3 inputs
    pipeline_input_a = InputParameter(name="a", value=1)
    pipeline_input_a.set_owner(MockPipeline())
    pipeline_input_b = InputParameter(name="b", value=2)
    pipeline_input_b.set_owner(MockPipeline())

    _pipeline_context.activate()
    _pipeline_context.clear()

    # add components to pipeline context
    a_plus_b = add(
        "a_plus_b",
        a=pipeline_input_a,
        b=pipeline_input_b,
    )

    a_plus_b_plus_2 = add(
        "a_plus_b_plus_2",
        a=a_plus_b.outputs["sum"],
        b=InputParameter("two", 2),
    )

    # close pipeline context
    _pipeline_context.deactivate()

    with WorkflowTemplate(
        name="test-parameter-component-workflow-template",
        entrypoint="test_dag",
        namespace="argo",
        arguments=[
            Parameter(name="a", value=1),
            Parameter(name="b", value=2),
        ],
    ) as wft:

        with DAG(name="test_dag"):
            a_plus_b.to_hera()
            a_plus_b_plus_2.to_hera()

    wft.to_file(test_output_dir)
