from bettmensch_ai import (
    PIPELINE_TYPE,
    InputParameter,
    OutputParameter,
    _pipeline_context,
    component,
)
from hera.workflows import DAG, Parameter, WorkflowTemplate


def test_hera_component(test_output_dir):
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
    pipeline_input_c = InputParameter(name="c", value=3)
    pipeline_input_c.set_owner(MockPipeline())

    _pipeline_context.activate()
    _pipeline_context.clear()

    # add components to pipeline context
    a_plus_b = add(
        "a_plus_b",
        a=pipeline_input_a,
        b=pipeline_input_b,
    )

    a_plus_b_plus_c = add(
        "a_plus_b_plus_c",
        a=a_plus_b.outputs["sum"],
        b=pipeline_input_c,
    )

    # close pipeline context
    _pipeline_context.deactivate()

    with WorkflowTemplate(
        name="test-parameter-component-workflow-template",
        entrypoint="test_dag",
        namespace="argo",
        arguments=[
            Parameter(name="a"),
            Parameter(name="b"),
            Parameter(name="c"),
        ],
    ) as wft:

        with DAG(name="test_dag"):
            a_plus_b.to_hera()
            a_plus_b_plus_c.to_hera()

    wft.to_file(test_output_dir)
