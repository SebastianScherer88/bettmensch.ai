import os
from typing import Callable, List

import pytest
from bettmensch_ai.pipelines.constants import ResourceType
from bettmensch_ai.pipelines.io import (
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
)
from pydantic import BaseModel


@pytest.fixture
def test_output_dir():
    return os.path.join(".", "sdk", "test", "unit", "outputs")


@pytest.fixture
def test_mock_pipeline():
    class MockPipeline:
        type = ResourceType.pipeline.value
        io_owner_name = ResourceType.pipeline.value

    return MockPipeline()


@pytest.fixture
def test_mock_component():
    class MockComponent:
        type = ResourceType.component.value
        name = "mock-component-0"
        io_owner_name = f"{type}.{name}"

    return MockComponent()


@pytest.fixture
def test_mock_script(test_function_and_task_inputs):
    test_function, _ = test_function_and_task_inputs

    class MockArgument(BaseModel):
        name: str

    class MockIO(BaseModel):
        parameters: List[MockArgument]
        artifacts: List[MockArgument]

    class MockScript:
        source: Callable = test_function
        add_cwd_to_sys_path: bool = False

        def _build_inputs(self):

            return MockIO(
                parameters=[
                    MockArgument(name="a"),
                    MockArgument(name="b"),
                    MockArgument(name="c"),
                ],
                artifacts=[MockArgument(name="d")],
            )

        def _build_outputs(self):

            return MockIO(
                parameters=[MockArgument(name="a_out")],
                artifacts=[MockArgument(name="b_out")],
            )

    return MockScript()


@pytest.fixture
def test_function_and_task_inputs(test_mock_pipeline, test_mock_component):
    def test_function(
        a: InputParameter,
        b: InputParameter,
        c: InputParameter,
        d: InputArtifact,
        a_out: OutputParameter,
        b_out: OutputArtifact,
    ):
        pass

    test_input_a = InputParameter("fixed", 1)
    test_input_b = InputParameter("mock_pipe_in", 1)
    test_input_b.set_owner(test_mock_pipeline)
    test_input_c = OutputParameter("mock_comp_out_param")
    test_input_c.set_owner(test_mock_component)
    test_input_d = OutputArtifact("mock_comp_out_art")
    test_input_d.set_owner(test_mock_component)

    task_inputs = {
        "a": test_input_a,
        "b": test_input_b,
        "c": test_input_c,
        "d": test_input_d,
    }

    return test_function, task_inputs
