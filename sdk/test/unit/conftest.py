import os

import pytest
from bettmensch_ai import (
    COMPONENT_TYPE,
    PIPELINE_TYPE,
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
)


@pytest.fixture
def test_output_dir():
    return os.path.join(".", "sdk", "test", "unit")


@pytest.fixture
def test_mock_pipeline():
    class MockPipeline:
        type = PIPELINE_TYPE
        io_owner_name = PIPELINE_TYPE

    return MockPipeline()


@pytest.fixture
def test_mock_component():
    class MockComponent:
        type = COMPONENT_TYPE
        name = "mock-component-0"
        io_owner_name = f"{type}.{name}"

    return MockComponent()


@pytest.fixture
def test_convert_to_artifact_function():
    def convert_to_artifact(
        a: InputParameter,
        b: InputParameter,
        a_art: OutputArtifact = None,
        b_art: OutputArtifact = None,
    ) -> None:

        with open(a_art.path, "w") as a_art_file:
            a_art_file.write(str(a))

        with open(b_art.path, "w") as b_art_file:
            b_art_file.write(str(b))

    return convert_to_artifact


@pytest.fixture
def test_show_artifact_function():
    def show_artifact(a: InputArtifact, b: InputArtifact) -> None:

        with open(a.path, "r") as a_art_file:
            a_content = a_art_file.read()

        with open(b.path, "r") as b_art_file:
            b_content = b_art_file.read()

        print(f"Content of input artifact a: {a_content}")
        print(f"Content of input artifact b: {b_content}")

    return show_artifact


@pytest.fixture
def test_add_function():
    def add(
        a: InputParameter = 1,
        b: InputParameter = 2,
        sum: OutputParameter = None,
    ) -> None:

        sum.assign(a + b)

    return add
