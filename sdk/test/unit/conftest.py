import os

import pytest
from bettmensch_ai.constants import COMPONENT_TYPE, PIPELINE_TYPE


@pytest.fixture
def test_output_dir():
    return os.path.join(".", "sdk", "test", "unit", "outputs")


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
