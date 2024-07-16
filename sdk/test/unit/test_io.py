import pytest
from bettmensch_ai.io import (
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
)
from hera.workflows import Artifact, Parameter, models


@pytest.mark.parametrize(
    "test_io,test_owner_fixture_name,test_expected_io_id",
    [
        (
            InputParameter("test_name"),
            "test_mock_pipeline",
            "{{workflow.parameters.test_name}}",
        ),
        (
            OutputArtifact("test_name"),
            "test_mock_component",
            "{{tasks.mock-component-0.outputs.artifacts.test_name}}",
        ),
        (
            OutputParameter("test_name"),
            "test_mock_component",
            "{{tasks.mock-component-0.outputs.parameters.test_name}}",
        ),
    ],
)
def test_set_owner_and_id(
    test_io, test_owner_fixture_name, test_expected_io_id, request
):
    test_owner = request.getfixturevalue(test_owner_fixture_name)
    test_io.set_owner(test_owner)

    assert test_io.id == test_expected_io_id


@pytest.mark.parametrize(
    "test_io",
    [
        InputArtifact("test_name"),
        InputParameter("test_name"),
        OutputArtifact("test_name"),
        OutputParameter("test_name"),
    ],
)
def test_raise_invalid_owner(test_io):

    with pytest.raises(TypeError, match="The specified parameter owner"):
        test_io.set_owner("invalid_owner_value")


@pytest.mark.parametrize(
    "test_io,test_owner_fixture_name,test_expected_hera_parameter",
    [
        (
            OutputArtifact("test_name"),
            "test_mock_component",
            Artifact(name="test_name", path="test_name"),
        ),
        (
            OutputParameter("test_name"),
            "test_mock_component",
            Parameter(
                name="test_name", value_from=models.ValueFrom(path="test_name")
            ),
        ),
    ],
)
def test_outputs_to_hera(
    test_io, test_owner_fixture_name, test_expected_hera_parameter, request
):
    """Test the OutputArtifact's and the OutputParameter's `to_hera` method for
    all currently supported scenarios."""
    test_owner = request.getfixturevalue(test_owner_fixture_name)
    test_io.set_owner(test_owner)

    assert test_io.to_hera() == test_expected_hera_parameter


@pytest.mark.parametrize(
    "test_source_owner_fixture_name,test_expected_hera_parameter,is_template",
    [
        (
            "test_mock_component",
            Artifact(name="test_name", path="test_name"),
            True,
        ),
        (
            "test_mock_component",
            Artifact(
                name="test_name",
                from_="{{tasks.mock-component-0.outputs.artifacts.test_source_name}}",  # noqa: E501
            ),
            False,
        ),
    ],
)
def test_input_artifact_to_hera(
    test_source_owner_fixture_name,
    test_expected_hera_parameter,
    is_template,
    request,
):
    """Test the InputArtifact's `to_hera` method for all currently supported
    scenarios."""
    test_source_owner = request.getfixturevalue(test_source_owner_fixture_name)
    test_source = OutputArtifact("test_source_name")
    test_source.set_owner(test_source_owner)

    test_io = InputArtifact("test_name")
    test_io.set_source(test_source)

    assert (
        test_io.to_hera(template=is_template) == test_expected_hera_parameter
    )  # noqa: E501


@pytest.mark.parametrize(
    "test_owner_fixture_name,test_source,test_source_owner_fixture_name,test_expected_hera_parameter",  # noqa: E501
    [
        (
            "test_mock_pipeline",
            None,
            None,
            Parameter(name="test_name", value="test_value"),
        ),
        (
            "test_mock_component",
            None,
            None,
            Parameter(name="test_name", value="test_value"),
        ),
        (
            "test_mock_component",
            OutputParameter("test_source_name"),
            "test_mock_component",
            Parameter(
                name="test_name",
                value="{{tasks.mock-component-0.outputs.parameters.test_source_name}}",  # noqa: E501
            ),
        ),
        (
            "test_mock_component",
            InputParameter("test_source_name"),
            "test_mock_pipeline",
            Parameter(
                name="test_name",
                value="{{workflow.parameters.test_source_name}}",
            ),
        ),
    ],
)
def test_input_parameter_to_hera(
    test_owner_fixture_name,
    test_source,
    test_source_owner_fixture_name,
    test_expected_hera_parameter,
    request,
):
    """Test the InputParameter's `to_hera` method for all currently supported
    scenarios."""

    test_io = InputParameter("test_name", value="test_value")
    test_owner = request.getfixturevalue(test_owner_fixture_name)
    test_io.set_owner(test_owner)

    if test_source is not None:
        test_source_owner = request.getfixturevalue(
            test_source_owner_fixture_name
        )
        test_source.set_owner(test_source_owner)
        test_io.set_source(test_source)

    assert test_io.to_hera() == test_expected_hera_parameter


@pytest.mark.parametrize(
    "test_io",
    [
        InputArtifact("test_name"),
        InputParameter("test_name"),
        OutputArtifact("test_name"),
        OutputParameter("test_name"),
    ],
)
def test_raise_invalid_source(test_io):

    with pytest.raises(TypeError, match="The specified parameter source"):
        test_io.set_source("invalid_owner_value")
