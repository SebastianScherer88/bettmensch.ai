from typing import NamedTuple, TypedDict

import pytest
from bettmensch_ai.pipelines.context import PipelineAssemblyContext
from bettmensch_ai.pipelines.exceptions import MissingRequiredInputError
from bettmensch_ai.pipelines.io_binding import PipelineInput
from bettmensch_ai.pipelines.task import TaskOutput, resource, task, uv
from bettmensch_ai.pipelines.task.decorators import (
    ResourceRequirements,
    UvRequirements,
)
from bettmensch_ai.pipelines.task.task import ReservedTaskParameterError


class DivModOutput(NamedTuple):
    quotient: int
    remainder: int


def test_task_call_outside_pipeline_runs_eagerly():
    @task
    def add(a: int, b: int) -> int:
        return a + b

    assert add(1, 2) == 3


def test_task_output_is_always_named_result():
    @task
    def add(a: int, b: int) -> int:
        return a + b

    assert add.output_names == ("result",)
    assert add.is_multi_output is False


def test_task_returning_a_named_tuple_has_one_output_per_field():
    @task
    def divmod_task(a: int, b: int) -> DivModOutput:
        return DivModOutput(quotient=a // b, remainder=a % b)

    assert divmod_task.output_names == ("quotient", "remainder")
    assert divmod_task.is_named_tuple_output is True
    assert divmod_task.is_typed_dict_output is False
    assert divmod_task.output_type_hints == {"quotient": int, "remainder": int}


class DivModDict(TypedDict):
    quotient: int
    remainder: int


def test_task_returning_a_typed_dict_has_one_output_per_key():
    @task
    def divmod_dict_task(a: int, b: int) -> DivModDict:
        return {"quotient": a // b, "remainder": a % b}

    assert divmod_dict_task.output_names == ("quotient", "remainder")
    assert divmod_dict_task.is_typed_dict_output is True
    assert divmod_dict_task.is_named_tuple_output is False
    assert divmod_dict_task.output_type_hints == {"quotient": int, "remainder": int}


def test_resource_and_uv_decorators_attach_requirements():
    @task
    @resource(cpu="1", memory="1Gi", gpu=1)
    @uv(["numpy==2.0.0"])
    def train(x: int) -> int:
        return x

    assert train.resource_requirements == ResourceRequirements(
        cpu="1", memory="1Gi", gpu=1
    )
    assert train.uv_requirements == UvRequirements(packages=("numpy==2.0.0",))


def test_task_call_inside_context_records_assembled_task():
    @task
    def add(a: int, b: int) -> int:
        return a + b

    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        output = add(1, 2)

    assert isinstance(output, TaskOutput)
    assert output == TaskOutput(task_name="add", output_name="result")
    assert "add" in context.assembled_tasks
    assembled_task = context.assembled_tasks["add"]
    assert assembled_task.static_inputs == {"a": 1, "b": 2}
    assert context.io_bindings == []


def test_task_call_inside_context_dedupes_names():
    @task
    def add(a: int, b: int) -> int:
        return a + b

    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        add(1, 2)
        add(3, 4)

    assert set(context.assembled_tasks) == {"add", "add-1"}


def test_task_call_with_task_output_creates_binding():
    @task
    def add(a: int, b: int) -> int:
        return a + b

    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        first = add(1, 2)
        add(first, 3)

    bindings = context.io_bindings
    assert len(bindings) == 1
    assert bindings[0].target.task_name == "add-1"
    assert bindings[0].target.input_name == "a"
    assert bindings[0].source == first


def test_named_tuple_task_call_inside_context_returns_a_named_tuple_of_task_outputs():
    @task
    def divmod_task(a: int, b: int) -> DivModOutput:
        return DivModOutput(quotient=a // b, remainder=a % b)

    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        output = divmod_task(7, 2)

    assert isinstance(output, DivModOutput)
    assert output.quotient == TaskOutput(task_name="divmod-task", output_name="quotient")
    assert output.remainder == TaskOutput(
        task_name="divmod-task", output_name="remainder"
    )


def test_typed_dict_task_call_inside_context_returns_a_dict_of_task_outputs():
    @task
    def divmod_dict_task(a: int, b: int) -> DivModDict:
        return {"quotient": a // b, "remainder": a % b}

    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        output = divmod_dict_task(7, 2)

    assert output == {
        "quotient": TaskOutput(task_name="divmod-dict-task", output_name="quotient"),
        "remainder": TaskOutput(
            task_name="divmod-dict-task", output_name="remainder"
        ),
    }


def test_task_call_inside_context_resolves_omitted_default_into_static_inputs():
    @task
    def add(a: int, b: int = 5) -> int:
        return a + b

    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        add(1)

    assert context.assembled_tasks["add"].static_inputs == {"a": 1, "b": 5}


def test_task_call_inside_context_missing_required_input_raises():
    @task
    def add(a: int, b: int) -> int:
        return a + b

    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        with pytest.raises(MissingRequiredInputError) as excinfo:
            add(1)

    error = excinfo.value
    assert error.task_name == "add"
    assert error.missing_input_names == ("b",)
    assert error.provided_inputs == {"a": 1}
    assert "'add'" in str(error)
    assert "'b'" in str(error)
    assert "a=literal value 1" in str(error)


def test_task_call_inside_context_missing_input_error_describes_owner_tasks():
    @task
    def add(a: int, b: int) -> int:
        return a + b

    @task
    def needs_three(x: int, y: int, z: int) -> int:
        return x + y + z

    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {"p": PipelineInput(name="p")}
        first = add(1, 2)
        with pytest.raises(MissingRequiredInputError) as excinfo:
            needs_three(first, PipelineInput(name="p"))

    error = excinfo.value
    assert error.missing_input_names == ("z",)
    assert "y=pipeline input 'p'" in str(error)
    assert "x=output 'result' of task 'add'" in str(error)


def test_reserved_parameter_name_raises_at_decoration_time():
    with pytest.raises(ReservedTaskParameterError):

        @task
        def add(a: int, resources: int) -> int:
            return a + resources

    with pytest.raises(ReservedTaskParameterError):

        @task
        def subtract(a: int, uv: int) -> int:
            return a - uv


def test_resources_override_applies_to_this_node_only():
    @task
    @resource(cpu="1")
    def add(a: int, b: int) -> int:
        return a + b

    override = ResourceRequirements(cpu="4", memory="8Gi")

    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        add(1, 2, resources=override)
        add(3, 4)

    overridden = context.assembled_tasks["add"]
    plain = context.assembled_tasks["add-1"]

    assert overridden.resource_override == override
    assert overridden.resource_requirements == override
    assert plain.resource_override is None
    assert plain.resource_requirements == ResourceRequirements(cpu="1")


def test_uv_override_applies_to_this_node_only():
    @task
    @uv(["numpy==1.0.0"])
    def add(a: int, b: int) -> int:
        return a + b

    override = UvRequirements(packages=("numpy==2.0.0",))

    with PipelineAssemblyContext() as context:
        context.pipeline_inputs = {}
        add(1, 2, uv=override)
        add(3, 4)

    overridden = context.assembled_tasks["add"]
    plain = context.assembled_tasks["add-1"]

    assert overridden.uv_override == override
    assert overridden.uv_requirements == override
    assert plain.uv_override is None
    assert plain.uv_requirements == UvRequirements(packages=("numpy==1.0.0",))


def test_resources_and_uv_kwargs_pass_through_unchanged_outside_a_trace():
    @task
    def add(a: int, b: int) -> int:
        return a + b

    override = ResourceRequirements(cpu="4")

    with pytest.raises(TypeError):
        add(1, 2, resources=override)


def test_registering_duplicate_task_name_raises():
    from bettmensch_ai.pipelines.context import PipelineAssemblyError
    from bettmensch_ai.pipelines.task.assembled_task import AssembledTask

    with PipelineAssemblyContext() as context:
        context.register_task(
            AssembledTask(
                name="dup", task=None, static_inputs={}, output_names=("result",)
            )
        )
        with pytest.raises(PipelineAssemblyError):
            context.register_task(
                AssembledTask(
                    name="dup",
                    task=None,
                    static_inputs={},
                    output_names=("result",),
                )
            )
