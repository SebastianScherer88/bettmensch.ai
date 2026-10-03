from typing import Any, NamedTuple, TypedDict

import pytest
from bettmensch_ai.pipelines.assembler import CyclicGraphError, IOBindingError
from bettmensch_ai.pipelines.assembler.assembler import Assembler
from bettmensch_ai.pipelines.context import PipelineAssemblyContext
from bettmensch_ai.pipelines.io_binding import (
    NO_DEFAULT,
    IOBinding,
    PipelineInput,
    TaskInput,
    TaskOutput,
)
from bettmensch_ai.pipelines.materializers import (
    MATERIALIZER_BY_NAME,
    BaseMaterializer,
)
from bettmensch_ai.pipelines.pipeline import Pipeline, pipeline
from bettmensch_ai.pipelines.pipeline.assembled_pipeline import AssembledPipeline
from bettmensch_ai.pipelines.task import task
from bettmensch_ai.pipelines.task.assembled_task import AssembledTask


class Unsupported:
    pass


class StubPickleMaterializer(BaseMaterializer):
    """A minimal stand-in for a pickle/cloudpickle-based materializer, used
    only to test that `Pipeline(default_materializer=...)` is actually
    honoured - not a real implementation.
    """

    name = "stub-pickle"

    @classmethod
    def supports_type(cls, type_hint: Any) -> bool:
        return True

    def supports(self, value: Any) -> bool:
        return True

    @property
    def s3_support(self) -> bool:
        return False

    def _save(self, value: Any, path: str) -> None:
        raise NotImplementedError

    def _load(self, path: str) -> Any:
        raise NotImplementedError


@task
def add(a: int, b: int) -> int:
    return a + b


class DivModOutput(NamedTuple):
    quotient: int
    remainder: int


@task
def divmod_task(a: int, b: int) -> DivModOutput:
    return DivModOutput(quotient=a // b, remainder=a % b)


def test_pipeline_decorator_defaults_to_eager_assembly():
    @pipeline
    def my_pipeline(a: int, b: int, c: int = 3):
        ab = add(a, b)
        return add(ab, c)

    assert isinstance(my_pipeline, AssembledPipeline)


def test_pipeline_decorator_can_opt_out_of_eager_assembly():
    @pipeline(assemble=False)
    def my_pipeline(a: int, b: int, c: int = 3):
        ab = add(a, b)
        return add(ab, c)

    assert isinstance(my_pipeline, Pipeline)

    assembled = my_pipeline.assemble()
    assert isinstance(assembled, AssembledPipeline)


def test_assemble_orders_tasks_topologically_and_wires_bindings():
    @pipeline
    def my_pipeline(a: int, b: int, c: int = 3):
        ab = add(a, b)
        return add(ab, c)

    assert [t.name for t in my_pipeline.tasks] == ["add", "add-1"]
    assert my_pipeline.task_ranks == (
        (my_pipeline.get_task("add"),),
        (my_pipeline.get_task("add-1"),),
    )
    assert my_pipeline.output.source == TaskOutput(
        task_name="add-1", output_name="result"
    )


def test_assemble_groups_independent_tasks_into_the_same_rank():
    @pipeline
    def my_pipeline(a: int, b: int):
        x = add(a, b)
        y = add(a, b)
        return add(x, y)

    ranks = [[t.name for t in rank] for rank in my_pipeline.task_ranks]

    assert ranks == [["add", "add-1"], ["add-2"]]


def test_assemble_carries_pipeline_input_defaults():
    @pipeline
    def my_pipeline(a: int, b: int, c: int = 3):
        ab = add(a, b)
        return add(ab, c)

    assert my_pipeline.get_input("a") == PipelineInput(name="a", default=NO_DEFAULT)
    assert my_pipeline.get_input("c") == PipelineInput(name="c", default=3)
    assert my_pipeline.get_input("a").required is True
    assert my_pipeline.get_input("c").required is False


def test_assemble_resolves_pipeline_input_materializers_from_type_hints():
    @pipeline
    def my_pipeline(a: int, b: int, c: int = 3):
        ab = add(a, b)
        return add(ab, c)

    assert set(my_pipeline.input_materializers) == {"a", "b", "c"}

    for materializer in my_pipeline.input_materializers.values():
        assert type(materializer).__name__ == "JsonMaterializer"


def test_assemble_untyped_pipeline_input_falls_back_to_default_materializer():
    @pipeline
    def my_pipeline(a):
        return add(a, 1)

    assert type(my_pipeline.input_materializers["a"]).__name__ == "DefaultMaterializer"


def test_assemble_resolves_materializers_for_every_input_and_output():
    @pipeline
    def my_pipeline(a: int, b: int):
        return add(a, b)

    add_task = my_pipeline.get_task("add")

    assert set(add_task.materializers) == {"a", "b", "result"}


def test_assemble_treats_named_tuple_return_as_multiple_named_outputs():
    @pipeline
    def my_pipeline(a: int, b: int):
        outputs = divmod_task(a, b)
        return outputs.quotient

    divmod_assembled = my_pipeline.get_task("divmod-task")

    assert divmod_assembled.output_names == ("quotient", "remainder")
    assert set(divmod_assembled.materializers) == {"a", "b", "quotient", "remainder"}
    assert my_pipeline.output.source == TaskOutput(
        task_name="divmod-task", output_name="quotient"
    )


class DivModDict(TypedDict):
    quotient: int
    remainder: int


@task
def divmod_dict_task(a: int, b: int) -> DivModDict:
    return {"quotient": a // b, "remainder": a % b}


def test_assemble_treats_typed_dict_return_as_multiple_named_outputs():
    @pipeline
    def my_pipeline(a: int, b: int):
        outputs = divmod_dict_task(a, b)
        return outputs["remainder"]

    divmod_assembled = my_pipeline.get_task("divmod-dict-task")

    assert divmod_assembled.output_names == ("quotient", "remainder")
    assert set(divmod_assembled.materializers) == {"a", "b", "quotient", "remainder"}
    assert my_pipeline.output.source == TaskOutput(
        task_name="divmod-dict-task", output_name="remainder"
    )


def test_pipeline_has_at_most_one_output():
    @pipeline
    def no_output_pipeline(a: int, b: int):
        add(a, b)

    assert no_output_pipeline.outputs == ()
    assert no_output_pipeline.output is None


def test_assemble_pipeline_output_referencing_unknown_task_output_raises():
    with pytest.raises(IOBindingError):

        @pipeline
        def bad_pipeline(a: int, b: int):
            add(a, b)
            return TaskOutput(task_name="add", output_name="bogus")


def test_topological_order_groups_independent_tasks_into_one_rank():
    t1 = AssembledTask(name="t1", task=None, static_inputs={}, output_names=("result",))
    t2 = AssembledTask(name="t2", task=None, static_inputs={}, output_names=("result",))

    context = PipelineAssemblyContext()
    context.pipeline_inputs = {}
    context.assembled_tasks = {"t1": t1, "t2": t2}
    context.io_bindings = []

    assert Assembler()._topological_order(context) == [["t1", "t2"]]


def test_topological_order_raises_on_cycle():
    t1 = AssembledTask(name="t1", task=None, static_inputs={}, output_names=("result",))
    t2 = AssembledTask(name="t2", task=None, static_inputs={}, output_names=("result",))

    context = PipelineAssemblyContext()
    context.pipeline_inputs = {}
    context.assembled_tasks = {"t1": t1, "t2": t2}
    context.io_bindings = [
        IOBinding(
            target=TaskInput(task_name="t1", input_name="x"),
            source=TaskOutput(task_name="t2", output_name="result"),
        ),
        IOBinding(
            target=TaskInput(task_name="t2", input_name="x"),
            source=TaskOutput(task_name="t1", output_name="result"),
        ),
    ]

    with pytest.raises(CyclicGraphError):
        Assembler()._topological_order(context)


def test_validate_bindings_raises_for_unassembled_target_task():
    context = PipelineAssemblyContext()
    context.pipeline_inputs = {}
    context.assembled_tasks = {}
    context.io_bindings = [
        IOBinding(
            target=TaskInput(task_name="missing", input_name="x"),
            source=TaskOutput(task_name="also-missing", output_name="result"),
        )
    ]

    with pytest.raises(IOBindingError):
        Assembler()._validate_bindings(context)


def test_unsupported_pipeline_return_type_raises():
    with pytest.raises(TypeError):

        @pipeline
        def bad_pipeline(a: int, b: int):
            add(a, b)
            return 42


@task
def identity_unsupported(a: Unsupported) -> Unsupported:
    return a


def test_pipeline_default_materializer_defaults_to_default_materializer():
    @pipeline
    def my_pipeline(a: Unsupported):
        return identity_unsupported(a)

    assert type(my_pipeline.input_materializers["a"]).__name__ == "DefaultMaterializer"

    task_ = my_pipeline.get_task("identity-unsupported")
    assert type(task_.materializers["a"]).__name__ == "DefaultMaterializer"
    assert type(task_.materializers["result"]).__name__ == "DefaultMaterializer"


def test_pipeline_default_materializer_override_applies_to_tasks_and_pipeline_inputs():
    @pipeline(default_materializer=StubPickleMaterializer)
    def my_pipeline(a: Unsupported):
        return identity_unsupported(a)

    assert isinstance(my_pipeline.input_materializers["a"], StubPickleMaterializer)

    task_ = my_pipeline.get_task("identity-unsupported")
    assert isinstance(task_.materializers["a"], StubPickleMaterializer)
    assert isinstance(task_.materializers["result"], StubPickleMaterializer)


def test_pipeline_default_materializer_override_does_not_affect_supported_types():
    @pipeline(default_materializer=StubPickleMaterializer)
    def my_pipeline(a: int, b: int):
        return add(a, b)

    assert type(my_pipeline.input_materializers["a"]).__name__ == "JsonMaterializer"


def test_pipeline_default_materializer_is_registered_for_after_the_fact_resolution():
    @pipeline(default_materializer=StubPickleMaterializer)
    def my_pipeline(a: Unsupported):
        return identity_unsupported(a)

    assert MATERIALIZER_BY_NAME["stub-pickle"] is StubPickleMaterializer
