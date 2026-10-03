from typing import NamedTuple

import polars
import pytest
from bettmensch_ai.pipelines.artifact_store import LocalArtifactStore, LocalArtifactStoreConfig
from bettmensch_ai.pipelines.materializers import DefaultMaterializer
from bettmensch_ai.pipelines.runner.exceptions import MaterializerMismatchWarning
from bettmensch_ai.pipelines.runner.task_execution import execute_task, get_captured_logs
from bettmensch_ai.pipelines.task import task


@pytest.fixture
def store(tmp_path):
    return LocalArtifactStore(LocalArtifactStoreConfig(root_dir=str(tmp_path / "artifacts")))


@task
def add_execution_fixture(a: int, b: int) -> int:
    return a + b


def test_execute_task_resolves_static_inputs_directly(store):
    execute_task(
        add_execution_fixture,
        store,
        static_inputs={"a": 1, "b": 2},
        input_keys={},
        output_keys={"result": "k/result"},
        default_materializer_cls=DefaultMaterializer,
    )

    from bettmensch_ai.pipelines.materializers import resolve_materializer_from_artifact

    materializer = resolve_materializer_from_artifact(store, "k/result")
    assert store.load(materializer, "k/result") == 3


def test_execute_task_loads_bound_inputs_by_key(store):
    from bettmensch_ai.pipelines.materializers import JsonMaterializer

    store.save(JsonMaterializer(), 5, "upstream/a")

    execute_task(
        add_execution_fixture,
        store,
        static_inputs={"b": 10},
        input_keys={"a": "upstream/a"},
        output_keys={"result": "k/result"},
        default_materializer_cls=DefaultMaterializer,
    )

    from bettmensch_ai.pipelines.materializers import resolve_materializer_from_artifact

    materializer = resolve_materializer_from_artifact(store, "k/result")
    assert store.load(materializer, "k/result") == 15


class DivModOutput(NamedTuple):
    quotient: int
    remainder: int


@task
def divmod_execution_fixture(a: int, b: int) -> DivModOutput:
    return DivModOutput(quotient=a // b, remainder=a % b)


def test_execute_task_materializes_each_multi_output_independently(store):
    from bettmensch_ai.pipelines.materializers import resolve_materializer_from_artifact

    execute_task(
        divmod_execution_fixture,
        store,
        static_inputs={"a": 7, "b": 2},
        input_keys={},
        output_keys={"quotient": "k/q", "remainder": "k/r"},
        default_materializer_cls=DefaultMaterializer,
    )

    assert store.load(resolve_materializer_from_artifact(store, "k/q"), "k/q") == 3
    assert store.load(resolve_materializer_from_artifact(store, "k/r"), "k/r") == 1


@task
def lying_about_int_execution_fixture(a: int) -> int:
    """Declares `-> int` but actually returns a `polars.DataFrame` -
    `resolve_materializer_for_type(int, ...)` resolves a materializer that
    can't serialize the real value, forcing `execute_task` to reconcile."""

    return polars.DataFrame({"a": [a]})


def test_execute_task_reconciles_a_materializer_mismatch(store):
    from bettmensch_ai.pipelines.materializers import resolve_materializer_from_artifact

    with pytest.warns(MaterializerMismatchWarning):
        execute_task(
            lying_about_int_execution_fixture,
            store,
            static_inputs={"a": 1},
            input_keys={},
            output_keys={"result": "k/result"},
            default_materializer_cls=DefaultMaterializer,
        )

    materializer = resolve_materializer_from_artifact(store, "k/result")
    result = store.load(materializer, "k/result")
    assert isinstance(result, polars.DataFrame)
    assert result.equals(polars.DataFrame({"a": [1]}))


def test_execute_task_captures_stdout_and_returns_it(store):
    @task
    def printer(a: int) -> int:
        print(f"got {a}")
        return a

    logs = execute_task(
        printer,
        store,
        static_inputs={"a": 5},
        input_keys={},
        output_keys={"result": "k/result"},
        default_materializer_cls=DefaultMaterializer,
    )

    assert logs is not None
    assert "got 5" in logs


def test_execute_task_returns_none_when_nothing_printed(store):
    logs = execute_task(
        add_execution_fixture,
        store,
        static_inputs={"a": 1, "b": 2},
        input_keys={},
        output_keys={"result": "k/result"},
        default_materializer_cls=DefaultMaterializer,
    )

    assert logs is None


@task
def failing_execution_fixture(a: int) -> int:
    print("about to fail")
    raise ValueError("boom")


def test_execute_task_attaches_captured_logs_to_the_original_exception_and_reraises(store):
    with pytest.raises(ValueError, match="boom") as excinfo:
        execute_task(
            failing_execution_fixture,
            store,
            static_inputs={"a": 1},
            input_keys={},
            output_keys={"result": "k/result"},
            default_materializer_cls=DefaultMaterializer,
        )

    logs = get_captured_logs(excinfo.value)
    assert logs is not None
    assert "about to fail" in logs
    assert "ValueError: boom" in logs


def test_get_captured_logs_returns_none_for_an_unrelated_exception():
    assert get_captured_logs(ValueError("not from execute_task")) is None
