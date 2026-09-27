import json
import os
import warnings
from typing import NamedTuple, TypedDict

import polars
import pytest
from bettmensch_ai.pipelines.artifact_store import (
    LocalArtifactStore,
    LocalArtifactStoreConfig,
)
from bettmensch_ai.pipelines.metadata_store import (
    LocalMetadataStore,
    LocalMetadataStoreConfig,
    RunStatus,
)
from bettmensch_ai.pipelines.pipeline import pipeline
from bettmensch_ai.pipelines.runner import (
    LocalRunner,
    MaterializerMismatchWarning,
    MissingPipelineInputError,
    UnknownPipelineInputError,
    run_locally,
)
from bettmensch_ai.pipelines.task import task


@pytest.fixture
def store(tmp_path):
    return LocalArtifactStore(LocalArtifactStoreConfig(root_dir=str(tmp_path / "artifacts")))


@pytest.fixture
def metadata_store(tmp_path):
    return LocalMetadataStore(
        LocalMetadataStoreConfig(db_path=str(tmp_path / "metadata.db"))
    )


@pytest.fixture
def runner(store, metadata_store):
    return LocalRunner(store, metadata_store)


@task
def add(a: int, b: int) -> int:
    return a + b


class DivMod(NamedTuple):
    quotient: int
    remainder: int


@task
def divmod_task(a: int, b: int) -> DivMod:
    return DivMod(quotient=a // b, remainder=a % b)


class DivModDict(TypedDict):
    quotient: int
    remainder: int


@task
def divmod_dict_task(a: int, b: int) -> DivModDict:
    return {"quotient": a // b, "remainder": a % b}


def test_run_executes_multi_task_pipeline_and_returns_result(runner):
    @pipeline
    def my_pipeline(a: int, b: int, c: int = 3):
        ab = add(a, b)
        return add(ab, c)

    result = runner.run(my_pipeline, a=1, b=2)

    assert result == 6


def test_run_executes_independent_tasks_in_the_same_rank_correctly(runner):
    @pipeline
    def my_pipeline(a: int, b: int):
        x = add(a, b)
        y = add(a, b)
        return add(x, y)

    result = runner.run(my_pipeline, a=1, b=2)

    assert result == 6


def test_run_uses_pipeline_input_default_when_omitted(runner):
    @pipeline
    def my_pipeline(a: int, c: int = 10):
        return add(a, c)

    result = runner.run(my_pipeline, a=5)

    assert result == 15


@task
def lying_about_int(a: int) -> int:
    """Declares `-> int` but actually returns a `polars.DataFrame` - an
    inaccurate type hint, used to exercise `LocalRunner`'s reconciliation.
    """

    return polars.DataFrame({"a": [a]})


def test_run_reconciles_a_materializer_that_does_not_match_the_actual_value(runner):
    @pipeline
    def my_pipeline(a: int):
        return lying_about_int(a)

    with pytest.warns(MaterializerMismatchWarning):
        result = runner.run(my_pipeline, a=1)

    assert isinstance(result, polars.DataFrame)
    assert result.equals(polars.DataFrame({"a": [1]}))


@task
def row_count(a: int) -> int:
    """Declares `a: int` but actually receives whatever the pipeline input
    was given - here, deliberately, a `polars.DataFrame`. Returns a real
    `int`, so this task's own output is never a source of mismatch: any
    warning raised in this test comes from materializing the mismatched
    pipeline input, not from this task's output.
    """

    return a.height


def test_run_reconciles_a_pipeline_input_that_does_not_match_its_declared_type(runner):
    @pipeline
    def my_pipeline(a: int):
        return row_count(a)

    with pytest.warns(MaterializerMismatchWarning):
        result = runner.run(my_pipeline, a=polars.DataFrame({"a": [1, 2, 3]}))

    assert result == 3


def test_run_does_not_warn_when_the_declared_type_matches(runner):
    @pipeline
    def my_pipeline(a: int, b: int):
        return add(a, b)

    with warnings.catch_warnings():
        warnings.simplefilter("error", MaterializerMismatchWarning)
        result = runner.run(my_pipeline, a=1, b=2)

    assert result == 3


def test_run_materializes_each_named_tuple_field_as_its_own_artifact(store, runner):
    @pipeline
    def my_pipeline(a: int, b: int):
        outputs = divmod_task(a, b)
        return outputs.quotient

    result = runner.run(my_pipeline, a=7, b=2)

    assert result == 3

    all_files = [p for p in store.root_dir.rglob("*") if p.is_file()]
    data_files = [p for p in all_files if not p.name.endswith(".metadata")]

    # a=7, b=2 (pipeline inputs), plus divmod_task's "quotient" (3) and
    # "remainder" (1) - each materialized separately, under its own key.
    assert {f.read_text() for f in data_files} == {"7", "2", "3", "1"}


def test_run_typed_dict_task_loads_each_key_independently(store, runner):
    @pipeline
    def my_pipeline(a: int, b: int):
        outputs = divmod_dict_task(a, b)
        return outputs["remainder"]

    result = runner.run(my_pipeline, a=7, b=2)

    assert result == 1


def test_run_materializes_every_artifact_through_the_store(store, runner):
    @pipeline
    def my_pipeline(a: int, b: int, c: int = 3):
        ab = add(a, b)
        return add(ab, c)

    runner.run(my_pipeline, a=1, b=2)

    all_files = [p for p in store.root_dir.rglob("*") if p.is_file()]
    data_files = [p for p in all_files if not p.name.endswith(".metadata")]
    metadata_files = [p for p in all_files if p.name.endswith(".metadata")]

    # Pipeline inputs a=1, b=2, c=3, materialized once each up front; "add"'s
    # output (3); "add-1"'s output (6). No task input is ever independently
    # saved - "add"'s a/b and "add-1"'s c are all pipeline-input reads, and
    # "add-1"'s other input is a TaskOutput read of "add"'s own output. 5
    # data artifacts in total, each saved alongside its own metadata sidecar.
    assert len(data_files) == 5
    assert len(metadata_files) == 5

    for data_file in data_files:
        assert data_file.read_text() in {"1", "2", "3", "6"}

    for metadata_file in metadata_files:
        metadata = json.loads(metadata_file.read_text())
        assert metadata["materializer"] == "json"


def test_run_materializes_a_shared_pipeline_input_exactly_once(store, runner):
    """A pipeline input consumed by more than one task must be saved once,
    not once per consuming task - no task input is ever independently
    materialized.
    """

    @pipeline
    def my_pipeline(a: int, b: int):
        x = add(a, b)
        return add(a, x)

    result = runner.run(my_pipeline, a=1, b=2)

    assert result == 4

    all_files = [p for p in store.root_dir.rglob("*") if p.is_file()]
    data_files = [p for p in all_files if not p.name.endswith(".metadata")]

    # a, b materialized once each; "add"'s output (3); "add-1"'s output
    # (4). 4 data artifacts, even though "a" feeds two different tasks.
    assert len(data_files) == 4
    assert {f.read_text() for f in data_files} == {"1", "2", "3", "4"}


def test_run_never_materializes_a_static_literal_task_input(store, runner):
    @pipeline
    def my_pipeline(a: int):
        # "10" is a literal argument to add() - a static task input, not a
        # pipeline input or another task's output.
        return add(a, 10)

    result = runner.run(my_pipeline, a=1)

    assert result == 11

    all_files = [p for p in store.root_dir.rglob("*") if p.is_file()]
    data_files = [p for p in all_files if not p.name.endswith(".metadata")]

    # Only "a" (a pipeline input) and "add"'s output are materialized; the
    # literal "10" never touches the store, since it isn't a pipeline input
    # or a task output - just a static value baked into the trace.
    assert {f.read_text() for f in data_files} == {"1", "11"}


def test_run_passes_pipeline_input_through_to_output(runner):
    @pipeline
    def passthrough_pipeline(a: int):
        return a

    result = runner.run(passthrough_pipeline, a=42)

    assert result == 42


def test_run_pipeline_with_no_output_returns_none(runner):
    @pipeline
    def no_output_pipeline(a: int, b: int):
        add(a, b)

    result = runner.run(no_output_pipeline, a=1, b=2)

    assert result is None


def test_run_missing_required_pipeline_input_raises(runner):
    @pipeline
    def my_pipeline(a: int, b: int):
        return add(a, b)

    with pytest.raises(MissingPipelineInputError) as excinfo:
        runner.run(my_pipeline, a=1)

    assert excinfo.value.pipeline_name == "my-pipeline"
    assert excinfo.value.missing_input_names == ("b",)


def test_run_unknown_pipeline_input_raises(runner):
    @pipeline
    def my_pipeline(a: int, b: int):
        return add(a, b)

    with pytest.raises(UnknownPipelineInputError) as excinfo:
        runner.run(my_pipeline, a=1, b=2, bogus=3)

    assert excinfo.value.unknown_input_names == ("bogus",)


def test_run_locally_convenience_function_uses_local_artifact_store_by_default():
    @pipeline
    def my_pipeline(a: int, b: int):
        return add(a, b)

    result = run_locally(my_pipeline, a=2, b=3)

    assert result == 5


def test_default_runner_uses_a_fresh_local_artifact_store_when_none_given():
    runner = LocalRunner()

    assert isinstance(runner.artifact_store, LocalArtifactStore)


def test_default_runner_uses_a_fresh_local_metadata_store_when_none_given():
    runner = LocalRunner()

    assert isinstance(runner.metadata_store, LocalMetadataStore)


def test_run_records_pipeline_and_task_run_success_in_the_metadata_store(
    runner, metadata_store
):
    @pipeline
    def my_pipeline(a: int, b: int, c: int = 3):
        ab = add(a, b)
        return add(ab, c)

    result = runner.run(my_pipeline, a=1, b=2)
    assert result == 6

    pipeline_runs = metadata_store.list_pipeline_runs("my-pipeline")
    assert len(pipeline_runs) == 1
    assert pipeline_runs[0].status == RunStatus.SUCCEEDED

    pipeline_run_id = pipeline_runs[0].pipeline_run_id
    task_runs = {
        r.task_name: r.status for r in metadata_store.list_task_runs(pipeline_run_id)
    }
    assert task_runs == {"add": RunStatus.SUCCEEDED, "add-1": RunStatus.SUCCEEDED}

    add_outputs = metadata_store.list_task_outputs(pipeline_run_id, "add")
    assert len(add_outputs) == 1
    assert add_outputs[0].output_name == "result"
    assert add_outputs[0].artifact_key == os.path.join(
        "my-pipeline", str(pipeline_run_id), "add", "result"
    )


def test_run_records_a_pipeline_assembly_and_references_it_from_the_run(
    runner, metadata_store
):
    @pipeline
    def my_pipeline(a: int, b: int, c: int = 3):
        ab = add(a, b)
        return add(ab, c)

    runner.run(my_pipeline, a=1, b=2)

    assemblies = metadata_store.list_pipeline_assemblies("my-pipeline")
    assert len(assemblies) == 1
    assert [t["name"] for t in assemblies[0].dag_structure["tasks"]] == ["add", "add-1"]

    pipeline_run = metadata_store.list_pipeline_runs("my-pipeline")[0]
    assert pipeline_run.pipeline_assembly_id == assemblies[0].pipeline_assembly_id


def test_run_does_not_duplicate_assembly_records_across_unchanged_runs(
    runner, metadata_store
):
    @pipeline
    def my_pipeline(a: int, b: int):
        return add(a, b)

    runner.run(my_pipeline, a=1, b=2)
    runner.run(my_pipeline, a=3, b=4)

    assemblies = metadata_store.list_pipeline_assemblies("my-pipeline")
    assert len(assemblies) == 1

    runs = metadata_store.list_pipeline_runs("my-pipeline")
    assert len({r.pipeline_assembly_id for r in runs}) == 1


def test_run_records_multi_output_task_outputs_in_the_metadata_store(
    runner, metadata_store
):
    @pipeline
    def my_pipeline(a: int, b: int):
        outputs = divmod_task(a, b)
        return outputs.quotient

    runner.run(my_pipeline, a=7, b=2)

    pipeline_run_id = metadata_store.list_pipeline_runs("my-pipeline")[0].pipeline_run_id
    outputs = {
        o.output_name: o.artifact_key
        for o in metadata_store.list_task_outputs(pipeline_run_id, "divmod-task")
    }

    assert set(outputs) == {"quotient", "remainder"}


@task
def failing_task(a: int) -> int:
    raise ValueError("boom")


def test_run_records_failure_and_reraises(runner, metadata_store):
    @pipeline
    def my_pipeline(a: int):
        return failing_task(a)

    with pytest.raises(ValueError, match="boom"):
        runner.run(my_pipeline, a=1)

    pipeline_run = metadata_store.list_pipeline_runs("my-pipeline")[0]
    assert pipeline_run.status == RunStatus.FAILED

    task_run = metadata_store.get_task_run(pipeline_run.pipeline_run_id, "failing-task")
    assert task_run.status == RunStatus.FAILED


@task
def printing_task(a: int) -> int:
    print(f"received {a}")
    return a


def test_run_captures_stdout_into_task_run_logs(runner, metadata_store):
    @pipeline
    def my_pipeline(a: int):
        return printing_task(a)

    runner.run(my_pipeline, a=7)

    pipeline_run_id = metadata_store.list_pipeline_runs("my-pipeline")[0].pipeline_run_id
    task_run = metadata_store.get_task_run(pipeline_run_id, "printing-task")

    assert task_run.logs is not None
    assert "received 7" in task_run.logs


def test_run_appends_traceback_to_logs_on_failure(runner, metadata_store):
    @pipeline
    def my_pipeline(a: int):
        return failing_task(a)

    with pytest.raises(ValueError, match="boom"):
        runner.run(my_pipeline, a=1)

    pipeline_run_id = metadata_store.list_pipeline_runs("my-pipeline")[0].pipeline_run_id
    task_run = metadata_store.get_task_run(pipeline_run_id, "failing-task")

    assert task_run.logs is not None
    assert "ValueError: boom" in task_run.logs


def test_run_leaves_logs_none_when_nothing_is_printed(runner, metadata_store):
    @pipeline
    def my_pipeline(a: int, b: int):
        return add(a, b)

    runner.run(my_pipeline, a=1, b=2)

    pipeline_run_id = metadata_store.list_pipeline_runs("my-pipeline")[0].pipeline_run_id
    task_run = metadata_store.get_task_run(pipeline_run_id, "add")

    assert task_run.logs is None
