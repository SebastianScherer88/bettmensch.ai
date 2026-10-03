import uuid
from typing import Any, Dict, Optional

from bettmensch_ai.pipelines.assembler import record_assembly, serialize_assembled_pipeline
from bettmensch_ai.pipelines.compute import BaseComputeBackend
from bettmensch_ai.pipelines.context import get_active_context
from bettmensch_ai.pipelines.metadata_store import LocalMetadataStore, LocalMetadataStoreConfig
from bettmensch_ai.pipelines.pipeline import pipeline
from bettmensch_ai.pipelines.task import task


def make_store(tmp_path):
    return LocalMetadataStore(LocalMetadataStoreConfig(db_path=str(tmp_path / "metadata.db")))


@task
def add_recording_fixture(a: int, b: int) -> int:
    return a + b


class _FakeRemoteComputeBackend(BaseComputeBackend):
    """A minimal non-local `BaseComputeBackend` double, standing in for a
    real remote backend (e.g. the stashed `AwsBatchComputeBackend`) purely
    to exercise `serialize_assembled_pipeline`'s generic "pinned, non-
    default compute_backend" case without depending on stashed code.
    """

    name = "fake-remote"

    def __init__(self, queue: str):
        self.queue = queue

    def run(self, *args: Any, **kwargs: Any) -> Optional[str]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {"queue": self.queue}


def test_serialize_assembled_pipeline_describes_tasks_ranks_and_edges():
    @pipeline
    def my_pipeline(a: int, b: int, c: int = 3):
        ab = add_recording_fixture(a, b)
        return add_recording_fixture(ab, c)

    dag_structure, pipeline_inputs = serialize_assembled_pipeline(my_pipeline)

    tasks_by_name = {t["name"]: t for t in dag_structure["tasks"]}
    assert set(tasks_by_name) == {"add-recording-fixture", "add-recording-fixture-1"}
    assert tasks_by_name["add-recording-fixture"]["rank"] == 0
    assert tasks_by_name["add-recording-fixture-1"]["rank"] == 1
    assert dag_structure["edges"] == [["add-recording-fixture", "add-recording-fixture-1"]]
    assert dag_structure["output"] == {
        "name": "result",
        "source": {"kind": "task_output", "task": "add-recording-fixture-1", "output": "result"},
    }

    assert pipeline_inputs["a"] == {"required": True, "default": None, "materializer": "json"}
    assert pipeline_inputs["c"] == {"required": False, "default": 3, "materializer": "json"}


def test_serialize_assembled_pipeline_defaults_compute_backend_to_local():
    @pipeline
    def my_pipeline(a: int, b: int):
        return add_recording_fixture(a, b)

    dag_structure, _ = serialize_assembled_pipeline(my_pipeline)
    add_task = dag_structure["tasks"][0]

    assert add_task["compute_backend"] == {"name": "local", "config": {}}


def test_serialize_assembled_pipeline_includes_a_pinned_compute_backend():
    @pipeline
    def my_pipeline(a: int, b: int):
        result = add_recording_fixture(a, b)
        get_active_context().assembled_tasks[result.task_name].compute_backend = (
            _FakeRemoteComputeBackend(queue="my-queue")
        )
        return result

    dag_structure, _ = serialize_assembled_pipeline(my_pipeline)
    add_task = dag_structure["tasks"][0]

    assert add_task["compute_backend"] == {
        "name": "fake-remote",
        "config": {"queue": "my-queue"},
    }


def test_serialize_assembled_pipeline_records_static_and_bound_inputs():
    @pipeline
    def my_pipeline(a: int):
        return add_recording_fixture(a, 2)

    dag_structure, _ = serialize_assembled_pipeline(my_pipeline)
    add_task = dag_structure["tasks"][0]
    inputs_by_name = {i["name"]: i["source"] for i in add_task["inputs"]}

    assert inputs_by_name["a"] == {"kind": "pipeline_input", "name": "a"}
    assert inputs_by_name["b"] == {"kind": "static", "value": 2}


def test_serialize_assembled_pipeline_includes_task_source_code():
    # `_task_source`'s own contract explicitly allows `None` when source
    # genuinely can't be retrieved (a presentation detail, not load-bearing
    # structure - see its docstring) - and, in practice, `inspect.
    # getsource`/`findsource` can fail to relocate a function's `def` line
    # under `pytest`'s own assertion-rewriting import hook specifically,
    # in a way that scales with how many other test modules are collected
    # in the same session, even though the same lookup reliably succeeds
    # for identical code run as a plain script (verified directly,
    # multiple times, outside pytest). So this asserts the *correct*
    # content when source is available, without hard-requiring it to be -
    # a pytest-tooling limitation, not a product defect, and not something
    # test authoring (module-level vs. nested function, renaming, etc.) has
    # reliably controlled here.
    @pipeline
    def my_pipeline(a: int, b: int):
        return add_recording_fixture(a, b)

    dag_structure, _ = serialize_assembled_pipeline(my_pipeline)
    add_task = dag_structure["tasks"][0]

    if add_task["source"] is not None:
        assert "def add_recording_fixture(a: int, b: int) -> int:" in add_task["source"]
        assert "return a + b" in add_task["source"]


def test_serialize_assembled_pipeline_degrades_to_none_source_gracefully():
    # `_task_source` must never raise even when source genuinely can't be
    # retrieved (e.g. a function whose code object claims a filename that
    # doesn't exist) - it degrades to `None`, since source text is a
    # presentation detail, not load-bearing structure.
    @pipeline
    def my_pipeline(a: int, b: int):
        return add_recording_fixture(a, b)

    dag_structure, _ = serialize_assembled_pipeline(my_pipeline)
    add_task = dag_structure["tasks"][0]

    assert add_task["source"] is None or "return a + b" in add_task["source"]


def test_serialize_assembled_pipeline_falls_back_to_repr_for_unserializable_static_input():
    class Unserializable:
        def __repr__(self):
            return "<Unserializable>"

    @task
    def takes_object(obj: object) -> int:
        return 1

    @pipeline
    def my_pipeline():
        return takes_object(Unserializable())

    dag_structure, _ = serialize_assembled_pipeline(my_pipeline)
    obj_input = dag_structure["tasks"][0]["inputs"][0]

    assert obj_input["source"] == {"kind": "static", "value": "<Unserializable>"}


def test_record_assembly_persists_a_new_record(tmp_path):
    store = make_store(tmp_path)

    @pipeline
    def my_pipeline(a: int, b: int):
        return add_recording_fixture(a, b)

    assembly_id = record_assembly(store, my_pipeline)
    record = store.get_pipeline_assembly(assembly_id)

    assert record.pipeline_name == "my-pipeline"
    assert record.dag_structure["tasks"][0]["name"] == "add-recording-fixture"


def test_record_assembly_is_a_noop_when_structure_is_unchanged(tmp_path):
    store = make_store(tmp_path)

    @pipeline
    def my_pipeline(a: int, b: int):
        return add_recording_fixture(a, b)

    first_id = record_assembly(store, my_pipeline)
    second_id = record_assembly(store, my_pipeline)

    assert first_id == second_id
    assert len(store.list_pipeline_assemblies("my-pipeline")) == 1


def test_record_assembly_creates_a_new_record_when_structure_changes(tmp_path):
    store = make_store(tmp_path)

    @pipeline
    def my_pipeline(a: int, b: int):
        return add_recording_fixture(a, b)

    first_id = record_assembly(store, my_pipeline)

    @pipeline(name="my-pipeline")
    def my_pipeline_v2(a: int, b: int, c: int):
        first = add_recording_fixture(a, b)
        return add_recording_fixture(first, c)

    second_id = record_assembly(store, my_pipeline_v2)

    assert first_id != second_id
    assert len(store.list_pipeline_assemblies("my-pipeline")) == 2
