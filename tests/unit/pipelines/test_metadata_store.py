import uuid

import pytest
from bettmensch_ai.pipelines.metadata_store import (
    BaseMetadataStore,
    LocalMetadataStore,
    LocalMetadataStoreConfig,
    RunStatus,
)


def make_store(tmp_path):
    config = LocalMetadataStoreConfig(db_path=str(tmp_path / "metadata.db"))
    return LocalMetadataStore(config)


def test_local_metadata_store_is_a_base_metadata_store(tmp_path):
    assert isinstance(make_store(tmp_path), BaseMetadataStore)


def test_start_pipeline_run_records_a_running_status(tmp_path):
    store = make_store(tmp_path)
    pipeline_run_id = uuid.uuid4()

    store.start_pipeline_run("my-pipeline", pipeline_run_id)
    record = store.get_pipeline_run(pipeline_run_id)

    assert record.pipeline_name == "my-pipeline"
    assert record.pipeline_run_id == pipeline_run_id
    assert record.status == RunStatus.RUNNING
    assert record.ended_at is None


def test_finish_pipeline_run_updates_status_and_ended_at(tmp_path):
    store = make_store(tmp_path)
    pipeline_run_id = uuid.uuid4()

    store.start_pipeline_run("my-pipeline", pipeline_run_id)
    store.finish_pipeline_run(pipeline_run_id, RunStatus.SUCCEEDED)
    record = store.get_pipeline_run(pipeline_run_id)

    assert record.status == RunStatus.SUCCEEDED
    assert record.ended_at is not None
    assert record.ended_at >= record.started_at


def test_get_pipeline_run_raises_for_unknown_id(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(KeyError):
        store.get_pipeline_run(uuid.uuid4())


def test_list_pipeline_runs_filters_by_pipeline_name(tmp_path):
    store = make_store(tmp_path)
    run_a1, run_a2, run_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    store.start_pipeline_run("pipeline-a", run_a1)
    store.start_pipeline_run("pipeline-a", run_a2)
    store.start_pipeline_run("pipeline-b", run_b)

    all_runs = store.list_pipeline_runs()
    a_runs = store.list_pipeline_runs("pipeline-a")

    assert {r.pipeline_run_id for r in all_runs} == {run_a1, run_a2, run_b}
    assert {r.pipeline_run_id for r in a_runs} == {run_a1, run_a2}


def test_task_run_lifecycle_is_recorded(tmp_path):
    store = make_store(tmp_path)
    pipeline_run_id = uuid.uuid4()
    store.start_pipeline_run("my-pipeline", pipeline_run_id)

    store.start_task_run(pipeline_run_id, "add")
    running = store.get_task_run(pipeline_run_id, "add")
    assert running.status == RunStatus.RUNNING
    assert running.ended_at is None

    store.finish_task_run(pipeline_run_id, "add", RunStatus.SUCCEEDED)
    finished = store.get_task_run(pipeline_run_id, "add")
    assert finished.status == RunStatus.SUCCEEDED
    assert finished.ended_at is not None
    assert finished.logs is None


def test_finish_task_run_records_logs(tmp_path):
    store = make_store(tmp_path)
    pipeline_run_id = uuid.uuid4()
    store.start_pipeline_run("my-pipeline", pipeline_run_id)
    store.start_task_run(pipeline_run_id, "add")

    store.finish_task_run(pipeline_run_id, "add", RunStatus.SUCCEEDED, logs="hello\n")
    finished = store.get_task_run(pipeline_run_id, "add")

    assert finished.logs == "hello\n"


def test_get_task_run_raises_for_unknown_task(tmp_path):
    store = make_store(tmp_path)
    pipeline_run_id = uuid.uuid4()
    store.start_pipeline_run("my-pipeline", pipeline_run_id)

    with pytest.raises(KeyError):
        store.get_task_run(pipeline_run_id, "bogus")


def test_list_task_runs_returns_every_task_for_a_pipeline_run(tmp_path):
    store = make_store(tmp_path)
    pipeline_run_id = uuid.uuid4()
    store.start_pipeline_run("my-pipeline", pipeline_run_id)

    store.start_task_run(pipeline_run_id, "add")
    store.start_task_run(pipeline_run_id, "add-1")

    task_names = {r.task_name for r in store.list_task_runs(pipeline_run_id)}

    assert task_names == {"add", "add-1"}


def test_record_task_output_and_list_task_outputs(tmp_path):
    store = make_store(tmp_path)
    pipeline_run_id = uuid.uuid4()
    store.start_pipeline_run("my-pipeline", pipeline_run_id)
    store.start_task_run(pipeline_run_id, "divmod-task")

    store.record_task_output(
        pipeline_run_id, "divmod-task", "quotient", "my-pipeline/run/divmod-task/quotient"
    )
    store.record_task_output(
        pipeline_run_id, "divmod-task", "remainder", "my-pipeline/run/divmod-task/remainder"
    )

    outputs = store.list_task_outputs(pipeline_run_id, "divmod-task")
    by_name = {o.output_name: o.artifact_key for o in outputs}

    assert by_name == {
        "quotient": "my-pipeline/run/divmod-task/quotient",
        "remainder": "my-pipeline/run/divmod-task/remainder",
    }


def test_list_task_outputs_is_empty_when_none_recorded(tmp_path):
    store = make_store(tmp_path)
    pipeline_run_id = uuid.uuid4()
    store.start_pipeline_run("my-pipeline", pipeline_run_id)
    store.start_task_run(pipeline_run_id, "add")

    assert store.list_task_outputs(pipeline_run_id, "add") == []


def test_store_persists_across_instances_pointed_at_the_same_db_file(tmp_path):
    config = LocalMetadataStoreConfig(db_path=str(tmp_path / "metadata.db"))
    pipeline_run_id = uuid.uuid4()

    LocalMetadataStore(config).start_pipeline_run("my-pipeline", pipeline_run_id)
    reopened = LocalMetadataStore(config)

    assert reopened.get_pipeline_run(pipeline_run_id).pipeline_name == "my-pipeline"


def test_register_pipeline_records_a_new_active_registration(tmp_path):
    store = make_store(tmp_path)
    dag_structure = {"tasks": ["add", "add-1"], "edges": [["add", "add-1"]]}
    pipeline_inputs = {"a": {"type": "int", "required": True}}

    registration_id = store.register_pipeline(
        "my-pipeline", "aws_stepfunctions", dag_structure, pipeline_inputs
    )
    record = store.get_pipeline_registration(registration_id)

    assert record.pipeline_name == "my-pipeline"
    assert record.backend == "aws_stepfunctions"
    assert record.dag_structure == dag_structure
    assert record.pipeline_inputs == pipeline_inputs
    assert record.is_active is True
    assert record.deregistered_at is None
    assert record.backend_metadata == {}


def test_register_pipeline_records_backend_metadata(tmp_path):
    store = make_store(tmp_path)
    backend_metadata = {"state_machine_arn": "arn:aws:states:::my-pipeline"}

    registration_id = store.register_pipeline(
        "my-pipeline", "aws_stepfunctions", {}, {}, backend_metadata
    )
    record = store.get_pipeline_registration(registration_id)

    assert record.backend_metadata == backend_metadata


def test_record_pipeline_assembly_and_get_it_back(tmp_path):
    store = make_store(tmp_path)
    dag_structure = {"tasks": [{"name": "add", "rank": 0}], "edges": []}
    pipeline_inputs = {"a": {"required": True, "default": None}}

    assembly_id = store.record_pipeline_assembly(
        "my-pipeline", dag_structure, pipeline_inputs
    )
    record = store.get_pipeline_assembly(assembly_id)

    assert record.pipeline_name == "my-pipeline"
    assert record.dag_structure == dag_structure
    assert record.pipeline_inputs == pipeline_inputs
    assert record.assembled_at is not None


def test_get_pipeline_assembly_raises_for_unknown_id(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(KeyError):
        store.get_pipeline_assembly(uuid.uuid4())


def test_list_pipeline_assemblies_filters_by_pipeline_name_most_recent_first(
    tmp_path,
):
    store = make_store(tmp_path)
    first = store.record_pipeline_assembly("pipeline-a", {"v": 1}, {})
    second = store.record_pipeline_assembly("pipeline-a", {"v": 2}, {})
    other = store.record_pipeline_assembly("pipeline-b", {}, {})

    all_assemblies = store.list_pipeline_assemblies()
    a_assemblies = store.list_pipeline_assemblies("pipeline-a")

    assert {a.pipeline_assembly_id for a in all_assemblies} == {first, second, other}
    assert [a.pipeline_assembly_id for a in a_assemblies] == [second, first]


def test_start_pipeline_run_records_its_pipeline_assembly_id(tmp_path):
    store = make_store(tmp_path)
    assembly_id = store.record_pipeline_assembly("my-pipeline", {}, {})
    pipeline_run_id = uuid.uuid4()

    store.start_pipeline_run("my-pipeline", pipeline_run_id, assembly_id)
    record = store.get_pipeline_run(pipeline_run_id)

    assert record.pipeline_assembly_id == assembly_id


def test_start_pipeline_run_pipeline_assembly_id_defaults_to_none(tmp_path):
    store = make_store(tmp_path)
    pipeline_run_id = uuid.uuid4()

    store.start_pipeline_run("my-pipeline", pipeline_run_id)
    record = store.get_pipeline_run(pipeline_run_id)

    assert record.pipeline_assembly_id is None


def test_get_pipeline_registration_raises_for_unknown_id(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(KeyError):
        store.get_pipeline_registration(uuid.uuid4())


def test_deregister_pipeline_marks_registration_inactive(tmp_path):
    store = make_store(tmp_path)
    registration_id = store.register_pipeline("my-pipeline", "airflow", {}, {})

    store.deregister_pipeline(registration_id)
    record = store.get_pipeline_registration(registration_id)

    assert record.is_active is False
    assert record.deregistered_at is not None


def test_list_pipeline_registrations_filters_by_name_and_active_only(tmp_path):
    store = make_store(tmp_path)
    active = store.register_pipeline("pipeline-a", "airflow", {}, {})
    retired = store.register_pipeline("pipeline-a", "airflow", {}, {})
    other = store.register_pipeline("pipeline-b", "airflow", {}, {})
    store.deregister_pipeline(retired)

    all_regs = store.list_pipeline_registrations()
    a_regs = store.list_pipeline_registrations("pipeline-a")
    a_active_regs = store.list_pipeline_registrations("pipeline-a", active_only=True)

    assert {r.pipeline_registration_id for r in all_regs} == {active, retired, other}
    assert {r.pipeline_registration_id for r in a_regs} == {active, retired}
    assert {r.pipeline_registration_id for r in a_active_regs} == {active}


def test_register_trigger_and_list_triggers(tmp_path):
    store = make_store(tmp_path)
    registration_id = store.register_pipeline("my-pipeline", "airflow", {}, {})

    trigger_id = store.register_trigger(
        registration_id, "cron", {"schedule": "0 0 * * *"}
    )
    triggers = store.list_triggers(registration_id)

    assert len(triggers) == 1
    assert triggers[0].trigger_id == trigger_id
    assert triggers[0].trigger_type == "cron"
    assert triggers[0].trigger_config == {"schedule": "0 0 * * *"}
    assert triggers[0].is_active is True


def test_deregister_trigger_marks_it_inactive_and_filters_from_active_only(tmp_path):
    store = make_store(tmp_path)
    registration_id = store.register_pipeline("my-pipeline", "airflow", {}, {})
    trigger_id = store.register_trigger(registration_id, "cron", {"schedule": "* * * * *"})

    store.deregister_trigger(trigger_id)

    all_triggers = store.list_triggers(registration_id)
    active_triggers = store.list_triggers(registration_id, active_only=True)

    assert len(all_triggers) == 1
    assert all_triggers[0].is_active is False
    assert active_triggers == []


def test_list_triggers_is_empty_when_none_registered(tmp_path):
    store = make_store(tmp_path)
    registration_id = store.register_pipeline("my-pipeline", "airflow", {}, {})

    assert store.list_triggers(registration_id) == []
