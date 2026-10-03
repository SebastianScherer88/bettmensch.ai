"""The `BaseMetadataStore` interface: bookkeeping of pipeline runs and the
task runs within them, plus bookkeeping of pipelines registered with a
remote backend orchestrator - independent of whatever actually orchestrates
execution or registration.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RunStatus(str, Enum):
    """The lifecycle status of a pipeline run or a task run within one."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class PipelineAssemblyRecord:
    """One successful assembly of a pipeline into an `AssembledPipeline`.

    A fresh record per *distinct* assembly, not one row updated in place per
    pipeline - mirroring `PipelineRegistrationRecord`, re-assembling a
    pipeline whose definition has changed creates a new record rather than
    overwriting the previous one, so assembly history is preserved. Unlike
    registration, nothing here is backend-specific: this is the same
    backend-agnostic structure a Layer 2 compiler would start from.

    Attributes:
        pipeline_assembly_id: This assembly's own id.
        pipeline_name: The name of the pipeline that was assembled.
        dag_structure: The assembled DAG's structure - tasks (grouped by
            topological rank), their IO bindings, and the dependency edges
            between them - as a plain, JSON-able dict. See
            `bettmensch_ai.pipelines.assembler.serialize_assembled_pipeline`
            for the concrete shape this is built from an `AssembledPipeline`
            with; left unshaped/generic here for the same reason
            `PipelineRegistrationRecord.dag_structure` is (this store
            doesn't depend on `AssembledPipeline` directly).
        pipeline_inputs: The pipeline's declared inputs at assembly time
            (required/default/materializer), as a plain, JSON-able dict.
        assembled_at: When this assembly was recorded.
    """

    pipeline_assembly_id: uuid.UUID
    pipeline_name: str
    dag_structure: Dict[str, Any]
    pipeline_inputs: Dict[str, Any]
    assembled_at: datetime


@dataclass(frozen=True)
class PipelineRunRecord:
    """One pipeline run's bookkeeping record.

    Attributes:
        pipeline_name: The name of the pipeline that was run.
        pipeline_run_id: This run's id.
        status: The run's current (or final) status.
        started_at: When the run started.
        ended_at: When the run finished, or `None` while still `RUNNING`.
        pipeline_assembly_id: The id of the `PipelineAssemblyRecord`
            snapshotting the exact DAG structure this run executed, or
            `None` if the run wasn't recorded against one (e.g. an older
            record from before assembly bookkeeping existed). Recorded per
            run, rather than re-derived by matching on `pipeline_name` after
            the fact, so a run's DAG visualization stays accurate even if
            the pipeline's definition changes later.
    """

    pipeline_name: str
    pipeline_run_id: uuid.UUID
    status: RunStatus
    started_at: datetime
    ended_at: Optional[datetime] = None
    pipeline_assembly_id: Optional[uuid.UUID] = None


@dataclass(frozen=True)
class TaskRunRecord:
    """One task run's bookkeeping record, within a pipeline run.

    Attributes:
        pipeline_run_id: The id of the pipeline run this task run belongs
            to.
        task_name: The (unique, assembled) name of the task.
        status: The task run's current (or final) status.
        started_at: When the task run started.
        ended_at: When the task run finished, or `None` while still
            `RUNNING`.
        logs: Whatever the task function wrote to stdout/stderr while it
            ran, plus a traceback if it failed - captured by `LocalRunner`,
            `None` if nothing was captured (or for a run recorded before
            this field existed).
    """

    pipeline_run_id: uuid.UUID
    task_name: str
    status: RunStatus
    started_at: datetime
    ended_at: Optional[datetime] = None
    logs: Optional[str] = None


@dataclass(frozen=True)
class TaskOutputRecord:
    """One task output's recorded artifact key, within a pipeline run.

    Attributes:
        pipeline_run_id: The id of the pipeline run this output belongs to.
        task_name: The (unique, assembled) name of the task that produced
            it.
        output_name: The output's own name (see `Task.output_names`).
        artifact_key: The `BaseArtifactStore` key the output was
            materialized under.
    """

    pipeline_run_id: uuid.UUID
    task_name: str
    output_name: str
    artifact_key: str


@dataclass(frozen=True)
class PipelineRegistrationRecord:
    """One registration of a pipeline with a remote backend orchestrator.

    A fresh record per registration event, not one row updated in place per
    pipeline: re-registering a pipeline (e.g. deploying a new DAG version)
    creates a new record rather than overwriting the previous one, so
    registration history is preserved - mirroring how a `pipeline_run_id`
    distinguishes separate runs of the "same" pipeline rather than reusing
    one row.

    Attributes:
        pipeline_registration_id: This registration's own id.
        pipeline_name: The name of the pipeline that was registered.
        backend: Which backend orchestrator this was registered with (e.g.
            `"aws_stepfunctions"`) - a store may hold registrations for more
            than one backend.
        dag_structure: The registered DAG's structure (task names and the
            dependencies between them), as a plain, JSON-able dict. Left
            unshaped/generic rather than typed against `AssembledPipeline`
            directly, so `BaseMetadataStore` doesn't need to depend on the
            rest of `pipelines` - a caller (a future Layer 2 compiler)
            derives this dict from an `AssembledPipeline` itself.
        pipeline_inputs: The pipeline's declared inputs at registration time
            (name, type, default, required), as a plain, JSON-able dict.
        backend_metadata: Backend-specific resource references for this
            registration (e.g. a Step Functions state machine ARN, a
            workflow template id), as a plain, JSON-able dict. Kept separate
            from `dag_structure` (which stays backend-agnostic) so a caller
            doesn't have to invent a convention for stuffing backend-only
            fields into it. Left unshaped for the same reason `dag_structure`
            is: shapes vary per backend, and no concrete backend orchestrator
            exists yet to validate a fuller schema against. Defaults to an
            empty dict, not `None`, so callers can always iterate/render it
            without a null check.
        registered_at: When this registration was created.
        deregistered_at: When this registration was retired (e.g.
            superseded by a newer one, or explicitly removed), or `None`
            while still active.
    """

    pipeline_registration_id: uuid.UUID
    pipeline_name: str
    backend: str
    dag_structure: Dict[str, Any]
    pipeline_inputs: Dict[str, Any]
    registered_at: datetime
    deregistered_at: Optional[datetime] = None
    backend_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Whether this registration hasn't been deregistered."""

        return self.deregistered_at is None


@dataclass(frozen=True)
class TriggerRecord:
    """One schedule/trigger registered against a pipeline registration.

    Attributes:
        trigger_id: This trigger's own id.
        pipeline_registration_id: The id of the pipeline registration this
            trigger fires runs of.
        trigger_type: A backend-defined discriminator for what kind of
            trigger this is (e.g. `"cron"`, `"event"`, `"webhook"`).
        trigger_config: The trigger's own configuration (e.g. a cron
            expression, or an event source's details), as a plain, JSON-able
            dict. Left unshaped rather than typed per `trigger_type`, since
            trigger shapes vary by both type and backend, and no concrete
            backend orchestrator exists yet to validate a fuller schema
            against.
        registered_at: When this trigger was registered.
        deregistered_at: When this trigger was retired, or `None` while
            still active.
    """

    trigger_id: uuid.UUID
    pipeline_registration_id: uuid.UUID
    trigger_type: str
    trigger_config: Dict[str, Any]
    registered_at: datetime
    deregistered_at: Optional[datetime] = None

    @property
    def is_active(self) -> bool:
        """Whether this trigger hasn't been deregistered."""

        return self.deregistered_at is None


class BaseMetadataStore(ABC):
    """Bookkeeping for pipeline runs and the task runs within them - status,
    timing, and which artifact key each task output was materialized
    under - independent of whatever actually orchestrates execution.

    Deliberately orchestrator-agnostic: `LocalRunner` and a future Layer 2
    remote/distributed orchestrator both record into the same interface, so
    a pipeline run's bookkeeping looks the same regardless of how it was
    actually run. `BaseMetadataStore` is the abstract interface;
    `LocalMetadataStore` is Layer 1's concrete, local (SQLite-backed)
    default. A shared, remotely-reachable flavour (e.g. backed by a hosted
    database, so genuinely distributed workers can all report into the same
    store) is Layer 2's job, mirroring the `BaseArtifactStore`/
    `LocalArtifactStore` split.

    Covers three related but separate concerns: *assembly* bookkeeping
    (`record_pipeline_assembly` through `list_pipeline_assemblies`) - a
    dated snapshot of a pipeline's backend-agnostic DAG structure each time
    it's successfully assembled, independent of whether it's ever run or
    registered; *run* bookkeeping (`start_pipeline_run` through
    `list_task_outputs`); and *registration* bookkeeping
    (`register_pipeline` onward) - pipelines that have been registered with
    a remote backend orchestrator, their DAG structure, declared inputs,
    backend-specific metadata, and any schedules/triggers registered against
    them. Registration only becomes meaningful once a real Layer 2 backend
    orchestrator exists to register pipelines with; until then it's simply
    unused by `LocalRunner`. Assembly bookkeeping, unlike registration, is
    used today: `LocalRunner` records one automatically before every run
    (see `assembler.record_assembly`), and it can also be recorded
    standalone for a pipeline that's been assembled but not (yet) run.
    """

    @abstractmethod
    def record_pipeline_assembly(
        self,
        pipeline_name: str,
        dag_structure: Dict[str, Any],
        pipeline_inputs: Dict[str, Any],
    ) -> uuid.UUID:
        """Records a new assembly of a pipeline.

        Args:
            pipeline_name: The name of the pipeline that was assembled.
            dag_structure: The assembled DAG's structure, as a plain,
                JSON-able dict.
            pipeline_inputs: The pipeline's declared inputs, as a plain,
                JSON-able dict.

        Returns:
            The new assembly record's id.
        """

    @abstractmethod
    def get_pipeline_assembly(
        self, pipeline_assembly_id: uuid.UUID
    ) -> PipelineAssemblyRecord:
        """Looks up one pipeline assembly's record.

        Args:
            pipeline_assembly_id: The id of the assembly to look up.

        Returns:
            That assembly's record.

        Raises:
            KeyError: If no assembly with that id has been recorded.
        """

    @abstractmethod
    def list_pipeline_assemblies(
        self, pipeline_name: Optional[str] = None
    ) -> List[PipelineAssemblyRecord]:
        """Lists recorded pipeline assemblies.

        Args:
            pipeline_name: If given, only assemblies of this pipeline;
                otherwise every recorded assembly.

        Returns:
            The matching assemblies, most recently assembled first.
        """

    @abstractmethod
    def start_pipeline_run(
        self,
        pipeline_name: str,
        pipeline_run_id: uuid.UUID,
        pipeline_assembly_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Records that a pipeline run has started, with status `RUNNING`.

        Args:
            pipeline_name: The name of the pipeline being run.
            pipeline_run_id: The id of this run.
            pipeline_assembly_id: The id of the `PipelineAssemblyRecord`
                snapshotting the DAG structure being run, if the caller
                recorded one (see `assembler.record_assembly`, which
                `LocalRunner` calls automatically before every run).
        """

    @abstractmethod
    def finish_pipeline_run(
        self, pipeline_run_id: uuid.UUID, status: RunStatus
    ) -> None:
        """Records that a pipeline run has finished.

        Args:
            pipeline_run_id: The id of the run that finished.
            status: The run's final status (`SUCCEEDED` or `FAILED`).
        """

    @abstractmethod
    def start_task_run(self, pipeline_run_id: uuid.UUID, task_name: str) -> None:
        """Records that a task run has started, with status `RUNNING`.

        Args:
            pipeline_run_id: The id of the pipeline run `task_name` belongs
                to.
            task_name: The (unique, assembled) name of the task.
        """

    @abstractmethod
    def finish_task_run(
        self,
        pipeline_run_id: uuid.UUID,
        task_name: str,
        status: RunStatus,
        logs: Optional[str] = None,
    ) -> None:
        """Records that a task run has finished.

        Args:
            pipeline_run_id: The id of the pipeline run `task_name` belongs
                to.
            task_name: The (unique, assembled) name of the task.
            status: The task run's final status (`SUCCEEDED` or `FAILED`).
            logs: Whatever the task captured on stdout/stderr (plus a
                traceback on failure), if any.
        """

    @abstractmethod
    def record_task_output(
        self,
        pipeline_run_id: uuid.UUID,
        task_name: str,
        output_name: str,
        artifact_key: str,
    ) -> None:
        """Records the artifact key one of a task's outputs was
        materialized under.

        Args:
            pipeline_run_id: The id of the pipeline run `task_name` belongs
                to.
            task_name: The (unique, assembled) name of the task.
            output_name: The output's own name (see `Task.output_names`).
            artifact_key: The `BaseArtifactStore` key the output was saved
                under.
        """

    @abstractmethod
    def get_pipeline_run(self, pipeline_run_id: uuid.UUID) -> PipelineRunRecord:
        """Looks up one pipeline run's record.

        Args:
            pipeline_run_id: The id of the run to look up.

        Returns:
            That run's record.

        Raises:
            KeyError: If no run with that id has been recorded.
        """

    @abstractmethod
    def list_pipeline_runs(
        self, pipeline_name: Optional[str] = None
    ) -> List[PipelineRunRecord]:
        """Lists recorded pipeline runs.

        Args:
            pipeline_name: If given, only runs of this pipeline; otherwise
                every recorded run.

        Returns:
            The matching runs, most recently started first.
        """

    @abstractmethod
    def get_task_run(
        self, pipeline_run_id: uuid.UUID, task_name: str
    ) -> TaskRunRecord:
        """Looks up one task run's record.

        Args:
            pipeline_run_id: The id of the pipeline run `task_name` belongs
                to.
            task_name: The (unique, assembled) name of the task.

        Returns:
            That task run's record.

        Raises:
            KeyError: If no such task run has been recorded.
        """

    @abstractmethod
    def list_task_runs(self, pipeline_run_id: uuid.UUID) -> List[TaskRunRecord]:
        """Lists every task run recorded for a pipeline run.

        Args:
            pipeline_run_id: The id of the pipeline run to list task runs
                for.

        Returns:
            That run's task runs, in the order they were started.
        """

    @abstractmethod
    def list_task_outputs(
        self, pipeline_run_id: uuid.UUID, task_name: str
    ) -> List[TaskOutputRecord]:
        """Lists every output recorded for one task run.

        Args:
            pipeline_run_id: The id of the pipeline run `task_name` belongs
                to.
            task_name: The (unique, assembled) name of the task.

        Returns:
            That task's recorded outputs.
        """

    @abstractmethod
    def register_pipeline(
        self,
        pipeline_name: str,
        backend: str,
        dag_structure: Dict[str, Any],
        pipeline_inputs: Dict[str, Any],
        backend_metadata: Optional[Dict[str, Any]] = None,
    ) -> uuid.UUID:
        """Records a new registration of a pipeline with a backend
        orchestrator.

        Args:
            pipeline_name: The name of the pipeline being registered.
            backend: Which backend orchestrator this is registered with.
            dag_structure: The registered DAG's structure, as a plain,
                JSON-able dict.
            pipeline_inputs: The pipeline's declared inputs, as a plain,
                JSON-able dict.
            backend_metadata: Backend-specific resource references for this
                registration, as a plain, JSON-able dict. Defaults to an
                empty dict if omitted.

        Returns:
            The new registration's id.
        """

    @abstractmethod
    def deregister_pipeline(self, pipeline_registration_id: uuid.UUID) -> None:
        """Marks a pipeline registration as retired.

        Args:
            pipeline_registration_id: The id of the registration to retire.
        """

    @abstractmethod
    def get_pipeline_registration(
        self, pipeline_registration_id: uuid.UUID
    ) -> PipelineRegistrationRecord:
        """Looks up one pipeline registration's record.

        Args:
            pipeline_registration_id: The id of the registration to look up.

        Returns:
            That registration's record.

        Raises:
            KeyError: If no registration with that id has been recorded.
        """

    @abstractmethod
    def list_pipeline_registrations(
        self, pipeline_name: Optional[str] = None, active_only: bool = False
    ) -> List[PipelineRegistrationRecord]:
        """Lists recorded pipeline registrations.

        Args:
            pipeline_name: If given, only registrations of this pipeline;
                otherwise every recorded registration.
            active_only: If `True`, excludes registrations that have been
                deregistered.

        Returns:
            The matching registrations, most recently registered first.
        """

    @abstractmethod
    def register_trigger(
        self,
        pipeline_registration_id: uuid.UUID,
        trigger_type: str,
        trigger_config: Dict[str, Any],
    ) -> uuid.UUID:
        """Records a new schedule/trigger against a pipeline registration.

        Args:
            pipeline_registration_id: The id of the pipeline registration
                this trigger fires runs of.
            trigger_type: A backend-defined discriminator for what kind of
                trigger this is (e.g. `"cron"`, `"event"`).
            trigger_config: The trigger's own configuration, as a plain,
                JSON-able dict.

        Returns:
            The new trigger's id.
        """

    @abstractmethod
    def deregister_trigger(self, trigger_id: uuid.UUID) -> None:
        """Marks a trigger as retired.

        Args:
            trigger_id: The id of the trigger to retire.
        """

    @abstractmethod
    def list_triggers(
        self, pipeline_registration_id: uuid.UUID, active_only: bool = False
    ) -> List[TriggerRecord]:
        """Lists triggers registered against a pipeline registration.

        Args:
            pipeline_registration_id: The id of the pipeline registration to
                list triggers for.
            active_only: If `True`, excludes triggers that have been
                deregistered.

        Returns:
            The matching triggers, in the order they were registered.
        """
