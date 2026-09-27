"""`PostgresMetadataStore`: a remote, shared metadata store backed by
PostgreSQL - Layer 2's flavour of `BaseMetadataStore`, reachable by
genuinely distributed workers/orchestrators rather than just one local
process.
"""

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from .base_metadata_store import (
    BaseMetadataStore,
    PipelineAssemblyRecord,
    PipelineRegistrationRecord,
    PipelineRunRecord,
    RunStatus,
    TaskOutputRecord,
    TaskRunRecord,
    TriggerRecord,
)

if TYPE_CHECKING:
    import psycopg

# One statement per entry, executed individually rather than as one
# multi-statement script: psycopg's extended query protocol doesn't
# guarantee multiple `;`-separated statements in a single `execute()` call
# the way `sqlite3.executescript` does, so this sidesteps that entirely.
# Native `UUID`/`TIMESTAMPTZ`/`JSONB` types replace the `TEXT`-encoded
# columns `LocalMetadataStore` uses for the same fields (SQLite has none of
# these natively) - psycopg adapts `uuid.UUID`/`datetime`/`dict` to and from
# them automatically, so no manual (de)serialization is needed here the way
# `LocalMetadataStore` needs `str()`/`json.dumps()`/`.isoformat()`.
_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS pipeline_assemblies (
        pipeline_assembly_id UUID PRIMARY KEY,
        pipeline_name TEXT NOT NULL,
        dag_structure JSONB NOT NULL,
        pipeline_inputs JSONB NOT NULL,
        assembled_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        pipeline_run_id UUID PRIMARY KEY,
        pipeline_name TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
        started_at TIMESTAMPTZ NOT NULL,
        ended_at TIMESTAMPTZ,
        pipeline_assembly_id UUID REFERENCES pipeline_assemblies (pipeline_assembly_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS task_runs (
        pipeline_run_id UUID NOT NULL REFERENCES pipeline_runs (pipeline_run_id),
        task_name TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
        started_at TIMESTAMPTZ NOT NULL,
        ended_at TIMESTAMPTZ,
        logs TEXT,
        PRIMARY KEY (pipeline_run_id, task_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS task_outputs (
        pipeline_run_id UUID NOT NULL,
        task_name TEXT NOT NULL,
        output_name TEXT NOT NULL,
        artifact_key TEXT NOT NULL,
        PRIMARY KEY (pipeline_run_id, task_name, output_name),
        FOREIGN KEY (pipeline_run_id, task_name)
            REFERENCES task_runs (pipeline_run_id, task_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_registrations (
        pipeline_registration_id UUID PRIMARY KEY,
        pipeline_name TEXT NOT NULL,
        backend TEXT NOT NULL,
        dag_structure JSONB NOT NULL,
        pipeline_inputs JSONB NOT NULL,
        registered_at TIMESTAMPTZ NOT NULL,
        deregistered_at TIMESTAMPTZ,
        backend_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS triggers (
        trigger_id UUID PRIMARY KEY,
        pipeline_registration_id UUID NOT NULL
            REFERENCES pipeline_registrations (pipeline_registration_id),
        trigger_type TEXT NOT NULL,
        trigger_config JSONB NOT NULL,
        registered_at TIMESTAMPTZ NOT NULL,
        deregistered_at TIMESTAMPTZ
    )
    """,
    # Adds columns introduced after a database's tables already existed -
    # `CREATE TABLE IF NOT EXISTS` alone leaves an existing table's columns
    # untouched. Unlike SQLite, Postgres supports `IF NOT EXISTS` directly on
    # `ADD COLUMN`, so no separate existence check is needed here.
    "ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS "
    "pipeline_assembly_id UUID REFERENCES pipeline_assemblies (pipeline_assembly_id)",
    "ALTER TABLE pipeline_registrations ADD COLUMN IF NOT EXISTS "
    "backend_metadata JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE task_runs ADD COLUMN IF NOT EXISTS logs TEXT",
    "CREATE INDEX IF NOT EXISTS idx_pipeline_assemblies_pipeline_name "
    "ON pipeline_assemblies (pipeline_name)",
    "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_pipeline_name "
    "ON pipeline_runs (pipeline_name)",
    "CREATE INDEX IF NOT EXISTS idx_pipeline_registrations_pipeline_name "
    "ON pipeline_registrations (pipeline_name)",
    "CREATE INDEX IF NOT EXISTS idx_triggers_pipeline_registration_id "
    "ON triggers (pipeline_registration_id)",
]


class PostgresMetadataStoreConfig(BaseSettings):
    """Configuration for a `PostgresMetadataStore`.

    Attributes:
        dsn: The PostgreSQL connection string (e.g.
            `"postgresql://user:password@host:5432/dbname"`). No default -
            unlike `LocalMetadataStoreConfig.db_path`, there is no
            meaningful "local" fallback for a remote database. Can also be
            set via the `bettmensch_ai_postgres_metadata_store_dsn`
            environment variable.
    """

    dsn: str

    model_config = SettingsConfigDict(
        env_prefix="bettmensch_ai_postgres_metadata_store_"
    )


class PostgresMetadataStore(BaseMetadataStore):
    """Remote, shared Layer 2 metadata store, backed by PostgreSQL.

    The `BaseArtifactStore`/`LocalArtifactStore` split's Layer 2
    counterpart for `BaseMetadataStore`: where `LocalMetadataStore` is a
    single local SQLite file suited to one local process, this is reachable
    by every worker in a genuinely distributed run, so a remote/distributed
    Layer 2 orchestrator's workers can all report into the *same* store.
    Implements the exact same interface as `LocalMetadataStore` - callers
    (like `LocalRunner`, or a future Layer 2 orchestrator) don't need to
    know or care which one they're talking to.

    Requires the `psycopg` package, which is *not* a hard dependency of
    this project (installing a PostgreSQL client library for everyone using
    only Layer 1's local pipeline assembly would be a needless cost) - it's
    imported lazily, in `__init__`, so importing this module (or the
    `metadata_store` package as a whole) never requires `psycopg` to be
    installed; only actually constructing a `PostgresMetadataStore` does.
    Install it with `pip install psycopg[binary]` (or this project's
    `postgres` extra, once published) before using this class.

    Schema is created (`CREATE TABLE IF NOT EXISTS`) on every
    initialization, mirroring `LocalMetadataStore` - fine for this project's
    current scope, but a real production deployment sharing this database
    across many services would more likely want explicit, versioned
    migrations instead of an app auto-creating its own schema on connect.
    """

    def __init__(self, config: Optional[PostgresMetadataStoreConfig] = None):
        """Initializes the store, creating its schema if it doesn't exist.

        Args:
            config: The store's configuration. Defaults to
                `PostgresMetadataStoreConfig()` (reading from the
                environment) if omitted.

        Raises:
            ImportError: If the `psycopg` package isn't installed.
        """

        try:
            import psycopg
            from psycopg.types.json import Jsonb
        except ImportError as exc:
            raise ImportError(
                "PostgresMetadataStore requires the `psycopg` package, "
                "which isn't installed. Install it with "
                "`pip install psycopg[binary]`."
            ) from exc

        self._psycopg = psycopg
        self._Jsonb = Jsonb
        self.config = config or PostgresMetadataStoreConfig()

        with self._connect() as connection:
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)

    @contextmanager
    def _connect(self) -> Iterator["psycopg.Connection"]:
        """Opens a connection to the database, committing and closing it
        afterwards.

        A fresh connection per call, mirroring `LocalMetadataStore._connect`
        - bookkeeping calls are infrequent, not a hot path, so this trades a
        modest per-call connection cost for not having to manage a
        long-lived connection's lifecycle (or a connection pool) here. A
        real production deployment under sustained load would more likely
        use `psycopg_pool` instead.

        Returns:
            An open connection, as a context manager.
        """

        connection = self._psycopg.connect(self.config.dsn)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def start_pipeline_run(
        self,
        pipeline_name: str,
        pipeline_run_id: uuid.UUID,
        pipeline_assembly_id: Optional[uuid.UUID] = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO pipeline_runs "
                "(pipeline_run_id, pipeline_name, status, started_at, "
                "pipeline_assembly_id) VALUES (%s, %s, %s, %s, %s)",
                (
                    pipeline_run_id,
                    pipeline_name,
                    RunStatus.RUNNING.value,
                    _now(),
                    pipeline_assembly_id,
                ),
            )

    def finish_pipeline_run(
        self, pipeline_run_id: uuid.UUID, status: RunStatus
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE pipeline_runs SET status = %s, ended_at = %s "
                "WHERE pipeline_run_id = %s",
                (status.value, _now(), pipeline_run_id),
            )

    def start_task_run(self, pipeline_run_id: uuid.UUID, task_name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO task_runs "
                "(pipeline_run_id, task_name, status, started_at) "
                "VALUES (%s, %s, %s, %s)",
                (pipeline_run_id, task_name, RunStatus.RUNNING.value, _now()),
            )

    def finish_task_run(
        self,
        pipeline_run_id: uuid.UUID,
        task_name: str,
        status: RunStatus,
        logs: Optional[str] = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE task_runs SET status = %s, ended_at = %s, logs = %s "
                "WHERE pipeline_run_id = %s AND task_name = %s",
                (status.value, _now(), logs, pipeline_run_id, task_name),
            )

    def record_task_output(
        self,
        pipeline_run_id: uuid.UUID,
        task_name: str,
        output_name: str,
        artifact_key: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO task_outputs "
                "(pipeline_run_id, task_name, output_name, artifact_key) "
                "VALUES (%s, %s, %s, %s)",
                (pipeline_run_id, task_name, output_name, artifact_key),
            )

    def get_pipeline_run(self, pipeline_run_id: uuid.UUID) -> PipelineRunRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT pipeline_run_id, pipeline_name, status, started_at, "
                "ended_at, pipeline_assembly_id FROM pipeline_runs "
                "WHERE pipeline_run_id = %s",
                (pipeline_run_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"No pipeline run recorded with id {pipeline_run_id!r}.")

        return _pipeline_run_record(row)

    def list_pipeline_runs(
        self, pipeline_name: Optional[str] = None
    ) -> List[PipelineRunRecord]:
        query = (
            "SELECT pipeline_run_id, pipeline_name, status, started_at, "
            "ended_at, pipeline_assembly_id FROM pipeline_runs"
        )
        params: tuple = ()

        if pipeline_name is not None:
            query += " WHERE pipeline_name = %s"
            params = (pipeline_name,)

        query += " ORDER BY started_at DESC"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        return [_pipeline_run_record(row) for row in rows]

    def get_task_run(self, pipeline_run_id: uuid.UUID, task_name: str) -> TaskRunRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT pipeline_run_id, task_name, status, started_at, "
                "ended_at, logs FROM task_runs "
                "WHERE pipeline_run_id = %s AND task_name = %s",
                (pipeline_run_id, task_name),
            ).fetchone()

        if row is None:
            raise KeyError(
                f"No task run recorded for task {task_name!r} in pipeline "
                f"run {pipeline_run_id!r}."
            )

        return _task_run_record(row)

    def list_task_runs(self, pipeline_run_id: uuid.UUID) -> List[TaskRunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT pipeline_run_id, task_name, status, started_at, "
                "ended_at, logs FROM task_runs WHERE pipeline_run_id = %s "
                "ORDER BY started_at ASC",
                (pipeline_run_id,),
            ).fetchall()

        return [_task_run_record(row) for row in rows]

    def list_task_outputs(
        self, pipeline_run_id: uuid.UUID, task_name: str
    ) -> List[TaskOutputRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT pipeline_run_id, task_name, output_name, "
                "artifact_key FROM task_outputs "
                "WHERE pipeline_run_id = %s AND task_name = %s",
                (pipeline_run_id, task_name),
            ).fetchall()

        return [
            TaskOutputRecord(
                pipeline_run_id=row[0],
                task_name=row[1],
                output_name=row[2],
                artifact_key=row[3],
            )
            for row in rows
        ]

    def record_pipeline_assembly(
        self,
        pipeline_name: str,
        dag_structure: Dict[str, Any],
        pipeline_inputs: Dict[str, Any],
    ) -> uuid.UUID:
        pipeline_assembly_id = uuid.uuid4()

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO pipeline_assemblies "
                "(pipeline_assembly_id, pipeline_name, dag_structure, "
                "pipeline_inputs, assembled_at) VALUES (%s, %s, %s, %s, %s)",
                (
                    pipeline_assembly_id,
                    pipeline_name,
                    self._Jsonb(dag_structure),
                    self._Jsonb(pipeline_inputs),
                    _now(),
                ),
            )

        return pipeline_assembly_id

    def get_pipeline_assembly(
        self, pipeline_assembly_id: uuid.UUID
    ) -> PipelineAssemblyRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT pipeline_assembly_id, pipeline_name, dag_structure, "
                "pipeline_inputs, assembled_at FROM pipeline_assemblies "
                "WHERE pipeline_assembly_id = %s",
                (pipeline_assembly_id,),
            ).fetchone()

        if row is None:
            raise KeyError(
                f"No pipeline assembly recorded with id {pipeline_assembly_id!r}."
            )

        return _pipeline_assembly_record(row)

    def list_pipeline_assemblies(
        self, pipeline_name: Optional[str] = None
    ) -> List[PipelineAssemblyRecord]:
        query = (
            "SELECT pipeline_assembly_id, pipeline_name, dag_structure, "
            "pipeline_inputs, assembled_at FROM pipeline_assemblies"
        )
        params: tuple = ()

        if pipeline_name is not None:
            query += " WHERE pipeline_name = %s"
            params = (pipeline_name,)

        query += " ORDER BY assembled_at DESC"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        return [_pipeline_assembly_record(row) for row in rows]

    def register_pipeline(
        self,
        pipeline_name: str,
        backend: str,
        dag_structure: Dict[str, Any],
        pipeline_inputs: Dict[str, Any],
        backend_metadata: Optional[Dict[str, Any]] = None,
    ) -> uuid.UUID:
        pipeline_registration_id = uuid.uuid4()

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO pipeline_registrations "
                "(pipeline_registration_id, pipeline_name, backend, "
                "dag_structure, pipeline_inputs, registered_at, "
                "backend_metadata) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    pipeline_registration_id,
                    pipeline_name,
                    backend,
                    self._Jsonb(dag_structure),
                    self._Jsonb(pipeline_inputs),
                    _now(),
                    self._Jsonb(backend_metadata or {}),
                ),
            )

        return pipeline_registration_id

    def deregister_pipeline(self, pipeline_registration_id: uuid.UUID) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE pipeline_registrations SET deregistered_at = %s "
                "WHERE pipeline_registration_id = %s",
                (_now(), pipeline_registration_id),
            )

    def get_pipeline_registration(
        self, pipeline_registration_id: uuid.UUID
    ) -> PipelineRegistrationRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT pipeline_registration_id, pipeline_name, backend, "
                "dag_structure, pipeline_inputs, registered_at, "
                "deregistered_at, backend_metadata FROM pipeline_registrations "
                "WHERE pipeline_registration_id = %s",
                (pipeline_registration_id,),
            ).fetchone()

        if row is None:
            raise KeyError(
                f"No pipeline registration recorded with id "
                f"{pipeline_registration_id!r}."
            )

        return _pipeline_registration_record(row)

    def list_pipeline_registrations(
        self, pipeline_name: Optional[str] = None, active_only: bool = False
    ) -> List[PipelineRegistrationRecord]:
        query = (
            "SELECT pipeline_registration_id, pipeline_name, backend, "
            "dag_structure, pipeline_inputs, registered_at, deregistered_at, "
            "backend_metadata FROM pipeline_registrations"
        )
        conditions: List[str] = []
        params: List[Any] = []

        if pipeline_name is not None:
            conditions.append("pipeline_name = %s")
            params.append(pipeline_name)

        if active_only:
            conditions.append("deregistered_at IS NULL")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY registered_at DESC"

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()

        return [_pipeline_registration_record(row) for row in rows]

    def register_trigger(
        self,
        pipeline_registration_id: uuid.UUID,
        trigger_type: str,
        trigger_config: Dict[str, Any],
    ) -> uuid.UUID:
        trigger_id = uuid.uuid4()

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO triggers "
                "(trigger_id, pipeline_registration_id, trigger_type, "
                "trigger_config, registered_at) VALUES (%s, %s, %s, %s, %s)",
                (
                    trigger_id,
                    pipeline_registration_id,
                    trigger_type,
                    self._Jsonb(trigger_config),
                    _now(),
                ),
            )

        return trigger_id

    def deregister_trigger(self, trigger_id: uuid.UUID) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE triggers SET deregistered_at = %s WHERE trigger_id = %s",
                (_now(), trigger_id),
            )

    def list_triggers(
        self, pipeline_registration_id: uuid.UUID, active_only: bool = False
    ) -> List[TriggerRecord]:
        query = (
            "SELECT trigger_id, pipeline_registration_id, trigger_type, "
            "trigger_config, registered_at, deregistered_at FROM triggers "
            "WHERE pipeline_registration_id = %s"
        )
        params: List[Any] = [pipeline_registration_id]

        if active_only:
            query += " AND deregistered_at IS NULL"

        query += " ORDER BY registered_at ASC"

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()

        return [_trigger_record(row) for row in rows]


def _now() -> datetime:
    """Returns the current UTC time as a real `datetime` - psycopg adapts
    this directly to/from `TIMESTAMPTZ`, no string encoding needed.
    """

    return datetime.now(timezone.utc)


def _pipeline_run_record(row: Any) -> PipelineRunRecord:
    """Builds a `PipelineRunRecord` from a `pipeline_runs` row. Unlike
    `LocalMetadataStore`'s equivalent, no type conversion is needed: psycopg
    already hands back a `uuid.UUID` and timezone-aware `datetime`s.
    """

    return PipelineRunRecord(
        pipeline_run_id=row[0],
        pipeline_name=row[1],
        status=RunStatus(row[2]),
        started_at=row[3],
        ended_at=row[4],
        pipeline_assembly_id=row[5],
    )


def _pipeline_assembly_record(row: Any) -> PipelineAssemblyRecord:
    """Builds a `PipelineAssemblyRecord` from a `pipeline_assemblies` row."""

    return PipelineAssemblyRecord(
        pipeline_assembly_id=row[0],
        pipeline_name=row[1],
        dag_structure=row[2],
        pipeline_inputs=row[3],
        assembled_at=row[4],
    )


def _task_run_record(row: Any) -> TaskRunRecord:
    """Builds a `TaskRunRecord` from a `task_runs` row."""

    return TaskRunRecord(
        pipeline_run_id=row[0],
        task_name=row[1],
        status=RunStatus(row[2]),
        started_at=row[3],
        ended_at=row[4],
        logs=row[5] if len(row) > 5 else None,
    )


def _pipeline_registration_record(row: Any) -> PipelineRegistrationRecord:
    """Builds a `PipelineRegistrationRecord` from a `pipeline_registrations`
    row. `dag_structure`/`pipeline_inputs` come back already deserialized
    into plain dicts - psycopg decodes `JSONB` automatically.
    """

    return PipelineRegistrationRecord(
        pipeline_registration_id=row[0],
        pipeline_name=row[1],
        backend=row[2],
        dag_structure=row[3],
        pipeline_inputs=row[4],
        registered_at=row[5],
        deregistered_at=row[6],
        backend_metadata=row[7] if len(row) > 7 and row[7] is not None else {},
    )


def _trigger_record(row: Any) -> TriggerRecord:
    """Builds a `TriggerRecord` from a `triggers` row."""

    return TriggerRecord(
        trigger_id=row[0],
        pipeline_registration_id=row[1],
        trigger_type=row[2],
        trigger_config=row[3],
        registered_at=row[4],
        deregistered_at=row[5],
    )
