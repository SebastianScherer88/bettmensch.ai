"""`LocalMetadataStore`: Layer 1's default, SQLite-backed metadata store."""

import json
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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

_DEFAULT_DB_PATH = str(Path(tempfile.gettempdir()) / "bettmensch_ai" / "metadata.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_assemblies (
    pipeline_assembly_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    dag_structure TEXT NOT NULL,
    pipeline_inputs TEXT NOT NULL,
    assembled_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    pipeline_run_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    pipeline_assembly_id TEXT
);

CREATE TABLE IF NOT EXISTS task_runs (
    pipeline_run_id TEXT NOT NULL,
    task_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    logs TEXT,
    PRIMARY KEY (pipeline_run_id, task_name)
);

CREATE TABLE IF NOT EXISTS task_outputs (
    pipeline_run_id TEXT NOT NULL,
    task_name TEXT NOT NULL,
    output_name TEXT NOT NULL,
    artifact_key TEXT NOT NULL,
    PRIMARY KEY (pipeline_run_id, task_name, output_name)
);

CREATE TABLE IF NOT EXISTS pipeline_registrations (
    pipeline_registration_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    backend TEXT NOT NULL,
    dag_structure TEXT NOT NULL,
    pipeline_inputs TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    deregistered_at TEXT,
    backend_metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS triggers (
    trigger_id TEXT PRIMARY KEY,
    pipeline_registration_id TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    trigger_config TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    deregistered_at TEXT
);
"""


class LocalMetadataStoreConfig(BaseSettings):
    """Configuration for a `LocalMetadataStore`.

    Attributes:
        db_path: The local SQLite database file to persist bookkeeping to.
            Defaults to a `bettmensch_ai/metadata.db` file under the system
            temp directory. Can also be set via the
            `bettmensch_ai_local_metadata_store_db_path` environment
            variable.
    """

    db_path: str = _DEFAULT_DB_PATH

    model_config = SettingsConfigDict(
        env_prefix="bettmensch_ai_local_metadata_store_"
    )


class LocalMetadataStore(BaseMetadataStore):
    """Default, backend-agnostic Layer 1 metadata store.

    Persists pipeline/task run bookkeeping, and pipeline registration/
    trigger bookkeeping, to a local SQLite database file - a real, if
    lightweight, relational store rather than hand-rolled JSON files, since
    bookkeeping is inherently about structured records that need to be
    queried back (by pipeline, by run, by task), not just written once and
    re-read whole. `sqlite3` is part of the standard library, so this adds
    no new dependency. `dag_structure`/`pipeline_inputs`/`trigger_config`
    are stored as JSON-encoded `TEXT` (SQLite has no native JSON type);
    `PostgresMetadataStore` stores the same fields as native `JSONB`
    instead. Remote, backend-specific flavours reachable by genuinely
    distributed Layer 2 workers (like `PostgresMetadataStore`) belong to
    Layer 2 conceptually, not here - mirroring `LocalArtifactStore`'s own
    scope; this remains Layer 1's local, single-process default.
    """

    def __init__(self, config: Optional[LocalMetadataStoreConfig] = None):
        """Initializes the store, creating its schema if this is a fresh
        database file.

        Args:
            config: The store's configuration. Defaults to
                `LocalMetadataStoreConfig()` (reading from the environment)
                if omitted.
        """

        self.config = config or LocalMetadataStoreConfig()
        self.db_path = Path(self.config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            self._migrate(connection)

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """Adds columns introduced after a database file's tables already
        existed - `CREATE TABLE IF NOT EXISTS` alone leaves an existing
        table's columns untouched, so an older on-disk database would
        otherwise miss e.g. `pipeline_runs.pipeline_assembly_id` forever.
        Each check is idempotent, so this is safe to run on every init.
        """

        self._ensure_column(
            connection, "pipeline_runs", "pipeline_assembly_id", "TEXT"
        )
        self._ensure_column(
            connection,
            "pipeline_registrations",
            "backend_metadata",
            "TEXT NOT NULL DEFAULT '{}'",
        )
        self._ensure_column(connection, "task_runs", "logs", "TEXT")

    def _ensure_column(
        self, connection: sqlite3.Connection, table: str, column: str, ddl_type: str
    ) -> None:
        """Adds `column` to `table` if it doesn't already exist.

        SQLite's `ALTER TABLE ... ADD COLUMN` has no `IF NOT EXISTS` clause
        (unlike Postgres), so this checks `PRAGMA table_info` itself first.
        """

        existing_columns = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in existing_columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Opens a connection to the database, committing and closing it
        afterwards.

        A fresh connection per call rather than one kept open for the
        store's lifetime: bookkeeping calls are infrequent (one per task/
        pipeline run transition, not a hot path), so the simplicity of not
        managing a long-lived connection's lifecycle outweighs the modest
        per-call connection overhead.

        Returns:
            An open connection, as a context manager.
        """

        connection = sqlite3.connect(self.db_path)
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
                "pipeline_assembly_id) VALUES (?, ?, ?, ?, ?)",
                (
                    str(pipeline_run_id),
                    pipeline_name,
                    RunStatus.RUNNING.value,
                    _now(),
                    str(pipeline_assembly_id) if pipeline_assembly_id else None,
                ),
            )

    def finish_pipeline_run(
        self, pipeline_run_id: uuid.UUID, status: RunStatus
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE pipeline_runs SET status = ?, ended_at = ? "
                "WHERE pipeline_run_id = ?",
                (status.value, _now(), str(pipeline_run_id)),
            )

    def start_task_run(self, pipeline_run_id: uuid.UUID, task_name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO task_runs "
                "(pipeline_run_id, task_name, status, started_at) "
                "VALUES (?, ?, ?, ?)",
                (str(pipeline_run_id), task_name, RunStatus.RUNNING.value, _now()),
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
                "UPDATE task_runs SET status = ?, ended_at = ?, logs = ? "
                "WHERE pipeline_run_id = ? AND task_name = ?",
                (status.value, _now(), logs, str(pipeline_run_id), task_name),
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
                "VALUES (?, ?, ?, ?)",
                (str(pipeline_run_id), task_name, output_name, artifact_key),
            )

    def get_pipeline_run(self, pipeline_run_id: uuid.UUID) -> PipelineRunRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT pipeline_run_id, pipeline_name, status, started_at, "
                "ended_at, pipeline_assembly_id FROM pipeline_runs "
                "WHERE pipeline_run_id = ?",
                (str(pipeline_run_id),),
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
        params: Tuple[str, ...] = ()

        if pipeline_name is not None:
            query += " WHERE pipeline_name = ?"
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
                "WHERE pipeline_run_id = ? AND task_name = ?",
                (str(pipeline_run_id), task_name),
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
                "ended_at, logs FROM task_runs WHERE pipeline_run_id = ? "
                "ORDER BY started_at ASC",
                (str(pipeline_run_id),),
            ).fetchall()

        return [_task_run_record(row) for row in rows]

    def list_task_outputs(
        self, pipeline_run_id: uuid.UUID, task_name: str
    ) -> List[TaskOutputRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT pipeline_run_id, task_name, output_name, "
                "artifact_key FROM task_outputs "
                "WHERE pipeline_run_id = ? AND task_name = ?",
                (str(pipeline_run_id), task_name),
            ).fetchall()

        return [
            TaskOutputRecord(
                pipeline_run_id=uuid.UUID(row[0]),
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
                "pipeline_inputs, assembled_at) VALUES (?, ?, ?, ?, ?)",
                (
                    str(pipeline_assembly_id),
                    pipeline_name,
                    json.dumps(dag_structure),
                    json.dumps(pipeline_inputs),
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
                "WHERE pipeline_assembly_id = ?",
                (str(pipeline_assembly_id),),
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
        params: Tuple[str, ...] = ()

        if pipeline_name is not None:
            query += " WHERE pipeline_name = ?"
            params = (pipeline_name,)

        # rowid tiebreaker: two assemblies recorded in quick succession can
        # land on the same isoformat timestamp (string precision, not true
        # collision-proof), which would otherwise leave "most recent first"
        # order undefined for that tie - rowid is SQLite's own monotonic
        # insertion order, so it always breaks the tie correctly.
        query += " ORDER BY assembled_at DESC, rowid DESC"

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
                "backend_metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(pipeline_registration_id),
                    pipeline_name,
                    backend,
                    json.dumps(dag_structure),
                    json.dumps(pipeline_inputs),
                    _now(),
                    json.dumps(backend_metadata or {}),
                ),
            )

        return pipeline_registration_id

    def deregister_pipeline(self, pipeline_registration_id: uuid.UUID) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE pipeline_registrations SET deregistered_at = ? "
                "WHERE pipeline_registration_id = ?",
                (_now(), str(pipeline_registration_id)),
            )

    def get_pipeline_registration(
        self, pipeline_registration_id: uuid.UUID
    ) -> PipelineRegistrationRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT pipeline_registration_id, pipeline_name, backend, "
                "dag_structure, pipeline_inputs, registered_at, "
                "deregistered_at, backend_metadata FROM pipeline_registrations "
                "WHERE pipeline_registration_id = ?",
                (str(pipeline_registration_id),),
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
        params: List[str] = []

        if pipeline_name is not None:
            conditions.append("pipeline_name = ?")
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
                "trigger_config, registered_at) VALUES (?, ?, ?, ?, ?)",
                (
                    str(trigger_id),
                    str(pipeline_registration_id),
                    trigger_type,
                    json.dumps(trigger_config),
                    _now(),
                ),
            )

        return trigger_id

    def deregister_trigger(self, trigger_id: uuid.UUID) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE triggers SET deregistered_at = ? WHERE trigger_id = ?",
                (_now(), str(trigger_id)),
            )

    def list_triggers(
        self, pipeline_registration_id: uuid.UUID, active_only: bool = False
    ) -> List[TriggerRecord]:
        query = (
            "SELECT trigger_id, pipeline_registration_id, trigger_type, "
            "trigger_config, registered_at, deregistered_at FROM triggers "
            "WHERE pipeline_registration_id = ?"
        )
        params: List[str] = [str(pipeline_registration_id)]

        if active_only:
            query += " AND deregistered_at IS NULL"

        query += " ORDER BY registered_at ASC"

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()

        return [_trigger_record(row) for row in rows]


def _now() -> str:
    """Returns the current UTC time, ISO-formatted, for storage as TEXT."""

    return datetime.now(timezone.utc).isoformat()


def _pipeline_run_record(row: Tuple) -> PipelineRunRecord:
    """Builds a `PipelineRunRecord` from a `pipeline_runs` row."""

    return PipelineRunRecord(
        pipeline_run_id=uuid.UUID(row[0]),
        pipeline_name=row[1],
        status=RunStatus(row[2]),
        started_at=datetime.fromisoformat(row[3]),
        ended_at=datetime.fromisoformat(row[4]) if row[4] is not None else None,
        pipeline_assembly_id=uuid.UUID(row[5]) if row[5] is not None else None,
    )


def _pipeline_assembly_record(row: Tuple) -> PipelineAssemblyRecord:
    """Builds a `PipelineAssemblyRecord` from a `pipeline_assemblies` row."""

    return PipelineAssemblyRecord(
        pipeline_assembly_id=uuid.UUID(row[0]),
        pipeline_name=row[1],
        dag_structure=json.loads(row[2]),
        pipeline_inputs=json.loads(row[3]),
        assembled_at=datetime.fromisoformat(row[4]),
    )


def _task_run_record(row: Tuple) -> TaskRunRecord:
    """Builds a `TaskRunRecord` from a `task_runs` row."""

    return TaskRunRecord(
        pipeline_run_id=uuid.UUID(row[0]),
        task_name=row[1],
        status=RunStatus(row[2]),
        started_at=datetime.fromisoformat(row[3]),
        ended_at=datetime.fromisoformat(row[4]) if row[4] is not None else None,
        logs=row[5] if len(row) > 5 else None,
    )


def _pipeline_registration_record(row: Tuple) -> PipelineRegistrationRecord:
    """Builds a `PipelineRegistrationRecord` from a `pipeline_registrations`
    row.
    """

    return PipelineRegistrationRecord(
        pipeline_registration_id=uuid.UUID(row[0]),
        pipeline_name=row[1],
        backend=row[2],
        dag_structure=json.loads(row[3]),
        pipeline_inputs=json.loads(row[4]),
        registered_at=datetime.fromisoformat(row[5]),
        deregistered_at=datetime.fromisoformat(row[6]) if row[6] is not None else None,
        backend_metadata=json.loads(row[7]) if len(row) > 7 and row[7] is not None else {},
    )


def _trigger_record(row: Tuple) -> TriggerRecord:
    """Builds a `TriggerRecord` from a `triggers` row."""

    return TriggerRecord(
        trigger_id=uuid.UUID(row[0]),
        pipeline_registration_id=uuid.UUID(row[1]),
        trigger_type=row[2],
        trigger_config=json.loads(row[3]),
        registered_at=datetime.fromisoformat(row[4]),
        deregistered_at=datetime.fromisoformat(row[5]) if row[5] is not None else None,
    )
