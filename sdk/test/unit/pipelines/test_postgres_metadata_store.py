"""Unit tests for `PostgresMetadataStore`: just the lazy-import failure
path, which needs no real Postgres and no `psycopg` installed to verify -
see sdk/test/integration/pipelines/test_postgres_metadata_store_integration.py
for tests against a real PostgreSQL server.
"""

import pytest
from bettmensch_ai.pipelines.metadata_store import (
    PostgresMetadataStore,
    PostgresMetadataStoreConfig,
)


def test_missing_psycopg_raises_a_clear_import_error(monkeypatch):
    """Deliberately not gated on `psycopg` actually being absent - it
    verifies the lazy-import failure path itself by simulating that
    absence via monkeypatching, so it behaves the same whether or not
    `psycopg` happens to be installed in the environment running it.
    """

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psycopg" or name.startswith("psycopg."):
            raise ImportError("simulated missing psycopg")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="psycopg"):
        PostgresMetadataStore(PostgresMetadataStoreConfig(dsn="postgresql://unused"))
