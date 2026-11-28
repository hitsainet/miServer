"""Migration 043 (ENH-011 X-1, Q1 under OQ-3): every probe, circuit and profile that exists when it runs is stamped
`recorded_before_text_tower`; a row made later is NULL; down drops the three columns.

On PostgreSQL through alembic (skips loudly without a server, fails under `MISERVER_REQUIRE_PG=1`), and the same
`upgrade()`/`downgrade()` on SQLite through alembic's operations, which runs anywhere. The refusal the stamp feeds
is `tests/unit/services/test_layer_meaning.py`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import uuid

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.ext.asyncio import create_async_engine

from tests.schema.test_weights_origin_migration import PG_BASE, _admin, _alembic, _pg_available, _query
from tests.support.env_flags import flag
from tests.support.package_root import package_path

MIGRATION = package_path("db", "migrations", "versions", "043_recorded_before_text_tower.py")
TABLES = ("probes", "circuits", "profiles")
COLUMN = "recorded_before_text_tower"


def _module():
    spec = importlib.util.spec_from_file_location("migration_043", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_revision_chain():
    module = _module()
    assert (module.revision, module.down_revision) == ("043", "042")
    assert module.TABLES == TABLES and module.COLUMN == COLUMN
    followers = sorted(p.name for p in MIGRATION.parent.glob("*.py") if 'down_revision = "042"' in p.read_text())
    assert followers == [MIGRATION.name], f"more than one migration follows 042 (a fork): {followers}"


def test_the_model_columns_match_043():
    from miserver.db.models.circuit import Circuit
    from miserver.db.models.probe import Probe
    from miserver.db.models.profile import Profile

    for model in (Probe, Circuit, Profile):
        column = model.__table__.columns[COLUMN]
        assert column.nullable is True and column.server_default is None, model.__tablename__


def _run(fn_name: str, engine: sa.Engine) -> None:
    module = _module()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(module, fn_name)()


def test_upgrade_stamps_every_existing_row_and_downgrade_drops_the_column_on_sqlite():
    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        for table in TABLES:
            conn.execute(sa.text(f"CREATE TABLE {table} (id VARCHAR PRIMARY KEY)"))
            conn.execute(sa.text(f"INSERT INTO {table} (id) VALUES ('old_1'), ('old_2')"))
    _run("upgrade", engine)
    with engine.begin() as conn:
        for table in TABLES:
            conn.execute(sa.text(f"INSERT INTO {table} (id) VALUES ('new_1')"))
            rows = dict(conn.execute(sa.text(f"SELECT id, {COLUMN} FROM {table} ORDER BY id")).all())
            stamps = {k: (json.loads(v) if v is not None else None) for k, v in rows.items()}
            assert stamps["new_1"] is None, table
            for old in ("old_1", "old_2"):
                assert stamps[old]["stamped_by"] == "migration 043", table
                assert "ENH-011 X-1" in stamps[old]["reason"]
    _run("downgrade", engine)
    with engine.connect() as conn:
        for table in TABLES:
            assert COLUMN not in {c["name"] for c in sa.inspect(conn).get_columns(table)}
            assert conn.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar() == 3, "rows are kept"


@pytest.fixture(scope="module")
def fresh_db():
    if not asyncio.run(_pg_available()):
        message = f"PostgreSQL is not reachable at {PG_BASE}"
        if flag("REQUIRE_PG") == "1":
            pytest.fail(f"{message} and MISERVER_REQUIRE_PG=1 — this guard did NOT run")
        pytest.skip(f"{message} — a migration round trip needs a real server")
    name = f"text_tower_{uuid.uuid4().hex[:8]}"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


async def _execute(db: str, *statements: str) -> None:
    engine = create_async_engine(f"{PG_BASE}/{db}")
    try:
        async with engine.begin() as conn:
            for statement in statements:
                await conn.execute(sa.text(statement))
    finally:
        await engine.dispose()


def _seed(db: str, suffix: str) -> None:
    asyncio.run(_execute(
        db,
        "INSERT INTO probes (id, name, definition, hf_id, d_model, n_layers, layer, rule, scope, rung) VALUES "
        f"('pr_{suffix}', 'p{suffix}', '{{}}'::jsonb, 'hf/x', 8, 16, 1, 'mean', 'all', 2)",
        "INSERT INTO circuits (id, name, circuit_meta, layers) VALUES "
        f"('crc_{suffix}', 'c{suffix}', '{{}}'::jsonb, '[1]'::jsonb)",
        f"INSERT INTO profiles (id, name) VALUES ('prof_{suffix}', 'f{suffix}')",
    ))


def _stamps(db: str, table: str) -> dict:
    return dict(asyncio.run(_query(db, f"SELECT id, {COLUMN} FROM {table} ORDER BY id")))


def test_043_stamps_the_old_rows_only_and_comes_down(fresh_db):
    _alembic(fresh_db, "upgrade", "042")
    _seed(fresh_db, "old")
    _alembic(fresh_db, "upgrade", "043")
    _seed(fresh_db, "new")
    for table, prefix in (("probes", "pr"), ("circuits", "crc"), ("profiles", "prof")):
        stamps = _stamps(fresh_db, table)
        assert stamps[f"{prefix}_new"] is None, table
        assert stamps[f"{prefix}_old"]["stamped_by"] == "migration 043", table
    types = asyncio.run(_query(fresh_db, "SELECT table_name, data_type, is_nullable FROM information_schema.columns "
                                         f"WHERE column_name = '{COLUMN}' ORDER BY table_name"))
    assert types == [("circuits", "jsonb", "YES"), ("probes", "jsonb", "YES"), ("profiles", "jsonb", "YES")]

    _alembic(fresh_db, "downgrade", "042")
    assert asyncio.run(_query(fresh_db, "SELECT table_name FROM information_schema.columns "
                                        f"WHERE column_name = '{COLUMN}'")) == []
    assert asyncio.run(_query(fresh_db, "SELECT count(*) FROM probes")) == [(2,)], "the rows are kept"
    _alembic(fresh_db, "upgrade", "head")
