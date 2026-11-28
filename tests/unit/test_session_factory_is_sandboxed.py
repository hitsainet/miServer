"""The unit suite never opens the application's own session factory against the workstation's database
(DEBT_TASKS 6.3, DEBT-043-c; `tests/support/session_factory_guard.py`).

Control: drop `session_factory_guard.install()` from `tests/conftest.py` -> red.
"""

from __future__ import annotations

import time

import pytest


def test_the_factory_and_its_copies_are_bound_to_the_sandbox():
    import miserver.api.dependencies as deps
    import miserver.db as db
    import miserver.db.base as base
    from miserver.core.config import settings
    from tests.support.session_factory_guard import SANDBOX_URL

    url = base.async_session_factory.kw["bind"].url
    assert url.render_as_string(hide_password=False) == SANDBOX_URL
    assert (url.host, url.port) == ("127.0.0.1", 1)
    assert base.engine is base.async_session_factory.kw["bind"]
    assert deps.async_session_factory is base.async_session_factory, "dependencies kept the real factory"
    assert db.async_session_factory is base.async_session_factory
    assert settings.DATABASE_URL != SANDBOX_URL, "the settings must be untouched"


@pytest.mark.asyncio
async def test_an_unpatched_open_is_refused_at_once():
    import sqlalchemy
    from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

    import miserver.db.base as base

    started = time.monotonic()
    with pytest.raises((OSError, OperationalError, InterfaceError, DBAPIError)):
        async with base.async_session_factory() as session:
            await session.execute(sqlalchemy.text("select 1"))
    assert time.monotonic() - started < 10
