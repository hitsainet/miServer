"""The application's own database session factory never reaches the workstation's database (DEBT_TASKS 6.3,
DEBT-043-c).

`miserver.db.base.async_session_factory` is bound at import to `settings.DATABASE_URL`, whose default is
`postgresql+asyncpg://postgres:postgres@localhost:5432/millm`. A test path that reaches the factory unpatched would
read or write whatever answers there. Measured 2026-11-26 with a guard that failed such a test: 69 tests failed and
221 errored across the unit tree, nearly all through code that catches the error (unload teardown, the detailed
health endpoint, lease routes). They did not write: in CI nothing listens on 5432, and on this workstation the
default credentials are refused. So each one already ran against a database that refuses it, and the risk was a
workstation where those credentials work.

`install()` binds the engine and the factory (and every module-level copy already imported) to `SANDBOX_URL`, a
TCP endpoint nothing listens on, so an unpatched open is refused at once everywhere, as it is in CI. `settings`
is untouched. A test that needs a database still patches the factory with its own
(`monkeypatch.setattr("miserver.db.base.async_session_factory", factory)`).
"""

from __future__ import annotations

import sys
from typing import Any

#: Nothing listens on port 1 (tcpmux); the connection is refused at once, as CI's is.
SANDBOX_URL = "postgresql+asyncpg://unpatched:unpatched@127.0.0.1:1/miserver_unpatched_test_factory"


def install() -> Any:
    import miserver.db.base as base
    from sqlalchemy.ext.asyncio import create_async_engine

    # The modules that copy the factory at import (`from miserver.db.base import async_session_factory`) are
    # imported HERE, before the rebind: one first imported later, inside a test that had patched the factory,
    # kept that test's factory for the rest of the process (2026-11-26, `miserver.api.dependencies`).
    import miserver.api.dependencies  # noqa: F401
    import miserver.db  # noqa: F401

    if getattr(base.engine, "url", None) is not None and str(base.engine.url.database) == "miserver_unpatched_test_factory":
        return base.async_session_factory
    real_engine, real_factory = base.engine, base.async_session_factory
    engine = create_async_engine(SANDBOX_URL, pool_pre_ping=False)
    factory = real_factory.__class__(engine, **{k: v for k, v in real_factory.kw.items() if k != "bind"})
    base.engine, base.async_session_factory = engine, factory
    # A module that did `from miserver.db.base import …` before this ran holds the real objects.
    for module in list(sys.modules.values()):
        if module is None:
            continue
        if getattr(module, "async_session_factory", None) is real_factory:
            module.async_session_factory = factory
        if getattr(module, "engine", None) is real_engine:
            module.engine = engine
    return factory
