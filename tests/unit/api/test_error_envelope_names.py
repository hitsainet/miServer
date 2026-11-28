"""No error body names an exception class (miServer 042 FTASKS 5.4; FTDD §5's "unverified" point).

042 renamed the base exception `MiLLMError` to `MiServerError` on the claim that the class name is (p9: keep, rename history)
not on the wire: a response carries `exc.code`, never the class. This proves the claim for a
`MiServerError` subclass of each status family, raised on a management route and on a `/v1`
route, through the handlers `create_app` registers. A body naming the class would make every
such rename a wire change.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from miserver.core.errors import MiServerError


class ZzOddRefusalError(MiServerError):
    code = "ZZ_ODD_REFUSAL"
    status_code = 409


class ZzOddFaultError(MiServerError):
    code = "ZZ_ODD_FAULT"
    status_code = 500


ERRORS = {"refusal": ZzOddRefusalError("a refusal"), "fault": ZzOddFaultError("a fault"),
          "base": MiServerError("the base class")}


@pytest.fixture(scope="module")
def client():
    from miserver.main import create_app

    app = create_app()
    for name, exc in ERRORS.items():
        for prefix in ("/api", "/v1"):
            async def endpoint(_exc=exc):
                raise _exc

            app.add_api_route(f"{prefix}/_zz/{name}", endpoint, methods=["GET"])
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("prefix", ["/api", "/v1"])
@pytest.mark.parametrize("name", sorted(ERRORS))
def test_no_body_names_the_exception_class(client, prefix, name):
    r = client.get(f"{prefix}/_zz/{name}")
    assert r.status_code >= 400, r.text
    for cls in type(ERRORS[name]).__mro__:
        if cls.__module__.startswith("miserver") or cls.__name__.startswith("Zz"):
            assert cls.__name__ not in r.text, (prefix, name, cls.__name__, r.text)
    assert "MiServerError" not in r.text and "MiLLMError" not in r.text  # p9: keep, the old class name must not appear
