"""Typed answer for the two sensing event-clear routes (044, documented-only typing).

Routes:
  * ``DELETE /api/sensing/events``          (``miserver/api/routes/management/sensing.py``)
  * ``DELETE /api/circuit-sensing/events``  (``miserver/api/routes/management/circuit_sensing.py``)

Both answer ``{"deleted": n}``, where ``n`` is ``int(result.rowcount or 0)`` from
``SensingRepository.clear`` / ``CircuitEdgeSensingRepository.clear``.
"""

from pydantic import BaseModel, Field


class EventsCleared(BaseModel):
    """The count of stored sensing events a clear removed."""

    deleted: int = Field(description="Number of event rows deleted (0 when none matched)")
