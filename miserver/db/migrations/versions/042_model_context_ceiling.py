"""A per-model GGUF context ceiling, and the approval action that writes it (2026-11-21).

`models.context_ceiling`: the most context a GGUF model is loaded with, in tokens. NULL is the
server's GGUF_CONTEXT_LENGTH (every row that exists when this migration runs, so nothing loads
differently); 0 is "max", as large a window as fits up to what the model declares. A load request
may still name its own ceiling for that load.

An AI agent's write of it waits in miServer's approval queue as the new action
`model_context_write` (a stored setting, like `probe_limits_write`): the action CHECK constraint is
dropped and created again with the eighth name, its list written out LITERALLY (a migration records
what was applied; `tests/schema/test_approvals_migration.py` pins it to the model).

Downgrade deletes every `model_context_write` request, restores 038's constraint and drops the
column.

Revision ID: 042
Revises: 041
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "042"
down_revision = "041"
branch_labels = None
depends_on = None

TABLE = "models"
COLUMN = "context_ceiling"

CONSTRAINT = "ck_approval_requests_action"
PREVIOUS = ("model_load", "model_unload", "model_delete",
            "probe_limits_write", "capacity_measure", "code_rule_import",
            "telemetry_write")
ACTIONS = PREVIOUS + ("model_context_write",)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def upgrade() -> None:
    op.add_column(TABLE, sa.Column(COLUMN, sa.Integer(), nullable=True))
    op.drop_constraint(CONSTRAINT, "approval_requests", type_="check")
    op.create_check_constraint(CONSTRAINT, "approval_requests", _in("action", ACTIONS))


def downgrade() -> None:
    op.execute("DELETE FROM approval_requests WHERE action = 'model_context_write'")
    op.drop_constraint(CONSTRAINT, "approval_requests", type_="check")
    op.create_check_constraint(CONSTRAINT, "approval_requests", _in("action", PREVIOUS))
    op.drop_column(TABLE, COLUMN)
