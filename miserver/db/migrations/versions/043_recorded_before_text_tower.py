"""ENH-011 X-1 (Q1): stamp the artefacts recorded before layer resolution took the text decoder first.

Before this release, layer resolution took the FIRST layer stack in the module tree. On a composite model whose
vision tower is registered before its language model (Gemma 3 and 4, LLaVA, Qwen2.5-VL) that was the vision
encoder, so a probe, circuit or cluster profile names indices into that stack, and after the fix the same index
means a text decoder layer. Operator decision Q1 (2026-11-15): refuse such an artefact for serving and arming until
it is re-validated; delete nothing. OQ-3 (2026-11-26): build the flag even though the inventory found nothing.

One nullable JSONB column, `recorded_before_text_tower`, on `probes`, `circuits` and `profiles`, and EVERY row that
exists when this runs is stamped. miServer's rows do not record their model's architecture (a circuit or cluster
document names only `hf_id`), so a migration cannot tell a composite model's rows apart without guessing; the
decision is made at arm and activation time on the live model, which answers it exactly
(`services/layer_meaning.py`): a stamped row is refused only at a layer whose module moved. On a single-tower model
nothing moves, so nothing that served before this migration is refused. Rows made afterwards are NULL.

Downgrade drops the three columns, and with them every stamp.

Revision ID: 043
Revises: 042
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "043"
down_revision = "042"
branch_labels = None
depends_on = None

COLUMN = "recorded_before_text_tower"
TABLES = ("probes", "circuits", "profiles")
JSONVariant = sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True), "postgresql")

REASON = (
    "recorded before ENH-011 X-1, when layer resolution took the first layer stack; on a composite model that was "
    "not the text decoder's, so this artefact is refused wherever its layer indices now name a different module"
)


def upgrade() -> None:
    bind = op.get_bind()
    for table in TABLES:
        op.add_column(table, sa.Column(COLUMN, JSONVariant, nullable=True))
    stamp = sa.bindparam("stamp", type_=JSONVariant)
    for table in TABLES:
        bind.execute(
            sa.text(f"UPDATE {table} SET {COLUMN} = :stamp WHERE {COLUMN} IS NULL").bindparams(stamp),
            {"stamp": {"stamped_by": f"migration {revision}", "reason": REASON}},
        )


def downgrade() -> None:
    for table in reversed(TABLES):
        op.drop_column(table, COLUMN)
