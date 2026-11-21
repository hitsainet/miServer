"""A privacy sentinel whose every 4-character piece is foreign to log text (rule-runner review 2026-11-21).

The leak checks look for any 4-character piece of a random sentinel on a surface, so a piece that
occurs in ordinary log text reads as a leak. Drawn from all 52 letters, a sentinel's piece is
all-lowercase one time in sixteen ("otal", "ncod", "link" ...). Against the 700,000 characters of
the hostile test's captured log, 0.27% of such sentinels matched with no leak at all. One did, in a
CI-sized container (the "logs" surface, `0xcc/reviews/rule_runner_resource_limit_2026-11-21.md`).
Strictly alternating case makes every piece two capitals and two small letters interleaved. No
English word, identifier, hex digest or time has that shape: 0 of 200,000 matched the same log.
"""

from __future__ import annotations

import random
import string


def sentinel(length: int = 14) -> str:
    """Random letters in strictly alternating case, starting with either."""
    rng = random.SystemRandom()
    first = rng.randrange(2)
    return "".join(
        rng.choice(string.ascii_uppercase if (i + first) % 2 else string.ascii_lowercase)
        for i in range(length))


def pieces(value: str) -> list[str]:
    """Every 4-character piece the leak checks look for."""
    return [value[i:i + 4] for i in range(len(value) - 3)]
