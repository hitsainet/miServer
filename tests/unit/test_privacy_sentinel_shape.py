"""The privacy sentinel's shape (`tests/support/sentinel.py`; rule-runner review 2026-11-21).

The leak checks fail on ANY 4-character piece of the sentinel. A piece that ordinary text holds
("otal", "link", a run of hex) is a false leak. The hostile test read one, on the "logs" surface,
with nothing leaked. Every piece must therefore be two capitals and two small letters,
interleaved, which no word, identifier, digest or time has.
"""

from __future__ import annotations

from tests.support.sentinel import pieces, sentinel

#: Text of the kinds the guarded surfaces carry, including the fragments the old draw matched.
ORDINARY = ("total encoded link income loaded torch transform priority deadbeef0123abcdef "
            "2026-11-21T10:29:00.1107282Z rule_signal_results executemany UPDATE SELECT True "
            "None JSON Json Bool aiosqlite DEBUG INFO WARNING http HTTP OK rs_7023a4c20737")


def test_every_piece_is_two_capitals_and_two_small_letters_interleaved():
    for _ in range(2000):
        value = sentinel()
        assert len(value) == 14 and value.isalpha()
        assert pieces(value) == [value[i:i + 4] for i in range(11)]
        for piece in pieces(value):
            shape = [c.isupper() for c in piece]
            assert shape in ([True, False, True, False], [False, True, False, True]), value


def test_no_piece_occurs_in_ordinary_log_text():
    for _ in range(2000):
        assert not any(piece in ORDINARY for piece in pieces(sentinel()))
