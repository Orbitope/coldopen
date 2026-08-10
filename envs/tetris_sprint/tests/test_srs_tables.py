"""The SRS tables in both implementations must match spec.md verbatim.

The spec reproduces the published guideline kick tables and piece shapes in
full, and requires this comparison. The literals below are copied from the
spec BY HAND — they are the third, independent transcription, so a slip in
either implementation's tables cannot hide behind the differential test
(which only proves the two implementations agree with each other).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# spec: SRS kick tables — J, L, S, T, Z
SPEC_JLSTZ = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
}
# spec: SRS kick tables — I
SPEC_I = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
}
# spec: Piece shapes — rot 0 spawn cells, and spawn placement
SPEC_SPAWN_CELLS = {
    0: {(0, 2), (1, 2), (2, 2), (3, 2)},           # I
    1: {(1, 1), (2, 1), (1, 2), (2, 2)},           # O
    2: {(0, 1), (1, 1), (2, 1), (1, 2)},           # T
    3: {(0, 1), (1, 1), (2, 1), (0, 2)},           # J
    4: {(0, 1), (1, 1), (2, 1), (2, 2)},           # L
    5: {(0, 1), (1, 1), (1, 2), (2, 2)},           # S
    6: {(1, 1), (2, 1), (0, 2), (1, 2)},           # Z
}
SPEC_SPAWN_POS = {0: (3, 18), 1: (3, 19), 2: (3, 19), 3: (3, 19),
                  4: (3, 19), 5: (3, 19), 6: (3, 19)}


def test_reference_tables_match_the_spec():
    from tetris_sprint import reference as ref

    assert ref.KICKS_JLSTZ == SPEC_JLSTZ
    assert ref.KICKS_I == SPEC_I
    for piece, cells in SPEC_SPAWN_CELLS.items():
        assert set(ref.CELLS[piece][0]) == cells, f"piece {piece} spawn cells"
    for piece, pos in SPEC_SPAWN_POS.items():
        assert ref.SPAWN[piece] == pos


def test_batched_tables_match_the_spec():
    from tetris_sprint import fast

    assert fast._KICKS_JLSTZ == SPEC_JLSTZ
    assert fast._KICKS_I == SPEC_I
    for piece, cells in SPEC_SPAWN_CELLS.items():
        assert set(fast._CELLS[piece][0]) == cells, f"piece {piece} spawn cells"
    for piece, pos in SPEC_SPAWN_POS.items():
        assert tuple(fast._SPAWN[piece]) == pos


def test_kick_tensor_encodes_the_tables_in_try_order():
    import torch

    from tetris_sprint import fast

    table = fast._kick_tensor()
    for from_rot in range(4):
        for direction, to_rot in ((0, (from_rot + 1) % 4), (1, (from_rot + 3) % 4)):
            expected_i = SPEC_I[(from_rot, to_rot)]
            expected_j = SPEC_JLSTZ[(from_rot, to_rot)]
            assert table[0, from_rot, direction].tolist() == [list(k) for k in expected_i]
            for piece in (2, 3, 4, 5, 6):
                assert table[piece, from_rot, direction].tolist() == [
                    list(k) for k in expected_j]
            assert (table[1, from_rot, direction] == 0).all()  # O: identity
