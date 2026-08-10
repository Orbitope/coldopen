"""minesweeper: a simulacrum environment package."""

from enum import IntEnum


class Slots(IntEnum):
    """RNG slots — MUST mirror the table in spec.md."""

    #: Reset-time (step 0), one draw per cell with ``index = cell``. A uniform
    #: uint32 key per cell; the mine layout is "the M eligible cells with the
    #: smallest keys". Realised as keys rather than as a sample of M positions
    #: because taking the M smallest is a pure function of the key vector and
    #: the exclusion set, so it cannot depend on draw order — sequential
    #: sampling without replacement is exactly the kind of rule that diverges
    #: between a scalar and a batched implementation.
    MINE_KEYS = 0
