"""tetris_sprint: a simulacrum environment package.

Single-player 40 LINES sprint at input level on a virtual clock, for
coldopen's skill-telemetry work. One step is one keystroke; time advances by
declared integer costs plus a per-environment LATENCY parameter, and gravity
and lock delay run on that virtual clock. Finesse exists because steps are
keystrokes; sprint time and PPS exist because the clock is state.
"""

from enum import IntEnum


class Slots(IntEnum):
    """RNG slots — MUST mirror the table in spec.md."""

    #: Reset-time bag draws (step 0, index 0..5): current piece + 5-deep queue,
    #: all six from the first 7-bag.
    BAG_RESET = 0
    #: Bag draw whenever a queue piece is consumed. step = pre-increment t;
    #: index = per-step consume counter (0 or 1 - an action-phase lock or
    #: first-hold may be followed by an auto-lock in the physics phase).
    BAG_REFILL = 1
