"""The distillation ladder has to actually be a ladder.

Two things make it one: snapshots taken where the student is still changing,
and a student that genuinely tracks the teacher rather than collapsing onto one
action. Both fail quietly - a schedule bunched at the end produces six rungs of
near-identical strength, and a collapsed student produces a ladder that ranks by
nothing at all.
"""

import torch

from coldopen.distill import snapshot_schedule


def test_snapshots_are_dense_where_the_student_changes_fastest():
    """Log spacing, because the first few hundred steps do most of the work."""
    points = snapshot_schedule(4000, count=16, lo=50)
    assert points[0] == 0
    assert points[-1] == 4000
    assert len(points) >= 10

    early = [p for p in points if p <= 500]
    late = [p for p in points if p > 500]
    assert len(early) > len(late) / 2, "too few rungs where strength moves fastest"


def test_the_schedule_is_strictly_increasing_and_unique():
    points = snapshot_schedule(4000)
    assert points == sorted(set(points))


def test_a_short_run_still_yields_usable_rungs():
    """A smoke-sized run must not collapse to two snapshots."""
    points = snapshot_schedule(200, count=16, lo=10)
    assert len(points) >= 6
    assert max(points) == 200


def test_nothing_is_scheduled_past_the_end_of_training():
    for total in (100, 1000, 7777):
        assert max(snapshot_schedule(total)) <= total


def test_the_real_ladder_spans_random_to_near_teacher():
    """The property that makes distillation worth using: it covers the range.

    Handicapping only reaches down from a finished agent and undertraining only
    reaches up from random. A student starts at chance and ends near its
    teacher, so one run covers both ends.
    """
    import json
    import pathlib

    log = pathlib.Path("coldopen/ladders/connect_four_distilled/train_log.json")
    if not log.exists():
        import pytest
        pytest.skip("distilled ladder not built in this checkout")

    agreement = [row["teacher_agreement"] for row in json.loads(log.read_text())]
    assert agreement[0] < 0.3, "the first snapshot should be near chance"
    assert agreement[-1] > 0.8, "the last should be close to the teacher"
    # Monotone enough to be a ladder: no large backward step.
    drops = [b - a for a, b in zip(agreement, agreement[1:]) if b < a]
    assert all(d > -0.05 for d in drops), f"student regressed sharply: {drops}"
