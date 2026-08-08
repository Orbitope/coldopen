"""The sprint telemetry extractor must count what actually happened.

The extractor detects locks from the outside (queue shifts plus terminal
corrections), which is exactly the kind of inference that can drift from the
truth silently. These tests drive the batched env with scripted action
streams whose lock/hold/clear counts are known, and check the rows.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "envs"))

from coldopen.tetris import FEATURE_NAMES, HUMAN_COMPARABLE, SprintTelemetry, rollout
from tetris_sprint.fast import TetrisSprintBatched


def make_env(n=4, latency=100):
    return TetrisSprintBatched(n, latency=latency)


def drive(env, script):
    """Run a per-step action script (list of ints, broadcast to the batch)."""
    env.reset(torch.arange(env.n, dtype=torch.int64))
    telemetry = SprintTelemetry(env)
    for a in script:
        actions = torch.full((env.n,), a, dtype=torch.int64)
        _, _, terminated, info = env.step(actions)
        telemetry.update(actions, terminated, info)
    return telemetry


def test_hard_drops_are_counted_as_locks():
    env = make_env()
    telemetry = drive(env, [8] * 10)
    assert (telemetry.locks == 10).all(), telemetry.locks


def test_movement_between_drops_counts_inputs_not_locks():
    env = make_env()
    telemetry = drive(env, [0, 1, 4, 8] * 5)
    assert (telemetry.locks == 5).all()
    assert (telemetry.steps == 20).all()


def test_first_hold_consumes_without_locking():
    env = make_env()
    telemetry = drive(env, [9, 8, 8])
    assert (telemetry.holds == 1).all()
    assert (telemetry.locks == 2).all()


def test_masked_second_hold_is_not_counted():
    env = make_env()
    telemetry = drive(env, [9, 9, 8])
    assert (telemetry.holds == 1).all(), "the second hold is masked and must not count"
    assert (telemetry.locks == 1).all()


def test_random_rollout_rows_are_internally_consistent():
    env = make_env(n=8)
    generator = torch.Generator().manual_seed(0)

    def random_policy(obs, env):
        return torch.randint(0, 10, (env.n,), generator=generator)

    rows = rollout(env, random_policy, total_episodes=8, seed=1)
    assert len(rows) == 8
    for row in rows:
        assert set(row) == set(FEATURE_NAMES)
        assert row["pieces"] > 0, "an episode with no locks should be impossible here"
        assert row["inputs"] >= row["pieces"], "every lock costs at least one input"
        total = row["singles"] + 2 * row["doubles"] + 3 * row["triples"] + 4 * row["quads"]
        assert total == row["lines"], f"clear sizes disagree with lines: {row}"
        assert row["time_ms"] > 0
        assert 0 <= row["quad_rate"] <= 1


def test_terminal_lock_is_counted_on_topout():
    """A run of hard drops ends in a T2/T3 top-out; the last lock must count."""
    env = make_env(n=2)
    env.reset(torch.arange(2, dtype=torch.int64))
    telemetry = SprintTelemetry(env)
    for _ in range(60):
        actions = torch.full((2,), 8, dtype=torch.int64)
        _, _, terminated, info = env.step(actions)
        telemetry.update(actions, terminated, info)
        if telemetry.rows:
            break
    assert telemetry.rows, "stacking in one spot must top out"
    row = telemetry.rows[0]
    # Every step was a hard drop, so pieces == inputs on the emitted episode.
    assert row["pieces"] == row["inputs"], row


def test_human_comparable_subset_is_a_subset():
    assert set(HUMAN_COMPARABLE) <= set(FEATURE_NAMES)
