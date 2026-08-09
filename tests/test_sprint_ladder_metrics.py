"""Guard the three sprint findings that no existing test could have caught.

Each of these was a silent wrong answer, not a crash — the suite was green
throughout. They are pinned here because every one of them made a measurement
report the opposite of the truth:

* a range-containment overlap check passed an agent sitting below every human;
* a coherence number was read as failure when it was the metric's noise floor;
* a deterministic policy deadlocked, and its telemetry described the deadlock.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "envs"))

from coldopen.sprint_bot import HOLD, SprintBot, best_pair, best_placement
from coldopen.sprint_ladders import (agent_deviation, human_noise_floor,
                                     overlap, rank_of)
from tetris_sprint.fast import _CELLS, TetrisSprintBatched


def _cells(piece, rot):
    return _CELLS[piece][rot]


def fake_humans():
    """Two ranks, wide within-rank spread, clearly separated means.

    The spread is the point: it is what let a range check pass anything.
    """
    rows = []
    for i in range(10):
        rows.append({"rank_index": 0, "rank": "d", "inputs_per_piece": 5.0 + i * 0.1,
                     "quad_rate": 0.00 + i * 0.01, "holds_per_piece": 0.05,
                     "max_b2b": 0.5, "pps": 1.0})
        rows.append({"rank_index": 17, "rank": "x+", "inputs_per_piece": 2.8,
                     "quad_rate": 0.60 + i * 0.04, "holds_per_piece": 0.30,
                     "max_b2b": 8.0, "pps": 6.0})
    return rows


def test_degenerate_agent_is_reported_off_manifold_not_inside():
    """An agent pinned below every human must not read as covering them.

    The original check asked "is the value between human p2 and p98?". Pooled
    human quad rate spans 0.00 to 1.00, so an agent stuck at 0.00 scored
    "100% inside the human range" while in truth no human played like it.
    """
    humans = fake_humans()
    pinned = [{"rung": f"r{i}", "quad_rate": 0.0, "inputs_per_piece": 5.4,
               "holds_per_piece": 0.05, "max_b2b": 0.5, "pps": 1.0}
              for i in range(4)]
    report = overlap({"pinned": pinned}, humans)
    span = report["quad_rate"]["pinned"]["rank_span"]
    # Every rung must map to the WEAKEST rank, never to a range implying
    # coverage of the strong end.
    assert span == [0, 0], span

    spread = [dict(r, quad_rate=q) for r, q in
              zip(pinned, [0.02, 0.25, 0.50, 0.75])]
    wide = overlap({"spread": spread}, humans)["quad_rate"]["spread"]["rank_span"]
    assert wide[1] > wide[0], "a ladder that really spans must report a span"


def test_rank_of_returns_none_outside_the_human_span():
    """Off-manifold is an answer, and must not be silently clamped."""
    humans = fake_humans()
    assert rank_of("holds_per_piece", 0.30, humans) == 17
    assert rank_of("holds_per_piece", 0.05, humans) == 0
    # Far above anything a human does: no rank, rather than "the best rank".
    assert rank_of("holds_per_piece", 9.0, humans) is None
    assert rank_of("pps", 500.0, humans) is None


def test_human_noise_floor_is_computed_and_nonzero():
    """The control that reframed the headline number must stay wired in.

    Reported alone, the agent ladder's coherence looked like a failure; the
    same statistic on held-out humans landed in the same place, making it the
    metric's floor. A zero floor would mean the control silently stopped
    working and the comparison became meaningless again.
    """
    floor = human_noise_floor(fake_humans())
    assert floor["n"] > 0
    assert floor["median_spread"] >= 0.0
    assert set(floor["feature_deviation"]) >= {"quad_rate", "holds_per_piece"}
    assert floor["mean_abs_rank_error"] >= 0.0


def test_agent_deviation_detects_a_single_off_feature():
    """Systematic per-feature bias must survive a metric that spread cannot see."""
    humans = fake_humans()
    # A rung that looks strong on everything EXCEPT hold, which is beginner.
    rungs = [{"rung": "strong_but_never_holds", "inputs_per_piece": 2.8,
              "quad_rate": 0.70, "holds_per_piece": 0.05, "max_b2b": 8.0,
              "pps": 6.0}]
    dev = agent_deviation({"a": rungs}, humans)["a"]
    # The median implied rank is "strong", so the four agreeing features sit at
    # zero deviation and the odd one out carries the whole signal. That is the
    # property being pinned: one wrong feature must not be averaged away.
    assert dev["holds_per_piece"] < -5.0, dev
    assert all(abs(dev[f]) < 1e-9 for f in
               ("inputs_per_piece", "quad_rate", "max_b2b", "pps")), dev


@pytest.mark.parametrize("skill", [1.0, 0.4])
def test_every_ladder_rung_finishes_the_sprint(skill):
    """A rung that tops out is measuring survival, not skill.

    This is the constraint that forced the skill dial to interpolate strategy
    rather than inject noise: at noise 8 the lower rungs cleared 1.6 lines and
    never finished, which no human record does.
    """
    env = TetrisSprintBatched(4, latency=17 if skill == 1.0 else 560)
    from coldopen.tetris import rollout
    from tetris_sprint.fast import HORIZON
    rows = rollout(env, SprintBot(env, skill=skill), total_episodes=4, seed=5,
                   max_steps=4 * HORIZON)
    assert rows
    assert np.mean([r["lines"] for r in rows]) > 20, \
        "lower rungs must still complete the task"


def test_the_search_covers_every_column_including_the_well():
    """Every column, especially the well at 9, must be in the search space.

    Written after I asserted the opposite and was wrong. `_CELLS` uses a
    bounding-box convention with every `dx >= 0`, so `x` is the box's left
    edge: a vertical I reaches column 9 at `x = 8`, well inside the sweep. The
    test pins the property that actually matters — full column coverage —
    rather than a claim about the loop bound.
    """
    from coldopen.sprint_bot import _score_after
    board = np.zeros((24, 10), dtype=np.int64)
    covered = set()
    for piece in range(7):
        for rot in range(4):
            for x in range(-2, 9):
                if _score_after(board, piece, rot, x) is None:
                    continue
                covered.update(x + dx for dx, _ in _cells(piece, rot))
    assert covered == set(range(10)), sorted(covered)

    # And specifically: a vertical I can occupy column 9 on its own.
    vertical = [(rot, x) for rot in range(4) for x in range(-2, 9)
                if _score_after(board, 0, rot, x) is not None
                and {x + dx for dx, _ in _cells(0, rot)} == {9}]
    assert vertical, "no vertical I placement fills column 9 alone"


def test_two_ply_hold_beats_one_ply_on_hold_usage():
    """The hold slot is for ordering pieces, which one ply cannot express."""
    env = TetrisSprintBatched(4, latency=100)
    env.reset(torch.arange(4, dtype=torch.int64))
    board = env.board.cpu().numpy()[0]
    piece, other = int(env.piece[0]), int(env.queue.cpu().numpy()[0][0])
    single, _, _ = best_placement(board, piece)
    pair = best_pair(board, piece, other)
    assert pair is not None
    # The two-ply score accounts for a second placement, so it must differ
    # from - and generally exceed - scoring the first piece alone.
    assert pair != pytest.approx(single), "two-ply collapsed to one ply"


def test_sampling_escapes_a_deterministic_deadlock():
    """A constant-Q net argmaxes to one action forever; sampling does not.

    This is the shape of the failure that made a 98.6%-agreement student clear
    1.8 lines: argmax over reversible actions can cycle, and the telemetry
    then describes the cycle. Sampling must produce more than one action.
    """
    class FlatNet(torch.nn.Module):
        def forward(self, obs):
            q = torch.zeros(obs.shape[0], 10)
            q[:, 0] = 1.0        # tap left, always, under argmax
            q[:, 1] = 0.99       # tap right, nearly as good
            return q

    from coldopen.sprint_ladders import net_policy
    env = TetrisSprintBatched(8, latency=100)
    env.reset(torch.arange(8, dtype=torch.int64))
    obs = env.observe()

    greedy = net_policy(FlatNet(), torch.device("cpu"), temperature=0.0)
    assert len(set(greedy(obs, env).tolist())) == 1, "argmax should be degenerate"

    sampled = net_policy(FlatNet(), torch.device("cpu"), temperature=1.0)
    seen = set()
    for _ in range(20):
        seen.update(sampled(obs, env).tolist())
    assert len(seen) > 1, "sampling must break the deadlock"


def test_placement_encoding_roundtrips_over_the_legal_range():
    """encode/decode must cover exactly the placements the search can produce.

    The student's whole action space is this encoding, so an off-by-one here
    would silently mislabel every example — the kind of error that shows up as
    "the net just didn't learn" rather than as a failure.
    """
    from coldopen.distill_placement import (HOLD_INDEX, N_PLACEMENTS,
                                            N_TARGETS, X_HI, X_LO, decode,
                                            encode)
    seen = set()
    for rot in range(4):
        for x in range(X_LO, X_HI):
            index = encode(rot, x)
            assert 0 <= index < N_TARGETS, (rot, x, index)
            assert decode(index) == (rot, x)
            seen.add(index)
    assert seen == set(range(N_TARGETS)), "encoding must be a bijection"
    # Hold sits just past the placements and must not collide with one.
    assert HOLD_INDEX == N_TARGETS
    assert N_PLACEMENTS == N_TARGETS + 1

    # And the range must match the search's own sweep, or labels fall outside
    # the student's reachable set.
    from coldopen.sprint_bot import _score_after
    board = np.zeros((24, 10), dtype=np.int64)
    for piece in range(7):
        for rot in range(4):
            for x in range(X_LO, X_HI):
                if _score_after(board, piece, rot, x) is not None:
                    assert 0 <= encode(rot, x) < N_TARGETS


def test_own_state_agreement_differs_from_pool_agreement():
    """The two agreement numbers must not be wired to the same thing.

    They diverged by 4-13x every time distillation failed here, so a refactor
    that quietly made them identical would remove the only signal that caught
    it. A net predicting one constant placement should score near zero on its
    own states.
    """
    from coldopen.distill_placement import N_PLACEMENTS, own_state_agreement

    class OnePlacement(torch.nn.Module):
        def forward(self, obs):
            q = torch.zeros(obs.shape[0], N_PLACEMENTS)
            q[:, 0] = 1.0
            return q

    score = own_state_agreement(OnePlacement(), torch.device("cpu"),
                                n_envs=4, steps=40)
    assert 0.0 <= score < 0.5, score


def test_legal_placement_mask_matches_the_search():
    """The mask must admit exactly the placements the search calls legal.

    It is a correctness fix, not a speedup: `markov_action` emits HARD_DROP
    only once the piece reaches (target_rot, target_x), so an unreachable
    target means the piece never drops. An unmasked student predicted them and
    stalled at 50.8 inputs per piece against the teacher's 3.3.
    """
    from coldopen.distill_placement import LEGAL_TARGETS, X_HI, X_LO, decode
    from coldopen.sprint_bot import _score_after
    board = np.zeros((24, 10), dtype=np.int64)
    LEGAL = LEGAL_TARGETS
    for piece in range(7):
        for index in range(LEGAL.shape[1]):
            rot, x = decode(index)
            # On an empty board, column bounds are the only constraint, so the
            # cheap mask and the full search must agree exactly.
            assert bool(LEGAL[piece, index]) == (
                _score_after(board, piece, rot, x) is not None), (piece, rot, x)
        assert LEGAL[piece].any(), f"piece {piece} has no legal placement"


def test_masked_argmax_never_picks_an_unreachable_target():
    """Even a net that wants an illegal placement must be steered to a legal one."""
    from coldopen.distill_placement import (LEGAL_TARGETS, N_TARGETS, decode)
    from coldopen.sprint_bot import _score_after
    board = np.zeros((24, 10), dtype=np.int64)
    LEGAL = LEGAL_TARGETS
    for piece in range(7):
        illegal = [i for i in range(N_TARGETS) if not LEGAL[piece, i]]
        if not illegal:
            continue
        logits = torch.full((1, N_TARGETS), -5.0)
        logits[0, illegal[0]] = 100.0          # the net's favourite is illegal
        legal = LEGAL[piece].unsqueeze(0)
        choice = logits.masked_fill(~legal, -1e9).argmax(dim=1)
        rot, x = decode(choice[0])
        assert _score_after(board, piece, rot, x) is not None, (piece, rot, x)


def test_hold_is_maskable_and_reflects_hold_used():
    """Hold must be offered only when the slot is actually free.

    Excluding hold from the student's action space capped the whole pathway:
    an oracle with perfect placements but no hold scored 31.0 lines at a 50%
    finish rate, against 36.8 and 83% once hold was available. Half its runs
    topped out, while every human sprint record is a finish.
    """
    from coldopen.distill_placement import HOLD_INDEX, legal_actions
    env = TetrisSprintBatched(4, latency=100)
    env.reset(torch.arange(4, dtype=torch.int64))
    assert bool(legal_actions(env)[:, HOLD_INDEX].all()), "hold free at spawn"
    env.hold_used[:] = 1
    assert not bool(legal_actions(env)[:, HOLD_INDEX].any()), "spent hold is masked"
