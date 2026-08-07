"""Telemetry features must mean exactly what their names say.

These are hand-built positions with one right answer, because the classifier
downstream is only as trustworthy as the features under it.
"""

import pytest
import torch

from coldopen.c4 import BatchedC4
from coldopen.telemetry import FEATURE_NAMES, aggregate, move_features


def test_missed_win_fires_only_when_a_win_was_available_and_declined():
    env = BatchedC4(2)
    # Both slots: player 0 stacks column 3, player 1 spreads harmlessly.
    for p1 in (0, 1, 5):
        env.step(torch.tensor([3, 3]))
        env.step(torch.tensor([p1, p1]))
    # Slot 0 takes the win on column 3; slot 1 plays elsewhere.
    feats = move_features(env, torch.tensor([3, 6]))
    assert feats["missed_win"].tolist() == [0.0, 1.0]
    assert feats["took_win"].tolist() == [1.0, 0.0]


def test_taking_a_win_is_never_scored_as_a_missed_block():
    """If you can win now, winning beats blocking. That is not a blunder."""
    env = BatchedC4(1)
    # Player 0 builds column 3; player 1 builds column 0. Player 0 moves with
    # both a win available and a threat against them.
    for _ in range(3):
        env.step(torch.tensor([3]))
        env.step(torch.tensor([0]))
    from coldopen.c4 import blocking_moves, winning_moves

    assert bool(winning_moves(env)[0, 3])
    assert bool(blocking_moves(env)[0, 0])

    feats = move_features(env, torch.tensor([3]))
    assert feats["took_win"].item() == 1.0
    assert feats["missed_block"].item() == 0.0
    assert feats["missed_win"].item() == 0.0


def test_missed_block_fires_when_a_threat_is_ignored():
    env = BatchedC4(1)
    # Player 1 stacks column 0; player 0 spreads and then ignores the threat.
    env.step(torch.tensor([6]))
    for c in (1, 2):
        env.step(torch.tensor([0]))  # this is player 1 on odd plies
        env.step(torch.tensor([c]))
    env.step(torch.tensor([0]))
    from coldopen.c4 import blocking_moves, winning_moves

    # Player 0 to move, facing three-in-column-0 with no win of their own.
    assert bool(blocking_moves(env)[0, 0])
    assert not bool(winning_moves(env).any())
    feats = move_features(env, torch.tensor([5]))
    assert feats["missed_block"].item() == 1.0
    assert feats["gave_win"].item() == 1.0, "ignoring it hands them the win"


def test_move_features_leave_the_board_untouched():
    """The look-ahead must not advance the real game."""
    env = BatchedC4(3)
    for a in (1, 4, 2, 5):
        env.step(torch.tensor([a, a, a]))
    before = env.clone_state()
    move_features(env, torch.tensor([3, 0, 6]))
    after = env.clone_state()
    for b, a in zip(before, after):
        assert torch.equal(b, a), "move_features mutated the environment"


def test_made_threat_detects_a_created_win():
    env = BatchedC4(1)
    # Player 0 has two in column 3; a third creates a live threat. Player 1 only
    # has two in column 0, so nothing is handed back.
    env.step(torch.tensor([3]))
    env.step(torch.tensor([0]))
    env.step(torch.tensor([3]))
    env.step(torch.tensor([0]))
    feats = move_features(env, torch.tensor([3]))
    assert feats["made_threat"].item() == 1.0
    assert feats["gave_win"].item() == 0.0, "the opponent is still a move short"

    # One more exchange and the same move does hand over a win.
    env.step(torch.tensor([6]))
    env.step(torch.tensor([0]))  # player 1 now has three in column 0
    feats = move_features(env, torch.tensor([5]))
    assert feats["gave_win"].item() == 1.0


def test_center_distance_is_measured_from_the_middle_column():
    env = BatchedC4(4)
    feats = move_features(env, torch.tensor([3, 2, 0, 6]))
    assert feats["center_distance"].tolist() == [0.0, 1.0, 3.0, 3.0]


def test_aggregate_averages_over_available_moves_and_keeps_the_sample_fixed():
    """Short games must stay in, averaged over what they have.

    Dropping them would make every N a different sample - long games are the
    ones between evenly matched players - so an accuracy-versus-N curve would
    move for two reasons at once.
    """
    features = torch.zeros(3, 10, len(FEATURE_NAMES))
    features[:, :, 0] = 1.0  # constant feature: the mean must be 1 regardless
    features[1, 4:, 0] = 99.0  # moves this game never actually made
    counts = torch.tensor([10, 4, 7])

    x, valid = aggregate(features, counts, n_moves=6)
    assert valid.tolist() == [True, True, True], "no game should be dropped"
    assert torch.allclose(x[:, 0], torch.ones(3)), "padding leaked into the mean"

    # The sample size is the same at every window length.
    for n in (1, 3, 6, 10):
        _, v = aggregate(features, counts, n_moves=n)
        assert v.sum().item() == 3

    # A game shorter than the window contributes its own moves, unchanged.
    features2 = torch.zeros(1, 10, len(FEATURE_NAMES))
    features2[0, :3, 0] = torch.tensor([1.0, 0.0, 2.0])
    x2, _ = aggregate(features2, torch.tensor([3]), n_moves=8)
    assert x2[0, 0].item() == pytest.approx(1.0)


def test_aggregate_excludes_games_with_no_recorded_moves():
    features = torch.zeros(2, 10, len(FEATURE_NAMES))
    _, valid = aggregate(features, torch.tensor([0, 5]), n_moves=4)
    assert valid.tolist() == [False, True]


def test_reference_features_are_zero_without_a_reference_net():
    env = BatchedC4(2)
    feats = move_features(env, torch.tensor([3, 3]), reference=None)
    assert feats["ref_agreement"].sum().item() == 0.0
    assert feats["ref_regret"].sum().item() == 0.0
