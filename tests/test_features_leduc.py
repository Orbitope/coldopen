"""Leduc features must mean exactly what their names say.

Almost every feature here is a conjunction - "raised while holding a pair" - and
conjunctions are where an off-by-one in whose card is whose hides comfortably.
The states are built by overwriting the deal on a real initial state, which is
the only part of Leduc worth controlling.

Cards are ranks 0, 1, 2 with two of each; ``_cards`` is
``[player 0, player 1, public]``; actions are 0 call, 1 raise, 2 fold.
"""

import jax
import numpy as np
import pytest
import torch

from coldopen import games
from coldopen.features import leduc_holdem as feat

CALL, RAISE, FOLD = feat.CALL, feat.RAISE, feat.FOLD


@pytest.fixture(scope="module")
def game():
    return games.make("leduc_holdem")


def situation(game, own, public, rnd=1, mover=0, last=-1, chips=(1, 1)):
    """A one-slot state with the deal and betting history set by hand."""
    state = game.init(1, seed=0)
    cards = np.array([[0, 0, 0]], dtype=np.int32)
    cards[0, mover] = own
    cards[0, 1 - mover] = 1  # the opponent's card is never read by a feature
    cards[0, 2] = public
    return state.replace(
        _cards=cards,
        _round=np.array([rnd], dtype=np.int32),
        _last_action=np.array([last], dtype=np.int32),
        _chips=np.array([list(chips)], dtype=np.int32),
        current_player=np.array([mover], dtype=np.int32),
    )


def extract(game, state, action):
    return feat.FEATURES.extract(
        game, state, torch.tensor([action]), jax.random.PRNGKey(0))


def test_the_three_actions_are_one_hot(game):
    state = situation(game, own=2, public=0)
    for action, name in [(RAISE, "raised"), (CALL, "called"), (FOLD, "folded")]:
        f = extract(game, state, action)
        for other in ("raised", "called", "folded"):
            assert f[other].item() == (1.0 if other == name else 0.0)


def test_a_pair_only_counts_once_the_public_card_is_out(game):
    """The public card is dealt at the start but not in play until round 1."""
    assert extract(game, situation(game, 2, 2, rnd=1), CALL)["has_pair"].item() == 1.0
    assert extract(game, situation(game, 2, 2, rnd=0), CALL)["has_pair"].item() == 0.0
    assert extract(game, situation(game, 2, 1, rnd=1), CALL)["has_pair"].item() == 0.0


def test_the_pair_belongs_to_whoever_is_actually_to_move(game):
    """Reading the wrong seat's card would still look plausible in aggregate."""
    as_first = situation(game, own=2, public=2, mover=0)
    as_second = situation(game, own=2, public=2, mover=1)
    assert extract(game, as_first, CALL)["has_pair"].item() == 1.0
    assert extract(game, as_second, CALL)["has_pair"].item() == 1.0

    # Player 1 holds the pair, but player 0 is to move and does not.
    state = game.init(1, seed=0).replace(
        _cards=np.array([[0, 2, 2]], dtype=np.int32),
        _round=np.array([1], dtype=np.int32),
        current_player=np.array([0], dtype=np.int32),
    )
    assert extract(game, state, CALL)["has_pair"].item() == 0.0


def test_value_slow_play_and_blunder_are_told_apart(game):
    """Same hand, three actions, three different verdicts."""
    paired = situation(game, own=2, public=2)
    assert extract(game, paired, RAISE)["raise_with_pair"].item() == 1.0
    assert extract(game, paired, CALL)["passive_with_pair"].item() == 1.0
    assert extract(game, paired, FOLD)["fold_with_pair"].item() == 1.0
    # Each is exclusive of the others.
    assert extract(game, paired, RAISE)["fold_with_pair"].item() == 0.0
    assert extract(game, paired, CALL)["raise_with_pair"].item() == 0.0


def test_a_bluff_is_the_worst_card_raised_without_a_pair(game):
    bluff = situation(game, own=0, public=2)
    assert extract(game, bluff, RAISE)["bluff_raise"].item() == 1.0
    # The same card paired is a value raise, not a bluff.
    value = situation(game, own=0, public=0)
    assert extract(game, value, RAISE)["bluff_raise"].item() == 0.0
    assert extract(game, value, RAISE)["raise_with_pair"].item() == 1.0
    # A strong card raised is neither.
    strong = situation(game, own=2, public=1)
    assert extract(game, strong, RAISE)["bluff_raise"].item() == 0.0


def test_folding_to_a_raise_is_distinguished_from_folding_unprompted(game):
    faced = situation(game, own=0, public=2, last=RAISE)
    unfaced = situation(game, own=0, public=2, last=CALL)
    assert extract(game, faced, FOLD)["fold_to_raise"].item() == 1.0
    assert extract(game, unfaced, FOLD)["fold_to_raise"].item() == 0.0
    assert extract(game, faced, CALL)["fold_to_raise"].item() == 0.0


def test_chips_committed_reads_the_movers_own_stack(game):
    state = situation(game, own=1, public=0, mover=1, chips=(3, 7))
    assert extract(game, state, CALL)["chips_committed"].item() == 7.0
