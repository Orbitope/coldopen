"""Backgammon features must mean exactly what their names say.

The board here is Pgx's 28-vector from the mover's point of view, so these tests
build boards directly - a position is just a list of signed checker counts, and
that is far clearer than a move sequence would be.

The move replay in ``apply_move`` is the piece most worth pinning down: it has
to agree with Pgx's own update, including the hit rule, or every feature that
looks at the position afterwards is wrong in the same direction.
"""

import jax
import pytest
import torch

from coldopen import games
from coldopen.features import backgammon as feat


@pytest.fixture(scope="module")
def game():
    return games.make("backgammon")


def board(**points):
    """A 28-vector board: ``board(p0=2, p5=-1)`` puts two of ours on 0, a blot
    of theirs on 5. Positive is the mover, negative the opponent."""
    b = torch.zeros(1, 28, dtype=torch.long)
    for name, count in points.items():
        key = name[1:] if name.startswith("p") else name
        index = {"bar": feat.BAR, "oppbar": feat.OPP_BAR,
                 "off": feat.OFF, "oppoff": feat.OPP_OFF}.get(key, None)
        b[0, int(key) if index is None else index] = count
    return b


def action(src_code, die):
    return torch.tensor([src_code * 6 + (die - 1)])


def test_decompose_matches_pgx_encoding():
    """src_code 0 is the no-op, 1 is the bar, and the rest are points."""
    src, die, tgt, noop = feat.decompose(action(0, 3))
    assert bool(noop) and int(die) == 3

    src, die, tgt, noop = feat.decompose(action(1, 4))  # off the bar with a 4
    assert int(src) == feat.BAR and int(tgt) == 3 and not bool(noop)

    src, die, tgt, noop = feat.decompose(action(2 + 5, 6))  # point 5, die 6
    assert int(src) == 5 and int(tgt) == 11

    src, die, tgt, noop = feat.decompose(action(2 + 22, 5))  # runs off the end
    assert int(src) == 22 and int(tgt) == feat.OFF


def test_a_hit_sends_the_opponent_checker_to_the_bar():
    b = board(**{"5": 1, "8": -1})
    src, die, tgt, noop = feat.decompose(action(2 + 5, 3))
    after, hit = feat.apply_move(b, src, tgt, noop)
    assert bool(hit)
    assert int(after[0, 5]) == 0, "the checker left its point"
    assert int(after[0, 8]) == 1, "the point changed hands"
    assert int(after[0, feat.OPP_BAR]) == -1, "their checker is on the bar"


def test_landing_on_two_of_theirs_is_not_a_hit():
    """Two checkers hold a point; nothing is sent anywhere."""
    b = board(**{"5": 1, "8": -2})
    src, die, tgt, noop = feat.decompose(action(2 + 5, 3))
    after, hit = feat.apply_move(b, src, tgt, noop)
    assert not bool(hit)
    assert int(after[0, feat.OPP_BAR]) == 0


def test_a_no_op_leaves_the_board_alone():
    b = board(**{"5": 2})
    src, die, tgt, noop = feat.decompose(action(0, 2))
    after, hit = feat.apply_move(b, src, tgt, noop)
    assert torch.equal(after, b)
    assert not bool(hit)


def test_blots_exposed_counts_only_the_ones_that_can_actually_be_hit(game):
    # Ours alone on 10 with theirs on 14: four pips away, a direct shot.
    exposed = feat._direct_shots(board(**{"10": 1, "14": -2}))
    assert float(exposed) == 1.0

    # Same blot, but their checker is now behind it and moving away.
    safe = feat._direct_shots(board(**{"10": 1, "4": -2}))
    assert float(safe) == 0.0

    # Seven pips away is out of range of a single die.
    far = feat._direct_shots(board(**{"10": 1, "17": -2}))
    assert float(far) == 0.0

    # Two checkers is a point, not a blot, however close they are.
    made = feat._direct_shots(board(**{"10": 2, "14": -2}))
    assert float(made) == 0.0


def test_a_checker_on_their_bar_covers_our_whole_home_board():
    """They re-enter into our home board, so any blot there is exposed."""
    exposed = feat._direct_shots(board(**{"20": 1, "oppbar": -1}))
    assert float(exposed) == 1.0
    # Outside the home board the bar checker is irrelevant.
    outside = feat._direct_shots(board(**{"10": 1, "oppbar": -1}))
    assert float(outside) == 0.0


def test_point_making_and_stacking_are_told_apart(game):
    """Second checker onto a point makes it; a fourth just piles up."""
    key = jax.random.PRNGKey(0)
    state = game.init(1, seed=0)

    makes = state.replace(_board=board(**{"5": 1, "8": 1}).numpy())
    f = feat.FEATURES.extract(game, makes, action(2 + 5, 3), key)
    assert f["makes_point"].item() == 1.0
    assert f["stacks_high"].item() == 0.0

    stacks = state.replace(_board=board(**{"5": 1, "8": 4}).numpy())
    f = feat.FEATURES.extract(game, stacks, action(2 + 5, 3), key)
    assert f["makes_point"].item() == 0.0
    assert f["stacks_high"].item() == 1.0
    assert f["max_stack"].item() == 5.0


def test_coming_in_from_the_bar_is_flagged(game):
    key = jax.random.PRNGKey(0)
    state = game.init(1, seed=0)

    entering = state.replace(_board=board(**{"bar": 1}).numpy())
    f = feat.FEATURES.extract(game, entering, action(1, 2), key)
    assert f["from_bar"].item() == 1.0

    ordinary = state.replace(_board=board(**{"5": 2}).numpy())
    f = feat.FEATURES.extract(game, ordinary, action(2 + 5, 3), key)
    assert f["from_bar"].item() == 0.0


def test_moving_the_back_checker_is_told_from_moving_a_front_one(game):
    """The die decides how far you go; you decide which checker goes.

    That is why this replaced ``pip_gain``, which was the die value and so was
    the dice's choice rather than the player's - noise a classifier can fit.
    """
    key = jax.random.PRNGKey(0)
    state = game.init(1, seed=0)
    position = board(**{"3": 2, "11": 3, "18": 2})

    assert int(feat.rearmost_point(position)) == 3

    back = state.replace(_board=position.numpy())
    f = feat.FEATURES.extract(game, back, action(2 + 3, 4), key)
    assert f["moves_rearmost"].item() == 1.0

    front = state.replace(_board=position.numpy())
    f = feat.FEATURES.extract(game, front, action(2 + 11, 4), key)
    assert f["moves_rearmost"].item() == 0.0

    # A no-op moves nothing, so it moves nothing rearmost either.
    f = feat.FEATURES.extract(game, back, action(0, 4), key)
    assert f["moves_rearmost"].item() == 0.0


def test_rearmost_point_ignores_the_opponents_checkers(game):
    """Negative counts are theirs; the back checker is ours."""
    assert int(feat.rearmost_point(board(**{"2": -3, "9": 1}))) == 9
    # With nothing on the board at all it reports past the last point.
    assert int(feat.rearmost_point(board(**{"off": 15}))) == feat.POINTS


def test_home_points_counts_made_points_in_our_home_board(game):
    key = jax.random.PRNGKey(0)
    state = game.init(1, seed=0)
    # Points 18-23 are the home board; 18 and 20 are made, 19 is a blot.
    held = state.replace(
        _board=board(**{"18": 2, "19": 1, "20": 3, "10": 2}).numpy())
    f = feat.FEATURES.extract(game, held, action(0, 1), key)
    assert f["home_points"].item() == 2.0
    assert f["blots_after"].item() == 1.0
