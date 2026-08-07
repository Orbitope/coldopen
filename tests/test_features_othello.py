"""Othello features must mean exactly what their names say.

Built on real positions reached by playing known move sequences, because the
Pgx state is not something to hand-assemble. The classifier downstream is only
as trustworthy as the features under it, and Othello's are judgement calls
encoded as arithmetic - "gave a corner away" has to be the thing that actually
happened, not something correlated with it.
"""

import jax
import pytest
import torch

from coldopen import games
from coldopen.features import othello as feat

SIZE = 8


def idx(row, col):
    return row * SIZE + col


@pytest.fixture(scope="module")
def game():
    return games.make("othello")


def one(game, seed=0):
    return game.init(1, seed=seed)


def advance(game, state, moves, key_seed=0):
    key = jax.random.PRNGKey(key_seed)
    for move in moves:
        legal = game.legal_mask(state)
        assert bool(legal[0, move]), f"move {move} is not legal here"
        key, sub = jax.random.split(key)
        state = game.step(state, torch.tensor([move]), sub)
    return state


def extract(game, state, action, seed=0):
    return feat.FEATURES.extract(
        game, state, torch.tensor([action]), jax.random.PRNGKey(seed))


def test_opening_move_flips_exactly_one_disc(game):
    """Othello's opening position has four discs and every legal reply flips one."""
    state = one(game)
    legal = game.legal_mask(state)
    move = int(torch.nonzero(legal[0, :64])[0])
    f = extract(game, state, move)
    assert f["discs_flipped"].item() == 1.0
    # Two of ours, plus the one placed, plus the one flipped, against their one.
    assert f["disc_share"].item() == pytest.approx(4 / 5)


def test_corner_x_and_c_squares_are_recognised_and_are_mutually_exclusive(game):
    """The three corner-related flags describe different squares."""
    state = one(game)
    for action, expect in [
        (idx(0, 0), "corner_taken"),
        (idx(1, 1), "x_square"),
        (idx(0, 1), "c_square"),
        (idx(1, 0), "c_square"),
    ]:
        f = extract(game, state, action)
        for name in ("corner_taken", "x_square", "c_square"):
            want = 1.0 if name == expect else 0.0
            assert f[name].item() == want, f"{action} -> {name}"


def test_an_x_square_stops_counting_once_its_corner_is_taken(game):
    """The square is only a blunder while the corner it guards is still empty."""
    state = one(game)
    empty = extract(game, state, idx(1, 1))["x_square"].item()
    assert empty == 1.0

    # Put a disc in the corner. The extractor reads the observation planes, so
    # that is what has to change - editing the internal board would leave the
    # cached observation stale and prove nothing.
    filled = state.replace(observation=state.observation.at[0, 0, 0, 0].set(True))
    assert extract(game, filled, idx(1, 1))["x_square"].item() == 0.0


def test_mobility_and_gave_corner_read_the_position_after_the_move(game):
    """Both look at what the opponent can do next, not at what we just did."""
    state = one(game)
    move = int(torch.nonzero(game.legal_mask(state)[0, :64])[0])
    f = extract(game, state, move)
    # Early Othello leaves the opponent a handful of replies and no corner.
    assert 1.0 <= f["opp_mobility"].item() <= 10.0
    assert f["gave_corner"].item() == 0.0

    nxt = advance(game, state, [move])
    assert f["opp_mobility"].item() == float(game.legal_mask(nxt)[0, :64].sum())


def test_frontier_counts_our_discs_touching_an_empty_square(game):
    """In the opening every disc on the board is on the frontier."""
    state = one(game)
    move = int(torch.nonzero(game.legal_mask(state)[0, :64])[0])
    f = extract(game, state, move)
    assert f["frontier"].item() == 4.0, "all four of our discs touch empty space"


def test_a_pass_is_flagged_and_flips_nothing(game):
    """Action 64 is a pass; it must not be scored as a zero-flip move."""
    state = one(game)
    f = extract(game, state, feat.PASS)
    assert f["is_pass"].item() == 1.0
    assert f["discs_flipped"].item() == 0.0
    assert f["corner_taken"].item() == 0.0
    assert f["x_square"].item() == 0.0


def test_extraction_does_not_advance_the_real_game(game):
    """The look-ahead runs on a copy - Pgx states are immutable, so prove it."""
    state = one(game)
    before = game.observe(state).clone()
    move = int(torch.nonzero(game.legal_mask(state)[0, :64])[0])
    extract(game, state, move)
    assert torch.equal(game.observe(state), before)
