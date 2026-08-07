"""The Pgx port of the Connect Four features must agree with the original.

``coldopen/telemetry.py`` computes these six features against the bespoke
``BatchedC4``, which is itself differential-tested against PettingZoo. The
generic pipeline recomputes them against Pgx so that all four games are profiled
by one code path - which is only worth anything if the numbers did not change on
the way. So rather than re-deriving the expected values by hand, this plays the
same moves through both and asserts they agree.

Both implementations lay the board out identically (row 0 at the top, pieces
settling toward row 5) and both observe from the mover's point of view, so the
same action sequence produces the same position in each.
"""

import jax
import pytest
import torch

from coldopen import games
from coldopen.c4 import COLS, BatchedC4
from coldopen.features import connect_four as feat
from coldopen.telemetry import move_features

BATCH = 24


def _both(seed):
    game = games.make("connect_four")
    return game, game.init(BATCH, seed=seed), BatchedC4(BATCH)


def _legal_actions(mask, generator):
    noise = torch.rand(mask.shape, generator=generator)
    return noise.masked_fill(~mask, -1).argmax(dim=1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_two_implementations_report_the_same_features(seed):
    game, state, env = _both(seed)
    gen = torch.Generator().manual_seed(seed)
    key = jax.random.PRNGKey(seed)

    for _ in range(30):
        legal_pgx = game.legal_mask(state)
        legal_own = env.legal_mask()
        live = ~env.done
        if not bool(live.any()):
            break
        assert torch.equal(legal_pgx[live], legal_own[live]), "positions diverged"

        actions = _legal_actions(legal_own, gen)
        # Finished games are held still on both sides; only compare live ones.
        key, sub = jax.random.split(key)
        ported = feat.FEATURES.extract(game, state, actions, sub)
        original = move_features(env, actions)

        for name in feat.FEATURES.names:
            a = ported[name][live]
            b = original[name][live]
            assert torch.equal(a, b), f"{name} disagrees"

        key, sub = jax.random.split(key)
        state = game.step(state, actions, sub)
        env.step(actions)
        # Pgx keeps terminated games frozen; mirror that rather than resetting.
        if bool(env.done.all()):
            break


def test_extraction_does_not_advance_the_real_game():
    """The look-ahead for gave_win/made_threat runs on a copy."""
    game = games.make("connect_four")
    state = game.init(4, seed=7)
    before = game.observe(state).clone()
    feat.FEATURES.extract(
        game, state, torch.tensor([3, 0, 6, 2]), jax.random.PRNGKey(0))
    assert torch.equal(game.observe(state), before)


def test_a_won_position_is_not_credited_with_further_threats():
    """After the winning move the game is over; nothing was 'given away'."""
    game = games.make("connect_four")
    state = game.init(1, seed=0)
    key = jax.random.PRNGKey(0)
    # Player to move stacks column 3, the other spreads harmlessly.
    for reply in (0, 1, 5):
        for col in (3, reply):
            key, sub = jax.random.split(key)
            state = game.step(state, torch.tensor([col]), sub)

    f = feat.FEATURES.extract(game, state, torch.tensor([3]), key)
    assert f["took_win"].item() == 1.0
    assert f["gave_win"].item() == 0.0
    assert f["made_threat"].item() == 0.0


def test_center_distance_is_measured_from_the_middle_column():
    game = games.make("connect_four")
    state = game.init(4, seed=0)
    f = feat.FEATURES.extract(
        game, state, torch.tensor([3, 2, 0, 6]), jax.random.PRNGKey(0))
    assert f["center_distance"].tolist() == [0.0, 1.0, 3.0, 3.0]
    assert COLS == 7
