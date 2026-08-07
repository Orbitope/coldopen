"""The game adapter has to mean the same thing for all four games.

Everything downstream - training, the league, telemetry - is written once
against this surface, so a game that reports its turn order or its rewards
differently from the others would produce a silently wrong ladder rather than
an error.
"""

import jax
import pytest
import torch

from coldopen import games

ALL = list(games.ALL_GAMES)


@pytest.fixture(scope="module")
def made():
    return {key: games.make(key) for key in ALL}


@pytest.mark.parametrize("key", ALL)
def test_shapes_agree_with_what_the_net_will_be_built_from(made, key):
    game = made[key]
    state = game.init(8, seed=0)
    obs = game.observe(state)
    assert obs.shape == (8, *game.info.obs_shape)
    assert game.legal_mask(state).shape == (8, game.info.n_actions)
    assert game.current_player(state).shape == (8,)


@pytest.mark.parametrize("key", ALL)
def test_rewards_are_normalised_into_the_value_head_range(made, key):
    """Backgammon pays up to 3 and Leduc up to 13; the Q head expects [-1, 1].

    Deliberately a lot of hands, and restarting rather than stopping at the
    first termination: the largest payout is also the rarest, and an earlier
    version of this test sampled too few to ever see Leduc's 13-chip pot. The
    scale constant was wrong for months of nothing.
    """
    game = made[key]
    batch = 512
    gen = torch.Generator().manual_seed(11)
    state = game.init(batch, seed=1)
    key_ = jax.random.PRNGKey(0)
    for _ in range(400):
        legal = game.legal_mask(state)
        noise = torch.rand(batch, game.info.n_actions, generator=gen)
        acts = noise.masked_fill(~legal, -1).argmax(dim=1)
        key_, sub = jax.random.split(key_)
        state = game.step(state, acts, sub)
        assert game.rewards(state).abs().max() <= 1.0 + 1e-6
        key_, sub = jax.random.split(key_)
        state = game.restart(state, game.finished(state), sub)


@pytest.mark.parametrize("key", ALL)
def test_restart_replaces_only_the_games_selected(made, key):
    game = made[key]
    state = game.init(8, seed=2)
    key_ = jax.random.PRNGKey(3)
    legal = game.legal_mask(state)
    acts = torch.rand(8, game.info.n_actions).masked_fill(~legal, -1).argmax(dim=1)
    key_, sub = jax.random.split(key_)
    state = game.step(state, acts, sub)

    before = game.observe(state)
    mask = torch.zeros(8, dtype=torch.bool)
    mask[:4] = True
    key_, sub = jax.random.split(key_)
    restarted = game.restart(state, mask, sub)
    after = game.observe(restarted)

    assert torch.equal(after[4:], before[4:]), "untouched games must not move"
    assert (game.step_count(restarted)[:4] == 0).all(), "restarted games start at zero"


@pytest.mark.parametrize("key", ALL)
def test_the_ply_cap_almost_never_decides_a_game(made, key):
    """The cap exists to bound a pathological rollout, not to end normal ones.

    Random backgammon is the only one that ever reaches it - two players who
    both refuse to bear off can shuffle checkers for a long time - and a cut
    game is scored as a draw, so a cap that bit often would quietly turn real
    results into draws.
    """
    game = made[key]
    batch = 128
    gen = torch.Generator().manual_seed(4)
    state = game.init(batch, seed=4)
    key_ = jax.random.PRNGKey(5)
    for _ in range(game.info.max_plies):
        if bool(game.terminated(state).all()):
            break
        legal = game.legal_mask(state)
        acts = (torch.rand(batch, game.info.n_actions, generator=gen)
                .masked_fill(~legal, -1).argmax(dim=1))
        key_, sub = jax.random.split(key_)
        state = game.step(state, acts, sub)
    cut = float((~game.terminated(state)).float().mean())
    assert cut <= 0.02, f"the cap ended {cut:.1%} of games; it is too tight"


def test_ensure_nonempty_only_touches_empty_rows():
    mask = torch.tensor([[False, False, False], [False, True, False]])
    out = games.ensure_nonempty(mask)
    assert out[0].tolist() == [True, False, False]
    assert out[1].tolist() == [False, True, False]


@pytest.mark.parametrize("key", ALL)
def test_the_declared_properties_match_the_game(made, key):
    info = made[key].info
    assert (info.information == "hidden") == (key == "leduc_holdem")
    assert info.chance == (key in ("backgammon", "leduc_holdem"))
