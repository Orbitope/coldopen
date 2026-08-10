"""The n-step target is where a two-player value sign can quietly go wrong.

These games pay only at the end, so an n-step return is the terminal result
seen through the product of the negamax signs between here and there. Get the
product backwards and the network learns to lose; get the episode boundary
wrong and it learns from the next episode's result. Neither shows up as an
error, only as a ladder that does not rank.
"""

import torch

from coldopen.selfplay import NStep


def step(sign, reward=0.0, done=False, tag=0.0, batch=1):
    """One ply for a batch, tagged so the emitted head can be identified."""
    return dict(
        obs=torch.full((batch, 3), tag),
        action=torch.zeros(batch, dtype=torch.long),
        reward=torch.full((batch,), reward),
        sign=torch.full((batch,), sign),
        done=torch.zeros(batch, dtype=torch.bool) | done,
        next_obs=torch.full((batch, 3), tag + 0.5),
        next_legal=torch.ones(batch, 2, dtype=torch.bool),
    )


def test_nothing_is_emitted_until_the_window_fills():
    acc = NStep(3)
    assert acc.push(**step(-1.0, tag=0)) is None
    assert acc.push(**step(-1.0, tag=1)) is None
    assert acc.push(**step(-1.0, tag=2)) is not None


def test_alternating_play_negates_once_per_ply():
    """Three plies of strict alternation: the bootstrap flips three times."""
    acc = NStep(3)
    acc.push(**step(-1.0, tag=0))
    acc.push(**step(-1.0, tag=1))
    out = acc.push(**step(-1.0, tag=2))
    assert out["obs"][0, 0].item() == 0.0, "the head of the window is emitted"
    assert not bool(out["done"][0])
    assert out["sign"][0].item() == -1.0, "(-1)^3"
    assert out["next_obs"][0, 0].item() == 2.5, "bootstrap from the end of the window"


def test_a_player_moving_twice_does_not_flip_the_sign():
    """Backgammon spends several actions on one turn; only the turn flips."""
    acc = NStep(3)
    acc.push(**step(1.0, tag=0))   # same player still to move
    acc.push(**step(-1.0, tag=1))  # turn passes
    out = acc.push(**step(1.0, tag=2))
    assert out["sign"][0].item() == -1.0, "1 * -1 * 1"


def test_a_win_inside_the_window_is_carried_back_through_the_signs():
    """A loss two plies ahead is a win for the player moving now."""
    acc = NStep(4)
    acc.push(**step(-1.0, tag=0))
    acc.push(**step(-1.0, tag=1))
    acc.push(**step(-1.0, reward=1.0, done=True, tag=2))
    out = acc.push(**step(-1.0, tag=3))
    assert bool(out["done"][0]), "the window terminated, so no bootstrap"
    # Two sign flips between the head and the win: (-1) * (-1) * (+1).
    assert out["reward"][0].item() == 1.0


def test_a_win_by_the_opponent_comes_back_as_a_loss():
    acc = NStep(4)
    acc.push(**step(-1.0, tag=0))
    acc.push(**step(-1.0, reward=1.0, done=True, tag=1))
    acc.push(**step(-1.0, tag=2))
    out = acc.push(**step(-1.0, tag=3))
    assert out["reward"][0].item() == -1.0, "one flip: their win is our loss"


def test_the_walk_stops_at_the_first_termination():
    """Everything past the end of the episode belongs to the next one."""
    acc = NStep(4)
    acc.push(**step(-1.0, tag=0))
    acc.push(**step(-1.0, reward=1.0, done=True, tag=1))
    # A fresh episode starts here and ends differently; it must not be read.
    acc.push(**step(-1.0, reward=-1.0, done=True, tag=2))
    out = acc.push(**step(-1.0, tag=3))
    assert out["reward"][0].item() == -1.0, "the second result leaked in"


def test_slots_terminate_independently():
    """Two games in the batch end at different plies and must not cross."""
    acc = NStep(3)
    a = step(-1.0, tag=0, batch=2)
    acc.push(**a)
    b = step(-1.0, tag=1, batch=2)
    b["done"] = torch.tensor([True, False])
    b["reward"] = torch.tensor([1.0, 0.0])
    acc.push(**b)
    out = acc.push(**step(-1.0, tag=2, batch=2))
    assert out["done"].tolist() == [True, False]
    assert out["reward"][0].item() == -1.0, "slot 0 lost one ply later"
    assert out["reward"][1].item() == 0.0, "slot 1 is still bootstrapping"
    assert out["sign"][1].item() == -1.0
