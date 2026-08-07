"""Handicapped policies must be weaker in the way they claim to be.

A handicap that does not actually degrade play produces a ladder with no
spread, and a handicap that degrades play by accident - through a bug rather
than the intended mechanism - produces a ladder that ranks fine and means
nothing. Both failures are silent, so each mechanism is checked directly rather
than only through its effect on strength.
"""

import torch

from coldopen import handicap
from coldopen.nets import epsilon_actions


class Fixed(torch.nn.Module):
    """A stand-in network whose preference order is known exactly."""

    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)
        self.seen = []

    def forward(self, obs):
        self.seen.append(obs.clone())
        return self.values.unsqueeze(0).repeat(obs.shape[0], 1)


def board(batch=4, channels=2, rows=6, cols=7):
    return torch.ones(batch, channels, rows, cols)


def legal_all(batch=4, actions=7):
    return torch.ones(batch, actions, dtype=torch.bool)


def test_every_handicap_at_zero_strength_plays_the_best_move():
    """Zero must be a true no-op, or the ladder's top rung is not the agent."""
    net = Fixed([0.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    gen = torch.Generator().manual_seed(0)
    for kind in handicap.KINDS:
        policy = handicap.make(kind, net, 0.0)
        acts = policy.act(board(), legal_all(), 0.0, gen, torch.device("cpu"))
        assert acts.tolist() == [1, 1, 1, 1], f"{kind} is not a no-op at zero"


def test_temperature_prefers_better_moves_but_not_always():
    """Errors stay ordered: the second-best move beats the worst."""
    net = Fixed([0.0, 1.0, 0.9, -5.0, -5.0, -5.0, -5.0])
    gen = torch.Generator().manual_seed(1)
    policy = handicap.make("temperature", net, 0.5)
    acts = policy.act(board(4000), legal_all(4000), 0.0, gen, torch.device("cpu"))
    counts = torch.bincount(acts, minlength=7).float()
    assert counts[1] > counts[2] > counts[3], "ordering by value was not preserved"
    assert counts[2] > 0, "a near-tie should sometimes be chosen"


def test_higher_temperature_flattens_the_choice():
    net = Fixed([0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0])
    gen = torch.Generator().manual_seed(2)
    shares = []
    for strength in (0.2, 5.0):
        policy = handicap.make("temperature", net, strength)
        acts = policy.act(board(2000), legal_all(2000), 0.0, gen, torch.device("cpu"))
        shares.append(float((acts == 1).float().mean()))
    assert shares[0] > shares[1], "hotter sampling should pick the best move less"


def test_epsilon_throws_the_move_away_at_full_strength():
    net = Fixed([0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gen = torch.Generator().manual_seed(3)
    policy = handicap.make("epsilon", net, 1.0)
    acts = policy.act(board(2000), legal_all(2000), 0.0, gen, torch.device("cpu"))
    # Uniform over seven columns, so the best move keeps roughly a seventh.
    assert 0.05 < float((acts == 1).float().mean()) < 0.25


def test_epsilon_respects_illegal_columns_even_when_blundering():
    net = Fixed([0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    legal = legal_all(500)
    legal[:, 3:] = False
    gen = torch.Generator().manual_seed(4)
    policy = handicap.make("epsilon", net, 1.0)
    acts = policy.act(board(500), legal, 0.0, gen, torch.device("cpu"))
    assert int(acts.max()) < 3, "a blunder must still be a legal move"


def test_blindspot_actually_hides_part_of_the_board():
    """The mechanism, not just the effect: the net must see a changed board."""
    net = Fixed([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gen = torch.Generator().manual_seed(5)
    policy = handicap.make("blindspot", net, 0.45)
    policy.act(board(1), legal_all(1), 0.0, gen, torch.device("cpu"))
    seen = net.seen[-1]
    hidden = int((seen == 0).sum())
    assert hidden > 0, "nothing was hidden"
    # 0.45 of seven columns is three, across two planes and six rows.
    assert hidden == 3 * 6 * 2


def test_blindspot_hides_more_as_strength_rises():
    net = Fixed([1.0] + [0.0] * 6)
    gen = torch.Generator().manual_seed(6)
    hidden = []
    for strength in (0.15, 0.9):
        policy = handicap.make("blindspot", net, strength)
        net.seen.clear()
        policy.act(board(1), legal_all(1), 0.0, gen, torch.device("cpu"))
        hidden.append(int((net.seen[-1] == 0).sum()))
    assert hidden[0] < hidden[1]


def test_policies_are_delegated_to_by_the_shared_action_helper():
    """The league and profiler call this, and must not know handicaps exist."""
    net = Fixed([0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    gen = torch.Generator().manual_seed(7)
    policy = handicap.make("temperature", net, 0.0)
    acts = epsilon_actions(policy, board(), legal_all(), 0.0, gen, torch.device("cpu"))
    assert acts.tolist() == [1, 1, 1, 1]


def test_a_ladder_is_ordered_weakest_first():
    net = Fixed([0.0] * 7)
    rungs = handicap.ladder(net, "temperature", [0.0, 2.0, 8.0])
    assert [r["strength"] for r in rungs] == [8.0, 2.0, 0.0]
    assert rungs[0]["id"] == "temperature_8"
