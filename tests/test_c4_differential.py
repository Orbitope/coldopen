"""Differential test: BatchedC4 against PettingZoo's connect_four_v3.

Same seeds, same action sequences, one game per batch slot. If these two ever
disagree on a mask, an observation, a termination or a reward, the fast env is
wrong and every number downstream of it is suspect.
"""

import numpy as np
import pytest
import torch
from pettingzoo.classic import connect_four_v3

from coldopen.c4 import BatchedC4, blocking_moves, winning_moves


def _play_reference(seed):
    """Run one PettingZoo game with a seeded random policy, recording everything."""
    rng = np.random.default_rng(seed)
    env = connect_four_v3.env()
    env.reset(seed=seed)
    trace, rewards = [], {}
    for agent in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        if term or trunc:
            # env.rewards is cleared once every agent has been retired, so take
            # each agent's final cumulative reward on its terminal step.
            rewards[agent] = reward
            env.step(None)
            continue
        legal = np.nonzero(obs["action_mask"])[0]
        action = int(rng.choice(legal))
        trace.append(
            {
                "agent": agent,
                "observation": obs["observation"].copy(),
                "action_mask": obs["action_mask"].copy(),
                "action": action,
            }
        )
        env.step(action)
    env.close()
    return trace, rewards


@pytest.mark.parametrize("seed", range(24))
def test_matches_pettingzoo(seed):
    trace, ref_rewards = _play_reference(seed)

    fast = BatchedC4(1)
    fast_reward_for_seat = {0: 0.0, 1: 0.0}
    for ply, rec in enumerate(trace):
        seat = 0 if rec["agent"] == "player_0" else 1
        assert int(fast.to_move.item()) == seat, f"seat drift at ply {ply}"
        assert not bool(fast.done.item()), f"fast env ended early at ply {ply}"

        got_mask = fast.legal_mask()[0].to(torch.int8).numpy()
        assert np.array_equal(got_mask, rec["action_mask"]), f"mask at ply {ply}"

        got_obs = fast.observe_pettingzoo()[0].numpy()
        assert np.array_equal(got_obs, rec["observation"]), f"observation at ply {ply}"

        reward, _ = fast.step(torch.tensor([rec["action"]]))
        fast_reward_for_seat[seat] += float(reward.item())

    assert bool(fast.done.item()), "fast env did not terminate when the reference did"

    # PettingZoo pays the loser -1; ours only reports the mover's +1, so derive
    # the two-sided outcome and compare that.
    winner = int(fast.winner.item())
    derived = {0: 0.0, 1: 0.0}
    if winner >= 0:
        derived[winner] = 1.0
        derived[1 - winner] = -1.0
    assert derived[0] == ref_rewards["player_0"]
    assert derived[1] == ref_rewards["player_1"]
    assert fast_reward_for_seat[winner] == 1.0 if winner >= 0 else True


def test_batch_slots_are_independent():
    """Two games in one batch must not leak into each other."""
    rng = np.random.default_rng(7)
    solo_states = []
    for seed in (11, 12):
        env = BatchedC4(1)
        r = np.random.default_rng(seed)
        for _ in range(10):
            legal = torch.nonzero(env.legal_mask()[0], as_tuple=True)[0].tolist()
            env.step(torch.tensor([int(r.choice(legal))]))
        solo_states.append(env.board.clone())

    both = BatchedC4(2)
    rngs = [np.random.default_rng(11), np.random.default_rng(12)]
    for _ in range(10):
        acts = []
        for b in range(2):
            legal = torch.nonzero(both.legal_mask()[b], as_tuple=True)[0].tolist()
            acts.append(int(rngs[b].choice(legal)))
        both.step(torch.tensor(acts))

    assert torch.equal(both.board[0:1], solo_states[0])
    assert torch.equal(both.board[1:2], solo_states[1])
    del rng


def test_reset_done_only_clears_finished_games():
    env = BatchedC4(2)
    # Slot 0: player 0 stacks column 0 and wins vertically on the fourth drop.
    # Slot 1: both players spread out, so nothing resolves.
    for p0_col, p1_col, s1_a, s1_b in [(0, 1, 2, 3), (0, 2, 4, 5), (0, 3, 6, 2)]:
        env.step(torch.tensor([p0_col, s1_a]))
        env.step(torch.tensor([p1_col, s1_b]))
    env.step(torch.tensor([0, 4]))  # player 0 completes the column-0 four
    assert bool(env.done[0].item())
    assert not bool(env.done[1].item())
    before = env.board[1].clone()
    plies_before = int(env.plies[1].item())

    env.reset_done()
    assert env.board[0].abs().sum().item() == 0
    assert int(env.plies[0].item()) == 0
    assert int(env.winner[0].item()) == -1
    assert torch.equal(env.board[1], before), "live game was disturbed by the reset"
    assert int(env.plies[1].item()) == plies_before
    assert int(env.to_move[0].item()) == 0


def test_immediate_win_and_block_probes():
    env = BatchedC4(1)
    # Player 0 stacks column 3. Player 1 spreads across the bottom row, which
    # gives them no vertical threat of their own.
    for p1_col in (0, 1, 5):
        env.step(torch.tensor([3]))  # player 0
        env.step(torch.tensor([p1_col]))  # player 1

    # Player 0 to move, three in column 3: exactly one immediate win, no block
    # required because player 1 threatens nothing yet.
    assert bool(winning_moves(env)[0, 3].item())
    assert int(winning_moves(env)[0].sum().item()) == 1
    assert int(blocking_moves(env)[0].sum().item()) == 0

    env.step(torch.tensor([6]))  # player 0 declines the win

    # Player 1 to move now faces that same column-3 threat and must block it.
    assert bool(blocking_moves(env)[0, 3].item())
    assert int(blocking_moves(env)[0].sum().item()) == 1
    assert int(winning_moves(env)[0].sum().item()) == 0, "player 1 has no win here"
