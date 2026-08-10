"""Beginner and intermediate boards must satisfy the same contract as expert.

The validation battery runs the factories' default parameters, which are
Expert (16x30, 99). The E4 ladders now also train on Beginner (9x9, 10) and
Intermediate (16x16, 40) because the sprint lesson says comparable agents must
WIN games and human leaderboards are per-difficulty. Same code, different
parameters — but "the differential test passed" is only true of parameters it
actually ran, so this pins a miniature differential at each extra size.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "envs"))

from minesweeper.fast import MinesweeperBatched
from minesweeper.reference import MinesweeperReference


@pytest.mark.parametrize("H,W,M", [(9, 9, 10), (16, 16, 40)])
@pytest.mark.parametrize("seed", [0, 3])
def test_differential_on_nondefault_boards(H, W, M, seed):
    """reference and fast must agree bit-for-bit at every step, n=1."""
    ref = MinesweeperReference(H=H, W=W, M=M)
    fast = MinesweeperBatched(1, H=H, W=W, M=M, debug=True)
    ref.reset(seed, 0)
    fast.reset(torch.tensor([seed], dtype=torch.int64))

    K = H * W
    generator = torch.Generator().manual_seed(seed * 17 + 1)
    episode = 0
    for step in range(300):
        action = int(torch.randint(0, 3 * K, (1,), generator=generator))
        _, r_ref, term_ref, _ = ref.step(action)
        _, r_fast, term_fast, _ = fast.step(torch.tensor([action]))

        ref_json = ref.to_json(ref.state)
        fast_json = fast.slice_to_json(0)
        if term_ref:
            # fast auto-resets; compare the pre-reset state via final_state
            # capture is exercised by the main battery — here compare scalars
            assert bool(term_fast[0]) == term_ref, (H, W, seed, step)
            assert float(r_fast[0]) == r_ref, (H, W, seed, step)
            episode += 1
            ref.reset(seed, episode)
            continue
        assert bool(term_fast[0]) == term_ref, (H, W, seed, step)
        assert float(r_fast[0]) == r_ref, (H, W, seed, step)
        assert ref_json == fast_json, (
            H, W, seed, step,
            {k: (ref_json[k], fast_json[k]) for k in ref_json
             if ref_json[k] != fast_json[k]})


@pytest.mark.parametrize("H,W,M", [(9, 9, 10), (16, 16, 40)])
def test_invariants_hold_on_nondefault_boards(H, W, M):
    """debug=True raises on any invariant break; drive random play through."""
    env = MinesweeperBatched(8, H=H, W=W, M=M, debug=True)
    env.reset(torch.arange(8, dtype=torch.int64))
    generator = torch.Generator().manual_seed(5)
    for _ in range(120):
        acts = torch.randint(0, 3 * H * W, (8,), generator=generator)
        env.step(acts)


def test_beginner_wins_are_reachable():
    """An oracle that reveals only safe cells must win a 9x9 quickly.

    Guards the reason these boards exist: agents must be able to WIN here so
    their records are comparable to human leaderboard entries.
    """
    env = MinesweeperBatched(8, H=9, W=9, M=10, emit_final_states=False)
    env.reset(torch.arange(8, dtype=torch.int64))
    won = torch.zeros(8, dtype=torch.bool)
    for _ in range(80):
        hidden_safe = ~env.mines & ~env.revealed
        # before generation nothing is a mine; reveal centre first
        target = torch.where(env.generated,
                             hidden_safe.float().argmax(dim=1),
                             torch.full((8,), 40, dtype=torch.int64))
        _, reward, term, _ = env.step(target)
        won = won | (reward > 500_000)
    assert won.all()
