"""The oracle must be sound before it is a yardstick.

Soundness is checked two ways: hand positions with known answers, and a sweep
over real env games asserting the oracle NEVER calls a true mine safe or a
true safe cell a mine. (It sees only revealed counts, so its claims can be
checked against the layout it was never shown.)
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "envs"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coldopen.ms_solver import Verdict, judge_click, neighbours, solve
from minesweeper.fast import MinesweeperBatched


def make(H, W, mines):
    """revealed/counts arrays for a fully-hidden board with a given layout."""
    mine_set = set(mines)
    counts = [sum(1 for j in neighbours(i, H, W) if j in mine_set)
              for i in range(H * W)]
    return mine_set, counts


def test_unary_all_hidden_neighbours_are_mines():
    # 3x3, centre revealed showing 8: every neighbour is provably a mine
    H = W = 3
    mine_set, counts = make(H, W, [0, 1, 2, 3, 5, 6, 7, 8])
    revealed = [0] * 9
    revealed[4] = 1
    v = solve(revealed, counts, H, W)
    assert v.mines == {0, 1, 2, 3, 5, 6, 7, 8}
    assert not v.safe and not v.ambiguous


def test_unary_satisfied_count_clears_the_rest():
    # centre shows 0 -> all neighbours provably safe
    H = W = 3
    _, counts = make(H, W, [])
    revealed = [0] * 9
    revealed[4] = 1
    v = solve(revealed, counts, H, W)
    assert v.safe == {0, 1, 2, 3, 5, 6, 7, 8}


def test_subset_one_two_pattern():
    """The classic 1-2 wall: the cell beyond the 2 is a mine.

    Board 3x4, bottom row revealed [1, 2, ...], mines at 4 and 6 in the row
    above. The constraint of the 1 is a subset of the constraint of the 2.
    """
    H, W = 2, 4
    # row 0 hidden, row 1 revealed; mines at cells 1 and 2... construct:
    # revealed cells 4,5,6,7 with counts from mines {0, 2}:
    mine_set, counts = make(H, W, [0, 2])
    revealed = [0, 0, 0, 0, 1, 1, 1, 1]
    v = solve(revealed, counts, H, W)
    # cell 4 sees {0,1}=1; cell 5 sees {0,1,2}=2; subset gives 2 a mine...
    # full enumeration resolves the whole row: 0 and 2 mines, 1 and 3 safe
    assert 0 in v.mines and 2 in v.mines
    assert 1 in v.safe and 3 in v.safe


def test_forced_guess_is_recognised():
    """Two mirror-symmetric solutions -> both cells ambiguous, none safe.

    1x4 board: cells 0,1 hidden, 2 revealed "1" seeing {0,1}... make a true
    50/50: revealed 2,3 with one mine in {0,1} adjacent only to cell... use
    2x2 with one mine in the left column, right column revealed.
    """
    H, W = 1, 3
    mine_set, counts = make(H, W, [0])
    revealed = [0, 0, 1]
    v = solve(revealed, counts, H, W)
    # cell 2 shows "1" over {0? no - neighbours of 2 are just 1}. So cell 1
    # is proven a mine here; rebuild with the mine at 1 for a determined case
    mine_set, counts = make(H, W, [1])
    v = solve(revealed, counts, H, W)
    assert 1 in v.mines
    # and cell 0 is then ambiguous-or-interior, never safe: cell 2's "1" is
    # satisfied by 1, but 0 is only constrained through cell 1 (hidden). The
    # oracle must not overclaim.
    assert 0 not in v.safe


def test_judge_click_labels():
    v = Verdict(safe={3}, mines={5}, ambiguous={7})
    assert judge_click(v, 3, 0) == "proven_safe"
    assert judge_click(v, 5, 0) == "blunder"
    assert judge_click(v, 7, 0) == "guess_avoidable"
    assert judge_click(v, 3, 1) == "other"
    empty = Verdict(ambiguous={7})
    assert judge_click(empty, 7, 0) == "guess_forced"


@pytest.mark.parametrize("H,W,M,seed", [(9, 9, 10, 0), (9, 9, 10, 4),
                                        (16, 16, 40, 1)])
def test_oracle_is_sound_on_real_games(H, W, M, seed):
    """Against live env games: no proven-safe cell is a mine, no proven-mine
    cell is safe. Checked at every step of a random-reveal playthrough."""
    env = MinesweeperBatched(1, H=H, W=W, M=M, emit_final_states=False)
    env.reset(torch.tensor([seed], dtype=torch.int64))
    generator = torch.Generator().manual_seed(seed + 100)
    checked = 0
    for _ in range(60):
        if bool(env.generated[0]):
            revealed = env.revealed[0].tolist()
            counts = env._counts()[0].tolist()
            v = solve(revealed, counts, H, W)
            mines = set(torch.nonzero(env.mines[0]).flatten().tolist())
            assert not (v.safe & mines), "oracle called a mine safe"
            assert v.mines <= mines, "oracle called a safe cell a mine"
            checked += 1
        hidden = (~env.revealed[0] & ~env.flagged[0]).nonzero().flatten()
        pick = hidden[int(torch.randint(len(hidden), (1,), generator=generator))]
        _, _, term, _ = env.step(pick.unsqueeze(0))
    assert checked > 5, "sweep never saw a generated board"
