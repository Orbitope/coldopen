"""The curriculum must start agents in states the validated env could reach.

Its mine placement is a re-derivation of fast.py's inline rule (the validated
env is deliberately not modified), so the first test pins the two
byte-for-byte — the copy is allowed to exist only because this test makes
silent drift impossible.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "envs"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coldopen.ms_curriculum import (CurriculumMinesweeper, Pacer,
                                    SyntheticStates, lay_mines)
from minesweeper.fast import MinesweeperBatched


def test_placement_parity_with_the_validated_env():
    """lay_mines must equal fast.py's placement for the same keys and click.

    Drive the real env to its first reveal and rebuild the layout from the
    same mine_keys with the curriculum's copy; any divergence in exclusion,
    sort stability, or tie-breaking shows up here as a mismatch.
    """
    env = MinesweeperBatched(8)
    env.reset(torch.arange(8, dtype=torch.int64))
    K = env.K
    cells = torch.randint(0, K, (8,), generator=torch.Generator().manual_seed(3))
    env.step(0 * K + cells)          # kind 0 = reveal, generates the board
    ours = lay_mines(env.mine_keys, cells, env.NEIGH, env.NVALID, env.M)
    assert torch.equal(ours, env.mines), "curriculum layout diverged from fast.py"


def test_synthetic_states_satisfy_the_env_invariants_except_i9():
    """Injected starts must be states the validated rules accept — bar I9.

    I9 defines a fresh `t == 0` episode as ungenerated; a curriculum start is
    deliberately mid-game, so I9 is the one invariant the mechanism breaks on
    purpose (hence debug=False in training). Every other invariant must hold
    on the injected batch — I3 (exactly M mines), I4 (opening guarantee) and
    I5 (no revealed mine) are the ones an injection bug would break.
    """
    source = SyntheticStates(k_max=5, seed=1)
    env = CurriculumMinesweeper(16, source=source)
    env.reset(torch.arange(16, dtype=torch.int64))
    hidden = (~env.mines & ~env.revealed).sum(dim=1)
    assert (hidden >= 1).all() and (hidden <= 5).all(), hidden
    for name in ("_i1", "_i2", "_i3", "_i4", "_i5", "_i6", "_i7", "_i8",
                 "_i10", "_i11", "_i12", "_i13"):
        held = getattr(env, name)()
        assert bool(torch.as_tensor(held).all()), f"{name} violated"
    assert not bool(env._i9().all()), "I9 should NOT hold — that is the point"


def test_curriculum_episodes_are_winnable_quickly():
    """An oracle revealing the hidden safe cells must win within k steps.

    This is the property the whole curriculum rests on: starts near the goal
    make the win bonus reachable. If the oracle cannot win from an injected
    start, the injection is wrong, not the learner.
    """
    source = SyntheticStates(k_max=3, seed=2)
    env = CurriculumMinesweeper(8, source=source, emit_final_states=False)
    env.reset(torch.arange(8, dtype=torch.int64))
    won = torch.zeros(8, dtype=torch.bool)
    for _ in range(4):
        hidden = ~env.mines & ~env.revealed
        target = hidden.float().argmax(dim=1)      # any hidden safe cell
        _, _, term, _ = env.step(0 * env.K + target)
        # `term` here means won: the oracle only reveals safe cells, so death
        # is impossible and the horizon is far. env.won cannot be read after
        # the step - auto-reset has already cleared it for winners.
        won = won | term
    assert won.all(), "oracle failed to finish a near-finished board"


def test_pacer_walks_backwards_and_only_backwards():
    source = SyntheticStates(k_max=3)
    pacer = Pacer(source, threshold=0.5, window=8, grow=2.0)
    assert not pacer.update([True] * 4)            # window not yet full
    assert pacer.update([True] * 4)                # 8/8 wins -> widen
    assert source.k_max == 6
    pacer.update([False] * 8)                      # losing: never narrows
    assert source.k_max == 6


def test_replay_states_round_trip_through_the_reference():
    """A hand-written replay must reproduce as visited states.

    5x5 board, 4 mines in the corners-ish, two reveals. Checks the contract
    end to end: validation, player split, state capture, source output.
    """
    from coldopen.human.minesweeper_replays import (ReplayStates, load,
                                                    split_players,
                                                    states_from_replay)
    import json, tempfile
    H = W = 5
    mines = [0] * 25
    for cell in (0, 4, 20, 24):
        mines[cell] = 1
    replay = {"player": "a" * 16, "rank": 1, "H": H, "W": W, "M": 4,
              "mines": mines,
              # flag first, then a reveal that floods the safe region. The
              # flagged safe cell BLOCKS the flood (spec: flags are never
              # flood-revealed), so the game does not win - both captured
              # states are live, which is what a curriculum source wants.
              "clicks": [{"t_ms": 100, "kind": 1, "cell": 3},
                         {"t_ms": 350, "kind": 0, "cell": 12}]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"replays": [replay, dict(replay, player="b" * 16)]}, fh)
        path = fh.name
    replays = load(path)
    train, evaluation = split_players(replays, eval_fraction=0.5)
    assert {r["player"] for r in train}.isdisjoint(
        {r["player"] for r in evaluation})

    states = states_from_replay(replays[0])
    assert len(states) == 2
    assert bool(states[0]["flagged"][3]) and not states[0]["terminated"]
    # the flag held back the flood: revealed everywhere safe except cell 3
    assert bool(states[1]["revealed"][12]) and not states[1]["terminated"]
    assert not bool(states[1]["revealed"][3])
    assert int(states[0]["mines"].sum()) == 4

    source = ReplayStates(train, tail=1.0)
    env = CurriculumMinesweeper(4, source=source, H=H, W=W, M=4)
    env.reset(torch.arange(4, dtype=torch.int64))
    assert (env.mines.sum(dim=1) == 4).all()
    assert env.generated.all()
