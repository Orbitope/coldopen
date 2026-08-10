import sys
from pathlib import Path

import pytest

ENV_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENV_ROOT.parent))

pytest_plugins = ["simulacrum.harness.plugin"]

H, W, M = 16, 30, 99
K = H * W
LATENCY = 200
REVEAL, FLAG, CHORD = 0, 1, 2


def _flag_everything(state, t):
    """Flag cells 0,1,2,... forever. Never reveals, so never generates a board.

    Return is exactly -(30 + LATENCY) per step for as long as it runs: a pure
    check that no-op-free flagging costs what the spec says and that an
    ungenerated board is never scored as a win.
    """
    return FLAG * K + (t % K)


def _reveal_row_zero(state, t):
    """Reveal along the top row until something ends the episode.

    Exercises the flood and the death path: the first reveal opens a region,
    and with 99 mines in 480 cells a later one lands on a mine with high
    probability.
    """
    return REVEAL * K + (t % K)


@pytest.fixture
def harness_config():
    from simulacrum.harness import (DiscreteActionSampler, HarnessConfig,
                                    ScriptedPolicy)

    from minesweeper.fast import MinesweeperBatched
    from minesweeper.reference import MinesweeperReference

    return HarnessConfig(
        name="minesweeper",
        root=ENV_ROOT,
        reference_factory=MinesweeperReference,
        batched_factory=lambda n, debug=False: MinesweeperBatched(n, debug=debug),
        # spec: Actions — 3 kinds x H*W cells
        action_sampler=DiscreteActionSampler(n_actions=3 * K),
        scripted_policies=[
            # 20 flags, no reveal: 20 steps x -(30+200), and no win despite the
            # board vacuously having no unrevealed safe cells.
            ScriptedPolicy(
                name="flag_only_never_wins",
                policy=_flag_everything,
                expected_return=-20 * (30 + LATENCY),
                tol=1e-6,
                episodes=8,
                max_steps=20,
            ),
        ],
        # Training shape: compiled step core, no per-step terminal JSON.
        # The battery's parity test bit-checks this against the eager env.
        benchmark_factory=lambda n, debug=False: MinesweeperBatched(
            n, debug=debug, compile=not debug),
    )
