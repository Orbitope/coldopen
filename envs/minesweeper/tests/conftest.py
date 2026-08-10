import sys
from pathlib import Path

import pytest

ENV_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENV_ROOT.parent))

pytest_plugins = ["simulacrum.harness.plugin"]


@pytest.fixture
def harness_config():
    from simulacrum.harness import DiscreteActionSampler, HarnessConfig

    from minesweeper.fast import MinesweeperBatched
    from minesweeper.reference import MinesweeperReference

    return HarnessConfig(
        name="minesweeper",
        root=ENV_ROOT,
        reference_factory=MinesweeperReference,
        batched_factory=lambda n, debug=False: MinesweeperBatched(n, debug=debug),
        action_sampler=DiscreteActionSampler(n_actions=2),  # TODO
        # scripted_policies=[...],  # strongly recommended (see ScriptedPolicy)
        # min_speedup=100.0,        # enforce the vectorization win
    )
