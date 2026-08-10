import sys
from pathlib import Path

import pytest

ENV_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENV_ROOT.parent))

pytest_plugins = ["simulacrum.harness.plugin"]


@pytest.fixture
def harness_config():
    from simulacrum.harness import DiscreteActionSampler, HarnessConfig

    from tetris_sprint.fast import TetrisSprintBatched
    from tetris_sprint.reference import TetrisSprintReference

    return HarnessConfig(
        name="tetris_sprint",
        root=ENV_ROOT,
        reference_factory=TetrisSprintReference,
        batched_factory=lambda n, debug=False: TetrisSprintBatched(n, debug=debug),
        action_sampler=DiscreteActionSampler(n_actions=10),
        # Training shape: compiled step core, no per-step terminal JSON.
        # The battery's parity test bit-checks this against the eager env.
        benchmark_factory=lambda n, debug=False: TetrisSprintBatched(
            n, debug=debug, compile=not debug),
    )
