"""One interface over four games, so everything downstream is game-agnostic.

The point of this project is not any single game - it is the question of *what
makes skill legible early*, which only has an answer if you can vary the game and
hold the method fixed. So training, leagues, telemetry and the classifier all
talk to this adapter and never to a specific environment.

The four games are chosen to vary three properties independently:

    game           information   chance   tactical density
    connect_four   perfect       none     high
    othello        perfect       none     low
    backgammon     perfect       dice     medium
    leduc_holdem   hidden        cards    n/a

"Tactical density" is not asserted here, it is measured - see
``tactics.tactical_density``. It is the fraction of positions in which the player
to move has a move that wins immediately, and it is the thing this project
expects to predict how far behavioural features can go.

Everything runs on Pgx, which is JAX. The networks are PyTorch. Observations
cross that boundary once per step via numpy, which is a view rather than a copy
on CPU.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pgx
import torch

#: Games are capped so a pathological rollout can't run forever. Backgammon
#: under random play regularly exceeds several hundred plies; trained agents
#: bear off far sooner, but the cap keeps batch rollouts bounded.
MAX_PLIES = {
    "connect_four": 42,
    "othello": 70,
    "backgammon": 512,
    "leduc_holdem": 16,
}


@dataclass
class GameInfo:
    key: str
    label: str
    information: str  # "perfect" or "hidden"
    chance: bool
    expected_tactics: str  # what we expect to find, to be checked against measurement
    max_plies: int
    obs_shape: tuple
    n_actions: int
    spatial: bool  # whether the observation is a board (conv) or a vector (MLP)


class Game:
    """A batched two-player game with a uniform, framework-free surface.

    All arrays crossing this boundary are torch tensors on ``device``; the JAX
    state stays inside. Every method takes and returns whole batches.
    """

    def __init__(self, key, device="cpu"):
        if key not in MAX_PLIES:
            raise KeyError(f"unknown game {key!r}; have {sorted(MAX_PLIES)}")
        self.key = key
        self.device = torch.device(device)
        self.env = pgx.make(key)
        self._init = jax.jit(jax.vmap(self.env.init))
        self._step = jax.jit(jax.vmap(self.env.step))

        probe = self._init(jax.random.split(jax.random.PRNGKey(0), 2))
        obs_shape = tuple(probe.observation.shape[1:])
        spatial = len(obs_shape) == 3
        self.info = GameInfo(
            key=key,
            label=key.replace("_", " "),
            information="hidden" if key in ("leduc_holdem", "kuhn_poker") else "perfect",
            chance=key in ("backgammon", "leduc_holdem", "kuhn_poker"),
            expected_tactics={"connect_four": "high", "othello": "low",
                              "backgammon": "medium", "leduc_holdem": "none"}.get(key, "?"),
            max_plies=MAX_PLIES[key],
            # Torch wants channels first; Pgx gives channels last.
            obs_shape=(obs_shape[2], obs_shape[0], obs_shape[1]) if spatial else obs_shape,
            n_actions=int(probe.legal_action_mask.shape[1]),
            spatial=spatial,
        )

    # -- lifecycle ---------------------------------------------------------

    def init(self, batch, seed=0):
        return self._init(jax.random.split(jax.random.PRNGKey(seed), batch))

    def step(self, state, actions, key):
        """Advance every unfinished game by one ply.

        Pgx ignores actions on terminated states, so finished games simply hold
        still and no masking is needed here.
        """
        a = jnp.asarray(actions.detach().cpu().numpy())
        batch = a.shape[0]
        return self._step(state, a, jax.random.split(key, batch))

    # -- views -------------------------------------------------------------

    def observe(self, state):
        """[B, *obs_shape] float, from the point of view of the side to move."""
        obs = np.asarray(state.observation, dtype=np.float32)
        if self.info.spatial:
            obs = np.transpose(obs, (0, 3, 1, 2))
        return torch.from_numpy(np.ascontiguousarray(obs)).to(self.device)

    def legal_mask(self, state):
        m = np.asarray(state.legal_action_mask)
        return torch.from_numpy(np.ascontiguousarray(m)).to(self.device).bool()

    def current_player(self, state):
        v = np.asarray(state.current_player, dtype=np.int64)
        return torch.from_numpy(v).to(self.device)

    def terminated(self, state):
        v = np.asarray(state.terminated)
        return torch.from_numpy(np.ascontiguousarray(v)).to(self.device).bool()

    def rewards(self, state):
        """[B, 2] indexed by absolute player id, not by seat-to-move."""
        v = np.asarray(state.rewards, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(v)).to(self.device)

    def reward_for(self, state, player):
        """[B] reward for the given per-batch player ids."""
        return self.rewards(state).gather(1, player.view(-1, 1).clamp(min=0)).squeeze(1)


def make(key, device="cpu"):
    return Game(key, device)


ALL_GAMES = ("connect_four", "othello", "backgammon", "leduc_holdem")


def describe():
    """Table of the four games, for the article and for sanity."""
    rows = []
    for key in ALL_GAMES:
        g = Game(key)
        rows.append({
            "key": g.info.key,
            "information": g.info.information,
            "chance": g.info.chance,
            "expected_tactics": g.info.expected_tactics,
            "obs_shape": list(g.info.obs_shape),
            "n_actions": g.info.n_actions,
            "spatial": g.info.spatial,
            "max_plies": g.info.max_plies,
        })
    return rows
