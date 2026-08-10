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


#: Divisor that brings raw observations into roughly unit range. Pgx hands back
#: boolean planes for the board games, but backgammon's is a checker count per
#: point (up to 15 on a stack), which a small net trains on badly untouched.
OBS_SCALE = {"backgammon": 5.0}

#: Divisor that brings terminal rewards into [-1, 1], which is where a Q head
#: with a negamax target belongs. Backgammon pays 1/2/3 for a plain win, a
#: gammon and a backgammon. Leduc pays the pot, which tops out at 13: the ante,
#: then up to two raises of 2 in the first round and two of 4 in the second.
#: That bound is worth deriving rather than sampling - a random rollout of a few
#: hundred hands never reaches it, and a scale set from one is quietly wrong
#: for exactly the biggest pots.
REWARD_SCALE = {"backgammon": 3.0, "leduc_holdem": 13.0}


def _tensor(value, device, dtype=None):
    """A writable Torch view of a JAX array.

    ``np.asarray`` on a JAX array hands back a read-only buffer, which Torch
    accepts with a warning and then cannot be indexed into. The copy is small -
    these are masks and scalars, not observations.
    """
    array = np.array(value, dtype=dtype, copy=True)
    return torch.from_numpy(array).to(device)


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
    obs_scale: float = 1.0
    reward_scale: float = 1.0


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
        self._restart = jax.jit(_restart_fn(self.env))

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
            obs_scale=OBS_SCALE.get(key, 1.0),
            reward_scale=REWARD_SCALE.get(key, 1.0),
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

    def restart(self, state, mask, key):
        """Replace the games selected by ``mask`` with freshly dealt ones.

        Pgx has no in-place reset, so this deals a whole new batch and selects
        between the two leaf by leaf. Cheap next to a network forward pass, and
        it keeps every slot of the batch busy.
        """
        if not bool(mask.any()):
            return state
        m = jnp.asarray(mask.detach().cpu().numpy())
        return self._restart(state, m, key)

    # -- views -------------------------------------------------------------

    def observe(self, state):
        """[B, *obs_shape] float network input, from the mover's point of view.

        Scaled by ``info.obs_scale``; feature extractors that want the literal
        board read the Pgx state instead (see ``coldopen/features``).
        """
        obs = np.asarray(state.observation, dtype=np.float32)
        if self.info.spatial:
            obs = np.transpose(obs, (0, 3, 1, 2))
        obs = np.ascontiguousarray(obs)
        if self.info.obs_scale != 1.0:
            obs = obs / self.info.obs_scale
        return torch.from_numpy(obs).to(self.device)

    def legal_mask(self, state):
        return _tensor(state.legal_action_mask, self.device).bool()

    def current_player(self, state):
        return _tensor(state.current_player, self.device, np.int64)

    def terminated(self, state):
        return _tensor(state.terminated, self.device).bool()

    def step_count(self, state):
        return _tensor(state._step_count, self.device, np.int64)

    def finished(self, state):
        """Terminated, or run past ``max_plies`` and cut off.

        Only backgammon reaches the cap in practice, and only under weak play -
        two policies that both refuse to bear off can shuffle checkers for a
        very long time. A cut game is scored as a draw, which is the honest
        reading: nobody won it.
        """
        return self.terminated(state) | (self.step_count(state) >= self.info.max_plies)

    def rewards(self, state):
        """[B, 2] indexed by absolute player id, not by seat-to-move.

        Divided by ``info.reward_scale``, so a win is at most 1 in magnitude
        whatever the game pays.
        """
        v = np.asarray(state.rewards, dtype=np.float32)
        v = np.ascontiguousarray(v) / self.info.reward_scale
        return torch.from_numpy(v).to(self.device)

    def reward_for(self, state, player):
        """[B] reward for the given per-batch player ids."""
        return self.rewards(state).gather(1, player.view(-1, 1).clamp(min=0)).squeeze(1)


def _restart_fn(env):
    """Select leaf-by-leaf between a freshly dealt batch and the live one."""
    init = jax.vmap(env.init)

    def restart(state, mask, key):
        batch = mask.shape[0]
        fresh = init(jax.random.split(key, batch))

        def pick(new, old):
            shape = (batch,) + (1,) * (new.ndim - 1)
            return jnp.where(mask.reshape(shape), new, old)

        return jax.tree_util.tree_map(pick, fresh, state)

    return restart


def ensure_nonempty(mask):
    """Force one legal bit on rows that have none.

    A terminated Pgx state can present an all-False action mask; a masked max
    over it would be -inf and poison the bootstrap. The rows this touches are
    exactly the ones whose target is the terminal reward, so the bit is never
    read - it just keeps the arithmetic finite.
    """
    empty = ~mask.any(dim=1)
    if empty.any():
        mask = mask.clone()
        mask[empty, 0] = True
    return mask


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
