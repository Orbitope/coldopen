"""Connect Four telemetry: exact tactics, settled by the rules.

The same six features the bespoke pipeline in ``coldopen/telemetry.py`` uses,
reimplemented against Pgx so that all four games are profiled by one code path
and their results are comparable. Keeping both is deliberate: the bespoke
implementation is differential-tested against PettingZoo, and running the two
side by side is a check that the generic pipeline did not quietly change what
the numbers mean.

Nothing here consults a network. A winning move is a winning move whatever any
agent thinks of it, so no information about the agents being profiled can leak
in through these.
"""

import numpy as np
import torch

ROWS, COLS, CONNECT = 6, 7, 4

#: Pgx lays the board out as [6, 7] with row 0 at the top and pieces settling
#: toward row 5, the same convention as ``coldopen/c4.py``.
TOP_ROW = 0


def _kernels(device):
    """Four convolutions whose response reaches 4 exactly on a completed line."""
    eye = torch.eye(CONNECT, device=device)
    return [
        torch.ones(1, 1, 1, CONNECT, device=device),
        torch.ones(1, 1, CONNECT, 1, device=device),
        eye.view(1, 1, CONNECT, CONNECT),
        eye.flip(1).view(1, 1, CONNECT, CONNECT),
    ]


def _has_line(plane, kernels):
    """[B] bool: does this binary plane contain four in a row?"""
    x = plane.unsqueeze(1)
    hit = torch.zeros(x.shape[0], dtype=torch.bool, device=plane.device)
    for k in kernels:
        resp = torch.nn.functional.conv2d(x, k)
        hit |= resp.flatten(1).max(dim=1).values > CONNECT - 0.5
    return hit


def _planes(state, device):
    """(mover, opponent) [B, 6, 7] float planes from the Pgx observation."""
    obs = np.asarray(state.observation, dtype=np.float32)
    mover = torch.from_numpy(np.ascontiguousarray(obs[..., 0])).to(device)
    opp = torch.from_numpy(np.ascontiguousarray(obs[..., 1])).to(device)
    return mover, opp


def _immediate_wins(own, occupied, kernels):
    """[B, 7] bool: columns where dropping one piece completes four for ``own``."""
    device = own.device
    batch = own.shape[0]
    heights = occupied.sum(dim=1)  # [B, 7] pieces already in each column
    playable = heights < ROWS
    out = torch.zeros(batch, COLS, dtype=torch.bool, device=device)
    for c in range(COLS):
        live = playable[:, c]
        if not bool(live.any()):
            continue
        idx = torch.nonzero(live, as_tuple=True)[0]
        landing = ROWS - 1 - heights[idx, c].long()
        trial = own[idx].clone()
        trial[torch.arange(len(idx), device=device), landing, c] = 1.0
        out[idx, c] = _has_line(trial, kernels)
    return out


class ConnectFourFeatures:
    names = (
        "missed_win",
        "missed_block",
        "gave_win",
        "made_threat",
        "took_win",
        "center_distance",
    )

    def extract(self, game, state, actions, key):
        device = game.device
        kernels = _kernels(device)
        mover, opp = _planes(state, device)
        occupied = mover + opp

        wins = _immediate_wins(mover, occupied, kernels)
        blocks = _immediate_wins(opp, occupied, kernels)
        chose = torch.nn.functional.one_hot(actions, COLS).bool()

        had_win = wins.any(dim=1)
        took_win = (wins & chose).any(dim=1)
        threatened = blocks.any(dim=1)
        blocked = (blocks & chose).any(dim=1)

        feats = {
            "missed_win": (had_win & ~took_win).float(),
            # Failing to block only counts when you had no winning move of your
            # own: taking the win instead of blocking is correct play.
            "missed_block": (threatened & ~blocked & ~took_win & ~had_win).float(),
            "took_win": (had_win & took_win).float(),
            "center_distance": (actions - COLS // 2).abs().float(),
        }

        # The last two need the position after the move. Pgx states are
        # immutable, so this cannot disturb the game in progress.
        nxt = game.step(state, actions, key)
        alive = ~game.terminated(nxt)
        n_mover, n_opp = _planes(nxt, device)
        n_occupied = n_mover + n_opp
        # The opponent is to move now: their immediate wins are ones this move
        # handed over, and the threats on the other plane are ones it created.
        feats["gave_win"] = (
            _immediate_wins(n_mover, n_occupied, kernels).any(dim=1) & alive).float()
        feats["made_threat"] = (
            _immediate_wins(n_opp, n_occupied, kernels).any(dim=1) & alive).float()
        return feats


FEATURES = ConnectFourFeatures()
