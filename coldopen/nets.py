"""Q networks for any of the four games.

Two shapes are enough. Connect Four and Othello hand back a board, where the
useful structure is local and translation-equivariant, so a small residual-free
convolutional stack does the job. Backgammon and Leduc hand back a flat vector -
checker counts per point plus dice, or a card and a betting history - where
convolution would be meaningless and a plain MLP is right.

Everything downstream (training, the league, telemetry) only ever calls
``make_net`` and ``masked_q``, so adding a fifth game means adding a shape here
and nothing else.

Q(s, a) is the expected result for the side to move at s, on the scale the game
adapter normalises rewards to, which is [-1, 1] for all four.
"""

import torch
import torch.nn as nn

NEG = -1e9


class BoardNet(nn.Module):
    """Conv stack over a [B, C, H, W] board."""

    def __init__(self, obs_shape, n_actions, channels=64):
        super().__init__()
        c, h, w = obs_shape
        self.body = nn.Sequential(
            nn.Conv2d(c, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * h * w, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.head(self.body(x))


class VectorNet(nn.Module):
    """MLP over a [B, D] observation.

    Wider than the conv head because backgammon's 156 actions and Leduc's
    partial observability both ask more of the last layer than the board games
    do, and because there is no spatial weight sharing to lean on.
    """

    def __init__(self, obs_shape, n_actions, channels=256):
        super().__init__()
        (d,) = obs_shape
        self.net = nn.Sequential(
            nn.Linear(d, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def make_net(info, channels=None):
    """The right network for a game, chosen by its observation shape."""
    if info.spatial:
        return BoardNet(info.obs_shape, info.n_actions, channels or 64)
    return VectorNet(info.obs_shape, info.n_actions, channels or 256)


def masked_q(net, obs, legal):
    """Q values with illegal actions driven to -inf so argmax/max ignore them."""
    return net(obs).masked_fill(~legal, NEG)


def greedy_actions(net, obs, legal):
    with torch.no_grad():
        return masked_q(net, obs, legal).argmax(dim=1)


def epsilon_actions(net, obs, legal, epsilon, generator, device):
    """Greedy under ``net``, or uniform over legal actions if ``net`` is None.

    ``net=None`` is the uniform-random anchor player the league is scaled
    against, and it is deliberately the same code path as exploration so that
    "random" means the same thing everywhere.

    ``generator`` lives on the CPU whatever ``device`` is - Torch has no
    seedable generator for MPS, and every caller here wants a reproducible
    stream more than it wants the noise drawn on the accelerator.
    """
    n, a = obs.shape[0], legal.shape[1]
    if net is None:
        noise = torch.rand(n, a, generator=generator).to(device)
        return noise.masked_fill(~legal, -1).argmax(dim=1)
    with torch.no_grad():
        acts = masked_q(net, obs, legal).argmax(dim=1)
    if epsilon > 0:
        explore = torch.rand(n, generator=generator).to(device) < epsilon
        if explore.any():
            noise = torch.rand(n, a, generator=generator).to(device)
            acts[explore] = noise.masked_fill(~legal, -1).argmax(dim=1)[explore]
    return acts


def save_checkpoint(path, net, info, plies, channels):
    torch.save(
        {
            "game": info.key,
            "plies": plies,
            "channels": channels,
            "spatial": info.spatial,
            "state": net.state_dict(),
        },
        path,
    )


def load_checkpoint(path, info, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    if blob.get("game") not in (None, info.key):
        raise ValueError(f"checkpoint is for {blob['game']}, not {info.key}")
    net = make_net(info, blob.get("channels")).to(device)
    net.load_state_dict(blob["state"])
    net.eval()
    return net
