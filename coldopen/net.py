"""The policy/value network shared by training, the league and telemetry."""

import torch
import torch.nn as nn

from coldopen.c4 import COLS, ROWS

NEG = -1e9


class C4Net(nn.Module):
    """Small conv net mapping a [B, 2, 6, 7] board to seven action values.

    Q(s, a) is the expected game result for the side to move, in [-1, 1].
    """

    def __init__(self, channels=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * ROWS * COLS, 128),
            nn.ReLU(),
            nn.Linear(128, COLS),
        )

    def forward(self, x):
        return self.head(self.body(x))


def masked_q(net, obs, legal):
    """Q values with illegal columns driven to -inf so argmax/max ignore them."""
    q = net(obs)
    return q.masked_fill(~legal, NEG)


def greedy_actions(net, obs, legal):
    with torch.no_grad():
        return masked_q(net, obs, legal).argmax(dim=1)
