"""Vectorized Connect Four.

Board layout matches PettingZoo's ``connect_four_v3``: a [6, 7] grid where row 0
is the top and pieces settle toward row 5. Cell values are 0 empty, 1 for
player 0, 2 for player 1. ``tests/test_c4_differential.py`` plays identical
action sequences through this and through ``connect_four_v3`` and asserts the
observations, action masks, terminations and rewards agree.

Everything runs as torch tensors over a batch of B independent games so the
policy network can score every live game in one forward pass.
"""

import torch

ROWS, COLS, CONNECT = 6, 7, 4


def _win_kernels(device):
    """Four conv kernels whose response is 4 exactly on a completed line."""
    eye = torch.eye(CONNECT, device=device)
    return [
        torch.ones(1, 1, 1, CONNECT, device=device),  # horizontal
        torch.ones(1, 1, CONNECT, 1, device=device),  # vertical
        eye.view(1, 1, CONNECT, CONNECT),  # "\" diagonal
        eye.flip(1).view(1, 1, CONNECT, CONNECT),  # "/" diagonal
    ]


class BatchedC4:
    """B independent Connect Four games advanced in lockstep.

    Each ``step`` advances every game by one ply, played by whichever seat is to
    move in that game. Finished games hold still until ``reset_done`` clears
    them.
    """

    def __init__(self, n, device="cpu"):
        self.n = n
        self.device = torch.device(device)
        self._kernels = _win_kernels(self.device)
        self.board = torch.zeros(n, ROWS, COLS, dtype=torch.int8, device=self.device)
        self.heights = torch.zeros(n, COLS, dtype=torch.long, device=self.device)
        self.to_move = torch.zeros(n, dtype=torch.long, device=self.device)
        self.done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.winner = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.plies = torch.zeros(n, dtype=torch.long, device=self.device)

    # -- state -------------------------------------------------------------

    def clone_state(self):
        return (
            self.board.clone(),
            self.heights.clone(),
            self.to_move.clone(),
            self.done.clone(),
            self.winner.clone(),
            self.plies.clone(),
        )

    def load_state(self, state):
        (
            self.board,
            self.heights,
            self.to_move,
            self.done,
            self.winner,
            self.plies,
        ) = [t.clone() for t in state]

    def reset_done(self):
        """Restart every finished game. Returns the mask of games restarted."""
        m = self.done.clone()
        if m.any():
            self.board[m] = 0
            self.heights[m] = 0
            self.to_move[m] = 0
            self.winner[m] = -1
            self.plies[m] = 0
            self.done[m] = False
        return m

    def reset_all(self):
        self.done[:] = True
        self.reset_done()

    # -- observation -------------------------------------------------------

    def legal_mask(self):
        """[B, 7] bool: columns that still have room. Finished games are all-False."""
        return (self.heights < ROWS) & ~self.done.unsqueeze(1)

    def observe(self):
        """[B, 2, 6, 7] float. Plane 0 is the side to move, plane 1 the opponent."""
        mover_piece = (self.to_move + 1).view(-1, 1, 1).to(torch.int8)
        opp_piece = (2 - self.to_move).view(-1, 1, 1).to(torch.int8)
        return torch.stack(
            [(self.board == mover_piece).float(), (self.board == opp_piece).float()],
            dim=1,
        )

    def observe_pettingzoo(self):
        """[B, 6, 7, 2] int8, the exact layout ``connect_four_v3`` returns."""
        return self.observe().permute(0, 2, 3, 1).to(torch.int8)

    # -- dynamics ----------------------------------------------------------

    def _has_line(self, piece_plane):
        """[B] bool: does this binary plane contain four in a row?"""
        x = piece_plane.unsqueeze(1)
        hit = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device)
        for k in self._kernels:
            resp = torch.nn.functional.conv2d(x, k)
            hit |= resp.flatten(1).max(dim=1).values > CONNECT - 0.5
        return hit

    def step(self, actions):
        """Drop a piece for the side to move in every unfinished game.

        Returns ``(reward, done)`` where reward is from the *mover's* point of
        view: +1 for a win, 0 for a draw or an unfinished game. A player can
        never lose on their own move in Connect Four, so no -1 arises here.
        """
        live = ~self.done
        actions = actions.to(self.device).long()
        if not torch.all(self.legal_mask()[live].gather(1, actions[live].unsqueeze(1))):
            raise ValueError("illegal move in batch")

        idx = torch.nonzero(live, as_tuple=True)[0]
        cols = actions[idx]
        rows = ROWS - 1 - self.heights[idx, cols]
        piece = (self.to_move[idx] + 1).to(torch.int8)
        self.board[idx, rows, cols] = piece
        self.heights[idx, cols] += 1
        self.plies[idx] += 1

        mover_plane = (self.board[idx] == piece.view(-1, 1, 1)).float()
        won = self._has_line(mover_plane)
        full = self.heights[idx].sum(dim=1) >= ROWS * COLS

        reward = torch.zeros(self.n, device=self.device)
        reward[idx] = won.float()
        self.winner[idx[won]] = self.to_move[idx[won]]
        finished = won | full
        self.done[idx[finished]] = True
        self.to_move[idx] = 1 - self.to_move[idx]
        return reward, self.done.clone()


def winning_moves(env):
    """[B, 7] bool: columns where the side to move completes four immediately.

    Used as an objective blunder probe for telemetry - it depends only on the
    board, never on any network's opinion.
    """
    return _immediate_wins(env, env.to_move)


def blocking_moves(env):
    """[B, 7] bool: columns where the *opponent* would win next turn.

    A move is a required block if the opponent has a winning move there; playing
    elsewhere (without winning outright) hands them the game.
    """
    return _immediate_wins(env, 1 - env.to_move)


def _immediate_wins(env, seat):
    piece = (seat + 1).view(-1, 1, 1).to(torch.int8)
    legal = env.legal_mask()
    out = torch.zeros(env.n, COLS, dtype=torch.bool, device=env.device)
    for c in range(COLS):
        playable = legal[:, c]
        if not playable.any():
            continue
        idx = torch.nonzero(playable, as_tuple=True)[0]
        rows = ROWS - 1 - env.heights[idx, c]
        plane = (env.board[idx] == piece[idx]).float()
        plane[torch.arange(len(idx), device=env.device), rows, c] = 1.0
        out[idx, c] = env._has_line(plane)
    return out
