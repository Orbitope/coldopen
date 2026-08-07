"""Othello telemetry: positional judgement, because there are no tactics to find.

Connect Four is decided by whether you saw the four-in-a-row. Othello almost
never contains a forced win until the very end, which is the whole reason it is
in this project: if skill is only legible through immediate tactics, the
accuracy curve here should collapse.

So the features are the things Othello players actually argue about, and each
one is a specific, well-known difference between a beginner and a club player:

* **Discs flipped.** The single most reliable beginner tell. New players take
  the move that flips the most discs; strong players are frequently *behind* on
  disc count through the whole midgame and do it on purpose.
* **Corners, X-squares and C-squares.** Corners can never be flipped, so they
  are worth taking and worth not giving away. The X-square (diagonally inside an
  empty corner) is the classic way to hand one over, and the two C-squares
  beside an empty corner are the second-classic way.
* **Mobility.** Strong play restricts what the opponent can do; the count of
  replies left to them after the move is the direct measure.
* **Frontier.** Discs sitting next to an empty square are the ones that give the
  opponent something to flip. Grabbing discs early inflates the frontier, which
  is the mechanism behind the disc-count trap rather than a restatement of it.

Board layout is Pgx's: 8x8, index ``row * 8 + col``, action 64 is a pass.
"""

import numpy as np
import torch

SIZE = 8
PASS = SIZE * SIZE

CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
#: Diagonally inside each corner - playing here usually gives the corner away.
X_SQUARES = {(0, 0): (1, 1), (0, 7): (1, 6), (7, 0): (6, 1), (7, 7): (6, 6)}
#: The two edge squares flanking each corner.
C_SQUARES = {
    (0, 0): [(0, 1), (1, 0)],
    (0, 7): [(0, 6), (1, 7)],
    (7, 0): [(6, 0), (7, 1)],
    (7, 7): [(7, 6), (6, 7)],
}


def _index(rc):
    return rc[0] * SIZE + rc[1]


def _mask_from(cells, device):
    m = torch.zeros(PASS, dtype=torch.bool, device=device)
    for rc in cells:
        m[_index(rc)] = True
    return m


def _planes(state, device):
    """(mover, opponent) [B, 8, 8] float planes from the Pgx observation."""
    obs = np.asarray(state.observation, dtype=np.float32)
    mover = torch.from_numpy(np.ascontiguousarray(obs[..., 0])).to(device)
    opp = torch.from_numpy(np.ascontiguousarray(obs[..., 1])).to(device)
    return mover, opp


def _frontier(own, occupied):
    """[B] count of own discs orthogonally or diagonally touching an empty square."""
    empty = (1.0 - occupied).unsqueeze(1)
    kernel = torch.ones(1, 1, 3, 3, device=own.device)
    kernel[0, 0, 1, 1] = 0.0
    neighbours = torch.nn.functional.conv2d(empty, kernel, padding=1).squeeze(1)
    return ((neighbours > 0).float() * own).flatten(1).sum(dim=1)


class OthelloFeatures:
    names = (
        "discs_flipped",
        "disc_share",
        "corner_taken",
        "x_square",
        "c_square",
        "gave_corner",
        "opp_mobility",
        "frontier",
        "is_pass",
    )

    def extract(self, game, state, actions, key):
        device = game.device
        batch = actions.shape[0]
        mover, opp = _planes(state, device)

        corner_mask = _mask_from(CORNERS, device)
        empty_corner = torch.stack(
            [(mover[:, r, c] + opp[:, r, c]) == 0 for r, c in CORNERS], dim=1
        )  # [B, 4]

        is_pass = actions == PASS
        safe = actions.clamp(max=PASS - 1)
        on_corner = corner_mask[safe] & ~is_pass

        # An X- or C-square only matters while the corner it guards is empty;
        # once the corner is taken the square is ordinary.
        x_hit = torch.zeros(batch, dtype=torch.bool, device=device)
        c_hit = torch.zeros(batch, dtype=torch.bool, device=device)
        for i, corner in enumerate(CORNERS):
            x_hit |= (safe == _index(X_SQUARES[corner])) & empty_corner[:, i]
            for cell in C_SQUARES[corner]:
                c_hit |= (safe == _index(cell)) & empty_corner[:, i]

        nxt = game.step(state, actions, key)
        # After a legal move the opponent normally has the move, but Othello
        # makes a player pass when they have nothing, in which case it comes
        # straight back. Read who is actually to move rather than assuming.
        opp_to_move = game.current_player(nxt) != game.current_player(state)
        n_first, n_second = _planes(nxt, device)
        own_after = torch.where(
            opp_to_move.view(-1, 1, 1), n_second, n_first)
        opp_after = torch.where(
            opp_to_move.view(-1, 1, 1), n_first, n_second)

        own_before_count = mover.flatten(1).sum(dim=1)
        own_after_count = own_after.flatten(1).sum(dim=1)
        opp_after_count = opp_after.flatten(1).sum(dim=1)
        # One of the new discs is the one just placed; the rest were flipped.
        flipped = (own_after_count - own_before_count - 1.0).clamp(min=0.0)
        total = (own_after_count + opp_after_count).clamp(min=1.0)

        legal_after = game.legal_mask(nxt)
        finished = game.finished(nxt)
        # Passing is always "legal" when nothing else is; it is not mobility.
        mobility = (legal_after[:, :PASS].sum(dim=1).float()
                    * (opp_to_move & ~finished).float())
        gave_corner = (
            (legal_after[:, :PASS] & corner_mask.unsqueeze(0)).any(dim=1)
            & opp_to_move & ~finished
        )

        return {
            "discs_flipped": torch.where(is_pass, torch.zeros_like(flipped), flipped),
            "disc_share": own_after_count / total,
            "corner_taken": on_corner.float(),
            "x_square": x_hit.float(),
            "c_square": c_hit.float(),
            "gave_corner": gave_corner.float(),
            "opp_mobility": mobility,
            "frontier": _frontier(own_after, own_after + opp_after),
            "is_pass": is_pass.float(),
        }


FEATURES = OthelloFeatures()
