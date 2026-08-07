"""Backgammon telemetry: risk management under dice.

Backgammon is the chance axis of this project. Its skill is not calculation of a
forced line - the dice will not let you have one - it is what you leave standing
when the dice go against you. The features are the standard checker-play
concepts:

* **Blots.** A single checker on a point can be hit and sent back to the bar.
  Counting them after the move measures how much risk was accepted, and
  counting only the ones an opponent checker can reach with one die measures
  how much of that risk was *real*: a blot nobody can hit is free.
* **Points made.** Two or more checkers hold a point against everything.
  Landing a second checker on a point you already had one on makes it; the
  count in your home board is the specific thing that traps checkers on the bar.
* **Hitting and entering.** Hitting sends an opponent checker back; coming in
  from the bar is forced but tells you the state you were in.
* **Over-stacking.** Piling five checkers on one point is the classic beginner
  shape - safe, and wasteful, because the checkers do nothing there.

The board is Pgx's 28-vector, always written from the point of view of the
player to move: entries 0-23 are the points in that player's direction of
travel, 24 their bar, 25 the opponent's, 26 their borne-off checkers and 27 the
opponent's. Positive counts are theirs, negative the opponent's, which means
"a blot" is exactly ``board == 1`` and no perspective bookkeeping is needed.

The post-move board is reconstructed here rather than read back from Pgx.
Pgx flips the board every time the turn changes, and a backgammon turn is two to
four actions, so reading it back would mean tracking whether this particular
action ended the turn. Replaying the move is both simpler and easier to check
against the rules.
"""

import numpy as np
import torch

POINTS = 24
BAR = 24
OPP_BAR = 25
OFF = 26
OPP_OFF = 27
HOME = list(range(18, 24))  # the mover's home board, always
MAX_DIE = 6


def _board(state, device):
    """[B, 28] int64 board from the mover's point of view."""
    b = np.asarray(state._board, dtype=np.int64)
    return torch.from_numpy(np.ascontiguousarray(b)).to(device)


def decompose(actions):
    """(src, die, tgt, is_noop) for Pgx backgammon actions.

    ``action = src_code * 6 + (die - 1)``, where src_code 0 is the no-op played
    when the dice are unusable, 1 is the bar, and anything else is point
    ``src_code - 2``.
    """
    src_code = torch.div(actions, MAX_DIE, rounding_mode="floor")
    die = actions % MAX_DIE + 1
    is_noop = src_code == 0
    src = torch.where(src_code == 1, torch.full_like(src_code, BAR), src_code - 2)
    from_bar = src_code == 1
    landing = src + die
    tgt = torch.where(
        from_bar,
        die - 1,
        torch.where(landing <= POINTS - 1, landing, torch.full_like(landing, OFF)),
    )
    return src, die, tgt, is_noop


def apply_move(board, src, tgt, is_noop):
    """The board after this checker move, replaying Pgx's own update.

    A target holding exactly one opponent checker is a hit: that checker goes to
    the opponent's bar and the point changes hands.
    """
    out = board.clone()
    rows = torch.arange(board.shape[0], device=board.device)
    hit = (board[rows, tgt] == -1) & ~is_noop & (tgt < POINTS)
    move = ~is_noop
    out[rows, OPP_BAR] -= hit.long()
    out[rows[move], src[move]] -= 1
    out[rows[move], tgt[move]] += 1 + hit[move].long()
    return out, hit


def _direct_shots(board):
    """[B] own blots an opponent checker could hit with a single die.

    The opponent travels from high index to low, so a checker on point j covers
    blots on j-1 .. j-6. A checker on their bar enters on points 18-23, so any
    blot in the mover's home board is exposed while they have one there.
    """
    device = board.device
    points = board[:, :POINTS]
    blots = points == 1
    opp = points < 0
    exposed = torch.zeros_like(blots)
    for d in range(1, MAX_DIE + 1):
        # An opponent checker at index i + d bears on a blot at index i.
        shooter = torch.zeros_like(opp)
        shooter[:, : POINTS - d] = opp[:, d:]
        exposed |= blots & shooter
    on_bar = (board[:, OPP_BAR] < 0).view(-1, 1)
    home = torch.zeros(POINTS, dtype=torch.bool, device=device)
    home[HOME] = True
    exposed |= blots & home.unsqueeze(0) & on_bar
    return exposed.sum(dim=1).float()


class BackgammonFeatures:
    names = (
        "hit",
        "bears_off",
        "from_bar",
        "makes_point",
        "stacks_high",
        "blots_after",
        "blots_exposed",
        "home_points",
        "max_stack",
        "pip_gain",
    )

    def extract(self, game, state, actions, key):
        device = game.device
        board = _board(state, device)
        src, die, tgt, is_noop = decompose(actions)
        after, hit = apply_move(board, src, tgt, is_noop)

        rows = torch.arange(board.shape[0], device=device)
        before_tgt = board[rows, tgt.clamp(max=OPP_OFF)]
        on_board = tgt < POINTS
        points = after[:, :POINTS]

        home = torch.zeros(POINTS, dtype=torch.bool, device=device)
        home[HOME] = True

        return {
            "hit": hit.float(),
            "bears_off": ((tgt == OFF) & ~is_noop).float(),
            "from_bar": ((src == BAR) & ~is_noop).float(),
            # Landing the second checker on a point you already held one of.
            "makes_point": ((before_tgt == 1) & on_board & ~is_noop).float(),
            # Landing on a point that already had three or more: safe and idle.
            "stacks_high": ((before_tgt >= 3) & on_board & ~is_noop).float(),
            "blots_after": (points == 1).sum(dim=1).float(),
            "blots_exposed": _direct_shots(after),
            "home_points": (points.ge(2) & home.unsqueeze(0)).sum(dim=1).float(),
            "max_stack": points.clamp(min=0).max(dim=1).values.float(),
            "pip_gain": torch.where(is_noop, torch.zeros_like(die), die).float(),
        }


FEATURES = BackgammonFeatures()
