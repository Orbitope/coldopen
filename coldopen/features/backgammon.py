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
* **Running the back checker, and how far you got.** ``moves_rearmost`` is the
  running game against building at home, and it is the clearer *choice* of the
  two. ``pip_gain`` - the die actually played - looks at first like the dice's
  choice rather than the player's, and was cut on that reasoning. Removing it
  cost 6.5 points of accuracy at one move and 5 at two, so the reasoning was
  wrong: which die you are still *able* to play is decided by the position you
  left yourself, and early on that is most of what separates the tiers. Both
  are kept. The reasoning was worth writing down because it was wrong in the
  direction this whole project is about - assuming you know which behaviour
  carries skill instead of measuring it.
* **Stacking.** Piling checkers on one point is safe and idle. This was written
  down as a beginner habit and the measurement disagrees: the top tier stacks
  *more* than the bottom (``stacks_high`` 0.077 -> 0.143, ``max_stack``
  4.72 -> 5.03). Kept because it separates the tiers, but it is separating them
  in the opposite direction to the received wisdom, and the received wisdom is
  not what this file gets to assert.

The board is Pgx's 28-vector, always written from the point of view of the
player to move: entries 0-23 are the points in that player's direction of
travel, 24 their bar, 25 the opponent's, 26 their borne-off checkers and 27 the
opponent's. Positive counts are theirs, negative the opponent's, which means
"a blot" is exactly ``board == 1`` and no perspective bookkeeping is needed.

One thing is deliberately absent. ``bears_off`` fired on 0.000 of profiled moves
in every tier: a backgammon game runs a couple of hundred plies and the
telemetry window is the first twenty moves, so nobody in it is bearing off yet.
It is a correct feature of the game and a dead column of this experiment.

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


def rearmost_point(board):
    """[B] index of the player's furthest-back occupied point, or 24 if none.

    "Furthest back" is the lowest index, since the mover always travels toward
    23 in this frame.
    """
    points = board[:, :POINTS]
    index = torch.arange(POINTS, device=board.device).unsqueeze(0)
    occupied = points >= 1
    return torch.where(occupied, index, torch.full_like(index, POINTS)).min(dim=1).values


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
        "from_bar",
        "makes_point",
        "stacks_high",
        "moves_rearmost",
        "pip_gain",
        "blots_after",
        "blots_exposed",
        "home_points",
        "max_stack",
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
            "from_bar": ((src == BAR) & ~is_noop).float(),
            # Landing the second checker on a point you already held one of.
            "makes_point": ((before_tgt == 1) & on_board & ~is_noop).float(),
            # Landing on a point that already had three or more: safe and idle.
            "stacks_high": ((before_tgt >= 3) & on_board & ~is_noop).float(),
            # Running the back checker rather than building at home.
            "moves_rearmost": (
                (src == rearmost_point(board)) & ~is_noop & (src < POINTS)).float(),
            # The die actually played. Not as free a choice as the one above,
            # but being unable to use the big die is a consequence of the
            # position you left, and it is the strongest single early signal.
            "pip_gain": torch.where(is_noop, torch.zeros_like(die), die).float(),
            "blots_after": (points == 1).sum(dim=1).float(),
            "blots_exposed": _direct_shots(after),
            "home_points": (points.ge(2) & home.unsqueeze(0)).sum(dim=1).float(),
            "max_stack": points.clamp(min=0).max(dim=1).values.float(),
        }


FEATURES = BackgammonFeatures()
