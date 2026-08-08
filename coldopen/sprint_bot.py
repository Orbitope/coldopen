"""A scripted sprint bot: placement search plus keystroke emission.

The DQN could not bootstrap line-clearing from keystroke-level exploration —
three million steps produced fast stacking and no clears, the same wall the
Tetris RL literature documents. This bot exists to break that wall the way
AutoAscend does for NetHack: a symbolic teacher built from the rules and
hand-chosen heuristics, with **no human data anywhere**, so a ladder distilled
from it stays cold-start legitimate.

It plays the obvious classical way. When a new piece appears, enumerate every
(rotation, column) placement, drop each, score the resulting board with
Dellacherie-style features — holes, aggregate height, bumpiness, lines
cleared — and commit to the best. Then emit the keystroke sequence for that
placement: rotations, then DAS to a wall when the target hugs one (that is
what DAS is for, and it is why the bot's finesse is decent), else taps, then
hard drop. It replans only when a new piece appears, so its actions per piece
are exactly the plan length: finesse is a property of the emitter, not the
search.

No hold logic in v1 — the bot's holds_per_piece is 0.0, which sits BELOW the
whole human range (weakest humans hold 0.07/piece). Recorded as a known
manifold gap rather than papered over.
"""

from __future__ import annotations

import numpy as np
import torch

from tetris_sprint.fast import _CELLS, _SPAWN

TAP_LEFT, TAP_RIGHT, DAS_LEFT, DAS_RIGHT = 0, 1, 2, 3
ROTATE_CW, ROTATE_CCW, ROTATE_180 = 4, 5, 6
HARD_DROP = 8

#: Board-quality weights, hand-chosen in the Dellacherie tradition. Holes are
#: catastrophic in sprint (they bury lines), height is pressure, bumpiness
#: blocks flat stacking, clears are the goal.
W_LINES = 6.0
W_HOLES = -8.0
W_HEIGHT = -0.8
W_BUMP = -1.0


def _score_after(board, piece, rot, x):
    """Score of dropping (piece, rot) at column-offset x on `board` [24,10].

    Returns None when the placement is off-board or blocked at spawn height.
    Plain numpy, called only when a plan is (re)built — once per piece.
    """
    cells = _CELLS[piece][rot]
    cols = np.array([x + dx for dx, dy in cells])
    rows_rel = np.array([dy for dx, dy in cells])
    if cols.min() < 0 or cols.max() > 9:
        return None

    heights = np.zeros(10, dtype=np.int64)
    for c in range(10):
        filled = np.nonzero(board[:, c])[0]
        heights[c] = filled.max() + 1 if len(filled) else 0

    # Drop y: for each occupied column, the piece's lowest cell in that column
    # must land on the stack.
    y = -5
    for c in np.unique(cols):
        low = rows_rel[cols == c].min()
        y = max(y, int(heights[c]) - int(low))
    rows = y + rows_rel
    if rows.max() > 23:
        return None

    trial = board.copy()
    trial[rows, cols] = 1
    full = trial.sum(axis=1) == 10
    lines = int(full.sum())
    if lines:
        trial = np.concatenate(
            [trial[~full], np.zeros((lines, 10), dtype=trial.dtype)])

    new_heights = np.zeros(10, dtype=np.int64)
    holes = 0
    for c in range(10):
        filled = np.nonzero(trial[:, c])[0]
        if len(filled):
            top = filled.max()
            new_heights[c] = top + 1
            holes += int(top + 1 - len(filled))
    bump = int(np.abs(np.diff(new_heights)).sum())

    score = (W_LINES * lines + W_HOLES * holes
             + W_HEIGHT * int(new_heights.sum()) + W_BUMP * bump)
    return score, y


def plan_keystrokes(board, piece, rot0, x0, noise=0.0, rng=None, fumble=0.0):
    """The action list for the best placement of the current piece.

    `noise` perturbs each candidate's score (weak judgement); `fumble` is the
    probability of adding a wasted tap-and-return pair (weak execution).
    """
    best, best_target = None, None
    n_rots = 1 if piece == 1 else 4  # O has one distinct rotation
    for rot in range(n_rots):
        for x in range(-2, 9):
            scored = _score_after(board, piece, rot, x)
            if scored is None:
                continue
            score, _ = scored
            if noise and rng is not None:
                score = score + float(rng.normal(0.0, noise))
            if best is None or score > best:
                best, best_target = score, (rot, x)
    if best_target is None:  # nowhere to go; drop in place
        return [HARD_DROP]
    rot, x = best_target

    actions = []
    # Rotations first (kicks are irrelevant on the open board top).
    delta_rot = (rot - rot0) % 4
    actions.extend({0: [], 1: [ROTATE_CW], 2: [ROTATE_180],
                    3: [ROTATE_CCW]}[delta_rot])
    # Horizontal: DAS when the target is the extreme reachable column for this
    # rotation (finesse: one input to the wall), taps otherwise.
    dx_min = -min(dx for dx, dy in _CELLS[piece][rot])
    dx_max = 9 - max(dx for dx, dy in _CELLS[piece][rot])
    if x <= dx_min - 0 and x < x0:
        actions.append(DAS_LEFT)
    elif x >= dx_max and x > x0:
        actions.append(DAS_RIGHT)
    else:
        step = TAP_RIGHT if x > x0 else TAP_LEFT
        actions.extend([step] * abs(x - x0))
    # Fumbles: a tap and its correction. The placement is unchanged, the
    # keystroke count is not - which is exactly what finesse measures.
    if fumble and rng is not None:
        while rng.random() < fumble:
            if x < dx_max:
                actions.extend([TAP_RIGHT, TAP_LEFT])
            else:
                actions.extend([TAP_LEFT, TAP_RIGHT])
    actions.append(HARD_DROP)
    return actions


class SprintBot:
    """Per-instance plans over a batched env; replans when a piece appears.

    A plan always ends in a hard drop, so an exhausted plan means a new piece
    has spawned — that is the replan trigger. The other trigger is an episode
    boundary: auto-reset swaps in a fresh board mid-plan, and applying the
    leftover keystrokes to it would place pieces by reference to a board that
    no longer exists. `t == 0` catches that.

    **`skill` degrades judgement, and it has to.** Measured: dialling the
    env's LATENCY alone changes speed and nothing else — gravity moves pieces
    only vertically and a hard drop lands them at the bottom regardless, so a
    slow bot still places perfectly. That produces slow-but-flawless players, a
    combination E1 found does not exist among humans (the slowest ranks have
    the *worst* finesse, 4.86 against 2.81 at the top). Speed and accuracy have
    to be coupled deliberately.

    `skill` in [0, 1] does that with two knobs at once:

    * **evaluation noise** — Gaussian noise on each candidate placement's
      score, so a weak bot sometimes prefers a worse square. Its mistakes stay
      *ordered* (a slightly worse placement is likelier than a disastrous one),
      which is the property E2 found separates temperature-style degradation
      from uniform noise.
    * **wasted keystrokes** — a weak player fumbles the input: extra taps that
      are walked back, costing finesse without changing the placement. This is
      the only knob that moves inputs-per-piece, and finesse is the human
      feature with the widest measured spread.
    """

    def __init__(self, env, skill=1.0):
        self.env = env
        self.skill = float(skill)
        self.plans = [[] for _ in range(env.n)]
        self._rng = np.random.default_rng(0)

    def __call__(self, obs, env):
        boards = env.board.cpu().numpy()
        pieces = env.piece.cpu().tolist()
        rots = env.rot.cpu().tolist()
        xs = env.x.cpu().tolist()
        ts = env.t.cpu().tolist()
        actions = torch.zeros(env.n, dtype=torch.int64)
        for i in range(env.n):
            if ts[i] == 0:
                self.plans[i] = []  # auto-reset: the old plan is meaningless
            if not self.plans[i]:
                weak = 1.0 - self.skill
                self.plans[i] = plan_keystrokes(
                    boards[i], pieces[i], rots[i], xs[i],
                    noise=8.0 * weak, rng=self._rng, fumble=0.55 * weak)
            actions[i] = self.plans[i].pop(0)
        return actions
