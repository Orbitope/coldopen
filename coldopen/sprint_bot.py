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

**v2 plays for quads, because v1's strategy was not a human strategy.**
Measured against 396 real sprint records: v1 sat at quad_rate 0.01 and max_b2b
0.31 at *every* skill setting, while humans run 0.03 (rank d) to 0.69 (rank
x+). Its finesse spanned 16 human ranks and its stacking spanned none, so a
top rung looked like an x-rank player by one feature and a d-rank player by
another — a shape no human has (median rank disagreement: 8 ranks).

The cause was the objective, not the search. `W_LINES = 6.0` paid for any
immediate clear, so the bot cashed singles the instant one appeared and never
held four rows. v2 keeps a **well** (column 9) open, pays heavily for quads,
and charges for partial clears while the stack is safe — the classical sprint
strategy, and the one humans are ranked on. A danger valve re-enables partial
clears when the stack gets tall, or the bot would top out defending a well.

The other v1 fault fixed here: **no hold at all**. v1's holds_per_piece was
0.0, *below every human alive* (the weakest rank holds 0.07/piece). v2 holds
when the swap improves the two-piece ordering, which both improves play and
puts the feature on the manifold.

**Correction, recorded because the claim was published before it was
checked.** An earlier version of this docstring said column 9 had been
unreachable, on the reasoning that a piece whose cells all sit at `dx = 0`
needs `x = 9`. That is false. `_CELLS` uses a bounding-box convention in which
`dx >= 0` always, so `x` is the box's left edge, not a cell offset: a vertical
I has `dx = 1` or `2` and reaches column 9 at `x = 8` or `x = 7`, both inside
the original `x in range(-2, 9)` sweep. Verified directly — widening the sweep
adds zero legal placements for all seven pieces, since the largest legal `x`
is 8. The quad improvement came entirely from the objective, not the search
space.
"""

from __future__ import annotations

import numpy as np
import torch

from tetris_sprint.fast import _CELLS, _SPAWN

TAP_LEFT, TAP_RIGHT, DAS_LEFT, DAS_RIGHT = 0, 1, 2, 3
ROTATE_CW, ROTATE_CCW, ROTATE_180 = 4, 5, 6
HARD_DROP = 8
HOLD = 9

#: Board-quality weights in the Dellacherie tradition, retuned for quad play.
#: Holes are catastrophic in sprint (they bury lines), height is pressure,
#: bumpiness blocks flat stacking.
W_HOLES = -8.0
W_HEIGHT = -0.8
W_BUMP = -1.0

#: The well: one column kept deliberately empty so a vertical I can clear four
#: rows at once. Column 9 by convention (right-handed players' habit, and it
#: keeps the well away from spawn).
WELL_COLUMN = 9
W_QUAD = 40.0        # four rows at once: the goal
W_PARTIAL = -4.0     # per line, for cashing 1-3 rows while the stack is safe
W_GREEDY = 6.0       # per line, under the beginner strategy: any clear will do
#: Per cell dropped into the well. Swept: at -14 the bot defended the well to
#: the death (10 lines, 0% finish) because filling it in was never worth
#: surviving; at -6 it finishes every sprint and still quads at 0.61.
W_WELL_FILL = -6.0
W_WELL_DEPTH = 1.2   # per row the well is open below the surrounding stack

#: Above this stack height the well is a luxury: take any clear on offer. Set
#: below the 20-row visible field so the valve opens before a topout, not after.
DANGER_HEIGHT = 12

# ---------------------------------------------------------------------------
# NO CONSTANT BELOW THIS LINE IS FITTED TO HUMAN TETRIS DATA.
#
# That restriction is the point of the whole project: if the agent ladder is
# tuned until its telemetry matches the human telemetry, then measuring the
# overlap afterwards proves nothing. Human sprint records are the *test set*,
# read once at the end.
#
# Two constants here were briefly fitted and have been reverted, because the
# reverted values are the honest ones and the difference is worth recording:
#
#   * hold margins 0.3/4.0, chosen because human hold usage never falls below
#     0.070/piece. Reverted to a round guess.
#   * `strategy = skill ** 0.7`, chosen because a linear blend put quad rate at
#     0.265 where the rank-matched humans sit at 0.487. Reverted to linear.
#
# Every constant that survives is set from a property of the *game or the
# agent*: the rungs must finish, they must be ordered, and they must differ in
# how they play rather than only in how fast.
# ---------------------------------------------------------------------------

#: How much better a swap must score before the bot uses hold. A strong player
#: uses the slot whenever it helps at all; a weak one rarely thinks about it.
#: Round guesses - not fitted.
HOLD_MARGIN_STRONG = 0.5
HOLD_MARGIN_WEAK = 12.0

#: Judgement noise and keystroke fumbles at the weak end of the dial. Noise is
#: deliberately SMALL: at 8.0 every rung below skill 0.7 topped out instead of
#: finishing. That bound comes from the agent, not from people - a ladder whose
#: lower rungs cannot complete the task is measuring survival, not skill.
NOISE_MAX = 2.0
FUMBLE_MAX = 0.6


def _score_after(board, piece, rot, x, strategy=1.0):
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

    well_fill = int((cols == WELL_COLUMN).sum())

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
    # Bumpiness EXCLUDES the well: the whole point of a well is to be a step
    # down from its neighbour, and charging for that would close it.
    stack = np.delete(new_heights, WELL_COLUMN)
    bump = int(np.abs(np.diff(stack)).sum())

    danger = int(stack.max()) >= DANGER_HEIGHT

    # The expert strategy: hold four rows, cash them with a vertical I.
    if lines >= 4:
        expert = W_QUAD
    elif lines:
        # While the stack is safe, cashing rows costs the quad they were being
        # saved for. Once it is not safe, survival outranks efficiency.
        expert = (W_GREEDY * lines) if danger else (W_PARTIAL * lines)
    else:
        expert = 0.0
    # Reward the well being open BELOW its neighbours - that gap is where the
    # I piece goes. Only counted while defending; in danger it is dead weight.
    depth = int(stack.min()) - int(new_heights[WELL_COLUMN])
    if not danger:
        expert += W_WELL_DEPTH * max(0, min(depth, 4))
    if lines < 4:
        expert += W_WELL_FILL * well_fill

    # The beginner strategy: take any clear that is going, keep no well. This
    # is not a crippled expert - it is what human rank d measurably does
    # (quad rate 0.029 against x+'s 0.689), and it survives perfectly well.
    beginner = W_GREEDY * lines

    clear_score = strategy * expert + (1.0 - strategy) * beginner
    score = (clear_score + W_HOLES * holes
             + W_HEIGHT * int(new_heights.sum()) + W_BUMP * bump)
    return score, y


def best_placement(board, piece, noise=0.0, rng=None, strategy=1.0):
    """(score, rot, x) for the best drop of `piece`, or (None, None, None).

    `x` is the LEFT EDGE of the piece's bounding box, not a cell offset —
    `_CELLS` keeps every `dx >= 0`. The largest legal `x` is therefore 8 (a
    vertical I at `x = 8` occupies column 9), and `range(-2, 9)` covers the
    whole reachable set. Stated explicitly because the opposite was once
    asserted here: the sweep was briefly widened to 11 on the theory that
    column 9 needed `x = 9`, which added no legal placement at all.
    """
    best, best_target = None, (None, None)
    n_rots = 1 if piece == 1 else 4  # O has one distinct rotation
    for rot in range(n_rots):
        for x in range(-2, 9):
            scored = _score_after(board, piece, rot, x, strategy=strategy)
            if scored is None:
                continue
            score, _ = scored
            if noise and rng is not None:
                score = score + float(rng.normal(0.0, noise))
            if best is None or score > best:
                best, best_target = score, (rot, x)
    return best, best_target[0], best_target[1]


def plan_keystrokes(board, piece, rot0, x0, noise=0.0, rng=None, fumble=0.0,
                    strategy=1.0):
    """The action list for the best placement of the current piece.

    `strategy` picks the objective (see `_score_after`); `noise` perturbs each
    candidate's score (weak judgement); `fumble` is the probability of adding a
    wasted tap-and-return pair (weak execution).
    """
    best, rot, x = best_placement(board, piece, noise=noise, rng=rng,
                                  strategy=strategy)
    if best is None:  # nowhere to go; drop in place
        return [HARD_DROP]

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


def _apply(board, piece, rot, x):
    """The board after dropping (piece, rot) at x, or None if illegal.

    Only needed by the two-ply hold search, which has to know what the board
    looks like *after* the first piece before it can score the second.
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
    y = -5
    for c in np.unique(cols):
        y = max(y, int(heights[c]) - int(rows_rel[cols == c].min()))
    rows = y + rows_rel
    if rows.max() > 23:
        return None
    out = board.copy()
    out[rows, cols] = 1
    full = out.sum(axis=1) == 10
    if full.any():
        out = np.concatenate(
            [out[~full], np.zeros((int(full.sum()), 10), dtype=out.dtype)])
    return out


def best_pair(board, first, second, noise=0.0, rng=None, strategy=1.0, top_k=6):
    """Best total score for placing `first` then `second`.

    A one-ply comparison cannot express why people actually hold: the slot is
    for *ordering* two pieces, and whether a swap helps depends on what happens
    after both are down. Measured consequence of getting this wrong — the
    one-ply bot ran 4.4 ranks below the human baseline on hold usage, the only
    feature of five that fell outside the spread real players show.

    Only the `top_k` best first placements are expanded, which keeps this near
    6x a single search instead of 44x. Weak candidates for the first piece do
    not become strong because of what follows them.
    """
    scored = []
    for rot in range(1 if first == 1 else 4):
        for x in range(-2, 9):
            got = _score_after(board, first, rot, x, strategy=strategy)
            if got is not None:
                scored.append((got[0], rot, x))
    if not scored:
        return None
    scored.sort(reverse=True)
    best = None
    for score, rot, x in scored[:top_k]:
        after = _apply(board, first, rot, x)
        if after is None:
            continue
        follow, _, _ = best_placement(after, second, strategy=strategy)
        total = score + (follow if follow is not None else 0.0)
        if noise and rng is not None:
            total += float(rng.normal(0.0, noise))
        if best is None or total > best:
            best = total
    return best


def markov_action(board, piece, rot, x, target_rot, target_x):
    """The single next keystroke toward (target_rot, target_x), from HERE.

    The plan-following emitter is not a Markov policy: which keystroke it plays
    depends on how far through its plan it is, which is not in the observation.
    That is fine for playing, and fatal for imitation — "what would the teacher
    do in this state?" has no well-defined answer, which is precisely the
    question DAgger relabelling asks. Measured consequence: a student at 98.6%
    agreement on the teacher's own states scored 25.8% on its own, and 39.6% of
    its relabelled targets were `hold` against the student's 1.1%.

    This derives the action from observable state alone, so teacher and relabel
    agree by construction.
    """
    if target_rot is None:
        return HARD_DROP
    delta_rot = (target_rot - rot) % 4
    if delta_rot:
        return {1: ROTATE_CW, 2: ROTATE_180, 3: ROTATE_CCW}[delta_rot]
    if x == target_x:
        return HARD_DROP
    # DAS when the target is the extreme reachable column for this rotation
    # (one input to the wall, which is what good finesse looks like).
    dx_min = -min(dx for dx, dy in _CELLS[piece][target_rot])
    dx_max = 9 - max(dx for dx, dy in _CELLS[piece][target_rot])
    if target_x <= dx_min and target_x < x:
        return DAS_LEFT
    if target_x >= dx_max and target_x > x:
        return DAS_RIGHT
    return TAP_RIGHT if target_x > x else TAP_LEFT


class SprintBot:
    """Per-instance plans over a batched env; replans when a piece appears.

    A plan always ends in a hard drop, so an exhausted plan means a new piece
    has spawned — that is the replan trigger. The other trigger is an episode
    boundary: auto-reset swaps in a fresh board mid-plan, and applying the
    leftover keystrokes to it would place pieces by reference to a board that
    no longer exists. `t == 0` catches that.

    **`skill` interpolates between two strategies, and that is the whole
    design.** Two measurements forced it:

    1. Dialling the env's LATENCY alone changes speed and nothing else —
       gravity moves pieces only vertically and a hard drop lands them at the
       bottom regardless, so a slow bot still places perfectly. That produces
       slow-but-flawless players, a combination E1 found does not exist among
       humans (the slowest ranks have the *worst* finesse, 4.86 against 2.81).
    2. Degrading judgement with noise instead does not work either. At
       `noise = 8` every rung below skill 0.7 topped out — 1.6 to 11.4 lines,
       0% finish. A ladder whose lower rungs cannot complete the task is
       measuring survival, not skill, so noise is capped where every rung
       still finishes.

    What varies with Tetris skill is the *strategy*: beginners cash singles
    the moment one appears, experts hold four rows and cash them with a
    vertical I. That is standard, widely documented Tetris knowledge — it is
    in the game's own tutorials — not something inferred from this project's
    human records, and it happens to be exactly the objective this bot shipped
    with in v1, which survived fine on it. So the dial interpolates the
    *objective*:

    * **strategy** — `_score_after` blends the expert well-and-quad score with
      the beginner take-any-clear score. Both are competent at staying alive;
      they differ in what they are trying to achieve, which is what separates
      human ranks.
    * **hold margin** — how much better a swap must look before using it.
      Using the hold slot well is a planning skill with no speed component, so
      it is driven from the dial rather than left to emerge. The margins are
      round guesses, deliberately not fitted (see the note above the
      constants).
    * **wasted keystrokes** — extra taps that are walked back, costing finesse
      without changing the placement. The only knob that moves
      inputs-per-piece.
    * **evaluation noise** — kept, but small (`NOISE_MAX = 2.0`), so mistakes
      stay *ordered* without becoming fatal.
    """

    def __init__(self, env, skill=1.0, use_hold=True, markov=False,
                 hold_lookahead=True):
        self.env = env
        self.skill = float(skill)
        self.use_hold = use_hold
        #: Decide holds by two-ply piece ordering rather than a one-ply "is the
        #: other piece better right now" check.
        #:
        #: Labelled plainly: this was added AFTER the human comparison showed
        #: hold usage was the one feature of five outside the human spread, so
        #: it is a test-guided change and the budget for those is small.
        #:
        #: It is kept because it improves the bot at the GAME, which is an
        #: agent-side justification that stands on its own — finish rate rose
        #: at every rung (0.92 -> 1.00 at skill 1.0, 0.83 -> 1.00 at 0.7,
        #: 0.67 -> 0.92 at 0.4). Piece ordering is what the hold slot is for,
        #: and one ply cannot express it. Costs ~4.5x per replan.
        self.hold_lookahead = hold_lookahead
        #: Emit one keystroke per call from observable state alone, instead of
        #: following a stored plan. Slower (it re-searches every step) but it
        #: makes the teacher a genuine Markov policy, which imitation needs.
        #: Distillation uses it; the ladder rungs do not need to.
        self.markov = markov
        self.plans = [[] for _ in range(env.n)]
        self._rng = np.random.default_rng(0)
        #: Markov mode re-derives the target placement on every keystroke, but
        #: the target depends only on (board, piece) and neither changes while
        #: a piece is falling. Memoising it per instance turns ~4 searches per
        #: piece back into 1, with no semantic change: the cache is pure
        #: memoisation of a deterministic function of observable state.
        self._target_cache = [None] * env.n

    @property
    def _dials(self):
        weak = 1.0 - self.skill
        return {
            "strategy": self.skill,
            "noise": NOISE_MAX * weak,
            "fumble": FUMBLE_MAX * weak,
            "hold_margin": (HOLD_MARGIN_STRONG * self.skill
                            + HOLD_MARGIN_WEAK * weak),
        }

    def _wants_hold(self, board, piece, hold, hold_used, queue_head, dials):
        """True when swapping the active piece scores materially better.

        The alternative to the current piece is whatever hold would hand over:
        the stored piece, or the next one from the queue if the slot is empty.
        A plain greedy comparison, gated by a margin that widens as skill drops
        so weak rungs rarely hold and strong ones often do. One hold per piece
        is enforced by the env's own mask (`hold_used`), so this cannot loop.
        """
        if not self.use_hold or hold_used:
            return False
        other = queue_head if hold < 0 else hold
        if other is None or other < 0:
            return False
        kw = dict(noise=dials["noise"], rng=self._rng, strategy=dials["strategy"])
        if self.hold_lookahead:
            # Which ORDER of the two available pieces leaves the better board?
            # That is the question the hold slot actually answers.
            mine = best_pair(board, piece, other, **kw)
            theirs = best_pair(board, other, piece, **kw)
        else:
            mine, _, _ = best_placement(board, piece, **kw)
            theirs, _, _ = best_placement(board, other, **kw)
        return (mine is not None and theirs is not None
                and theirs > mine + dials["hold_margin"])

    def __call__(self, obs, env):
        boards = env.board.cpu().numpy()
        pieces = env.piece.cpu().tolist()
        rots = env.rot.cpu().tolist()
        xs = env.x.cpu().tolist()
        ts = env.t.cpu().tolist()
        holds = env.hold.cpu().tolist()
        hold_used = env.hold_used.cpu().tolist()
        queue = env.queue.cpu().numpy()
        actions = torch.zeros(env.n, dtype=torch.int64)
        dials = self._dials
        if self.markov:
            for i in range(env.n):
                key = (boards[i].tobytes(), pieces[i])
                cached = self._target_cache[i]
                if cached is None or cached[0] != key:
                    if self._wants_hold(boards[i], pieces[i], holds[i],
                                        hold_used[i], int(queue[i][0]), dials):
                        actions[i] = HOLD
                        self._target_cache[i] = None
                        continue
                    _, t_rot, t_x = best_placement(
                        boards[i], pieces[i], noise=dials["noise"],
                        rng=self._rng, strategy=dials["strategy"])
                    self._target_cache[i] = (key, t_rot, t_x)
                _, t_rot, t_x = self._target_cache[i]
                actions[i] = markov_action(boards[i], pieces[i], rots[i],
                                           xs[i], t_rot, t_x)
            return actions
        for i in range(env.n):
            if ts[i] == 0:
                self.plans[i] = []  # auto-reset: the old plan is meaningless
            if not self.plans[i]:
                if self._wants_hold(boards[i], pieces[i], holds[i],
                                    hold_used[i], int(queue[i][0]), dials):
                    # A one-action plan: it exhausts immediately, so the next
                    # call replans for whichever piece the swap handed over.
                    self.plans[i] = [HOLD]
                else:
                    self.plans[i] = plan_keystrokes(
                        boards[i], pieces[i], rots[i], xs[i],
                        noise=dials["noise"], rng=self._rng,
                        fumble=dials["fumble"], strategy=dials["strategy"])
            actions[i] = self.plans[i].pop(0)
        return actions
