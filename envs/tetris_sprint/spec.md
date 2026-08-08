# tetris_sprint — environment spec

Single source of truth. Both `reference.py` and `fast.py` are written from
this document — never from each other. Every rule below must be traceable in
both implementations.

**Purpose.** Single-player 40 LINES sprint at **input level**, on a **virtual
clock**: one env step is one keystroke, and time advances by declared integer
costs per action. Finesse (inputs per piece) exists because steps are
keystrokes; sprint time and PPS exist because the clock is real state; the
speed/accuracy coupling emerges because each keystroke costs latency. Skill
telemetry (finesse, quad rate, B2B, PPS, sprint time) is derived from
trajectories, not stored in state.

**Virtual time, not wall time.** There are no frames and no real-time physics.
The clock `time_ms` advances by `cost(action) + LATENCY` per step, where the
costs are declared constants and `LATENCY` is a per-environment parameter (the
"thinking time" of the player — the ladder's skill dial). Gravity and lock
delay run on this virtual clock. Everything is integer arithmetic; the whole
environment is a pure function of (action sequence, bag seed, parameters).

**Fidelity notes and documented deviations from TETR.IO:**

1. **Constants are declared, not measured.** DAS/ARR/gravity values below are
   TETR.IO-plausible but frozen as env constants; the exact client defaults
   were not verifiable from primary sources. This is acceptable because all
   agent↔human comparison is standardised within population (the E1
   discipline) — self-consistency across the ladder is what matters.
2. **No 180-kick table.** `rotate_180` succeeds in place or not at all.
3. **No T-spin detection.** Sprint telemetry needs clear sizes, not spin
   bonuses.
4. **Soft drop is instant** (SDF = ∞), which is standard sprint handling.
5. **At most one auto-lock per step**: physics ends at the first
   gravity-induced lock. Guaranteed reachable-by-construction under the
   declared parameter bounds; documented rather than emergent.
6. **Input counting is ours, not TETR.IO's.** Every env step counts as one
   input. TETR.IO's `inputs` field has its own counting; absolute finesse
   numbers are not directly comparable — standardise within population.
7. **Lock-delay reset cap simplification**: at the 15-reset cap the timer
   simply stops resetting (the piece locks 500 virtual ms after its last
   rest), rather than guideline's lock-on-next-contact.

## Parameters

Frozen constants (same for every instance, part of the environment identity):

| constant | value | meaning |
|---|---|---|
| G | 1000 | ms of virtual time per one-cell gravity fall |
| LOCK | 500 | ms resting before auto-lock |
| RESETS_MAX | 15 | lock-delay resets available per piece |
| DAS | 167 | ms cost of a das action (hold-to-wall), before ARR travel |
| ARR | 33 | ms per additional cell of das travel beyond the first |
| CLEAR_BONUS | 10000 | reward per line cleared |
| TOPOUT_PENALTY | 200000 | reward penalty for block-out / lock-out |
| HORIZON | 1200 | inputs per episode (termination T4) |

Constructor parameter (the ladder dial; **not** state):

| parameter | dtype | range | default | meaning |
|---|---|---|---|---|
| LATENCY | int (ms) | [17, 2000] | 100 | added to every action's cost: the player's per-keystroke thinking/motor time |

Bounds matter: with LATENCY ≤ 2000 the maximum per-step cost is
`2000 + DAS + 9·ARR = 2464` ms, so `time_ms ≤ 1200 · 2464 < 3,000,000` and all
rewards/returns stay well inside float32's exact-integer range (2^24).

## State space

Board coordinates: 10 columns (0 = leftmost), 24 rows (**0 = bottom**).
Rows 0–19 are the visible field; rows 20–23 are the spawn/overflow buffer.

Piece ids: `0=I, 1=O, 2=T, 3=J, 4=L, 5=S, 6=Z`.

| field       | dtype | shape   | bounds          | meaning                                             |
|-------------|-------|---------|-----------------|-----------------------------------------------------|
| t           | int32 | []      | [0, 1200]       | in-episode step counter (= inputs so far)           |
| time_ms     | int32 | []      | [0, 3000000]    | virtual clock                                       |
| board       | int8  | [24,10] | {0,1}           | locked cells; row 0 is the bottom                   |
| piece       | int8  | []      | [0, 6]          | active piece id                                     |
| rot         | int8  | []      | [0, 3]          | rotation state: 0=spawn, 1=CW (R), 2=180, 3=CCW (L) |
| x           | int8  | []      | [-2, 8]         | bounding-box bottom-left, column                    |
| y           | int8  | []      | [-2, 23]        | bounding-box bottom-left, row                       |
| hold        | int8  | []      | [-1, 6]         | held piece id, -1 = empty                           |
| hold_used   | int8  | []      | {0,1}           | 1 = hold spent since the last lock                  |
| queue       | int8  | [5]     | [0, 6]          | next pieces, queue[0] is next                       |
| bag         | int8  | []      | [1, 127]        | bitmask of piece ids remaining in the dealing bag (bit p = piece p); refilled to 127 the moment it empties, so 0 is unreachable |
| lines       | int8  | []      | [0, 43]         | total lines cleared this episode                    |
| fall_ms     | int32 | []      | [0, 999]        | virtual ms accrued toward the next gravity fall     |
| lock_ms     | int32 | []      | [0, 499]        | virtual ms accrued resting toward auto-lock         |
| lock_resets | int8  | []      | [0, 15]         | lock-delay resets consumed by the active piece      |

The active piece is **not** part of `board`. Scalar bounds on `x`/`y` are
loose envelopes — the sharp constraint is invariant I2.

**Resting** means ghost distance 0: the piece cannot move down one cell.
**Airborne** means ghost distance ≥ 1.

### Piece shapes

`CELLS[piece][rot]` is a set of four `(dx, dy)` offsets, **dy up**, added to
`(x, y)`. Derived from the guideline SRS diagrams (drawn y-down) by flipping
vertically; boxes are 3×3 for T/J/L/S/Z and 4×4 for I/O.

| piece | rot 0 (spawn) | rot 1 (R) | rot 2 | rot 3 (L) |
|---|---|---|---|---|
| I | (0,2)(1,2)(2,2)(3,2) | (2,0)(2,1)(2,2)(2,3) | (0,1)(1,1)(2,1)(3,1) | (1,0)(1,1)(1,2)(1,3) |
| O | (1,1)(2,1)(1,2)(2,2) | same as rot 0 | same as rot 0 | same as rot 0 |
| T | (0,1)(1,1)(2,1)(1,2) | (1,0)(1,1)(2,1)(1,2) | (1,0)(0,1)(1,1)(2,1) | (1,0)(0,1)(1,1)(1,2) |
| J | (0,1)(1,1)(2,1)(0,2) | (1,0)(1,1)(1,2)(2,2) | (2,0)(0,1)(1,1)(2,1) | (0,0)(1,0)(1,1)(1,2) |
| L | (0,1)(1,1)(2,1)(2,2) | (1,0)(2,0)(1,1)(1,2) | (0,0)(0,1)(1,1)(2,1) | (1,0)(1,1)(0,2)(1,2) |
| S | (0,1)(1,1)(1,2)(2,2) | (2,0)(1,1)(2,1)(1,2) | (0,0)(1,0)(1,1)(2,1) | (1,0)(0,1)(1,1)(0,2) |
| Z | (1,1)(2,1)(0,2)(1,2) | (1,0)(1,1)(2,1)(2,2) | (1,0)(2,0)(0,1)(1,1) | (0,0)(0,1)(1,1)(1,2) |

(Rot 0 matches the guideline spawn diagram read top-down: S = `.XX / XX.`,
Z = `XX. / .XX`. Cell order within an entry is irrelevant — a piece is a set.)

### Spawn placement

All pieces spawn at rot 0 with box position:

| piece     | (x, y)  | occupied cells            |
|-----------|---------|---------------------------|
| I         | (3, 18) | columns 3–6, row 20       |
| O         | (3, 19) | columns 4–5, rows 20–21   |
| T,J,L,S,Z | (3, 19) | columns 3–5, rows 20–21   |

Spawning is **blocked** (top-out) if any spawn cell overlaps `board`.
A freshly spawned or held-in piece always starts with
`fall_ms = 0, lock_ms = 0, lock_resets = 0`.

## Actions

10 discrete actions. One step = one action = **one input**. An action whose
effect is impossible is a **no-op** for the board but still advances the
clock by its cost and still counts as an input (humans mash; mashing is an
input).

| id | name        | effect (instantaneous, at the moment of the keypress) | cost (before +LATENCY) |
|----|-------------|-------------------------------------------------------|------------------------|
| 0  | tap_left    | x ← x−1 if valid, else no-op                          | 0 |
| 1  | tap_right   | x ← x+1 if valid, else no-op                          | 0 |
| 2  | das_left    | repeat x ← x−1 until invalid (0+ cells)               | DAS + max(0, cells−1)·ARR |
| 3  | das_right   | repeat x ← x+1 until invalid (0+ cells)               | DAS + max(0, cells−1)·ARR |
| 4  | rotate_cw   | SRS rotation rot→(rot+1)%4, kick table below          | 0 |
| 5  | rotate_ccw  | SRS rotation rot→(rot+3)%4, kick table below          | 0 |
| 6  | rotate_180  | rot→(rot+2)%4 at offset (0,0) only, else no-op        | 0 |
| 7  | soft_drop   | y ← y − ghost_d (drop to rest; does NOT lock)         | 0 |
| 8  | hard_drop   | drop to rest, LOCK (see Lock sequence)                | 0 |
| 9  | hold        | swap with hold slot (see Hold sequence)               | 0 |

Total step cost = table cost + LATENCY, always, including no-ops and
terminal steps.

"Valid" everywhere means: every cell `(x+dx, y+dy)` has `0 ≤ column ≤ 9`,
`0 ≤ row ≤ 23`, and `board[row][column] == 0`.

**Ghost distance** `ghost_d`: the largest `d ≥ 0` such that the piece at
`(x, y−d)` is valid.

**Action mask**: all actions available except `hold` (id 9), masked iff
`hold_used == 1`. A masked action stepped anyway is a no-op costing its
normal cost.

**Lock-delay reset on action success**: if a movement/rotation action
(ids 0–6) *changes* the piece's position or rotation AND `lock_ms > 0`:
if `lock_resets < RESETS_MAX`, then `lock_ms ← 0` and
`lock_resets ← lock_resets + 1`; at the cap, `lock_ms` is left unchanged.
`soft_drop` never resets the timer. (das that moves 0 cells changes nothing
and is a no-op for this rule.)

**Leaving the ground discards lock delay**: at the end of the action phase,
if the episode did not terminate and the piece is airborne, `lock_ms ← 0` —
without consuming a reset. (A kick or a slide off a ledge can lift a resting
piece; accrued lock delay does not follow it into the air. Without this rule
the at-the-cap case above strands `lock_ms` on an airborne piece and
invariant I11 cannot hold.)

### SRS kick tables

Offsets `(kx, ky)`, **ky up**, applied to the box position; tried strictly
left to right; the first offset where the rotated piece is valid is taken; if
none is valid the rotation is a no-op. These are the published guideline SRS
tables, reproduced because this document is the source of truth; `tests/`
must compare the implementation's tables against this section verbatim.

**J, L, S, T, Z:**

| transition | kicks |
|---|---|
| 0→1 | (0,0) (−1,0) (−1,+1) (0,−2) (−1,−2) |
| 1→0 | (0,0) (+1,0) (+1,−1) (0,+2) (+1,+2) |
| 1→2 | (0,0) (+1,0) (+1,−1) (0,+2) (+1,+2) |
| 2→1 | (0,0) (−1,0) (−1,+1) (0,−2) (−1,−2) |
| 2→3 | (0,0) (+1,0) (+1,+1) (0,−2) (+1,−2) |
| 3→2 | (0,0) (−1,0) (−1,−1) (0,+2) (−1,+2) |
| 3→0 | (0,0) (−1,0) (−1,−1) (0,+2) (−1,+2) |
| 0→3 | (0,0) (+1,0) (+1,+1) (0,−2) (+1,−2) |

**I:**

| transition | kicks |
|---|---|
| 0→1 | (0,0) (−2,0) (+1,0) (−2,−1) (+1,+2) |
| 1→0 | (0,0) (+2,0) (−1,0) (+2,+1) (−1,−2) |
| 1→2 | (0,0) (−1,0) (+2,0) (−1,+2) (+2,−1) |
| 2→1 | (0,0) (+1,0) (−2,0) (+1,−2) (−2,+1) |
| 2→3 | (0,0) (+2,0) (−1,0) (+2,+1) (−1,−2) |
| 3→2 | (0,0) (−2,0) (+1,0) (−2,−1) (+1,+2) |
| 3→0 | (0,0) (+1,0) (−2,0) (+1,−2) (−2,+1) |
| 0→3 | (0,0) (−1,0) (+2,0) (−1,+2) (+2,−1) |

**O:** rotation is the identity on cells; it "succeeds" at (0,0) and only
changes `rot` (this counts as a successful rotation for the reset rule).

## Step algorithm

Processing action `a` from a state with counter `t = k`. `consumed` counts
pieces taken from the queue this step (it is the BAG_REFILL draw index).

1. `consumed ← 0`.
2. **Action phase** (instantaneous): apply the effect from the Actions table.
   `hard_drop` and `hold` may run the Lock/Hold sequences below, may draw
   from the bag (index `consumed`, then `consumed ← consumed + 1`), and may
   terminate the episode (T1/T2/T3).
3. **Cost**: `cost ← table_cost(a) + LATENCY` (das cost uses the cells
   actually moved in step 2).
4. **Physics phase** — skipped if the episode terminated in step 2:
   `remaining ← cost`; loop:
   - **airborne** (`ghost_d ≥ 1`): `need ← G − fall_ms`.
     If `remaining ≥ need`: the piece falls one cell (`y ← y−1`),
     `fall_ms ← 0`, `lock_ms ← 0`, `remaining ← remaining − need`; continue.
     Else `fall_ms ← fall_ms + remaining`; exit loop.
   - **resting**: `fall_ms ← 0`; `need ← LOCK − lock_ms`.
     If `remaining ≥ need`: **auto-lock** — run the Lock sequence (no initial
     drop needed; the piece is already at rest), then **end the physics phase
     regardless of remaining time**. Else `lock_ms ← lock_ms + remaining`;
     exit loop.
5. `time_ms ← time_ms + cost` (always, terminal or not).
6. `reward ← CLEAR_BONUS · n_cleared_this_step − cost − TOPOUT_PENALTY ·
   topped_out_this_step` where `n_cleared_this_step` sums over both an
   action-phase lock and an auto-lock, and `topped_out` means T2 or T3 fired
   this step.
7. `t ← k + 1`. If `t == HORIZON` and the episode has not already
   terminated: terminate (T4).

### Lock sequence (shared by hard_drop and auto-lock)

1. (hard_drop only) `y ← y − ghost_d`.
2. Merge the four cells into `board`.
3. **Lock-out** (T3): if ALL four merged cells have row ≥ 20, terminate.
4. **Clear**: remove all full rows (all 10 cells set, any of the 24 rows);
   rows above shift down; `n` = number removed (0–4);
   `lines ← min(lines + n, 43)`.
5. **Success** (T1): if `lines ≥ 40`, terminate. No spawn.
6. `hold_used ← 0`.
7. **Spawn**: `piece ← queue[0]`; `queue[0..3] ← queue[1..4]`;
   `queue[4] ← bag_draw(slot=BAG_REFILL, step=k, index=consumed)`;
   `consumed ← consumed + 1`; `rot ← 0`; `(x, y) ←` spawn placement;
   `fall_ms ← 0`; `lock_ms ← 0`; `lock_resets ← 0`.
8. **Block-out** (T2): if any spawn cell overlaps `board`, terminate.

### Hold sequence

1. If `hold_used == 1`: no-op (mask normally prevents this).
2. If `hold == −1`: `hold ← piece`; `piece ← queue[0]`; shift queue;
   `queue[4] ← bag_draw(slot=BAG_REFILL, step=k, index=consumed)`;
   `consumed ← consumed + 1`. Else swap `hold ↔ piece` (no draw).
3. `rot ← 0`; `(x, y) ←` spawn placement; `fall_ms ← 0`; `lock_ms ← 0`;
   `lock_resets ← 0`.
4. **Block-out** (T2): overlap ⇒ terminate.
5. `hold_used ← 1`.

### Bag draw (the only random primitive)

`bag_draw(slot, step, index)`:

1. `n ← popcount(bag)` (≥ 1 by invariant I5).
2. `r ← draw_randint(episode_key, step, slot, n, index)` — uniform on [0, n).
3. The drawn piece is the `r`-th **set** bit of `bag` in ascending piece-id
   order. Clear that bit.
4. If `bag == 0`, `bag ← 127`.

All integer; bit-identical across implementations by construction.

## Observations

`float32`, shape `[9, 24, 10]`, values exactly 0.0 or 1.0. Planes are built
as 0/1 integers and cast to float32 once, at the end — there is no float
computation anywhere else in the environment.

| channel | contents |
|---------|----------|
| 0 | `board` (locked cells) |
| 1 | active piece cells at `(x, y, rot)` |
| 2 | ghost cells: active piece cells at `(x, y − ghost_d)` |
| 3 | `hold` piece rendered at ITS spawn placement, rot 0; all-zero if `hold == −1` |
| 4–8 | `queue[0..4]`, each rendered at its spawn placement, rot 0 |

The timers (`fall_ms`, `lock_ms`, `lock_resets`, `time_ms`) are deliberately
**not** observed in v1: humans do not track milliseconds either, and a policy
that soft/hard-drops promptly is unaffected. Revisit if hesitation-drift
turns out to need policy awareness.

## Rewards

Integer-valued float32. Per step, exactly as in Step algorithm item 6:

```
reward = 10000 · lines_cleared_this_step − cost − 200000 · topped_out
```

Maximizing return is **minimizing virtual time to 40 lines** — the human
sprint objective, not an analogue of it: finishing yields `+400000 −
total_ms − Σ` (a constant minus elapsed time), stalling bleeds cost forever,
and the top-out penalty makes suicide strictly worse than stalling. All
quantities are integers below 2^24, so float32 holds them exactly and **no
`x-atol` is declared anywhere** — the differential test requires bit-exact
equality on every field.

## Termination

| id | condition | where |
|----|-----------|-------|
| T1 | `lines ≥ 40` (success) | Lock sequence step 5 |
| T2 | block-out: spawn cells overlap `board` | Lock step 8, Hold step 4 |
| T3 | lock-out: all four locked cells in rows ≥ 20 | Lock step 3 |
| T4 | `t == 1200` after the step's increment | Step algorithm item 7 |

All four are terminations (no separate truncation channel; T4 is the bounded
horizon the auto-reset battery needs). Random policies hard-drop ~1/10 steps
and top out via T2 long before T4; with high LATENCY, auto-locks add T2/T3
paths through gravity alone.

## Reset

1. `board ← 0`, `lines ← 0`, `hold ← −1`, `hold_used ← 0`, `t ← 0`,
   `time_ms ← 0`, `fall_ms ← 0`, `lock_ms ← 0`, `lock_resets ← 0`,
   `bag ← 127`.
2. Six sequential draws `bag_draw(slot=BAG_RESET, step=0, index=k)` for
   `k = 0..5`: the first fills `piece`, the next five fill `queue[0..4]`.
   All six come from the first 7-bag: six **distinct** ids,
   `popcount(bag) == 1` afterwards.
3. `rot ← 0`; `(x, y) ←` spawn placement. (Empty board: cannot be blocked.)

## Invariants

Checked batch-wide every step in debug mode; each becomes an
`@simulacrum.invariant` on the batched env.

1. **I1 board binary** — every `board` cell is 0 or 1.
2. **I2 active piece legal** — in every non-terminal state, all four active
   piece cells are in bounds (columns 0–9, rows 0–23) and on empty cells.
3. **I3 no full rows** — no `board` row has all 10 cells set.
4. **I4 line counter** — `0 ≤ lines ≤ 43`; `lines < 40` in non-terminal
   states.
5. **I5 bag never empty** — `1 ≤ bag ≤ 127`.
6. **I6 id ranges** — `piece ∈ [0,6]`; `queue` entries `∈ [0,6]`;
   `hold ∈ [−1,6]`; `hold_used ∈ {0,1}`; `rot ∈ [0,3]`.
7. **I7 reset support** — `t == 0` ⇒ empty board, `lines == 0`,
   `hold == −1`, `hold_used == 0`, all four timers zero, and
   `{piece} ∪ queue` six distinct ids with `popcount(bag) == 1`.
8. **I8 counter bound** — `0 ≤ t ≤ 1200`; `t == 1200` ⇒ terminal.
9. **I9 clock bounds** — `0 ≤ time_ms ≤ 3000000`, and
   `time_ms ≤ t · 2464` (each step costs at most `2000 + DAS + 9·ARR`).
10. **I10 gravity residual** — `0 ≤ fall_ms ≤ 999`; resting (in a
    non-terminal state) ⇒ `fall_ms == 0`.
11. **I11 lock residual** — `0 ≤ lock_ms ≤ 499`; airborne (in a
    non-terminal state) ⇒ `lock_ms == 0`.
12. **I12 reset cap** — `0 ≤ lock_resets ≤ 15`.

## RNG slots

| slot | name       | used at | distribution |
|------|------------|---------|--------------|
| 0    | BAG_RESET  | reset (step 0), index k = 0..5 | k-th draw: uniform over pieces remaining in the bag (7−k choices) |
| 1    | BAG_REFILL | any step whose action or auto-lock consumes a queue piece; step = pre-increment counter `t`, index = per-step consume counter (0 or 1) | uniform over pieces remaining in the bag |

Up to **two** BAG_REFILL draws can occur in one step (an action-phase lock or
first-hold, followed by an auto-lock in the physics phase), distinguished by
index 0 and 1. BAG_RESET and BAG_REFILL share step 0 on an episode's first
step but are distinct slots, so no draw collides. The slot table is mirrored
by the `Slots` IntEnum in `__init__.py`.

## Telemetry notes (not part of the env contract)

Derived from trajectories downstream, never stored in state: finesse = steps
between locks per piece; quad rate = share of clearing locks with `n == 4`;
B2B = consecutive quad locks; holds per piece; **sprint time** = `time_ms` at
T1; **PPS** = pieces placed / (`time_ms`/1000). LATENCY is a chosen dial, so
absolute agent PPS is chosen too — within-population standardisation is
mandatory when comparing to humans.
