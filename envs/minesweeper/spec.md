# minesweeper — environment spec

Single source of truth. Both `reference.py` and `fast.py` are written from
this document — never from each other. Every rule below must be traceable in
both implementations.

Expert minesweeper as a solo timed environment. The purpose is measuring
**per-click judgement** — was a provably safe cell available, and did the agent
click one — alongside speed (3BV/s) and click efficiency. The deduction oracle
that answers the judgement question lives in the telemetry layer, not here:
constraint solving on every step would make the batched env unusable, and the
env does not need the answer to compute its own dynamics.

## Parameters

| name | type | range | default | meaning |
|---|---|---|---|---|
| `H` | int | [5, 32] | 16 | board rows |
| `W` | int | [5, 32] | 30 | board columns |
| `M` | int | [1, H·W − 9] | 99 | mine count (Expert) |
| `LATENCY` | int (ms) | [10, 2000] | 200 | added to every action's cost: the player's per-click motor and thinking time. **The speed dial.** |
| `HORIZON` | int | [1, 4000] | 2000 | step cap |

`M ≤ H·W − 9` is required so a full 3×3 exclusion zone can always be carved out
of the board at first click (see Reset).

### Action costs

| action kind | cost (ms) |
|---|---|
| `reveal` | 30 |
| `flag` | 30 |
| `chord` | 60 |

Every action additionally costs `LATENCY`, so one step advances the clock by
`COST[kind] + LATENCY`.

**Why the costs differ, and why it matters.** If every action cost the same,
elapsed time would be exactly `t · (COST + LATENCY)`, so 3BV/s would equal
`efficiency / (COST + LATENCY)` — the speed axis and the efficiency axis would
be the *same number* wearing different units. E6 on tetris_sprint showed
exactly what that costs: a one-dimensional agent population produced collinear
telemetry, a fitted model split weight arbitrarily among redundant predictors,
and multivariate transfer to humans collapsed. Distinct per-kind costs break
the algebraic identity, so speed and efficiency can vary independently.

## State space

Serialized as flat row-major arrays of length `H·W` (row `r`, column `c` is
index `r·W + c`). Mirrored in `schema.json` `$defs/state`.

| field | dtype | shape | bounds | meaning |
|---|---|---|---|---|
| `mines` | int8 | [H·W] | {0,1} | mine layout. All zero until `generated == 1`. Hidden from the agent. |
| `revealed` | int8 | [H·W] | {0,1} | revealed cells |
| `flagged` | int8 | [H·W] | {0,1} | flagged cells |
| `generated` | int8 | [] | {0,1} | 1 once the first `reveal` has placed mines |
| `first_cell` | int16 | [] | [−1, H·W−1] | cell index of the first `reveal`; −1 before generation |
| `dead` | int8 | [] | {0,1} | a mine has been revealed |
| `won` | int8 | [] | {0,1} | every non-mine cell is revealed |
| `time_ms` | int32 | [] | [0, HORIZON·(60+2000)] | virtual clock |
| `t` | int64 | [] | [0, HORIZON] | in-episode step counter |

Adjacent-mine counts are **not** state: `count(i)` is derived from `mines` as
the number of mines among cell `i`'s ≤8 neighbours, recomputed wherever needed.
Storing it would be a second copy of the same fact and a chance for the two
implementations to disagree.

`time_ms` is state rather than derived because per-kind costs make it a
function of the action *history*, not of `t`.

## Actions

`3 · H · W` discrete actions. With `K = H·W`, action `a` decodes as
`kind = a // K`, `cell = a % K`:

| kind | range | name | semantics |
|---|---|---|---|
| 0 | `[0, K)` | `reveal` | reveal `cell` (see Reveal sequence) |
| 1 | `[K, 2K)` | `flag` | toggle the flag on `cell` |
| 2 | `[2K, 3K)` | `chord` | mass-reveal around `cell` |

Every action is legal at every step. An action whose preconditions do not hold
is a **no-op that still costs its time** — the clock advances and `t`
increments. Wasted clicks are the point: efficiency is defined as work done per
click, so an env that silently discarded useless clicks could not measure it.

**`flag`**: if `revealed[cell]`, no-op. Otherwise `flagged[cell] ^= 1`.

**`reveal`**: if `revealed[cell]` or `flagged[cell]`, no-op. Otherwise run the
Reveal sequence on `cell`.

**`chord`**: no-op unless *all* of: `revealed[cell]`, `count(cell) > 0`, and the
number of flagged neighbours of `cell` equals `count(cell)`. When it applies,
run the Reveal sequence on every neighbour that is neither flagged nor
revealed, in **ascending cell-index order**. The order is specified because a
chord can uncover a mine: which mine is found first decides nothing about the
outcome (`dead` is set either way) but fixing the order keeps the two
implementations bit-identical.

### Reveal sequence (on cell `i`)

1. If `generated == 0`: place mines (see Reset → Mine placement), set
   `generated ← 1` and `first_cell ← i`.
2. Set `revealed[i] ← 1`.
3. If `mines[i] == 1`: set `dead ← 1` and stop.
4. If `count(i) == 0`, flood: repeat until a full pass changes nothing —
   for every cell `j` with `revealed[j] == 1` and `count(j) == 0`, set
   `revealed[k] ← 1` for every neighbour `k` with `flagged[k] == 0`.

The flood runs to a fixpoint rather than for a fixed number of iterations, so
both implementations agree regardless of region shape. Flagged cells are never
revealed by the flood, matching standard play.

## Observations

`float32`, shape `[11, H, W]`, every value exactly `0.0` or `1.0`. Planes are
built as 0/1 integers and cast to `float32` once, at the end. There is no float
arithmetic anywhere, so no `x-atol` is declared.

| channel | contents |
|---|---|
| 0 | unrevealed and unflagged |
| 1 | flagged |
| 2–10 | revealed with `count == 0 … 8` respectively (one-hot) |

`mines` is never observable. A revealed mine appears only through `dead`, and
the episode ends there.

## Rewards

Integer-valued `float32`. Per step:

```
reward = −(COST[kind] + LATENCY) + 1000000·won_this_step − 1000000·dead_this_step
```

Maximizing return is **minimizing virtual time to clear the board**, with death
strictly worse than any amount of dithering: a win yields `+1000000 − elapsed`,
and the largest possible elapsed cost is `HORIZON·(60 + 2000) = 4120000`, so a
death at `−1000000` can never be preferable to surviving to the horizon.

Every quantity is an integer below `2^24`, so `float32` represents them
exactly and the differential test requires bit-equality on every field.

## Termination

| id | condition | when checked |
|---|---|---|
| T1 | `dead == 1` — a mine was revealed | after the action phase |
| T2 | `won == 1` — every cell with `mines[i] == 0` has `revealed[i] == 1` | after the action phase |
| T3 | `t == HORIZON` | after `t` increments |

`won` is evaluated only when `generated == 1`; an ungenerated board is not a
win even though it vacuously has no unrevealed safe cells.

Episodes terminate readily under random actions: a uniformly random action is a
`reveal` one third of the time, and with 99 mines in 480 cells an unlucky
reveal arrives quickly, so the auto-reset test sees terminations well inside
the horizon.

## Reset

`mines ← 0`, `revealed ← 0`, `flagged ← 0`, `generated ← 0`, `first_cell ← −1`,
`dead ← 0`, `won ← 0`, `time_ms ← 0`, `t ← 0`.

Reset draws `H·W` values from slot `MINE_KEYS` at step 0, one per cell
(`index = cell`). The board is **not** laid out yet — only the keys that will
decide it are fixed.

### Mine placement (at the first `reveal`, on cell `i`)

1. `excluded` = the closed 3×3 neighbourhood of `i` (cell `i` plus its ≤8
   neighbours), clipped at the board edges.
2. `eligible` = every cell not in `excluded`. `|eligible| ≥ H·W − 9 ≥ M`.
3. `mines` = the `M` eligible cells with the smallest `MINE_KEYS` values, ties
   broken by smaller cell index (a *stable* ascending sort gives exactly this).

**The key is masked to its low 32 bits, and that is load-bearing.**
`draw_bits` returns a uint64 as a Python int in `[0, 2^64)`, while its torch
counterpart returns the same 64 bits reinterpreted as *signed* int64 — the
value prints negative whenever bit 63 is set. Sorting those two ascending
produces different orderings for half the key space, so the two
implementations would place mines differently while both "using the same
draw". The low 32 bits are identical under either reading, so masking removes
the ambiguity. Over 480 cells a 32-bit key collides with probability ~3e-5 per
episode, and the index tie-break resolves collisions identically on both
sides.

Drawing the keys at reset but *applying* them at the first click makes the
board a deterministic function of `(seed, first_cell)` — reproducible, and
independent of how many flags were placed beforehand.

Excluding the full 3×3 means `count(first_cell) == 0` by construction, so the
first reveal always opens a region. That is the "guaranteed opening" rule, and
it removes first-move luck from a measurement that is supposed to be about
judgement.

## Invariants

Hold in every reachable state, including terminal states.

1. `revealed[i] == 1` implies `flagged[i] == 0` — no cell is both.
2. `generated == 0` implies `mines` is all zero, `revealed` is all zero, and
   `first_cell == −1`.
3. `generated == 1` implies exactly `M` entries of `mines` are 1, and
   `0 ≤ first_cell < H·W`.
4. `generated == 1` implies `count(first_cell) == 0` — the opening guarantee.
5. `dead == 0` implies no revealed cell is a mine.
6. `dead == 1` implies at least one revealed cell is a mine.
7. `won == 1` implies every non-mine cell is revealed, and `generated == 1`.
8. `won` and `dead` are never both 1.
9. `t == 0` implies `generated == 0`, `time_ms == 0`, and `flagged` all zero.
10. `0 ≤ t ≤ HORIZON`.
11. `0 ≤ time_ms ≤ HORIZON · (60 + LATENCY)`, and `time_ms` never decreases.
12. The number of flagged cells is at most `H·W` minus the number of revealed
    cells (flags and reveals are disjoint by 1, so this is a count check on the
    same fact and catches mask corruption).
13. `generated == 1` and `dead == 0` and `won == 0` implies at least one
    non-mine cell is unrevealed.

## RNG slots

| slot | name | used at | distribution |
|---|---|---|---|
| 0 | `MINE_KEYS` | reset (step 0), `index = cell` for each of `H·W` cells | `rng.draw_bits(...) & 0xFFFFFFFF` — the low 32 bits |

One stochastic decision only: the mine layout. It is realised as a key per cell
rather than as a sample of `M` positions because "take the `M` smallest keys
among eligible cells" is a pure function of the key vector and the exclusion
set — order-independent, trivially vectorizable, and identical in both
implementations. Sampling `M` positions without replacement would need a
sequential procedure whose result depends on draw order, which is exactly the
kind of rule that diverges between a scalar and a batched implementation.

No per-step draws: after reset the environment is deterministic given the
action sequence.
