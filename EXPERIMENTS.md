# Experiments

Working plan. The project's question is whether a player's skill can be read
from how they play before their results say anything, and whether that can be
learned from agents rather than from people.

There are two tracks and **both are committed**, regardless of how either turns
out:

- a **lightweight track** — can skill be read from cheap, domain-general
  aggregate metrics that most real-time games already expose?
- an **agent track** — can a ladder of trained agents, built with no human data
  at all, predict human skill, and how early?

These answer different questions. The lightweight track asks whether a free
prior exists. The agent track asks *why* skill is legible — which properties of
a game make it legible early — and reaches move-level resolution that aggregate
metrics cannot. A working heuristic does not make the agent work redundant; it
makes it **measurable**, because the heuristic becomes the baseline the agent
approach has to beat. A result of "58% from three rounds" means nothing on its
own and a great deal against a known simple-baseline number.

Ordering principle: cheapest first, so the baseline exists before the work it is
meant to calibrate, and so results land early. Nothing below is sequenced on the
assumption that an earlier result cancels a later one. The two tracks also
interleave well in practice — the lightweight track is mostly waiting on network
I/O, the agent track is mostly compute.

A hard constraint runs through all of it: **nothing that requires the target
game's human data may be used to build the method.** If the simulator has to be
configured from real player telemetry, the cold-start problem has been relocated
rather than solved, because at an actual launch that telemetry does not exist.
Human data is for testing, once, at the end.

| # | experiment | track | needs |
|---|---|---|---|
| E1 | cross-game speed/efficiency heuristic | lightweight | public data only |
| E2 | handicapping vs undertraining | agent | nothing new — runs today |
| E3 | chess transfer sanity check | agent | Lichess dumps |
| E4 | minesweeper, per-move | agent | scraper, solver, ladder |
| E5 | TETR.IO simulation | agent | versus sim, attack table |
| E6 | baseline vs agent, head to head | both | E1 + E4/E5 |
| E7 | deployment framing | both | E6 |

---

## Settled so far

Four games are built end to end through one pipeline — Connect Four, Othello,
backgammon, Leduc Hold'em — with measured ladders and per-game telemetry. See
the README for results. Four findings carry forward because they will recur:

1. **A ladder has to be checked for being a ladder.** One-step targets over long
   episodes produced a backgammon ladder spanning 32 Elo with the untrained
   network ranked third. Nothing errored. `league.monotonicity` caught it.
2. **The scoring metric must match what the game settles.** Counting hands won
   would have ranked a fold-everything Leduc policy first.
3. **Bounds should be derived, not sampled.** Leduc's pot maxes at 13, not the
   11 a few hundred random hands show.
4. **Priors about which behaviours carry skill are unreliable.** Of three
   features cut for "not earning their place", one — backgammon's `pip_gain` —
   turned out to be the strongest early signal in the game. Inside a fully
   specified game. This is the argument for measuring rather than asserting, and
   it applies with more force to humans.

---

## E1 — Is the speed/efficiency heuristic domain-general?

**Question.** Most real-time games expose something like an actions-per-minute
metric and something like an efficiency metric. Does the mapping from those two
numbers to skill have a common shape across games? If it does, it is a free
cold-start prior for most real-time games, needing no simulation and no
per-game feature engineering.

**Why first.** No RL, no environment, no training — pure public data collection
and regression, so it produces a result quickly. Its output is the **baseline
every later experiment is scored against**: whatever E4 and E5 achieve on a
given game, the number that matters is how much they beat this. It also supplies
the speed-accuracy coupling that E5's ladder needs, from games that are not the
target.

**Method.**
1. For each game, pull rank-labelled records exposing a speed axis and an
   efficiency axis.
2. Standardise within game (rank → percentile; each metric → z-score within
   game) so the comparison is of *shape*, not units.
3. Fit skill ~ f(speed, efficiency) per game.
4. The actual test: fit on game A, predict rank percentile on game B. Repeat
   over all ordered pairs. A heuristic that transfers is one where the
   off-diagonal barely degrades.

**Candidate games and their two axes:**

| game | speed axis | efficiency axis | label | status |
|---|---|---|---|---|
| TETR.IO | PPS | APP = APM/(PPS·60) | TR (Glicko-2) | confirmed, public API |
| Minesweeper | 3BV/s | efficiency % (clicks vs 3BV) | rank ladder | confirmed, ranked category |
| Jstris | time / blocks | finesse | leaderboard position | confirmed, public API |
| StarCraft II | APM | spending quotient | league | SkillCraft1 + replays |
| osu! | pieces… n/a | accuracy, unstable rate | pp | public API |
| Rocket League | boost/min, speed | rotation, first-touch | rank | ballchasing.com |

Minesweeper is the most valuable entry because it is *not* a versus game and not
real-time in the same sense — if the relationship holds there too, it is a
statement about skill rather than about action games.

**Outcomes, both useful.** If cross-game prediction holds up, there is a free
prior applicable to most real-time games — a standalone result, and a demanding
baseline for the agent track. If it collapses to chance, the free-lunch idea
dies cleanly and the per-game work becomes the only route, with the null result
worth reporting. Neither outcome changes whether E4 and E5 happen; it changes
what their numbers mean.

**Watch for.** Within a person, speed and accuracy trade off; *across* people
they correlate positively. The cross-person direction is the one being tested,
and mixing the two would produce a sign error that looks like a null result.

### Result: the shape does not transfer. The axes are useful anyway.

Three games ingested: **TETR.IO** (450 players, 25,707 rounds, all 18 ranks),
**SkillCraft1** (3,395 rated StarCraft II players) and **Lichess** (6,152 blitz
games at 180+0, 12,304 player-rows, speed from clock deltas and efficiency from
Stockfish annotations). Transfer matrix, Spearman ρ against each game's own
rating, ridge on two standardised axes:

|  | → lichess | → skillcraft | → tetrio |
|---|---|---|---|
| fit on lichess | 0.355\* | 0.436 | 0.927 |
| fit on skillcraft | 0.073 | 0.658\* | 0.911 |
| fit on tetrio | 0.317 | 0.606 | 0.974\* |

\* within-game, cross-validated grouped by player.

**The ablation kills the strong reading of that table.**

| feature set | →lichess | →skillcraft | →tetrio |
|---|---|---|---|
| speed only | 0.102 | 0.661 | 0.926 |
| efficiency only | 0.358 | 0.436 | 0.927 |
| **no fit at all** | **0.102 / 0.358** | **0.661 / 0.436** | **0.926 / 0.927** |

The last row ranks players by the z-scored axis with no model whatsoever. It is
**numerically identical** to the fitted single-axis transfer scores, and the
fitted scores are identical regardless of which game they were fitted on — every
column above is constant across source games. After within-game standardisation
there is nothing left for a one-axis model to carry, so single-axis "transfer" is
not transfer; it is the target game's own correlation wearing a costume.

That leaves the two-axis combination as the only place a real learned
relationship could live, and it **fails in five of six ordered pairs**:

| pair | both | best single axis |
|---|---|---|
| lichess→skillcraft | 0.436 | 0.661 |
| lichess→tetrio | 0.927 | 0.927 |
| skillcraft→lichess | 0.073 | 0.358 |
| skillcraft→tetrio | 0.911 | 0.927 |
| tetrio→lichess | 0.317 | 0.358 |
| tetrio→skillcraft | 0.606 | 0.661 |

Within a game the pair is complementary — TETR.IO scores 0.974 against 0.926 for
speed alone. Across games, importing the weighting costs accuracy every time.

**Why: the two axes are not equally portable.** With four games the asymmetry is
clear.

| game | speed | efficiency |
|---|---|---|
| chess (fixed time control) | 0.102 | 0.358 |
| NetHack | 0.173 | **0.654** |
| StarCraft II | **0.661** | 0.436 |
| TETR.IO | 0.926 | 0.927 |

**Efficiency — output per action — is the portable axis.** It is positive and
useful in all four games. **Speed is the game-specific one**: near-worthless in
chess at a fixed time control and in untimed NetHack, dominant in StarCraft,
interchangeable with efficiency in Tetris. That fits the obvious mechanism —
actions-per-second can only matter where the game applies time pressure.

And importing the wrong emphasis is actively destructive rather than merely
unhelpful. SkillCraft is speed-dominant, so a SkillCraft-fitted model
over-weights speed; applied to NetHack, where speed is worth 0.173 and
efficiency 0.654, the pair collapses to **0.158** — four times worse than
ignoring SkillCraft entirely and using NetHack's own efficiency axis.

**Verdict.** The strong hypothesis — a transferable speed/efficiency *shape* — is
refuted. What replaces it is more useful than the weak version stated earlier:

> Take the game's output-per-action ratio, standardise it within the game, and
> rank by it. No model, no training, no data from the target game. Add a speed
> axis only if the game applies time pressure, and never import another game's
> weighting between the two.

That is worth 0.93 in Tetris, 0.65 in NetHack, 0.44 in StarCraft and 0.36 in
chess. It is the baseline the agent track must beat, and it is a low bar in
chess and a very high one in Tetris.

### The cold-start curve, on real humans

NetHack is the only corpus here with many games per identified player, so it is
the only place the actual question can be asked without a proxy: **predict how
good somebody turns out to be, from the first games they ever played.** Career
labels are held out *in time* — skill is scored from the second half of a
career and predicted from the first, so the observation window is never part of
what it is predicting.

| games seen | ρ against later career | 
|---|---|
| 1 | 0.303 |
| 3 | 0.369 |
| 5 | 0.409 |
| 10 | 0.497 |
| 20 | 0.547 |

967 careers, 46,660 games. One game buys a third of the correlation that half a
career does. This is the number the agent track is really competing with.

### Also established

The cross-person speed-efficiency correlation the E5 ladder design assumes is
confirmed rather than asserted: **ρ = 0.878** between PPS and attack-per-piece
across players. Slow-but-efficient players essentially do not exist, which is
why E5's ladder must couple its handicap axes rather than grid them.

**Observation budget, TETR.IO, 450 players:**

| rounds seen | ρ | mean error (percentile) |
|---|---|---|
| 1 | 0.900 | 0.086 |
| 3 | 0.939 | 0.072 |
| 10 | 0.965 | 0.055 |
| all | 0.974 | 0.048 |

**One prior did not survive.** `VS / APM` was expected to isolate downstacking
skill, since VS counts garbage cleared as well as attack sent. Across ranks its
mean is flat at ~1.9–2.1 from D to X+; the variation in individual records is
within-rank noise, not between-rank signal.

### Caveats and what is still needed

**Sampling is stratified** to hold ranks equal, which makes the task easier than
a natural population concentrated in the middle ranks, and the percentile label
is computed within that stratified sample. Do not compare these ρ values to one
computed on an unstratified corpus.

**SkillCraft is one row per player**, so it contributes to the shape test but
cannot contribute to the observation-budget curve.

**SkillCraft's efficiency axis is a proxy.** It has no output-per-action measure,
so the axis is hotkey selections per action — efficiency of execution, the direct
analogue of Tetris finesse, but not a measurement of output. Raw
`SelectByHotkeys` correlates 0.82 with APM and would have been speed under
another name; normalising by APM is what makes it an efficiency measure at all.
Some of the weak `tetrio→skillcraft` transfer is likely this proxy rather than
the hypothesis.

**Candidate games must have a skill label not derived from speed.** That rules
out Jstris (a sprint rating *is* the completion time), osu! (pp is computed from
accuracy) and typing tests, all of which would predict skill tautologically.
Rocket League was considered and dropped: its telemetry is positional physics
rather than an action rate, so the speed axis would not mean the same thing, and
replay parsing is a project in itself.

**Minesweeper remains the best fourth entry**, because `efficiency %` is
literally board value per click — output per action, measured rather than
proxied — and the rank ladder is a third quantity. That makes E4's scraper a
dependency of E1 as well.

---

## E2 — Handicapping versus undertraining, in the games already built

**Question.** A ladder can be generated by catching one agent mid-training, or
by taking one strong agent and degrading it — restricted lookahead, added
reaction delay, a bounded view, an action-rate cap. Do the two produce the same
telemetry-versus-skill relationship?

**Why second.** It needs no new game, no new data and no permission: it runs
today against the existing Connect Four ladder. And it gates the design of every
later ladder, including TETR.IO's.

**Why it matters.** Undertrained networks are *uniformly* bad — they err at
similar rates everywhere. Humans are non-uniformly bad: strong at drilled
patterns, weak elsewhere. So an undertrained ladder may vary along one axis
where humans vary along several, and a classifier trained on it learns a
one-dimensional notion of badness.

**Method.** Build a second Connect Four ladder by handicapping the strongest
checkpoint — cap search/lookahead, inject reaction delay, restrict the observed
board — with rungs chosen so measured Elo matches the existing ladder's rungs.
Then compare, at matched Elo: per-tier feature means, the accuracy-versus-N
curve, and the classifier's confusion structure.

**Prediction to write down now.** The handicapped ladder will show *more*
structured errors — blunders concentrated in specific position types rather than
spread — and its accuracy-versus-N curve will rise more slowly, because the
signal is in which mistakes are made rather than how many.

**What it unblocks.** Which generator to use for TETR.IO, and whether the
existing four-game results are an artefact of the generator.

---

## E3 — Chess as the transfer sanity check

**Question.** The load-bearing one: can features derived from agents rank
*humans*? This is the cheapest place to find out, on the richest data.

**Why third.** Highest stakes, low cost, and the most favourable possible
conditions — if it fails here it fails everywhere, and that is worth knowing
before E4 and E5 are built.

**Method.** Lichess publishes monthly database dumps: every rated game, with
per-move clock times. Port the existing tactical-feature vocabulary, train on an
agent-derived ladder, test against rating-labelled human games. Report manifold
overlap *before* accuracy — whether the agent tiers even occupy the same region
of feature space as the human bands.

**Benchmarks.** "Chess Rating Estimation from Moves and Clock Times Using a
CNN-LSTM" (arXiv 2409.11506) does per-move rating prediction on humans directly;
Maia (McIlroy-Young et al.) found engine-strength-matched agents predict human
moves badly, which is close to this project's null hypothesis and should be
treated as the result to beat rather than a curiosity.

**Prediction to write down now.** Transfer will *compress*: humans will pile
into middle tiers rather than spreading across them, because the agent ladder
varies along fewer axes than people do.

---

## E4 — Minesweeper, for the per-move question

**Question.** The original thesis at move resolution, in a game where per-action
human data is actually obtainable and the key feature is decidable from the
rules alone.

**Why here.** Minesweeper has the cleanest analogue of `missed_win` of anything
surveyed: *was there a provably safe cell, and did the player click one?* That
is settled by a solver, not by a network's opinion, so no yardstick-independence
argument is needed. Efficiency (clicks versus the board's 3BV) is a second
rules-defined measure.

**Method.**
1. Solver providing the oracle: given a board state, which cells are provably
   safe, provably mines, or genuinely ambiguous.
2. Features: guessed-when-a-deduction-existed, flagged-a-safe-cell, efficiency,
   3BV/s, time-to-first-guess, chord usage.
3. Agent ladder — undertrained and/or handicapped per E2's verdict.
4. Human corpus from minesweeper.online.

**Compliance.** `robots.txt` allows everything needed (only `/chat`,
`/chat-history`, password-reset and invoice paths are disallowed); the site has
no Terms of Service, only a Privacy Policy, which is silent on automated access.
The scraping prohibitions found during research belong to `minesweeper.org` and
`minesweeper-online.org`, unrelated lookalike domains. Contact is
`support@minesweeper.online`; worth emailing to ask about an export and a
preferred request rate. Pages are JS-driven, so ingestion targets internal XHR
JSON endpoints, which must be discovered first.

**Handling.** Hash player IDs at ingest. The pipeline needs a stable key and a
skill label, never a username. Publish aggregates only.

---

## E5 — TETR.IO simulation

**Question.** Can an agent ladder trained entirely offline predict the skill of
real TETR.IO players from their first few rounds?

**Why later.** It is the largest build here, and E1–E3 each sharpen it: E1
supplies the speed-accuracy coupling for the ladder and the baseline to beat, E2
decides how the ladder is generated, E3 says what transfer failure looks like
before it is expensive to discover. Later in sequence, not contingent — this is
the centrepiece of the agent track and it happens whatever E1 returns.

**Available without any replay access.** `GET /users/:user/records/league/recent`
returns per-match records containing **both players' stats broken down per
round**: `apm`, `pps`, `vsscore`, `garbagesent`, `garbagereceived`, `btb`,
`kills`, `lifetime`. A first-to-7 match yields up to 13 sequential observations,
so the accuracy-versus-N curve survives with N as rounds observed. Confirmed
populated across the whole ladder, from a 499 TR player at 9.9 APM to a
24,733 TR player at 216 APM. The label is continuous TR, so no Bradley-Terry and
no tier-picking is needed on the human side.

Replays are *not* available: the documented public API has no download endpoint,
and the main game API is off-limits without written consent, on pain of account
suspension. Do not build on tools that reach it. Asking for research consent is
reasonable and would reopen the per-move question here.

**Derived features worth building first:**
- `APP = APM / (PPS · 60)` — attack per piece. Measured on real records it runs
  ~0.15 at D+ to ~0.89 at X+, roughly a sixfold spread. The efficiency axis.
- `VS / APM` — not redundant with APM: the observed ratio ranges 1.66–2.62. The
  gap is the garbage-cleared term, so this is how much of the work was
  downstacking rather than attacking.
- round-to-round variance of each — consistency is itself a skill marker.

**The correctness gate.** TETR.IO's attack table and Multiplier combo system
(`base × (1 + 0.25 × combo)`) must be implemented exactly, or simulated APM and
VS are not the same quantity as the API's and nothing downstream compares.
Differential-test the table against real records, the way `c4.py` is tested
against PettingZoo. This uses real data for *rule correctness*, not calibration,
and is the only legitimate pre-launch use of it.

**Ladder generation.** Walk a **single coupled skill parameter** that moves
lookahead depth, speed cap and reaction delay together — not an independent grid
over those axes. Gridding them produces agents that are slow but near-perfect, a
combination that does not exist in people. The coupling is justified pre-data by
the cross-person speed-accuracy correlation; its steepness can come from E1.

**Candidate engines.** Cold Clear and Cold Clear 2 (open-source Tetris *versus*
bots, so garbage and attack are already implemented; C API; ~14-move search)
handicapped along the coupled parameter. TetrisBattle is a two-player Gym
environment for the RL route — note its sparse-reward warning, which is the same
credit-assignment wall that produced the flat backgammon ladder, so n-step
targets from the start.

**Open question.** How far back `records/league/recent` paginates. If it reaches
only recent matches, the honest framing becomes "N rounds from an established
account" rather than literal cold start.

---

## E6 — Baseline versus agent telemetry, head to head

**Question.** On a game where both tracks have run, how much does the agent
ladder add over the domain-general heuristic from E1?

This is where the two tracks meet, and it is the experiment that makes the whole
project's claim precise. Three numbers per game, at each observation budget N:
the E1 heuristic alone, the agent-derived features alone, and both together.

The interesting cases are not the obvious one. If the agent features add a lot,
the per-game investment is vindicated. If they add little *on aggregate metrics*
but a lot at **move resolution** (E4's minesweeper features, E5's per-round
`btb` and downstacking signals), that localises where the value is — and says
the agent track's contribution is resolution rather than accuracy. If they add
nothing anywhere, that is a real and publishable finding about how much of
"skill legibility" is just speed and efficiency wearing a costume.

---

## E7 — The deployment question

**Question.** The one a matchmaker actually cares about: given N observations,
how much better is a telemetry prior than the default of starting everyone in
the middle?

Accuracy over tiers is a proxy. The real measure is matchmaking quality —
expected mismatch in the first N games, and how many games it takes a
telemetry-seeded rating to reach the accuracy an unseeded one needs many more to
reach. Worth stating in those terms because it is the claim the project is
actually making, and because a modest accuracy can still be a large improvement
over no prior at all.

---

## Cross-cutting risks

**Off-manifold agents.** The central threat. Agent tiers may occupy regions of
feature space no human occupies. It is a hypothesis to test, not something to
engineer away pre-data — so the first thing reported at every transfer test is
distribution overlap, before any accuracy number.

**Compression.** Agent ladders likely vary along fewer axes than human skill
does. Predicted symptom: humans collapse into middle tiers. Written down in
advance so it counts as a prediction.

**Non-independence within a session.** In TETR.IO, rounds of one match are not
independent — a player who is losing eats more garbage and their stats shift.
Group cross-validation by *match*, not only by opponent.

**Generator artefact.** Everything currently known about telemetry-versus-skill
comes from undertrained checkpoints. E2 exists to find out how much of it is a
property of that choice.

**Personal data.** Player records are personal data regardless of what
`robots.txt` permits. Hash identifiers at ingest; publish aggregates only; keep
no usernames in any exported artefact.

---

## Data source status

| source | access | granularity | label | compliance |
|---|---|---|---|---|
| TETR.IO | public API, no auth | per round | TR, rank | documented; ~1 req/s, honour cache |
| TETR.IO replays | **prohibited** without written consent | per keystroke | — | do not use; ask first |
| Minesweeper.online | scrape (JS/XHR) | per click | rank | robots.txt clear, no ToS |
| Jstris | public API | per record | leaderboard | open, undocumented terms |
| Lichess | bulk dumps | per move + clock | rating | published for research |
