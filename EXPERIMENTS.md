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
