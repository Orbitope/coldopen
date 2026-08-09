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

**Minesweeper remains a good further entry**, because `efficiency %` is literally
board value per click — output per action, measured rather than proxied — and the
rank ladder is a third quantity. An email to `support@minesweeper.online` asking
about an export is outstanding.

**The Atari Grand Challenge dataset is not currently obtainable.** It would have
been the best entry available — five games collected under one protocol, which
is the only way to remove the "different game *and* different collection method"
confound that every pair in the table above still carries. But
`atarigrandchallenge.com` has been re-registered as an unrelated content site and
the `/data` path is gone; the [code repository](https://github.com/yobibyte/atarigrandchallenge)
survives but holds collection code, not data. A HyperAI mirror and a
[processor repo](https://github.com/Rowing0914/Atari-Grand-Challenge-Processor)
are untried leads. Worth knowing before chasing them: the trajectory format is
`(episode, frame, reward, score, terminal, action)` with **no player identifier**,
so it would support the cross-game shape test and a within-episode budget curve,
but not the per-career question NetHack answers.

### Where this stands

E1 is answered well enough to write up. The strong hypothesis is refuted, the
replacement rule is stated above and holds across four games spanning real-time
and turn-based, timed and untimed. The remaining work — a fifth game, the
single-protocol Atari corpus — would sharpen the estimate rather than change the
conclusion.

The agent track (E2 onward) is untouched and now has its baselines: **0.93 in
Tetris, 0.65 in NetHack, 0.44 in StarCraft, 0.36 in chess**, plus the NetHack
cold-start curve of 0.303 from a single game. E2 remains the cheapest next step
and still runs against the Connect Four ladder that already exists.

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

### Result: the generator matters more than the game does

Three handicaps applied to the strongest Connect Four checkpoint, each swept to
span a comparable Elo range, measured by the same league and profiled by the same
classifier:

- **epsilon** — throw the move away at random. The control, and the one computer
  chess already knows feels least like a human, because the errors are
  uncorrelated with the position.
- **temperature** — sample from a softmax over action values, so mistakes stay
  *ordered*: a slightly worse move is much likelier than a disastrous one.
- **blindspot** — hide a band of the board before the network looks. Models
  attention rather than decision noise.

Five generators, all through one code path, with **tiers pinned to the same six
ratings** so that tier spacing cannot explain the difference. An earlier version
matched only the total Elo range, which let the trained ladder's tiers sit 1.4×
further apart — the exact confound this document raises about Leduc and then
failed to apply to itself. Corrected, the finding is stronger, because the
trained ladder now has one of the *narrowest* gaps and still wins:

| generator | tier gap | concordance | n=1 | n=3 | n=5 | n=20 |
|---|---|---|---|---|---|---|
| **trained** (undertrained RL) | 121 | 0.912 | **0.522** | 0.601 | 0.519 | 0.634 |
| distilled (imitation) | 122 | 0.971 | 0.283 | 0.333 | 0.436 | 0.479 |
| blindspot | 126 | 1.000 | 0.308 | 0.361 | 0.504 | 0.481 |
| temperature | 139 | 0.978 | 0.292 | 0.298 | 0.312 | 0.410 |
| epsilon | 130 | 1.000 | 0.249 | 0.283 | 0.286 | 0.382 |

**Undertrained RL checkpoints are not one generator among five — they are the
outlier.** Every other method lands in a 0.25–0.31 band at one move; the RL
ladder nearly doubles it. The four agree with each other and disagree with it.

The decisive part is **distillation**, which is also a form of undertraining — a
student snapshotted partway to competence — and which behaves like the handicaps
rather than like the RL ladder. So what makes the original ladder legible is not
"partly trained" in general. It is something specific to RL checkpoints.

The likely mechanism: a DQN checkpoint early in training has a systematically
*distorted* value function — it has learned something about the centre column and
nothing at all about blocking — so its blunder profile is a recognisable
fingerprint of what has not been learned yet. A behavioural-cloning student at
60% teacher agreement is wrong in scattered places instead, because what it has
missed is whichever positions were rare in the demonstrations.

That reading is supported by the error concentration, which the earlier writeup
got backwards and which now separates the generators cleanly:

| generator | error concentration, weakest → strongest tier |
|---|---|
| trained | 0.33, 0.34, 0.46, 0.25, 0.46, 0.39 |
| distilled | 0.37, 0.31, 0.24, 0.14, 0.11, 0.12 |
| blindspot | 0.31, 0.36, 0.19, 0.16, 0.12, 0.16 |
| temperature | 0.16, 0.15, 0.12, 0.12, 0.13, 0.49 |
| epsilon | 0.21, 0.13, 0.12, 0.11, 0.17, 0.32 |

The trained ladder's errors stay concentrated **at every strength**. Distillation
and blindspot are concentrated only at the weak end and become uniform as they
improve — which is the more plausible description of a person: a beginner has
systematic gaps, while a strong player's remaining mistakes are idiosyncratic.

**Three curve families**, which is the practical output:

- *trained* — high from move one, stays high
- *distilled and blindspot* — low start, a jump around five moves, plateau ~0.48
- *temperature and epsilon* — low start, slow steady climb to ~0.40

Those are fingerprints. Computing the same two signatures on a human corpus says
which family people belong to, and therefore which generator a ladder should use.
The human side of that needs no agents at all.

It also sharpens the standing E3 prediction: if people are not undertrained RL
networks — and four of five generators say that is a distinctive thing to be —
achievable accuracy from one move is nearer 0.3 than 0.5.

**A secondary finding.** The trained ladder has the *worst* concordance of the
five (0.912 against 0.97–1.00). A dial controls strength directly; training only
correlates with it. If the point of a ladder is graded strength, every other
method here is a better instrument.

### The prediction was wrong, and the reversal is more useful

I predicted handicapping would produce *more structured* errors than
undertraining. Measuring the concentration of each tier's blunder profile —
0 meaning mistakes spread evenly across missed wins, missed blocks and handed-over
wins, 1 meaning they all pile into one kind — gives the opposite:

| generator | error concentration, weakest tiers |
|---|---|
| undertrained | 0.33, 0.34, 0.25, 0.37 |
| blindspot | 0.31, 0.36, 0.19, 0.14 |
| epsilon | 0.21, 0.13, 0.12, 0.11 |
| temperature | 0.16, 0.15, 0.12, 0.12 |

**Undertraining produces the most concentrated errors; noise handicaps produce
the most uniform ones.** Obvious in hindsight: an undertrained network has
systematic gaps — it has learned something about the centre and nothing about
blocking — so its mistakes have a characteristic shape. Noise applied evenly
across positions spreads mistakes evenly across error types.

That reverses the mechanism but strengthens the conclusion. Undertrained agents
are easy to classify early *because* their errors are systematic: one move can
reveal a recognisable profile of what the agent has not learned. Noise-degraded
agents are hard because their errors are unbiased, so you must accumulate
observations to estimate a rate.

**Blindspot behaves differently from both**, and its curve is the giveaway: flat
until three moves, then a jump to 0.536 at five, then a plateau. Attention
failures show up as occasional catastrophic misses, so a few moves reveal
nothing and then one does.

### What this gives the agent track

The three generators have **distinguishable fingerprints** — the shape of the
accuracy curve and the concentration of the error profile. That is a method, not
just a caveat: compute the same two signatures on a human corpus, and the
generator whose fingerprint matches is the one to build the ladder with. It turns
"which failure model are humans?" from a guess into a measurement, and it is
cheap because the human side needs no agents at all.

Neither pure generator is obviously right. Human beginners have systematic gaps,
like an undertrained network, *and* noisy execution, like a temperature policy.
A ladder combining both is the likely answer, and now there is a way to check.

**A secondary finding.** All three handicapped ladders rank at least as well as
the trained one — concordance 1.000 for epsilon and blindspot, 0.978 for
temperature, against 0.912 — because a handicap dial controls strength directly
while training only correlates with it. If the point of a ladder is graded
strength, handicapping is the better instrument.

**A limit worth recording.** Blindspot cannot reach the bottom of the ladder:
blind to most of the board it still scores about 0.80 against random, because a
*consistent* policy beats random play whatever it is consistent about. That is
this project's opening trap — an untrained network beats random 85% of the time —
reappearing as a floor on how bad attention-limitation alone can make you.

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

### Scoped down: single-player sprint, not versus

The versus engine is the expensive part — SRS-X kick tables, T-spin corner
rules, garbage cancelling order, the multiplier combo system, the eight-line
garbage cap — and it is days of work in which a subtle error silently voids the
comparison rather than raising one.

It is also not needed. A **40 LINES** record already carries everything the
experiment wants:

```
pps 7.088   inputs 256   piecesplaced 103   holds 13
clears: singles 7, doubles 5, triples 1, quads 5   topbtb 5
```

`inputs / piecesplaced` is **finesse** — keystrokes spent per piece against the
minimum the placement needed. It is rules-derived, needs no opponent, no attack
table and no engine, and it is the same class of feature as `missed_win`:
decidable from the rules, so no agent's opinion can leak into it.

Measured across the ladder (3 players per rank, 47 of 54 have a sprint record,
coverage at every rank):

| rank | inputs/piece | quad rate |
|---|---|---|
| d | 6.71 | 0.14 |
| c | 4.23 | 0.35 |
| b+ | 3.52 | 0.44 |
| s+ | 3.20 | 0.36 |
| x | 2.86 | 0.60 |
| x+ | **2.60** | **0.76** |

Finesse spans 2.6×, quad rate 5×, both monotone.

**The design this permits avoids every trap hit so far:**

- **Label from versus, features from sprint.** Skill is league TR, earned in a
  different game mode from the one the features come from, so the label cannot
  be a restatement of the features. This is what disqualified Jstris, where a
  sprint rating *is* the completion time.
- **No yardstick.** Finesse and quad rate need only the rules, so the inverted
  reference that broke Othello cannot occur.
- **Features fire from the first piece**, so the window mismatch that broke
  Othello cannot occur either.

**What is given up** is the garbage and versus dynamics, and specifically
`VS / APM` — the downstacking signal. E1 already measured that as flat across
every rank, carrying no information beyond APM. The part sacrificed is the part
already shown to be inert.

**One design consequence.** The simulator must be **input-level, not
placement-level**: finesse only exists if the agent emits keystrokes rather than
teleporting pieces into place. That also makes the handicap axis natural, since
"does not know finesse" is exactly "spends more keystrokes than the placement
needs".

**Reference implementations** for the parts that are easy to get subtly wrong:
[MisaMino-Tetrio](https://github.com/chouhy/MisaMino-Tetrio) already carries
TETR.IO's attack table and garbage cap,
[OpenSolver](https://github.com/MochBot/OpenSolver) is an open-source TETR.IO
engine, and MinusKelvin's Tetris Bot Protocol makes boards and bots
interchangeable. All are C++/JS, so they are references rather than drop-ins.

**Validation gate**, in the same discipline as `c4.py` against PettingZoo:
differential-test the simulator's computed statistics against real API records
before trusting any comparison built on them.

### Status: simulator built and validated

`envs/tetris_sprint/` — spec, readable reference, batched torch implementation,
full simulacrum battery **PASS** (bit-exact differential over 8 seeds × 300
steps, batch independence, twelve invariants swept, auto-reset, determinism,
replay, compiled-parity).

The design changed once during speccing, for the better: instead of a timeless
input-level game, the env runs a **virtual clock** — every keystroke costs a
declared number of integer milliseconds plus a per-agent LATENCY parameter, and
gravity (1 cell/s) and guideline lock delay (500 ms, 15 move-resets) run on
that clock. Consequences: sprint time and PPS are real quantities for agents;
the reward (clear bonus − elapsed ms) *is* the human sprint objective; the
speed/accuracy coupling emerges from physics (a high-latency agent should plan
more per keypress); and hesitation-drift misdrops — the weak-human error the
timeless design could not produce — exist. The whole env is integer arithmetic:
no `x-atol` anywhere, bit-exact everywhere.

One spec bug was found by the invariant sweep, not by the differential test —
both implementations had faithfully implemented the same broken rule (a piece
lifted off a ledge at the lock-delay reset cap kept stale lock delay). Fixed in
the spec first, then in both implementations. That is the division of labour
working as designed: differential testing catches divergent readings, invariant
sweeps catch consistent misreadings.

Throughput, measured honestly: the harness's headline "1× vs reference" is an
artifact — its reference loop skips observation building, which training needs
every step. With observations included the reference does 13.8k steps/s and the
batched env 63k eager / 74k compiled at n=1024: a **~5× training-shape
speedup**, dispatch-bound on CPU (the fixed-trip ghost/DAS/kick/physics probes
are ~500 small tensor ops per step). Not the 100× of Connect Four, and not a
bottleneck either: the c4 ladder trained at ~7k plies/s wall rate, an order of
magnitude below what this env sustains. If it ever binds, the known lever is
the 24-trip ghost probe (cache it between step and observe, or height-map the
no-overhang case).

(The battery claim above was re-verified later, after the committed report
turned out to be a benchmark-only artifact — see *A process failure worth
recording* below. It does pass: 12 tests, overall PASS.)

What came next, and how much of it was wrong: **LATENCY as the coupled dial
did not work** — it is a pure speed dial, producing slow-but-flawless players
that do not exist among humans — and **distillation from the strong sprint
policy did not work either**, at either the keystroke or the placement level
without DAgger. What did work was interpolating the teacher's *objective*
between expert and beginner strategy. The rest of E5 is that story.

### The human target, measured

396 best-40L records across all 18 ranks (~22 per rank), via the documented
public API. Every feature is monotone or near-monotone in rank:

| rank | in/pc | quad rate | holds/pc | max b2b | pps | time |
|---|---|---|---|---|---|---|
| d | 4.86 | 0.03 | 0.070 | 0.6 | 0.84 | 134.6s |
| c+ | 4.48 | 0.22 | 0.136 | 2.8 | 1.00 | 109.8s |
| b+ | 4.06 | 0.43 | 0.185 | 4.5 | 1.41 | 78.0s |
| s+ | 3.49 | 0.49 | 0.272 | 5.8 | 2.40 | 43.6s |
| x+ | 2.81 | 0.69 | 0.305 | 8.0 | 6.01 | 17.1s |

An unpredicted find: **hold usage rises ~4× with skill**. Holds-per-piece is a
pure planning signal — the hold slot is free to use but only useful if you are
thinking ahead — which makes it a fifth human-comparable feature with no speed
component. This table is the manifold the agent ladders must land on, and the
overlap check against it comes before any accuracy claim (the Othello lesson).

### The overlap check was measuring the wrong thing

The first agent-versus-human comparison reported every generator as "100%
inside the human range" on quad rate, hold usage and B2B. All three were
false, and the metric was the reason: it asked whether agent values fell
between the human p2 and p98, and the *pooled* human quad rate spans 0.00 to
1.00. An agent pinned at 0.00 — below every human alive — passes that test.
A range check cannot detect degeneracy inside a wide range.

Replaced with two questions that have answers:

* **coverage** — map each rung's value to the human rank that produces it
  (`rank_of` in `coldopen/sprint_ladders.py`). Values outside the human span
  return no rank at all, which is reported as `off_manifold_share` rather than
  silently clamped.
* **coherence** — for one rung, do all five features agree about which rank it
  is? Reported as the spread of implied ranks. A real player is one player;
  an agent that implies rank 16 on one feature and rank 0 on another is a
  shape nobody has, however well each feature scores alone.

Coherence is the sharper instrument, and it is not Tetris-specific — any
multi-feature agent-versus-human comparison can run it. On the v1 sprint bot
it read **8.0 ranks of median disagreement**: finesse spanning human ranks
0–16 while quad rate, B2B and hold usage sat at rank 0 or below it.

### The v1 teacher's strategy was not a human strategy

The gap was in the objective, not the search. `W_LINES = 6.0` paid for any
immediate clear, so the bot cashed singles the moment one appeared and never
stacked four rows. Its finesse was excellent and its stacking was a beginner's
— exactly the incoherent shape coherence was built to catch.

One further fault surfaced on reading the code: **no hold logic**, leaving
holds-per-piece at 0.0, below the weakest human rank's 0.07.

> **Correction.** This section previously also claimed that column 9 was
> unreachable, because the search ran `x in range(-2, 9)` and a piece whose
> cells all sit at `dx = 0` would need `x = 9`. **That was wrong, and it was
> written up before it was checked.** `_CELLS` uses a bounding-box convention
> in which every `dx >= 0`, so `x` is the box's left edge rather than a cell
> offset: a vertical I has `dx = 1` or `2` and reaches column 9 at `x = 8` or
> `x = 7`, both inside the original sweep. Verified directly — widening the
> sweep to 11 added zero legal placements for all seven pieces, since the
> largest legal `x` is 8. The sweep has been restored to `range(-2, 9)`, and
> `tests/test_sprint_ladder_metrics.py` now pins the property that actually
> matters (every column is covered, and a vertical I can fill column 9 alone)
> rather than a claim about a loop bound. **The entire quad improvement came
> from the objective, not the search space.**

v2 keeps a well at column 9, pays `W_QUAD` for four-row clears, charges for
partial clears while the stack is safe, and holds when the swap scores
materially better. A danger valve re-enables partial clears above
`DANGER_HEIGHT`, without which the bot defends the well to the death.

That valve's price is the one parameter that mattered. Swept over
`DANGER_HEIGHT × W_PARTIAL × W_WELL_FILL × W_HOLES`:

| well-fill penalty | lines | finish | quad rate |
|---|---|---|---|
| −14 | 10.2 | 0.00 | 0.62 |
| −6 | 41.4 | 1.00 | 0.61 |

At −14 the bot would rather top out than fill its own well; at −6 it finishes
every sprint with the same quad rate. The strategy was never the problem — the
refusal to abandon it was.

### Where the cold-start line runs, and where it was crossed

The rule for this project is that nothing requiring the **target game's**
human data may be used to *build* the method; the human records are a test
set, read once at the end. Fixing the teacher put real pressure on that rule,
so it is worth stating exactly where the line falls.

Legitimate, and kept:

* **Playing for quads at all.** Well-and-quad play is standard, widely
  documented Tetris strategy — it is in the game's own tutorials. It does not
  require anyone's telemetry to know about.
* **Making column 9 reachable and adding hold.** Bug fixes. A bot that cannot
  place a vertical I in the rightmost column is broken as a Tetris player,
  independent of what humans do.
* **Capping evaluation noise so every rung finishes.** An agent-side
  criterion: a ladder whose lower rungs cannot complete the task is measuring
  survival, not skill.

Crossed, and reverted:

* `HOLD_MARGIN` set to 0.3/4.0 *because* human hold usage never falls below
  0.070/piece.
* `strategy = skill ** 0.7` *because* a linear blend put quad rate at 0.265
  where the rank-matched humans sit at 0.487.

Both were fitted to the 396-record table and both are gone; the constants now
in `coldopen/sprint_bot.py` are round guesses, under a comment banner saying
so. The fitted values are recorded here rather than deleted, because the gap
between the two is itself the measurement: it is how much of the manifold
overlap is real and how much would have been manufactured.

This matters more than a tidiness point. Tuning an agent ladder until its
telemetry matches human telemetry and then reporting the overlap is circular —
it proves the tuning worked, not that the method transfers. The Othello
failure (E6a) is only informative *because* nothing had been fitted to it.

### The cold-start ladder against real players: the first honest reading

Seven rungs, no constant fitted to human data, measured against all 396
records. **Coverage** first, since that is the gate Othello failed:

| feature | agent range | looks like human ranks | off-manifold |
|---|---|---|---|
| inputs per piece | 3.29 – 4.38 | c+ … u | 0% |
| quad rate | 0.07 – 0.69 | d+ … x+ | 0% |
| holds per piece | 0.04 – 0.24 | d … s− | 0% |
| max B2B | 0.95 – 5.40 | d+ … s+ | 0% |
| pps | 0.40 – 7.57 | d+ … x | 14% |

This is the result E5 was built to get. Every feature now traverses most of the
human ladder and essentially nothing falls outside it, against a v1 that had
quad rate pinned at rank d and hold usage below every human alive. Sprint
telemetry from agents and from people occupies the same region — which is what
Othello could not manage, and it is the precondition for anything downstream.

**Coherence** read 8.0 ranks of median disagreement, unchanged from v1, and
the first instinct was to call that a failure. It is not — the metric needed
a control, and the control changes the reading entirely.

Score each of the 396 human records the same way, leave-one-out (so a record
is never compared against a rank table it helped build):

| | median rank disagreement |
|---|---|
| a single **human** sprint record | **8.0** (quartiles 5 / 8 / 11) |
| the agent ladder | **8.0** |

**8.0 is the noise floor of the metric, not the agent's error.** One 40-line
sprint is a small sample, and any single player's five features disagree about
their rank by a median of eight ranks. The agent rungs are no less internally
coherent than a real game is. (Sanity check on the same pass: the median
implied rank of a human record is unbiased against its true rank, mean
absolute error 2.2 ranks.)

What survives the control is *systematic per-feature bias*, which spread
cannot see. Comparing each feature's signed deviation from its own record's
median implied rank, agents against the human baseline:

| feature | human control | agent ladder | verdict |
|---|---|---|---|
| quad rate | −1.0 | +0.4 | within human spread |
| max B2B | −0.3 | −1.3 | within human spread |
| pps | +0.4 | +1.7 | within human spread |
| inputs per piece | +1.1 | +2.9 | off by +1.8 |
| holds per piece | +0.3 | **−4.1** | **off by −4.4** |

So the reading at this stage was: **four of five features inside the spread
real players show, and exactly one genuinely off-distribution.**

* **Hold usage** (−4.4 against control) is the real gap. The bot's hold policy
  is a one-ply greedy comparison — swap if the other piece places better right
  now. Humans use the hold slot to *plan*, which one ply cannot express, and
  their usage stays meaningful all the way down to rank d (0.070/piece). This
  is exactly the constant that was fitted and reverted above; left un-fitted it
  is a specific mechanical thing to fix (a deeper hold policy), not a number to
  tune.
* **Finesse** (+1.8) is mild and has an obvious cause: the emitter computes the
  exact keystroke sequence for its chosen placement, so even weak rungs are
  tidier than the humans they otherwise resemble.
* **Quad rate** — the feature v1 got most wrong, and the hardest strategic one
  — is the best calibrated of the five, with nothing fitted to achieve it.

The methodological point generalises past Tetris: **a coherence or overlap
statistic is uninterpretable without running it on held-out humans first.**
Reported alone, 8.0 looked like a failure; against its control it is the
noise floor, and the single real defect was hiding underneath it.

### Fixing the hold policy, and the state of the ladder now

The hold gap had a mechanical cause, so it got a mechanical fix. `best_pair`
scores placing the current piece and the next one **both ways round** and holds
when the swap wins, expanding only the top 6 first placements so it costs about
4.5× one search rather than 44×. Piece *ordering* is what the slot is for, and
one ply cannot express it.

This was chosen after seeing which feature mismatched, which is model selection
on the test set; the budget for that is small and it is labelled as such in the
code. It survives on an agent-side justification too — it makes the bot better
at the game, independent of any human comparison. Finish rate rose at every
rung (0.92 → 1.00 at skill 1.0, 0.83 → 1.00 at 0.7, 0.67 → 0.92 at 0.4).

The resulting ladder, seven rungs, 20 episodes each:

| skill | time | in/pc | quad | hold | pps | finish |
|---|---|---|---|---|---|---|
| 1.00 | 15.3s | 3.34 | 0.669 | 0.341 | 7.21 | 1.00 |
| 0.80 | 36.1s | 3.59 | 0.449 | 0.182 | 3.09 | 0.85 |
| 0.60 | 122.3s | 3.90 | 0.146 | 0.127 | 0.88 | 0.95 |
| 0.40 | 272.1s | 4.45 | 0.049 | 0.094 | 0.39 | 0.90 |

**Coverage:** every feature 0% off-manifold except pps at 14%, spanning ranks
d…x+ on quad rate, d+…x on hold usage and c+…u on finesse.

**Coherence: 7.0 against the human floor of 8.0** — the agent ladder is now
*more* internally consistent than a single real sprint record, which is the
strongest form this result can take.

| feature | human | agent | excess | before the fix |
|---|---|---|---|---|
| holds per piece | +0.3 | −0.9 | **−1.2** | −4.4 |
| inputs per piece | +1.1 | +2.2 | +1.1 | +1.8 |
| pps | +0.4 | +1.3 | +0.9 | +1.2 |
| quad rate | −1.0 | +0.5 | +1.5 | +1.5 |
| max B2B | −0.3 | −2.4 | **−2.1** | −1.0 |

Hold moved from the one clear outlier to inside the human spread. The largest
remaining gap is **max B2B at −2.1**: the bot quads often (0.669) but chains
them less than people do, because the danger valve takes a partial clear
between quads and breaks the streak. That is smaller than the hold gap was and
has the same character — a specific mechanism, not a tuning knob.

Nothing here says the telemetry *predicts* skill yet; it says agents and people
now occupy the same feature space, coherently, which is the precondition E6a
showed cannot be assumed.

---

### Distillation failed, and agreement was the wrong thing to watch

The distilled generator — the one E2 nominated as closest to how people are
bad — does not work here. Every symptom pointed away from the cause:

| what was measured | value |
|---|---|
| agreement on the **teacher's** states | 0.986 |
| agreement on the **student's own** states | 0.258 |
| lines cleared | 1.8 |
| inputs per piece | 60.2 |

Sixty inputs per piece is not a student misplacing pieces; it is a student
never locking them. And adding DAgger rounds made it *worse* (3.2 lines
against plain cloning's 4.3), which is the opposite of what DAgger is for.

The cause is that the teacher **was not a Markov policy**. It followed a
stored plan, so which keystroke it played depended on how far through that
plan it was — and plan progress is not in the observation. "What would the
teacher do in this state?" therefore had no well-defined answer, and that is
precisely the question DAgger relabelling asks.

Hold made the damage visible. The teacher re-decided to hold on every step the
student declined to, so **39.6% of relabelled targets came back `hold`**
against the student's 1.1% — a single persistent disagreement flooding the
aggregated training set.

The fix is `markov=True`: derive the action from observable state alone, pick
the target placement, emit the one keystroke that moves toward it. It plays
identically to the plan-following version (39.2 lines, in/pc 3.27, quad 0.674,
hold 0.222 either way), so correctness here costs nothing but a re-search per
step.

The transferable lesson is about the *metric*, not about Tetris: **imitation
agreement measured on the teacher's own state distribution is close to
meaningless**. It was 98.6% while the policy was unusable. The number worth
watching is agreement on the student's own states, which is cheap to compute
and was off by a factor of four.

#### A second measurement error, and the verdict after fixing it

The Markov teacher fixed the labels but not the students, and for a while it
looked like it had not fixed anything — because **every distilled number logged
up to this point was also wrong**. `evaluate` took the argmax over Q, and a
deterministic policy here falls into a **limit cycle**: tap left, the new
state's argmax is tap right, oscillate until gravity locks the piece somewhere
arbitrary. Same checkpoint, only the action rule changed:

| policy | lines | inputs per piece |
|---|---|---|
| argmax | 0.7 | 138.0 |
| T = 1.0 sampling | 5.1 | 13.7 |

138 keystrokes for a piece the teacher places in 3.3 is a policy *stuck*, not a
policy misplacing pieces. This can happen to any deterministic policy over
reversible actions and is worth checking for by default.

Re-scored with sampling, the keystroke ladder is a clean negative:

| steps | 362 | 1,380 | 3,364 | 8,203 | 20,000 |
|---|---|---|---|---|---|
| lines | 2.0 | **6.7** | 2.9 | 3.2 | 2.1 |

It peaks at 6.7 lines against the teacher's 41, never finishes a sprint, and is
**not monotone** — so it fails as a *ladder* as well as a policy. Keystroke-level
cloning does not work in this environment.

#### Why, and what replaces it

Predicting the next keystroke asks one network to do two jobs: run a 44-way
placement search, *and* track whether the piece has already arrived at a column
it must itself have computed. Getting the second slightly wrong is
unrecoverable — the piece never reaches the exact state where the teacher says
"hard drop", so the student taps forever.

`coldopen/distill_placement.py` splits them, the same way they come apart in
people:

* **placement judgement** is learned — a 44-way classification over
  (rotation, column), which is a spatial question about a board picture and the
  kind of thing conv nets are good at;
* **motor execution** uses the teacher's own emitter, with `fumble` as an
  explicit dial.

The cost is stated rather than hidden: the rungs then vary in *judgement*, and
finesse is imposed instead of learned. That is a real step down from the
scripted ladder, where both fall out of one `skill` number. It is accepted
because the E2 question this generator exists to answer — *is a part-trained
imitator a distinct kind of bad?* — is a question about judgement, and because
the alternative is a generator that does not work at all.

#### The answer: distillation cannot be a ladder here, and the reason generalises

DAgger at the placement level worked exactly as advertised **on the metric it
targets**, and changed nothing about the play:

| step | own-state agreement | lines |
|---|---|---|
| 1,117 (pre-DAgger) | 0.273 | 1.0 |
| 2,887 | 0.581 | 1.1 |
| 7,463 | 0.811 | 1.0 |
| 12,000 | **0.834** | **1.3** |

Agreement tripled, tracking each round; lines stayed flat against an oracle
that scores 36.8 on the same pathway. Rather than guess why, corrupt the oracle
by a known random fraction — same pathway, same emitter, same teacher, only the
per-decision error rate moving:

| per-placement accuracy | lines | finish rate | quad rate |
|---|---|---|---|
| 1.00 | 36.8 | 0.83 | 0.621 |
| 0.98 | 40.2 | 0.92 | 0.494 |
| 0.95 | 27.8 | 0.42 | 0.472 |
| 0.90 | 10.2 | **0.00** | 0.185 |
| 0.80 | 5.6 | 0.00 | 0.069 |
| 0.60 | 0.8 | 0.00 | 0.000 |
| 0.20 | 0.2 | 0.00 | 0.000 |

**Nothing finishes below 95% per-placement accuracy.** Every one of the 396
human records is a finish, so the entire human-relevant range is compressed
into accuracy 0.95–1.00, and everything below 0.90 is indistinguishable
rubble. A sprint is ~100 placements; at 90% accuracy that is ten bad
placements, each leaving a hole, and the board fills long before 40 lines.

That is the disqualifying property, stated generally: **a ladder needs a dial
whose intermediate settings produce intermediate play.** Imitation accuracy is
not such a dial in a game with long correlated action sequences — it is a step
function. And this is exactly why the scripted dial *does* work: it varies
**strategy**, which changes performance smoothly all the way down, rather than
**accuracy**, which changes nothing until it is nearly perfect.

A second result falls out of the same data. The DAgger student at 0.834
agreement cleared **1.3** lines where *random* corruption at 0.80 clears
**5.6**. A learned policy is worse than noise at the same error rate, because
its errors are **systematic rather than independent** — it is wrong in the same
way on similar boards, so mistakes compound instead of averaging out. E2 found
the same asymmetry from the other side (undertraining produces more structured
errors than handicapping); here it is measured directly.

Recorded in `analysis/tetris_sprint/accuracy_cliff.json`.

#### What it cost to learn that, and the check that would have shortened it

Six separate faults in this generator's plumbing, each producing numbers that
read as "the model is not learning":

1. argmax limit cycles (the policy oscillates and never locks a piece);
2. agreement measured on the teacher's state distribution, not the student's;
3. predicted placements that were illegal for the piece;
4. predicted targets that changed as the piece fell, so the emitter chased;
5. legal targets unreachable from beside a tall stack, with no fallback;
6. **hold missing from the action space entirely**, capping the pathway at 31
   lines and a 50% finish rate.

Number 6 is the instructive one, because no amount of training or debugging
would have revealed it — the ceiling was below the human manifold before the
student made a single mistake. `oracle_ceiling()` catches that whole class in
about a minute by running the pathway with the teacher's own answers
substituted for the student's. It now runs at the top of every distillation and
its result is stored beside the training log, so no learning curve can be read
without the ceiling it is being judged against.

The rule worth keeping: **before training a student, measure what the pathway
scores with perfect predictions.** "The student has not learned" and "the
pathway cannot do better" are indistinguishable from the outside, and only one
of them is fixed by more training.

Plain cloning at the placement level then reproduced the same failure one level
up, which is worth recording because the numbers look so different from the
outside:

| | value |
|---|---|
| agreement on the teacher's pool | 91.5% |
| agreement on the student's **own** boards | **21.5%** |
| illegal placements predicted | 2.5% |
| mean board-score gap when legal | 15.66 (≈ two holes) |

The placements were legal and simply *bad*. So the fix is DAgger again — and
at this level it is well posed in the way the keystroke version never was: the
teacher's placement is a pure function of `(board, piece)`, with no plan state
to be out of sync with and no way for one disagreement to flood the labels.
`own_state_agreement()` is now logged beside pool agreement and printed first,
because the gap between them is the whole story every time this project
mistook a well-fitted student for a working one.

### What "early" can and cannot mean with this data

The project's question is skill *before results exist*. Sprint answers a strong
form of that: a 40 LINES run is a solo time trial, so **one game** produces a
full telemetry row, and the comparison is against a TETR.IO rank that takes
many ranked matches to establish. Predicting rank from a single sprint is the
cold-start question in its natural form for this game.

What this data cannot answer is the *within-game* version — "how many pieces
in can we tell?" The public API returns game-level summary statistics, not
replays, so a human record cannot be truncated to its first ten pieces. Agent
episodes can be truncated freely, but with nothing to compare against, that
would be measuring the simulator rather than testing it. Recorded as a limit
of the dataset, not of the method: a game with replay access (minesweeper, or
TETR.IO replays if they ever become available) could run it unchanged.

### A process failure worth recording: the validation gate was never green

`envs/tetris_sprint/validation_report.json` was committed reading
`overall_pass: false`, with all six required tests listed as not passed and a
single entry — `test_benchmark_factory_parity` — in the results. The standalone
benchmark run done while measuring throughput had overwritten the battery's
report before the commit picked it up.

The commit message claimed "the full validation battery — PASS, eligible for
training". That was true of console output seen at the time and false of the
artifact, and the project rule is specifically that *a fresh passing report is
the gate*. Everything trained on this environment since was trained ungated.

Re-run: **12 passed, 1 skipped, overall PASS** — bit-exact differential, batch
independence, invariant sweep, auto-reset, determinism and replay all green. So
the environment was in fact sound and no result changes. The lesson is about
the gate, not the env: a gate checked by reading console output is not a gate,
and a run that writes to the same artifact path can silently revoke one.

## E6a — Othello: the first agent-versus-human comparison, and it fails

Othello turned out to be the one game where both halves already existed. The
French Othello Federation's **WTHOR** database publishes 125,000 tournament
games over forty years and 3,650 players, free, with full move sequences — and
Othello already had a measured agent ladder, a reference network and a feature
extractor. So agents and people could be put through identical code.

Two things had to be handled to get there, both silent when wrong. WTHOR
**omits passes**, so replaying the move list literally desynchronises the board
the first time a pass was skipped and every later move is attributed to the
wrong player; passes are reinserted by checking legality at each ply. And WTHOR
has **no ratings**, so skill is fitted from the results themselves with the same
Bradley-Terry code the agent league uses, which keeps both populations on one
scale and covers every player rather than the ranked elite.

Validation passed: 60 of 60 games replayed legally through Pgx, and the fitted
human Elo spread is sensible (p5 −269, median 444, p95 1030).

**Then the comparison failed, in three ways at once.**

| feature | weak human | strong human | agent tier 0 → 5 |
|---|---|---|---|
| corner_taken | 0.001 | 0.000 | 0.003 → 0.060 |
| x_square | 0.000 | 0.000 | 0.082 → 0.004 |
| c_square | 0.009 | 0.007 | 0.101 → 0.002 |
| gave_corner | 0.000 | 0.000 | 0.167 → 0.009 |
| ref_agreement | 0.271 | **0.258** | 0.105 → 0.309 |
| ref_regret | 0.030 | 0.030 | 0.096 → 0.029 |

**1. The corner features never fire.** They are nonzero in 0.013% of human move
rows. The observation window is each player's first twenty moves — about forty
plies — and in competent Othello corners are contested in the *endgame*. The
feature set that separates agents is measuring events that do not occur in the
window humans are observed over. The agents only produce them because they play
badly enough to give corners away in the opening.

**2. The reference feature is inverted.** Stronger humans agree with the
reference network *less* (0.258 against 0.271). This is the yardstick problem
appearing in real data rather than in theory: `coldopen/othello_reference` is a
mediocre RL agent, and tournament players deviate from it because they are
better than it. A yardstick has to be stronger than everyone it measures, and
this one is weaker than all of them.

**3. There is no overlap in strength.** The agent ladder tops out at 641 Elo
above a random anchor. Every WTHOR player is a tournament entrant. The two
populations do not meet.

**This is the manifold-overlap risk named as the project's central threat,
measured, on the first real attempt.** It is also the most valuable negative
result so far, because each of the three failures says exactly what to change:

- **The window or the features must match.** Either profile later moves, where
  corners are live, or replace the corner family with features that vary in the
  opening. The current set was tuned on agents that blunder early.
- **The reference must be stronger than the profiled population.** For Othello
  that means a real engine (Edax) rather than a self-play DQN.
- **The ladder must reach the humans.** Distillation from a strong teacher is
  the route, and Othello has one — the same recipe proposed for NetHack.

None of this was visible from agent-only experiments, all four of which looked
healthy. The cheapest way to find it was to run the overlap check first, and it
should gate every remaining E6 attempt.

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
