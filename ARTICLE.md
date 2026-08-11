# Four Ways to Fool Yourself When Testing Cold-Start Skill Prediction

A new account has no results. That is the cold-start problem in ranked games:
the system must place a player before it has any evidence about them, and the
first few matches are miserable for everyone while it guesses. The appealing
fix is to read skill from *how* somebody plays rather than from whether they
won — and since nobody has labelled human data for a game that has not
launched, to learn that reading from reinforcement-learning agents instead.
Train a ladder of agents spanning weak to strong, measure their behaviour,
learn the mapping from behaviour to skill, and apply it to people.

This is a report on trying that. It contains one useful positive result and
four ways the evaluation lies to you, each of which I walked into. The
negative results are the point: every one of them produced a number that
looked decisive and pointed the wrong way.

Two environments were built for it, both differentially validated against
independent reference implementations: an input-level Tetris 40 LINES sprint
and expert/intermediate/beginner Minesweeper. Code and data:
`github.com/Orbitope/coldopen`.

---

## Trap 1: the baseline that peeked

The headline test: fit a model on agent telemetry only, apply it once to 396
real TETR.IO sprint records, measure Spearman correlation against the
players' actual ranks.

| method | ρ vs true rank |
|---|---|
| **`pps` alone, no model** | **+0.932** |
| `inputs_per_piece` alone | +0.773 |
| human-fitted 5-feature ridge (5-fold CV) | +0.923 |
| **agent-fitted 5-feature ridge** | **+0.660** |

Ranking players by pieces-per-second — one number, no model, no simulator,
no agents — beat the whole apparatus by 0.27. I wrote that up as a clean
negative: the simulator earned nothing.

That conclusion was wrong, and the error is worth dwelling on because it is
easy to make and hard to see.

**To rank players by `pps`, you must first know that `pps` is the
discriminating axis and which direction it runs.** That knowledge comes from
looking at players whose rank you already know. It is exactly the thing a
cold start does not have. The baseline was allowed to consult the test set's
labels; the method under test was not. Comparing them measures nothing.

Look again at what the agent-fitted model learned from **zero human labels**:
weights ≈ `[0, 0, 0, 0, +0.73]`. It discovered on its own that `pps` is the
axis and the other four features are noise. That is not a failure to beat the
baseline — *it is the baseline, derived without the labels the baseline
needs.*

### The honest question, and the positive result

If the baseline needs labels, the right question is how many. Same held-out
records for every method, 200 random splits per sample size:

| labelled human games | human-fitted | **agent-fitted (0 labels)** |
|---|---|---|
| 5 | +0.828 | **+0.927** |
| 10 | +0.883 | **+0.927** |
| 20 | +0.907 | **+0.927** |
| 80 | +0.918 | **+0.927** |
| 300 | +0.920 | **+0.922** |

**A model fitted only on agents is not beaten by a model fitted on humans at
any sample size tested.** The simulator is worth more than 300 labelled
games, and at the sample sizes that actually matter for a cold start it leads
by about 0.10.

The methodological point generalises past this project: **cold-start methods
should be evaluated on label-efficiency, not on raw correlation against a
label-informed baseline.** The question is never "does it beat the best
predictor?" — it is "how much labelled data would you need to do this well
without it?"

---

## Trap 2: the metric that could not see degeneracy

Before any accuracy claim, the agents have to occupy the same region of
feature space as the people. My first overlap check asked whether agent
values fell between the human 2nd and 98th percentiles. It reported every
generator as **100% inside the human range** on quad rate, hold usage and
back-to-back count.

All three were false. Pooled human quad rate spans 0.00 to 1.00, so an agent
pinned at 0.00 — below every human alive — passes a range test. A range
check cannot detect degeneracy inside a wide range.

Replacing it with "which human rank does this rung look like?" showed the
truth immediately: quad rate pinned at the weakest rank, hold usage *below
every human*, while finesse spanned sixteen ranks. The bot had a top
player's keystroke efficiency and a beginner's stacking — a shape no human
has.

The cause was not the search but the objective. The bot's board score paid
for any immediate line clear, so it cashed singles and never stacked four
rows. Rewriting it to keep a well and play for quads moved quad rate from
0.01 to 0.61.

---

## Trap 3: the number with no floor

The natural follow-up metric: for one agent rung, do all five features agree
about which rank it is? Call the spread of implied ranks its *coherence*. The
agent ladder scored **8.0 ranks of median disagreement**, which reads as
obviously broken.

Then I ran the same statistic on the humans, leave-one-out.

| | median rank disagreement |
|---|---|
| a single **human** sprint record | **8.0** |
| the agent ladder | **8.0** |

**8.0 is the metric's noise floor, not the agent's error.** One 40-line run is
a small sample; any real player's five features disagree about their rank by
a median of eight ranks. The agent rungs were no less internally consistent
than a real game.

What survived the control was *systematic per-feature bias*, which spread
cannot see. Against the human baseline, four of five features sat inside the
spread real players show; exactly one was genuinely off — hold usage, 4.4
ranks weak, because the bot's hold policy was a one-ply greedy check while
humans use the slot to plan. A two-ply fix closed it to 1.2 and, tellingly,
also raised the bot's finish rate at every skill level: it made the agent
better at the *game*, not merely more human-shaped.

**Any overlap or coherence statistic is uninterpretable without running it on
held-out humans first.** Reported alone, 8.0 looked like failure; against its
control it was the floor, with the single real defect hiding underneath.

---

## Trap 4: the dial that was a step function

The plan called for three kinds of agent ladder: undertrained RL, handicapped
strong agents, and part-trained imitators. The imitation ladder never worked,
and the reason is more interesting than the failure.

A student reached **98.6% agreement** with its teacher and cleared 1.8 lines
against the teacher's 41. On its own states, agreement was **25.8%** — the
first number was measured on the *teacher's* state distribution, which the
student never visits. That mistake recurred three times in different guises
before I started logging own-state agreement first, permanently.

But even with well-posed labels and DAgger, task performance would not move.
So I corrupted a perfect policy by a known fraction to isolate accuracy from
everything else:

| per-decision accuracy | lines cleared | finish rate |
|---|---|---|
| 1.00 | 36.8 | 0.83 |
| 0.98 | 40.2 | 0.92 |
| 0.95 | 27.8 | 0.42 |
| **0.90** | 10.2 | **0.00** |
| 0.80 | 5.6 | 0.00 |
| 0.20 | 0.2 | 0.00 |

**Nothing finishes below 95% accuracy.** Every human record is a completed
sprint, so the entire human-relevant range compresses into accuracy
0.95–1.00, and everything below 0.90 is indistinguishable rubble.

A ladder needs a dial whose *intermediate settings produce intermediate
play*. Imitation accuracy is not one in a game with long correlated action
sequences — it is a step function. This is also why the ladder that *did*
work varies **strategy** (expert well-and-quad play blended toward beginner
take-any-clear), which degrades smoothly all the way down.

A second result fell out of the same data: a learned policy at 83% accuracy
cleared 1.3 lines where *random* corruption at 80% cleared 5.6. **A trained
policy is worse than noise at the same error rate**, because its errors are
systematic rather than independent — it is wrong the same way on similar
boards, so mistakes compound instead of averaging out.

---

## The design lesson: one-dimensional ladders produce one-dimensional telemetry

The agent-fitted model's failure in Trap 1 had a specific mechanism worth
separating from the baseline error. Its learned weight on `pps` was
**−0.399** — backwards, in a population where `pps` correlates +0.99 with
skill.

Agent features were far more collinear than human ones, because a single
coupled `skill` dial moved all of them together:

| feature pair | agents | humans |
|---|---|---|
| holds × pps | +0.90 | +0.46 |
| quad × pps | +0.85 | +0.47 |

Ridge splits weight arbitrarily among redundant predictors, and that
arbitrary split does not transfer. Decoupling into a **skill × latency grid**
fixed the sign (+0.73) and recovered the trivial solution.

The second environment was built with this in mind. Minesweeper's ladder has
three independent dials — training progress, judgement lapses, and input
latency — and across 192 oracle-judged rungs:

| pair | Minesweeper | Sprint |
|---|---|---|
| speed × judgement | **−0.03** | +0.85 |
| speed × win rate | +0.03 | — |
| judgement × win rate | **+0.64** | — |

Win rate rides the judgement axis, not the speed axis. **Build the ladder
with at least as many independent axes as the skill you are trying to
model** — and verify it, because the collinearity is invisible until you
measure it.

---

## An aside: what actually trains an agent here

Two environments produced the same lesson from opposite directions.

Deep Q-learning on Minesweeper ran 2,000,000 steps to a **0.000 win rate at
every checkpoint**, settling into a policy that flags cells forever and never
reveals. That was rational under the reward: an uninformed reveal is a mine
with prior ~21%, expected value ≈ −210,000, against flagging at −230 per
step. Repricing the death penalty and weighting the progress bonus unlocked a
reverse curriculum that had never advanced — but still did not teach the
policy to *locate* safety.

What worked was changing the *density* of the signal, not its shape. The
environment knows where its mines are, so every visited state yields ~480
labelled cells for free. Same network, same environment, same observations:

| training signal | result |
|---|---|
| 1 sparse scalar per action | 0 safe cells after 2,000,000 env steps |
| 480 dense labels per state | 81 safe cells after 16,000 updates |

On the beginner board the supervised net goes from no wins at all, through 3%
at its first learning checkpoint, to a **67% win rate** — the first agents in the project producing records
directly comparable to a human leaderboard entry.

---

## What this does not show

The decisive claim — that agent-trained telemetry predicts human skill where
a naive method cannot — has been tested on exactly **one** game, and that game
turned out to be nearly the worst possible test case. A 40 LINES sprint is a
time trial: rank correlates 0.93 with a single trivially observable number,
and an exhaustive search over all sixteen feature subsets, fitted on humans
with cross-validation, found nothing that beats `pps` alone. There was no
headroom for any method to fill.

That suggests a screen worth running *before* building an environment for a
game: **how well does the best single observable feature already predict
rank?** At 0.93, the ceiling is the floor.

### The screen, run in anger

TETR.IO's *versus* mode was the obvious next environment. Its label is
attractive in exactly the way sprint's was not: **TR is a Glicko rating fed by
match outcomes**, not a direct function of any per-round statistic. The data
was already in hand — 25,707 rounds, 450 players, with per-round APM, PPS,
attack-per-piece, garbage sent and received.

| setting | best single observable | ρ vs rank |
|---|---|---|
| sprint (known dead) | `pps` | 0.932 |
| versus, aggregated over ~57 rounds | `vsscore` | 0.985 |
| **versus, a single round — the cold-start setting** | **`pps`** | **0.922** |

It fails. A latent outcome-fed rating did not create headroom, because the
thing the rating measures is still mostly speed: **Tetris skill is
speed-limited in every mode.** Building the versus environment — garbage,
attack tables, two-player dynamics — would have inherited the exact ceiling
that made sprint uninformative. The screen cost one script; skipping it would
have cost the environment.

### What the screen implies about *labels*

The generalisation is sharper than "check your features", and it is
uncomfortable:

> A label that is itself a performance measure — a time, a score, a
> words-per-minute — will be predictable from that measure's behavioural
> correlates. Headroom lives in labels that are **latent and multi-causal**.

That disqualifies whole families without any ingestion work. Typing sites rank
*by* words-per-minute, so rank ≈ WPM by construction. Rhythm games compute
their rating as a deterministic function of accuracy and chart difficulty. The
entire speed/accuracy-pair genre fails identically.

It also lands on this project's own remaining hope. minesweeper.online's
ranking page sorts by **time**, and 3BV/s is 3BV ÷ time — so a best-time
leaderboard risks being the same trap a third time. The label with genuine
headroom there is **win rate**: surviving an Expert board is dominated by
guess-avoidance, which is judgement, while time is dominated by clicking
speed. Those are precisely the two axes the three-dial ladder was built to
separate. The right request to make of a data holder is therefore not "your
leaderboard" but *"per-player win rate, and losing games as well as winning
ones."*

Minesweeper was chosen because its headline feature — *was a provably safe
cell available, and did the player click one?* — is decided by a solver from
the rules, not by a model's opinion, and is not reducible to speed. Its agent
side is complete: validated environment, five ladders, deduction oracle,
decorrelated three-dial grid. The human half is blocked on data access, and
until it arrives the central question has one data point.

One game is an anecdote.

### Where that leaves the approach

Stacking the constraints together explains why this stalled, and it is
structural rather than bad luck. A game must be **simulatable** to build agent
ladders, must have **headroom** above its obvious statistic, and must have
**accessible** human telemetry. Public sources fail at least one every time:
games rich enough to have interesting telemetry (shooters, MOBAs, card games)
are far too complex to simulate honestly, while games simple enough to
simulate are simple *because* their skill is thin — and the one candidate that
passed on paper serves its data only through an endpoint that writes to its
own database.

A game studio dissolves all three at once: they own the simulator, they hold
latent matchmaking ratings rather than leaderboard times, they record losses,
and access is a conversation. They also have the motive, since cold start is
their problem. Which suggests the honest framing for this work is not "here is
a solved method" but **"here is the method, the toolkit, and — most usefully —
the one-script screen that tells you in minutes whether your game has any room
for it."**

---

## Corrections

Three claims in this project were published and then retracted, and they are
kept in the repository's log rather than quietly fixed:

- **"Column 9 was unreachable"** — asserted before checking. The cell array
  uses bounding-box coordinates, so the move I claimed was impossible was
  always available. The improvement came entirely from the objective.
- **"The environment passed its validation gate"** — true of console output I
  had seen, false of the committed artifact, which a later benchmark run had
  overwritten. Everything trained on it was trained ungated. Re-run: it does
  pass, so no result changed. A gate checked by reading console output is not
  a gate.
- **"Distillation agreement was 98.6%"** — true, and describing a policy stuck
  in a tap-oscillation deadlock. Every distilled number before that fix was
  measuring the deadlock rather than the student.

Each was caught by a control, not by a test suite that was green throughout.
That is the through-line: in this kind of work the failure mode is not code
that crashes, it is a number that looks decisive and points the wrong way.
