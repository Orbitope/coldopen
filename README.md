# coldopen

Reading a player's skill from their opening moves, using reinforcement-learning
checkpoints as the labelled training set.

Every skill rating starts from nothing. A new account has no results, so
matchmaking puts it in the middle of the distribution and waits — which is both
a bad experience for everyone in those lobbies and the cheapest possible escape
hatch for anyone who wants to farm beginners: make a new account.

But results are not the only evidence. *How* somebody plays is visible on move
one. The obstacle has always been labels — to learn the mapping from behaviour
to skill you need a lot of games from players whose strength you already know,
and before launch you have none.

So generate them. Train an agent by self-play, snapshot it throughout training,
and **measure** the strength of each snapshot with a round robin rather than
assuming it improved. That gives a labelled skill ladder before a single human
has played the game. Harvest the ladder's telemetry, train a classifier, and
point it at real players on day one.

## The questions this project is trying to answer

1. **How early is skill legible?** On Connect Four most of the signal is there on
   the first move — but that number turns out to be mostly a fact about the
   *agents*, not the game. Rebuilding the ladder four other ways at matched
   strength drops one-move accuracy from 52% to 25-31%, and undertrained RL
   checkpoints are the outlier among the five. See `EXPERIMENTS.md`, E2.
2. **What property of a game decides that?** Three axes: tactical density (does a
   position usually contain a forced win or loss?), chance, and hidden
   information. Tactical density is now measured rather than asserted, and it
   separates Connect Four from Othello in the predicted direction.
3. **Does any of it transfer to humans?** Still the load-bearing question, and
   still unanswered. A network's blunders and a beginner's blunders are both
   blunders, but they are not the same distribution. Four human corpora are now
   ingested and a free baseline is measured on them, but no game yet has *both*
   an agent ladder and human data, so the two halves have not met. Until they do
   this is a proof of pipeline and not a proof of concept.

## Status

All four games are built end to end. The human-transfer test is not, and it is
still the one that decides whether any of this means anything.

Every game gets a 19-checkpoint ladder, six tiers picked by measured Elo, and a
classifier scored by cross-validation grouped by opponent, over six tiers with a
16.7% baseline. "own features only" drops the two reference-network features and
leaves the game's own vocabulary.

| game | information | chance | Elo range | tier gap | 1 move | 20 moves | own only, 1 move |
|---|---|---|---|---|---|---|---|
| Connect Four | perfect | none | 736 | 194 | 46.8% | **63.1%** | 39.8% |
| Othello | perfect | none | 796 | 159 | 48.0% | 52.9% | 30.1% |
| Backgammon | perfect | dice | 184 | 42 | 49.2% | 36.2% | 47.4% |
| Leduc Hold'em | hidden | cards | 46 | 15 | 29.9% | **62.1%** | 24.1% |

**Read the tier gap column before the accuracy columns.** The ladders are not
equally spread out: Connect Four's six tiers are ~194 Elo apart and Leduc's are
~15 Elo apart, so Leduc's classifier is separating players who are an order of
magnitude closer in strength. Its 62.1% is the more impressive number on this
table, not the less — and its 90.3% within-one-tier is the best of the four.

**And read the whole table as a statement about undertrained networks, not about
the games.** Every ladder here was built by snapshotting self-play. Rebuilding
Connect Four's four other ways — imitation, and three kinds of handicap — puts
one-move accuracy at 0.25-0.31 against this table's 0.47, at matched tier
spacing. Whether the games differ from each other is a separate question from
whether any of these numbers survive contact with people.

Three things fall out of it:

**Hidden information delays legibility, it does not prevent it.** Leduc is the
only game where one move says almost nothing — 29.9% against a 16.7% baseline —
and the only one whose accuracy climbs the whole way, doubling to 62.1% by move
twenty. In the other three, most of what you are going to learn arrives in the
first two or three moves. A poker player has to be watched; a Connect Four
player gives it away immediately.

**Where there are no tactics, the game's own features carry much less.** Strip
the reference-network features and Connect Four still gets 39.8% from one move,
because "you had a win and didn't take it" is a fact about the rules. Othello's
positional vocabulary — corners, X-squares, mobility, frontier — manages 30.1%,
and needs the reference network to reach parity. This is the tactical-density
hypothesis surviving its first test.

**Backgammon runs backwards, and that is the interesting one.** It is the only
game where watching longer makes the prediction *worse*: 49.2% from one move,
60.6% from two, then a steady decline to 36.2% by twenty. Backgammon starts from
a fixed position, so the first couple of moves are a near-pure skill signal —
after that the dice scatter players into positions that have little in common,
and averaging over them dilutes the signal instead of accumulating it. Its
ladder is also the least trustworthy of the four (concordance 0.73, and the
untrained network still places mid-table), so some of the ceiling is the ladder
rather than the game.

What each tier's play actually looks like, bottom tier to top:

| game | the tells |
|---|---|
| Connect Four | hands over an immediate win 19.2% → 1.7% of moves; ignores a threat 16.2% → 1.0%; plays the centre column much more |
| Othello | takes a corner 0.3% → 6.0%; plays an X-square 8.2% → 0.4%; gives a corner away 16.7% → 0.9% |
| Backgammon | leaves 3.06 → 2.60 blots, of which 2.59 → 2.25 are actually within range of being hit |
| Leduc Hold'em | bluff-raises 11.2% → 0.7% of moves; folds the best hand 1.3% → 0.1%; calls far more, raises far less |

Those are the features moving in the direction the games' own theory says they
should, which is the check that they measure what their names claim.

### Tactical density, measured

The table above used to assert this column. `coldopen/tactics.py` searches for
it instead: sample positions across whole games, and for each one search two
plies exhaustively for a move that wins on the spot, and for a move that lets
the opponent win on the spot.

| game | has a winning move | can blunder into a loss | was asserted |
|---|---|---|---|
| Connect Four | 19.8% | 25.7% | high ✓ |
| Othello | 0.0% | 0.0% | low ✓ |
| Backgammon | 0.5% | 0.2% | medium ✗ |
| Leduc Hold'em | 8.5% | 12.7% | not comparable |

Backgammon was labelled "medium" and is not: winning on the spot means bearing
off your last checker, which almost never sits one move away. Its difficulty is
real but it is not the kind this measures. Leduc's numbers are computed the same
way and should not be read across at all — a hand ends when somebody folds or
calls, so "a move that wins immediately" is usually the last call of a hand that
was already won, not a tactic anyone had to see.

The clean comparison is the top two rows, which hold everything else fixed —
perfect information, no chance, tiers ~160-190 Elo apart. Connect Four has a
forced win available in a fifth of its positions and its own features get 39.8%
from a single move; Othello has one in none of them and gets 30.1%, needing the
reference network to catch up. That is the tactical-density hypothesis
surviving one honest test, on a sample of two games.

## Layout

```
coldopen/
  games.py      one adapter over all four Pgx games; everything below is generic
  nets.py       conv net for board games, MLP for vector games
  selfplay.py   self-play DQN with n-step negamax targets -> a ladder
  ladder.py     round robin -> Bradley-Terry -> Elo, and the monotonicity gate
  profile.py    sessions of observed moves -> a feature matrix
  features/     the per-game feature sets, one module each
  predict.py    tier prediction from the first N moves
  tactics.py    measured tactical density, by two-ply search
  export.py     results -> docs/data.js

  c4.py         the original Connect Four, differential-tested against PettingZoo
  net.py        \
  train.py      | the bespoke Connect Four pipeline the first result came from,
  league.py     | kept as the check that the generic stack means the same thing
  telemetry.py  |
  classifier.py /
docs/           the article
```

Connect Four is deliberately built twice. The bespoke pipeline is validated
against PettingZoo move by move; the generic one has to reproduce it, and
`tests/test_features_connect_four.py` plays the same games through both and
asserts every feature agrees. Without that, "Othello scores lower than Connect
Four" could just as easily be a bug in the rewrite.

## Four traps this code is built around

**Training step is not skill.** A randomly initialised network beats a
uniform-random opponent about 85% of the time, because any *consistent* policy
does. The first five Connect Four checkpoints are statistically
indistinguishable from each other. Tiers are chosen by measured league Elo, and
`league.monotonicity` reports where the ladder actually starts behaving like
one.

**The yardstick must not be a contestant.** Telemetry features that compare a
move to a strong network's opinion need that network to be independent of the
agents being profiled — otherwise the top tier scores a perfect match with
itself and the classifier reads its own answer key. `coldopen/reference` is
trained from a different seed for exactly this reason.

Cross-validation is also grouped by opponent, because a weak opponent leaves
more winning moves lying around and makes whoever is playing them look sharper.

**A ladder has to be checked for being a ladder.** Adding backgammon exposed
this. These games pay out only at the end, so a one-step target walks the result
back one ply per sweep — fine over 42 plies of Connect Four, hopeless over a
couple of hundred plies of backgammon. The first backgammon run produced
nineteen checkpoints spanning 32 Elo in which the *untrained* network ranked
third: everything trained, nothing learned, and no error anywhere. `league.
monotonicity` is what caught it, which is the argument for measuring the ladder
rather than assuming training produced one. The targets are now n-step, with n
set per game against how long its episodes are.

**Winning the most hands is not winning.** Leduc is scored in chips, and a
player who folds every hand wins the majority of *hands* while losing all of the
money — so a league built on win counts would have ranked the worst policy
first. Pairwise results are fitted on the mean normalised result instead, which
for the two games that are simply won or lost is identically `wins + draws/2`
and changes nothing, and for the other two is the metric the game actually
settles on.

## Running it

```bash
uv venv && uv pip install -e ".[dev]"
pytest
```

The bespoke Connect Four pipeline:

```bash
python -m coldopen.train --out coldopen/checkpoints
python -m coldopen.train --seed 1234 --plies 3000000 --out coldopen/reference
python -m coldopen.league --games 300 --out analysis/league.json
python -m coldopen.classifier --out analysis/telemetry.json
```

Any of the four games, through the generic one. The reference run is what the
`ref_*` features are scored against and must be trained from a different seed:

```bash
python -m coldopen.selfplay --game othello
python -m coldopen.selfplay --game othello --seed 1234 --plies 3000000 \
    --out coldopen/ladders/othello_reference
python -m coldopen.ladder --game othello --games 300
python -m coldopen.predict --game othello
```

Then the cross-game measurements and the article's data file:

```bash
python -m coldopen.tactics --out analysis/tactics.json
python -m coldopen.export --out docs/data.js
```

Ply budgets, network sizes and the n-step horizon default per game — see
`selfplay.PROFILES`. Training picks MPS for the board games and CPU for the
vector ones, which is measured rather than assumed: a batch-512 update costs
107 ms on CPU against 6.5 ms on MPS for the Othello conv stack, and 1.8 ms
either way for the backgammon MLP.

## Related

The wager-based cheat detector this grew out of lives in
[smurf-hunting](https://github.com/orbitope/smurf-hunting). That project needs a
dozen matches before it can say anything; this one is the answer to "what about
an account with none?".
