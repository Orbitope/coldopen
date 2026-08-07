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

1. **How early is skill legible?** On Connect Four the answer so far is
   uncomfortably early — most of the signal is there on the first move — but a
   large part of that is opening choice, which is also the easiest thing for
   somebody to imitate.
2. **What property of a game decides that?** The plan is to vary three axes:
   tactical density (does a position usually contain a forced win or loss?),
   chance, and hidden information.
3. **Does any of it transfer to humans?** This is the load-bearing question. A
   network's blunders and a beginner's blunders are both blunders, but they are
   not the same distribution. Until that is tested against real human games with
   known ratings, this is a proof of pipeline and not a proof of concept.

## Status

Connect Four is built end to end. The other three games and the human-transfer
test are in progress.

| game | information | chance | tactical density | status |
|---|---|---|---|---|
| Connect Four | perfect | none | high | ladder + telemetry + classifier |
| Othello | perfect | none | low | planned |
| Backgammon | perfect | dice | medium | planned |
| Leduc Hold'em | hidden | cards | n/a | planned |

Connect Four so far: a 19-checkpoint ladder spanning **1,131 Elo**, and a
classifier that reaches **56.6%** exact-tier accuracy from a single move and
**82.6%** within-one-tier over six tiers, against a 16.7% baseline.

## Layout

```
coldopen/
  c4.py         vectorized Connect Four, differential-tested against PettingZoo
  net.py        the small conv net shared by training, leagues and telemetry
  train.py      self-play DQN, checkpointed into a ladder
  league.py     round robin -> Bradley-Terry -> Elo, and the monotonicity gate
  telemetry.py  per-move behavioural features
  classifier.py tier prediction from the first N moves
  export.py     results -> docs/data.js
docs/           the article
```

## Two traps this code is built around

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

## Running it

```bash
uv venv && uv pip install -e ".[dev]"
pytest
```

```bash
python -m coldopen.train --out coldopen/checkpoints
python -m coldopen.train --seed 1234 --plies 3000000 --out coldopen/reference
python -m coldopen.league --games 300 --out analysis/league.json
python -m coldopen.classifier --out analysis/telemetry.json
python -m coldopen.export --out docs/data.js
```

## Related

The wager-based cheat detector this grew out of lives in
[smurf-hunting](https://github.com/orbitope/smurf-hunting). That project needs a
dozen matches before it can say anything; this one is the answer to "what about
an account with none?".
