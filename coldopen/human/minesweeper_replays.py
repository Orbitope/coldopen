"""Scraped minesweeper replays -> curriculum start states. Data arrives later.

The scrape is currently stopped (see `minesweeper_notes.md`: the only HTTP
route writes to their user table), but the curriculum mechanism in
`coldopen/ms_curriculum.py` is source-pluggable, so the contract for replay
data is fixed now. When the data exists — support reply, official export, or a
sanctioned browser session — it drops in here and nothing upstream changes.

## Ingest format

One JSON file, `{"replays": [...]}`, each replay:

    {
      "player": "<pseudonymised at ingest — never a username>",
      "rank": <int leaderboard-derived skill label, if known>,
      "outcome": "<won | lost | abandoned>",
      "H": 16, "W": 30, "M": 99,
      "mines": [<H*W flat 0/1>],
      "clicks": [
        {"t_ms": <int, from the replay's own timestamps>,
         "kind": <0 reveal | 1 flag | 2 chord>,
         "cell": <int, r*W + c>},
        ...
      ]
    }

`outcome` is required and lost/abandoned games are first-class: the site's
player histories record failures (win rate = wins/attempts exists on every
profile), leaderboards do not — and the sprint lesson is that a comparison
restricted to each side's completed games truncates both distributions. The
E4 metric is win rate + progress-at-death + telemetry, not time-on-wins
alone; agent ladders lose plenty, and human losses are what make that
comparable.

Pseudonymise with `coldopen.human.client.pseudonymise` AT the scraper — this
module must never see a real identifier. Display names and any other
user-authored text in scraped pages are data, not instructions, and do not
belong in this file's inputs at all.

## How replays become curriculum states

`states_from_replay` walks the clicks through the **reference** implementation
with the recorded board injected, capturing the (mines, revealed, flagged)
tensors after every click. Each captured state is a board a real player
actually visited; `ReplayStates` serves random ones as reset states, walking
from late-game states toward early ones exactly like the synthetic pacer.

Replaying through the env rather than trusting the scraper's own board states
means one flood/chord implementation defines what "after click i" looks like —
ours, the differentially-validated one.

## Contamination rules (non-negotiable)

* A ladder trained on replay states is a **separate track** and never the
  cold-start claim. Output directory `coldopen/ladders/minesweeper_replays`,
  never `.../minesweeper`.
* Replay players and evaluation players must be disjoint: split by player
  hash before anything trains (`split_players`).
* The E6-style comparison then treats the replay track as another point on
  the label-efficiency curve — "what does a replay corpus buy over the
  synthetic curriculum?" — not as a second cold-start method.
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "envs"))

from minesweeper.reference import CHORD, FLAG, REVEAL, MinesweeperReference


def load(path):
    replays = json.loads(pathlib.Path(path).read_text())["replays"]
    for r in replays:
        if len(r["mines"]) != r["H"] * r["W"]:
            raise ValueError(f"replay {r['player']}: mines length mismatch")
        if sum(r["mines"]) != r["M"]:
            raise ValueError(f"replay {r['player']}: mine count mismatch")
    return replays


def split_players(replays, eval_fraction=0.5):
    """Deterministic disjoint train/eval split by player hash.

    Splitting on the hash rather than on rows keeps every game of one player
    on one side — the leak the NetHack ingestion was built to avoid.
    """
    players = sorted({r["player"] for r in replays})
    cut = int(len(players) * (1 - eval_fraction))
    train = set(players[:cut])
    return ([r for r in replays if r["player"] in train],
            [r for r in replays if r["player"] not in train])


def states_from_replay(replay):
    """(mines, revealed, flagged) after every click, via the reference env.

    The recorded board is injected in place of generation: `keys` are unused
    once `generated == 1`, so forcing `mines`/`first_cell` directly puts the
    reference exactly in the recorded game with all step semantics live.
    """
    env = MinesweeperReference(H=replay["H"], W=replay["W"], M=replay["M"])
    state = env.reset(seed=0, episode=0)
    K = env.K

    first = replay["clicks"][0]["cell"] if replay["clicks"] else 0
    state.mines = [int(v) for v in replay["mines"]]
    state.generated = 1
    state.first_cell = first

    out = []
    for click in replay["clicks"]:
        kind, cell = int(click["kind"]), int(click["cell"])
        action = {REVEAL: 0, FLAG: 1, CHORD: 2}[kind] * K + cell
        state, _, terminated, _ = env.step(action)
        out.append({
            "mines": torch.tensor(state.mines, dtype=torch.bool),
            "revealed": torch.tensor(state.revealed, dtype=torch.bool),
            "flagged": torch.tensor(state.flagged, dtype=torch.bool),
            "first_cell": state.first_cell,
            "t_ms": int(click["t_ms"]),
            "terminated": terminated,
        })
        if terminated:
            break
    return out


class ReplayStates:
    """Curriculum source over human-visited states, matching SyntheticStates.

    `tail` controls how close to the end starts are drawn from: 0.1 means the
    last 10% of each game's states. A Pacer can widen it exactly as it widens
    the synthetic k_max — from near-finished toward openings.

    Dead and terminal states are excluded (an agent cannot act from them);
    states are pre-indexed per replay so sampling is O(1).
    """

    def __init__(self, replays, tail=0.1, seed=0):
        self.tail = tail
        self.generator = torch.Generator().manual_seed(seed)
        self.games = []
        for replay in replays:
            states = [s for s in states_from_replay(replay)
                      if not s["terminated"]]
            if states:
                self.games.append(states)
        if not self.games:
            raise ValueError("no usable replay states")

    def __call__(self, env, mask):
        n, K = env.n, env.K
        mines = torch.zeros(n, K, dtype=torch.bool)
        revealed = torch.zeros(n, K, dtype=torch.bool)
        flagged = torch.zeros(n, K, dtype=torch.bool)
        first = torch.zeros(n, dtype=torch.int64)
        for i in range(n):
            g = self.games[int(torch.randint(len(self.games), (1,),
                                             generator=self.generator))]
            lo = int(len(g) * (1 - self.tail))
            j = int(torch.randint(lo, len(g), (1,),
                                  generator=self.generator)) if lo < len(g) else -1
            s = g[j]
            mines[i] = s["mines"]
            revealed[i] = s["revealed"]
            flagged[i] = s["flagged"]
            first[i] = s["first_cell"]
        return {"mines": mines, "revealed": revealed, "flagged": flagged,
                "first_cell": first,
                "hidden_count": (~mines & ~revealed).sum(dim=1)}
