"""Measure how strong each checkpoint of any game's run actually is.

Same argument as ``league.py``, which does this for the bespoke Connect Four
implementation: training step is a poor proxy for skill, so every checkpoint
plays a round robin and the results are fitted with Bradley-Terry onto an Elo
scale anchored by a uniform-random player. The maths is shared - only the
game-playing is rewritten against the Pgx adapter.

One thing does change across games. Two deterministic networks playing Connect
Four produce the same game every time, so a round robin there is one game per
pair dressed up as many, and the fix is a few random opening plies. Backgammon
and Leduc deal themselves fresh randomness every episode, and there a forced
random opening would be a real handicap rather than a decorrelation, so it is
applied only to the games that need it.

The other thing that changes is what counts as winning. Connect Four and Othello
are won or lost; backgammon pays 1, 2 or 3; Leduc pays a pot. Counting hands won
would rank a Leduc player who folds everything above one who plays well, since
folding wins the majority of *hands* and loses the money. So the Bradley-Terry
matrix is fed the mean normalised result, ``(r + 1) / 2`` on the adapter's
[-1, 1] reward scale, rather than a win count. For the two games that are simply
won or lost this is identically ``wins + draws / 2`` and nothing changes; for
the other two it is the metric that game actually settles on.

    python -m coldopen.ladder --game othello --games 200
"""

import argparse
import json
import pathlib
import re

import jax
import torch

from coldopen import games as game_mod
from coldopen.league import RANDOM_PLAYER, bradley_terry, monotonicity, pick_tiers, to_elo
from coldopen.nets import epsilon_actions, load_checkpoint


def load_ladder(directory, info, device="cpu"):
    """Every saved network for this game, ordered by plies, plus a random anchor."""
    directory = pathlib.Path(directory)
    entries = []
    for path in sorted(directory.glob("ckpt_*.pt")):
        plies = int(re.search(r"ckpt_(\d+)\.pt", path.name).group(1))
        entries.append({
            "id": f"ckpt_{plies}",
            "plies": plies,
            "net": load_checkpoint(path, info, device),
        })
    entries.sort(key=lambda e: e["plies"])
    return [{"id": RANDOM_PLAYER, "plies": -1, "net": None}] + entries


def play_pair(game, net_a, net_b, n_games, seed=0, epsilon=0.03, opening_plies=None):
    """Play ``n_games`` between two policies, seats alternating.

    Returns ``(wins_a, wins_b, draws, score_a)``, where ``score_a`` is the mean
    of ``(r + 1) / 2`` over A's normalised results - the quantity the league
    fits, equal to ``(wins_a + draws / 2) / n_games`` whenever the game is
    simply won or lost. A game that runs past the ply cap without terminating
    counts as a draw; only badly played backgammon manages it.
    """
    info = game.info
    device = game.device
    if opening_plies is None:
        opening_plies = 0 if info.chance else 2

    gen = torch.Generator().manual_seed(seed)
    state = game.init(n_games, seed=seed)
    key = jax.random.PRNGKey(seed + 1)
    seat_a = (torch.arange(n_games, device=device) % 2).long()

    for _ in range(opening_plies):
        legal = game.legal_mask(state)
        acts = epsilon_actions(None, game.observe(state), legal, 0.0, gen, device)
        key, sub = jax.random.split(key)
        state = game.step(state, acts, sub)

    result = torch.zeros(n_games, device=device)
    settled = game.finished(state)
    for _ in range(info.max_plies):
        if bool(settled.all()):
            break
        obs, legal = game.observe(state), game.legal_mask(state)
        a_turn = game.current_player(state) == seat_a
        acts = torch.zeros(n_games, dtype=torch.long, device=device)
        if a_turn.any():
            acts[a_turn] = epsilon_actions(
                net_a, obs[a_turn], legal[a_turn], epsilon, gen, device)
        if (~a_turn).any():
            acts[~a_turn] = epsilon_actions(
                net_b, obs[~a_turn], legal[~a_turn], epsilon, gen, device)
        key, sub = jax.random.split(key)
        state = game.step(state, acts, sub)
        # Pgx zeroes the rewards of a state stepped after it terminated, so the
        # outcome has to be latched the moment each game finishes.
        just = game.finished(state) & ~settled
        if just.any():
            result[just] = game.reward_for(state, seat_a)[just]
            settled |= just

    wins_a = int((result > 0).sum())
    wins_b = int((result < 0).sum())
    score_a = float(((result.clamp(-1.0, 1.0) + 1.0) / 2.0).mean())
    return wins_a, wins_b, n_games - wins_a - wins_b, score_a


def run_league(game, entries, n_games=200, seed=0, epsilon=0.03):
    n = len(entries)
    matrix = [[0.0] * n for _ in range(n)]
    records = []
    for i in range(n):
        for j in range(i + 1, n):
            wa, wb, dr, score = play_pair(
                game, entries[i]["net"], entries[j]["net"], n_games,
                seed=seed + i * 97 + j, epsilon=epsilon,
            )
            matrix[i][j] += score * n_games
            matrix[j][i] += (1.0 - score) * n_games
            records.append({
                "a": entries[i]["id"], "b": entries[j]["id"],
                "wins_a": wa, "wins_b": wb, "draws": dr,
                "score_a": score,
            })
            print(f"  {entries[i]['id']:>16} vs {entries[j]['id']:<16} "
                  f"{wa:4}-{wb:4}-{dr:4}  score {score:.3f}", flush=True)

    strengths = bradley_terry(matrix)
    elos = to_elo(strengths)
    table = [
        {"id": e["id"], "plies": e["plies"], "elo": round(elos[k], 1),
         "strength": strengths[k]}
        for k, e in enumerate(entries)
    ]
    return {"table": table, "pairs": records, "games_per_pair": n_games}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="othello", choices=list(game_mod.ALL_GAMES))
    ap.add_argument("--checkpoints")
    ap.add_argument("--out")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--tiers", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt_dir = args.checkpoints or f"coldopen/ladders/{args.game}"
    out = args.out or f"analysis/{args.game}/league.json"

    game = game_mod.make(args.game, args.device)
    entries = load_ladder(ckpt_dir, game.info, args.device)
    print(f"{args.game}: {len(entries)} players (including the random anchor)", flush=True)

    result = run_league(game, entries, n_games=args.games, seed=args.seed)
    result["game"] = args.game
    result["monotonicity"] = monotonicity(result["table"])
    result["tiers"] = pick_tiers(result["table"], args.tiers)

    path = pathlib.Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))

    print(f"\n{'id':>18} {'plies':>10} {'elo':>8}")
    for row in sorted(result["table"], key=lambda r: r["elo"]):
        print(f"{row['id']:>18} {row['plies']:>10} {row['elo']:>8.1f}")
    m = result["monotonicity"]
    print(f"\nmonotone: {m['strictly_monotone']}  inversions "
          f"{m['adjacent_inversions']}/{m['adjacent_pairs']}  "
          f"concordance {m['rank_concordance']:.3f}  range {m['elo_range']:.0f} Elo")
    print("\ntiers: " + ", ".join(f"{t['tier']}={t['id']}({t['elo']:.0f})"
                                 for t in result["tiers"]))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
