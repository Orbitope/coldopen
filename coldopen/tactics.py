"""Measure how tactical each game is, instead of asserting it.

The project's claim is that some property of a game decides how early skill
becomes legible, and the candidate property is tactical density: how often a
position contains something concrete to see. It would be circular to assert
"Connect Four is tactical, Othello is not" and then explain the accuracy curves
with it, so this measures two things directly, by search rather than by opinion:

**win_available** - the fraction of positions in which the player to move has a
move that ends the game in their favour on the spot. This is the thing
``missed_win`` in the Connect Four feature set detects, and the reason that
feature has anything to detect.

**blunder_available** - the fraction of positions in which *some* legal move
lets the opponent end the game in their favour immediately afterwards. This is
the thing ``missed_block`` detects: a position where it is possible to lose on
the next ply by choosing badly. It is the more interesting of the two, because
punishing a blunder is what separates a mid tier from a low one.

Both are two-ply exhaustive searches, so both are exact for the positions
sampled and neither depends on any network.

Leduc is measured the same way but does not mean the same thing, and its numbers
should not be read across: a hand there ends when somebody folds or calls, so
"a move that ends the game in your favour" is usually just the last call of a
hand you were already winning, not a tactic anybody had to see. Positions are drawn from play at a
chosen exploration rate: at epsilon 1 they describe the game as beginners meet
it, and with a policy supplied they describe the game as it is actually played,
which is not the same distribution - a strong player steers away from positions
with cheap tactics in them.

    python -m coldopen.tactics --out analysis/tactics.json
"""

import argparse
import json
import pathlib

import jax
import torch

from coldopen import games as game_mod
from coldopen.nets import epsilon_actions


def _wins_now(game, state, key):
    """[B] bool: does the player to move have a move that wins immediately?

    Exhaustive over the action space. Pgx states are immutable, so each trial is
    just another call to step.
    """
    device = game.device
    legal = game.legal_mask(state)
    mover = game.current_player(state)
    found = torch.zeros(state.terminated.shape[0], dtype=torch.bool, device=device)
    for action in range(game.info.n_actions):
        playable = legal[:, action]
        if not bool(playable.any()):
            continue
        key, sub = jax.random.split(key)
        trial = game.step(state, torch.full_like(mover, action), sub)
        won = game.terminated(trial) & (game.reward_for(trial, mover) > 0)
        found |= won & playable
    return found


def position_stats(game, samples=192, epsilon=1.0, net=None, seed=0,
                   measure_points=10):
    """Two-ply tactical statistics over positions drawn from the whole game.

    Sampling matters more here than it looks. An earlier version measured the
    first few plies and reported that Connect Four contains no tactics at all,
    which is true of the opening and of nothing else: nobody can have four in a
    row on move two. So this walks a full episode's worth of plies, restarting
    games as they end, and measures at ``measure_points`` evenly spaced stops
    along the way - opening, middlegame and endgame in the proportions the game
    actually produces them.

    The walk is cheap; only the stops are expensive, since each one is a two-ply
    exhaustive search over the action space.
    """
    device = game.device
    gen = torch.Generator().manual_seed(seed)
    state = game.init(samples, seed=seed)
    key = jax.random.PRNGKey(seed + 1)

    walk = game.info.max_plies
    stride = max(1, walk // measure_points)

    win_hits, blunder_hits, seen = 0, 0, 0
    legal_total = 0.0
    stops = 0
    for ply in range(walk):
        if stops >= measure_points:
            break
        if ply % stride:
            acts = epsilon_actions(
                net, game.observe(state), game.legal_mask(state), epsilon, gen, device)
            key, sub = jax.random.split(key)
            state = game.step(state, acts, sub)
            key, sub = jax.random.split(key)
            state = game.restart(state, game.finished(state), sub)
            continue
        stops += 1

        live = ~game.finished(state)
        if not bool(live.any()):
            break
        legal = game.legal_mask(state)
        mover = game.current_player(state)

        key, sub = jax.random.split(key)
        has_win = _wins_now(game, state, sub)

        # A blunder is available when some legal move hands the opponent an
        # immediate win. Try each of our moves, then search the reply.
        #
        # "The reply" is only the opponent's when the move actually passed the
        # turn. It often does not: a backgammon turn spends one action per die,
        # and Othello makes a player pass when they have nothing. Searching
        # those without the guard would find *our own* winning follow-up and
        # score it as a blunder.
        gives = torch.zeros(samples, dtype=torch.bool, device=device)
        for action in range(game.info.n_actions):
            playable = legal[:, action]
            if not bool(playable.any()):
                continue
            key, sub = jax.random.split(key)
            after = game.step(state, torch.full_like(mover, action), sub)
            handed_over = (game.current_player(after) != mover) & ~game.terminated(after)
            key, sub = jax.random.split(key)
            reply_wins = _wins_now(game, after, sub) & handed_over
            gives |= reply_wins & playable

        win_hits += int((has_win & live).sum())
        blunder_hits += int((gives & live).sum())
        seen += int(live.sum())
        legal_total += float(legal[live].sum())

        acts = epsilon_actions(
            net, game.observe(state), legal, epsilon, gen, device)
        key, sub = jax.random.split(key)
        state = game.step(state, acts, sub)
        key, sub = jax.random.split(key)
        state = game.restart(state, game.finished(state), sub)

    return {
        "positions": seen,
        "measured_at_plies": stops,
        "win_available": win_hits / max(seen, 1),
        "blunder_available": blunder_hits / max(seen, 1),
        "mean_legal_moves": legal_total / max(seen, 1),
    }


def episode_length(game, samples=256, seed=0):
    """Mean plies per episode under random play, for context on the ply budgets."""
    device = game.device
    gen = torch.Generator().manual_seed(seed)
    state = game.init(samples, seed=seed)
    key = jax.random.PRNGKey(seed + 2)
    for ply in range(game.info.max_plies):
        if bool(game.finished(state).all()):
            break
        acts = epsilon_actions(
            None, game.observe(state), game.legal_mask(state), 1.0, gen, device)
        key, sub = jax.random.split(key)
        state = game.step(state, acts, sub)
    return float(game.step_count(state).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/tactics.json")
    ap.add_argument("--samples", type=int, default=192)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for key in game_mod.ALL_GAMES:
        game = game_mod.make(key, args.device)
        print(f"{key} ...", flush=True)
        stats = position_stats(game, samples=args.samples, seed=args.seed)
        stats.update({
            "game": key,
            "expected_tactics": game.info.expected_tactics,
            "information": game.info.information,
            "chance": game.info.chance,
            "n_actions": game.info.n_actions,
            "mean_episode_plies": round(episode_length(game, seed=args.seed), 1),
        })
        rows.append(stats)
        print(f"  win available {stats['win_available']:.4f}  "
              f"blunder available {stats['blunder_available']:.4f}  "
              f"legal moves {stats['mean_legal_moves']:.1f}  "
              f"episode {stats['mean_episode_plies']:.0f} plies", flush=True)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
