"""Human Othello from the WTHOR database, replayed move by move.

This is the first corpus in the project that closes the gap the whole thing
turns on. Every other human dataset is for a game with no agent ladder, and
every agent ladder is for a game with no human data. WTHOR is Othello, and
Othello already has a measured ladder, a reference network and a feature
extractor - so the agents and the people can finally be put through *identical*
code and compared.

The French Othello Federation publishes 125,000 tournament games spanning forty
years and 3,650 players, free to download. Each record gives the two players,
the final disc count and the full move sequence, which is enough to replay the
game and compute exactly the features the agents are profiled with.

Two things have to be handled, and both are silent if you get them wrong.

**WTHOR omits passes.** Othello forces a player to pass when they have no legal
move, and the database records only the moves actually played. Replaying the
list literally desynchronises the board the first time a pass was skipped, and
from then on the moves still *look* plausible while belonging to the wrong
player. Passes are therefore reinserted by checking legality at every ply.

**There are no ratings in the file.** WTHOR gives results, not strengths. Rather
than joining against an external rating list - which would only cover the
tournament elite - skill is fitted from the games themselves with the same
Bradley-Terry code the agent league uses. That keeps the human labels and the
agent labels on one scale by construction, and it covers every player in the
database rather than the ranked few.

    python -m coldopen.human.wthor --years 2010-2024
"""

import argparse
import json
import pathlib
import struct
import urllib.request

import jax
import numpy as np
import torch

from coldopen import games as game_mod
from coldopen.features import othello as othello_features
from coldopen.human.client import USER_AGENT, pseudonymise
from coldopen.league import bradley_terry, to_elo

BASE = "https://www.ffothello.org/wthor/base/"
HEADER = 16
RECORD = 68
#: The move list is a fixed 60-byte field; a zero means the game ended earlier.
MOVE_BYTES = 60
PASS = 64


def fetch(name, cache_dir):
    """Download one WTHOR file, or reuse it. The archive is static."""
    path = pathlib.Path(cache_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    request = urllib.request.Request(BASE + name, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        blob = response.read()
    path.write_bytes(blob)
    return blob


def parse_players(blob):
    """Player names, indexed as the game records refer to them."""
    names = {}
    for index in range((len(blob) - HEADER) // 20):
        start = HEADER + 20 * index
        raw = blob[start:start + 20].decode("latin-1")
        names[index] = raw.replace("\x00", "").strip()
    return names


def parse_games(blob):
    """Games as (black, white, black_discs, moves) from one year file."""
    count = struct.unpack("<I", blob[4:8])[0]
    out = []
    for index in range(count):
        start = HEADER + RECORD * index
        record = blob[start:start + RECORD]
        if len(record) < RECORD:
            break
        _tournament, black, white = struct.unpack("<HHH", record[0:6])
        discs = record[6]
        moves = [m for m in record[8:8 + MOVE_BYTES] if m]
        if len(moves) < 20:
            continue
        out.append({"black": black, "white": white, "discs": discs,
                    "moves": [((m // 10) - 1) * 8 + ((m % 10) - 1) for m in moves]})
    return out


def replay(game, batch, reference=None, max_moves=20, seed=0):
    """Replay a batch of games, collecting each player's own move features.

    Games are stepped in lockstep so the environment work is batched; the only
    per-game logic is choosing whether the next recorded move can be played or a
    pass has to be inserted first.
    """
    device = game.device
    size = len(batch)
    state = game.init(size, seed=seed)
    key = jax.random.PRNGKey(seed + 3)
    names = list(othello_features.FEATURES.names)

    pointer = [0] * size
    alive = [True] * size
    rows = [[] for _ in range(size)]

    for _ in range(120):
        if not any(alive):
            break
        legal = game.legal_mask(state)
        actions = torch.zeros(size, dtype=torch.long, device=device)
        passing = [False] * size
        for i in range(size):
            if not alive[i] or pointer[i] >= len(batch[i]["moves"]):
                alive[i] = alive[i] and pointer[i] < len(batch[i]["moves"])
                actions[i] = PASS if bool(legal[i, PASS]) else 0
                passing[i] = True
                continue
            want = batch[i]["moves"][pointer[i]]
            if bool(legal[i, want]):
                actions[i] = want
            elif bool(legal[i, PASS]):
                actions[i] = PASS
                passing[i] = True
            else:
                alive[i] = False
                actions[i] = PASS if bool(legal[i, PASS]) else 0
                passing[i] = True

        key, sub = jax.random.split(key)
        feats = othello_features.FEATURES.extract(game, state, actions, sub)
        if reference is not None:
            from coldopen.profile import _reference_features
            feats.update(_reference_features(
                reference, game.observe(state), legal, actions))
        else:
            zeros = torch.zeros(size, device=device)
            feats["ref_agreement"], feats["ref_regret"] = zeros, zeros.clone()
        stacked = torch.stack([feats[n] for n in names
                               + ["ref_agreement", "ref_regret"]], dim=1)

        mover = game.current_player(state)
        for i in range(size):
            if alive[i] and not passing[i] and len(rows[i]) < max_moves:
                # Pgx seats the first player randomly, so who is "black" is read
                # from the seat rather than assumed.
                side = "black" if int(mover[i]) == 0 else "white"
                rows[i].append((side, stacked[i].tolist()))
                pointer[i] += 1
            elif alive[i] and not passing[i]:
                pointer[i] += 1

        key, sub = jax.random.split(key)
        state = game.step(state, actions, sub)

    return rows, names + ["ref_agreement", "ref_regret"]


def fit_ratings(games):
    """Bradley-Terry strengths over the recorded results, on an Elo scale.

    WTHOR stores the final disc count, so a game is a win, a loss or a 32-32
    draw. Players with very few games are dropped afterwards rather than here,
    because their games still carry information about the opponents they played.
    """
    players = sorted({g["black"] for g in games} | {g["white"] for g in games})
    index = {p: i for i, p in enumerate(players)}
    matrix = [[0.0] * len(players) for _ in players]
    for game in games:
        b, w = index[game["black"]], index[game["white"]]
        if game["discs"] > 32:
            matrix[b][w] += 1.0
        elif game["discs"] < 32:
            matrix[w][b] += 1.0
        else:
            matrix[b][w] += 0.5
            matrix[w][b] += 0.5
    strengths = bradley_terry(matrix)
    elos = to_elo(strengths, anchor_index=0, anchor_rating=0.0)
    # Centre on the median so the scale reads like a rating rather than an
    # offset from whichever player happened to be first in the list.
    middle = float(np.median(elos))
    return {p: elos[index[p]] - middle for p in players}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2010-2024")
    ap.add_argument("--max-games", type=int, default=8000)
    ap.add_argument("--min-games-per-player", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--reference", default="coldopen/ladders/othello_reference")
    ap.add_argument("--cache", default="data/cache/wthor")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="data/human/wthor.json")
    args = ap.parse_args()

    start, end = (int(x) for x in args.years.split("-"))
    players = parse_players(fetch("WTHOR.JOU", args.cache))
    games = []
    for year in range(start, end + 1):
        try:
            games.extend(parse_games(fetch(f"WTH_{year}.wtb", args.cache)))
        except Exception as err:  # a missing year is not worth stopping for
            print(f"  ! {year}: {err}")
    print(f"{len(games):,} games, {len(players):,} names on file", flush=True)

    ratings = fit_ratings(games)
    counts = {}
    for game in games:
        counts[game["black"]] = counts.get(game["black"], 0) + 1
        counts[game["white"]] = counts.get(game["white"], 0) + 1
    keep = {p for p, n in counts.items() if n >= args.min_games_per_player}
    print(f"  {len(keep):,} players with >= {args.min_games_per_player} games; "
          f"Elo spread {min(ratings[p] for p in keep):.0f} .. "
          f"{max(ratings[p] for p in keep):.0f}", flush=True)

    games = [g for g in games if g["black"] in keep and g["white"] in keep]
    games = games[:args.max_games]
    print(f"  replaying {len(games):,} games", flush=True)

    game_env = game_mod.make("othello", args.device)
    reference = None
    if pathlib.Path(args.reference).exists():
        from coldopen.predict import load_reference
        reference = load_reference(args.reference, game_env.info, args.device)
    print(f"  reference {'loaded' if reference is not None else 'MISSING'}",
          flush=True)

    rows, names = [], None
    for start_index in range(0, len(games), args.batch):
        chunk = games[start_index:start_index + args.batch]
        replayed, names = replay(game_env, chunk, reference=reference)
        for game, moves in zip(chunk, replayed):
            for side, values in moves:
                who = game["black"] if side == "black" else game["white"]
                rows.append({
                    "player": pseudonymise(f"wthor-{who}"),
                    "group": f"{game['black']}v{game['white']}",
                    "skill": ratings[who],
                    "features": values,
                })
        if (start_index // args.batch) % 5 == 0:
            print(f"    {start_index + len(chunk):,}/{len(games):,} games, "
                  f"{len(rows):,} move rows", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"game": "othello_human", "feature_names": names,
                               "rows": rows}, separators=(",", ":")))
    print(f"\n{len(rows):,} move rows from {len({r['player'] for r in rows}):,} "
          f"players\nwrote {out}")


if __name__ == "__main__":
    main()
