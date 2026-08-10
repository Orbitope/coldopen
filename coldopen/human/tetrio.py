"""Ingest real TETR.IO players: rank labels and per-round behaviour.

This is the human half of the TETR.IO experiment, and it uses only the public
TETRA CHANNEL API - no authentication, no replays. That restriction is not a
compromise: replay downloads run through the main game API, which is off limits
without written consent, and the documented endpoints turn out to carry more
than expected anyway.

A league record contains both players' statistics *broken down per round*, so a
first-to-seven match yields up to thirteen sequential observations of each
player. That is what keeps an accuracy-versus-N curve meaningful here, with N
counting rounds observed rather than moves.

What comes back per round, per player:

    apm              attack per minute
    pps              pieces per second
    vsscore          TETR.IO's own composite
    garbagesent      lines pushed to the opponent
    garbagereceived  lines taken
    btb              back-to-back count: did they chain difficult clears
    kills            knockouts
    lifetime         milliseconds survived

``wins`` is also present and is deliberately not collected. The whole premise is
reading skill from behaviour rather than from results, and a feature set with
the result in it would answer a different, easier question.

Players are pseudonymised at ingest. Grouping records by player and attaching a
rank needs a stable key, not a name.

    python -m coldopen.human.tetrio --players 400 --out data/human/tetrio.json
"""

import argparse
import json
import pathlib

from coldopen.human.client import PoliteClient, pseudonymise

API = "https://ch.tetr.io/api"

#: TETRA LEAGUE ranks from worst to best. Ordinal, so a classifier can be scored
#: on distance rather than only on exact match.
RANKS = ["d", "d+", "c-", "c", "c+", "b-", "b", "b+", "a-", "a", "a+",
         "s-", "s", "s+", "ss", "u", "x", "x+"]
RANK_INDEX = {r: i for i, r in enumerate(RANKS)}

#: Per-round fields worth keeping. Deliberately excludes `wins` and anything
#: else that describes the outcome rather than the play.
ROUND_STATS = ("apm", "pps", "vsscore", "garbagesent", "garbagereceived",
               "btb", "kills")


def make_client(cache_dir="data/cache/tetrio"):
    # The API asks for roughly one request a second and for its cache windows to
    # be respected; records are historical, so anything on disk stays good.
    return PoliteClient(API, cache_dir, min_interval=1.0, session_id="coldopen-ingest")


#: Where to enter the leaderboard to find each rank. The list is sorted by
#: rating, so walking it from the top yields nothing but the very best players -
#: a first attempt at this collected sixty accounts and every one was X+, which
#: is precisely the population the cold-start problem is *not* about. The
#: `after` cursor takes a rating, so instead of walking, drop in at eighteen
#: places chosen to land in each rank. Values found by probing; ``None`` means
#: start at the top.
PROBE_TR = {
    "x+": None, "x": 24000, "u": 22000, "ss": 20000, "s+": 18000, "s": 16000,
    "s-": 14000, "a+": 12500, "a": 12000, "a-": 10000, "b+": 8000, "b": 6000,
    "b-": 5000, "c+": 3000, "c": 2200, "c-": 1400, "d+": 700, "d": 250,
}


def sample_ladder(client, per_rank=25, batch=100):
    """Players from every rank, by entering the leaderboard once per rank.

    A probe does not always return the rank it was aimed at - rating boundaries
    move, and the entry points are approximate - so whatever comes back is kept
    and the per-rank cap is applied across the union. The probes exist to
    guarantee coverage, not to classify.
    """
    found, seen = {}, {}
    for rank, tr in PROBE_TR.items():
        params = {"limit": batch}
        if tr is not None:
            params["after"] = f"{tr}:0:0"
        body = client.get_json("/users/by/league", params)
        for player in body.get("data", {}).get("entries", []):
            actual = player.get("league", {}).get("rank")
            if actual not in RANK_INDEX or player["_id"] in found:
                continue
            if seen.get(actual, 0) >= per_rank:
                continue
            seen[actual] = seen.get(actual, 0) + 1
            found[player["_id"]] = player
        del rank
    return list(found.values()), seen


def player_rounds(client, player, limit=10):
    """Every round of this player's recent league matches.

    Returns one row per round. The opponent is carried as a pseudonym so that
    cross-validation can group by opponent and by match - a weak opponent leaves
    more room to look good, and rounds within one match are not independent
    because the player who is losing is eating more garbage.
    """
    user_id = player["_id"]
    body = client.get_json(f"/users/{user_id}/records/league/recent",
                           {"limit": limit})
    entries = body.get("data", {}).get("entries", [])

    me = pseudonymise(user_id)
    rows = []
    for match in entries:
        rounds = match.get("results", {}).get("rounds") or []
        for round_index, seats in enumerate(rounds):
            mine = next((s for s in seats if s.get("id") == user_id), None)
            theirs = next((s for s in seats if s.get("id") != user_id), None)
            if mine is None:
                continue
            stats = mine.get("stats") or {}
            if not stats.get("pps"):
                # A round the player never really started - no pieces placed,
                # so every rate statistic is either zero or undefined.
                continue
            rows.append({
                "player": me,
                "match": match["_id"],
                "opponent": pseudonymise(theirs["id"]) if theirs else None,
                "round": round_index,
                "lifetime": mine.get("lifetime"),
                **{name: stats.get(name) for name in ROUND_STATS},
            })
    return rows


def ingest(per_rank=25, matches=10, cache_dir="data/cache/tetrio"):
    client = make_client(cache_dir)
    print(f"sampling {len(PROBE_TR)} points across the ladder ...", flush=True)
    kept, spread = sample_ladder(client, per_rank=per_rank)
    print(f"  {len(kept)} players across {len(spread)} of {len(RANKS)} ranks",
          flush=True)
    print("  " + "  ".join(f"{r}:{spread[r]}" for r in RANKS if r in spread),
          flush=True)

    labels, rows = [], []
    for i, player in enumerate(kept, 1):
        league = player["league"]
        labels.append({
            "player": pseudonymise(player["_id"]),
            "rank": league["rank"],
            "rank_index": RANK_INDEX[league["rank"]],
            "tr": league["tr"],
            "glicko": league.get("glicko"),
            "rd": league.get("rd"),
            "gxe": league.get("gxe"),
            "gamesplayed": league.get("gamesplayed"),
        })
        rows.extend(player_rounds(client, player, limit=matches))
        if i % 25 == 0:
            print(f"  {i}/{len(kept)} players, {len(rows)} rounds "
                  f"(cache {client.stats['hits']}h/{client.stats['misses']}m)",
                  flush=True)

    return {
        "game": "tetrio",
        "labels": labels,
        "rounds": rows,
        "fetch_stats": client.stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-rank", type=int, default=25,
                    help="players to keep per rank; 18 ranks, so x18 in total")
    ap.add_argument("--matches", type=int, default=10,
                    help="recent matches per player")
    ap.add_argument("--cache", default="data/cache/tetrio")
    ap.add_argument("--out", default="data/human/tetrio.json")
    args = ap.parse_args()

    result = ingest(per_rank=args.per_rank, matches=args.matches,
                    cache_dir=args.cache)
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    print(f"\n{len(result['labels'])} players, {len(result['rounds'])} rounds")
    print(f"network: {result['fetch_stats']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
