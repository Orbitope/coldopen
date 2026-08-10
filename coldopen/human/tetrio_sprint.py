"""Human 40 LINES records for the players already sampled across the ladder.

The league ingestion (`tetrio.py`) established the sample: ~25 players per
rank, whole ladder. This adds each player's best 40L sprint — the single-player
mode whose statistics line up with what `tetris_sprint` agents produce:
`inputs`, `piecesplaced`, `holds`, the clears breakdown, `pps`, final time.

The label stays the league rank/TR (earned in versus), and the features come
from sprint — different modes, so the label cannot restate the features. Both
caveats from `coldopen/tetris.py` apply: input counting differs, so finesse is
compared standardised-within-population only.

    python -m coldopen.human.tetrio_sprint --out data/human/tetrio_sprint.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

from coldopen.human.client import pseudonymise
from coldopen.human.tetrio import RANK_INDEX, make_client, sample_ladder


def sprint_row(entry):
    results = entry.get("results", {})
    stats = results.get("stats", {})
    clears = stats.get("clears", {})
    pieces = stats.get("piecesplaced") or 0
    inputs = stats.get("inputs") or 0
    if not pieces or not inputs:
        return None
    finesse = stats.get("finesse") or {}
    total_clears = sum(clears.get(k, 0) for k in
                       ("singles", "doubles", "triples", "quads"))
    time_ms = None
    aggregate = results.get("aggregatestats", {})
    # 40L time arrives as stats["finaltime"] in ms on current records.
    if isinstance(stats.get("finaltime"), (int, float)):
        time_ms = float(stats["finaltime"])
    return {
        "inputs": inputs,
        "pieces": pieces,
        "inputs_per_piece": inputs / pieces,
        "holds": stats.get("holds") or 0,
        "holds_per_piece": (stats.get("holds") or 0) / pieces,
        "singles": clears.get("singles", 0),
        "doubles": clears.get("doubles", 0),
        "triples": clears.get("triples", 0),
        "quads": clears.get("quads", 0),
        "quad_rate": clears.get("quads", 0) / max(total_clears, 1),
        "max_b2b": stats.get("topbtb", 0),
        "lines": stats.get("lines", 0),
        "time_ms": time_ms,
        "pps": aggregate.get("pps"),
        "finesse_faults": (finesse.get("faults")
                           if isinstance(finesse, dict) else None),
    }


def ingest(per_rank=25, cache_dir="data/cache/tetrio"):
    client = make_client(cache_dir)
    players, spread = sample_ladder(client, per_rank=per_rank)
    print(f"{len(players)} players across {len(spread)} ranks (from cache)",
          flush=True)

    rows = []
    for i, player in enumerate(players, 1):
        league = player["league"]
        body = client.get_json(f"/users/{player['_id']}/records/40l/top",
                               {"limit": 1})
        entries = body.get("data", {}).get("entries", [])
        row = sprint_row(entries[0]) if entries else None
        if row is not None:
            row.update({
                "player": pseudonymise(player["_id"]),
                "rank": league["rank"],
                "rank_index": RANK_INDEX[league["rank"]],
                "tr": league["tr"],
            })
            rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(players)}: {len(rows)} with sprint records "
                  f"(cache {client.stats['hits']}h/{client.stats['misses']}m)",
                  flush=True)
    return {"game": "tetrio_sprint", "rows": rows, "fetch_stats": client.stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-rank", type=int, default=25)
    ap.add_argument("--cache", default="data/cache/tetrio")
    ap.add_argument("--out", default="data/human/tetrio_sprint.json")
    args = ap.parse_args()

    result = ingest(per_rank=args.per_rank, cache_dir=args.cache)
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    ranks = {}
    for row in result["rows"]:
        ranks[row["rank"]] = ranks.get(row["rank"], 0) + 1
    print(f"\n{len(result['rows'])} sprint records; by rank: {ranks}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
