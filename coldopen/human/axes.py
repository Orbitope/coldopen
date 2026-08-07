"""Reduce every game to the same two numbers, so they can be compared.

The hypothesis under test in E1 is that most real-time games expose something
like a *speed* and something like an *efficiency*, and that the mapping from
those two to skill has a shape that is shared across games. If it does, it is a
cold-start prior available to any game that logs them, with no simulation and no
per-game feature engineering.

Testing that requires a common representation. Each source is mapped to:

    speed        how much a player does per unit time
    efficiency   how much they get out of each thing they do
    skill        the game's own rating, as a percentile within the game

Percentiles rather than raw ratings, and z-scores within game rather than raw
metrics, because 3 pieces per second and 160% minesweeper efficiency are not
otherwise commensurable. What survives standardisation is the *shape* of the
relationship, which is the only thing that could transfer.

The pairing matters more than either axis alone. Across people speed and
efficiency correlate positively - better players are both faster and more
accurate - even though within one person they trade off. It is that cross-person
regularity the heuristic is trying to exploit.
"""

import numpy as np

#: How each source maps onto the two axes. Adding a game means adding a row here
#: and an ingestion module; nothing downstream changes.
AXES = {
    "tetrio": {
        "speed": "pieces per second",
        "efficiency": "attack per piece (apm / (pps * 60))",
        "skill": "Tetra Rating",
    },
    "minesweeper": {
        "speed": "3BV per second",
        "efficiency": "clicks used against board 3BV",
        "skill": "rank ladder position",
    },
    "jstris": {
        "speed": "blocks per second",
        "efficiency": "finesse (keystrokes against minimum)",
        "skill": "leaderboard position",
    },
}


def tetrio_axes(rounds, labels):
    """Per-round speed and efficiency for TETR.IO, joined to each player's rating.

    Efficiency is attack per piece, which is derived rather than reported:
    ``APP = APM / (PPS * 60)``. It is the quantity that separates a player who is
    fast and wasteful from one who is fast and economical, and on real records it
    runs from about 0.15 at the bottom of the ladder to about 0.89 at the top.

    ``vs_over_apm`` is carried alongside because it is not redundant with either
    axis: VS counts garbage cleared as well as attack sent, so the ratio is how
    much of the work was digging out rather than attacking. It has no analogue in
    the board games and is worth keeping to see whether it survives the transfer.
    """
    rating = {row["player"]: row["tr"] for row in labels}
    rank = {row["player"]: row["rank_index"] for row in labels}

    out = []
    for r in rounds:
        pps, apm = r.get("pps"), r.get("apm")
        if not pps or apm is None or r["player"] not in rating:
            continue
        pieces_per_minute = pps * 60.0
        app = apm / pieces_per_minute if pieces_per_minute > 0 else 0.0
        vs = r.get("vsscore") or 0.0
        out.append({
            "player": r["player"],
            "group": r.get("match"),
            "opponent": r.get("opponent"),
            "index": r.get("round", 0),
            "speed": float(pps),
            "efficiency": float(app),
            "skill": float(rating[r["player"]]),
            "rank_index": rank[r["player"]],
            "vs_over_apm": float(vs / apm) if apm > 0 else 0.0,
        })
    return out


def to_percentile(values):
    """Rank-transform to [0, 1]. Ties share the average position."""
    values = np.asarray(values, dtype=float)
    order = values.argsort()
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # Average the positions of tied values so a rating shared by many players
    # does not get an arbitrary ordering baked in.
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks / max(len(values) - 1, 1)


def standardise(rows, clip=4.0):
    """Z-score the two axes within a game and percentile-rank the skill label.

    Clipping guards the cross-game fit against a handful of outliers - a player
    with one absurd round - defining the scale for everyone else. Applied after
    standardisation so the threshold means the same thing in every game.
    """
    if not rows:
        return rows
    out = [dict(row) for row in rows]
    for axis in ("speed", "efficiency"):
        values = np.array([row[axis] for row in out], dtype=float)
        spread = values.std()
        centred = (values - values.mean()) / (spread if spread > 0 else 1.0)
        for row, value in zip(out, np.clip(centred, -clip, clip)):
            row[axis] = float(value)
    for row, pct in zip(out, to_percentile([row["skill"] for row in out])):
        row["skill_pct"] = float(pct)
    return out


def per_player(rows, budget=None):
    """Collapse rounds to one row per player, over their first ``budget`` rounds.

    The observation budget is the whole point: a matchmaker acting on a new
    account has seen a handful of rounds, not a career. Ordering is by the index
    each source assigned, so "first" means first observed rather than best.
    """
    by_player = {}
    for row in sorted(rows, key=lambda r: (r["player"], r.get("index", 0))):
        bucket = by_player.setdefault(row["player"], [])
        if budget is None or len(bucket) < budget:
            bucket.append(row)

    out = []
    for player, bucket in by_player.items():
        head = bucket[0]
        out.append({
            "player": player,
            "n": len(bucket),
            "speed": float(np.mean([r["speed"] for r in bucket])),
            "efficiency": float(np.mean([r["efficiency"] for r in bucket])),
            # Consistency is itself a skill marker, and it is free here.
            "speed_sd": float(np.std([r["speed"] for r in bucket])),
            "efficiency_sd": float(np.std([r["efficiency"] for r in bucket])),
            "skill": head["skill"],
            "skill_pct": head.get("skill_pct"),
            "rank_index": head.get("rank_index"),
        })
    return out
