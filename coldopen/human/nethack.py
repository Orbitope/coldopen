"""NetHack players from the nethack.alt.org public archive.

This is the only corpus found so far with **many games by the same identified
player**, which is what the cold-start question actually needs. Everywhere else
the observation budget has to be faked - N rounds of one match, N moves of one
game - because a player is only seen once. Here a player has a career, so the
experiment can be the real one: *predict how good somebody will turn out to be,
from the first few games they ever played.*

The archive publishes an xlogfile per server era: one line per game, key=value,
every game the server ever recorded. 8.6 million games in under 100 MB, which
makes this the cheapest human dataset in the project by a wide margin.

    speed        turns per second of wall-clock - how fast they actually play
    efficiency   score per turn - output per action
    skill        their mean score over *later* games, which the features never see

**The label is held out in time, not just in rows.** A player's skill is scored
from the second half of their career and predicted from the first. Using their
whole history as the label would leak the answer into the features, since the
observation window would be part of what it is being asked to predict. That
mistake would not look like a bug - it would look like a very good result.

**One caveat for cross-game use.** NetHack's efficiency axis and its skill label
are both built from score, at different times and on different games. That is
legitimate for the budget curve, and it is exactly what a matchmaker does, but
it makes NetHack's numbers *not* comparable with a game like TETR.IO where the
two are different quantities. Flagged rather than fixed, because the alternative
axes in NetHack are worse.

Two parsing traps the archive documents, both honoured here: the 3.4.3-era file
uses colon separators where every other file uses tabs, and the field set widens
over the years, so fields are read by name and every one is optional.

    python -m coldopen.human.nethack --xlogfile data/human/nh361.txt
"""

import argparse
import json
import math
import pathlib
from collections import defaultdict

from coldopen.human.client import pseudonymise

ARCHIVE = "https://archive.alt.org/archive/"

#: A game has to be long enough to have behaviour in it. The median NetHack
#: game on a public server lasts one turn - somebody starting a character and
#: quitting - and those carry no information about how anyone plays.
MIN_TURNS = 100
MIN_REALTIME = 10
#: Scores are wildly skewed (this file's maximum is 10^8, an exploit rather than
#: a game), so everything score-shaped is used on a log scale.
MAX_POINTS = 10 ** 7


def parse_xlogfile(path):
    """Games as dicts. Handles both the tab and the older colon convention."""
    games = []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            # The 3.4.3-era files predate the tab convention. Detect per line
            # rather than per file, since the archive warns the format is a
            # property of the era and not of the download.
            parts = line.split("\t") if "\t" in line else line.split(":")
            game = {}
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    game[key] = value
            if game:
                games.append(game)
    return games


def _number(game, key, default=0.0):
    try:
        return float(game.get(key, default))
    except (TypeError, ValueError):
        return default


def usable(game):
    """Long enough, timed, and not an obvious scoring exploit."""
    turns = _number(game, "turns")
    realtime = _number(game, "realtime")
    points = _number(game, "points")
    return (turns >= MIN_TURNS and realtime >= MIN_REALTIME
            and 0 <= points <= MAX_POINTS and game.get("name"))


def careers(games, min_games=20):
    """Group usable games by player, in the order they were played."""
    by_player = defaultdict(list)
    for game in games:
        if usable(game):
            by_player[game["name"]].append(game)
    for player, played in list(by_player.items()):
        if len(played) < min_games:
            del by_player[player]
            continue
        played.sort(key=lambda g: _number(g, "starttime"))
    return by_player


def nethack_rows(games, min_games=20):
    """Observation rows from the first half of each career, labelled by the second.

    The split is chronological and per player, so the label is always games the
    features were not computed from and could not have been.
    """
    rows = []
    for player, played in careers(games, min_games).items():
        split = len(played) // 2
        early, late = played[:split], played[split:]
        if not early or not late:
            continue

        # Skill: the score they go on to average, on a log scale because raw
        # NetHack scores span eight orders of magnitude.
        later = [_number(g, "points") for g in late]
        skill = math.log10(1.0 + sum(later) / len(later))

        key = pseudonymise(player)
        for index, game in enumerate(early):
            turns = _number(game, "turns")
            realtime = _number(game, "realtime")
            points = _number(game, "points")
            rows.append({
                "player": key,
                "group": key,
                "opponent": None,
                "index": index,
                "speed": turns / realtime,          # turns per wall-clock second
                "efficiency": points / turns,        # score per turn
                "skill": skill,
                "rank_index": int(min(9, max(0, skill * 1.5))),
                "maxlvl": _number(game, "maxlvl"),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlogfile", default="data/human/nh361.txt",
                    help=f"decompressed xlogfile; archives at {ARCHIVE}")
    ap.add_argument("--min-games", type=int, default=20,
                    help="career length needed to both observe and label a player")
    ap.add_argument("--out", default="data/human/nethack.json")
    args = ap.parse_args()

    games = parse_xlogfile(args.xlogfile)
    rows = nethack_rows(games, min_games=args.min_games)
    players = len({row["player"] for row in rows})
    print(f"{len(games):,} games -> {len(rows):,} observation rows "
          f"from {players:,} careers")

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"game": "nethack", "rows": rows}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
