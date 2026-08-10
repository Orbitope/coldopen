"""StarCraft II players from the SkillCraft1 dataset.

The second game in the E1 panel, and chosen because it is a *download* rather
than a crawl: 3,395 rated players with screen-movement telemetry, published for
research by the authors who introduced the perception-action cycle measure.

It earns its place by having a skill label that is genuinely a third thing.
`LeagueIndex` comes from the StarCraft ladder - Bronze through Professional,
decided by winning and losing - while `APM` is measured from the replay. Nothing
about the label is computed from the behaviour, which is exactly the property
that disqualified Jstris, where a sprint rating *is* the completion time and
speed would predict skill tautologically.

Two honest weaknesses, both of which make this a weaker instrument than TETR.IO
rather than an invalid one:

**There is no direct output-per-action measure.** In Tetris, efficiency is
attack per piece: a quantity the game itself produces. StarCraft II has no
equivalent in this dataset - no damage dealt, no resources denied - so the
closest analogue is the share of a player's actions that are efficient hotkey
selections rather than slow mouse work. That is a real efficiency-of-execution
measure and the direct analogue of Tetris finesse, but it is a proxy for output
rather than a measurement of it.

**One row per player.** The dataset is already aggregated over a game, so this
source can contribute to the cross-game shape test but not to the
observation-budget curve, which needs repeated observations of one player.
"""

import csv
import pathlib

#: Bronze(1) through Professional(8). Ordinal and ladder-derived.
LEAGUES = {1: "bronze", 2: "silver", 3: "gold", 4: "platinum",
           5: "diamond", 6: "master", 7: "grandmaster", 8: "professional"}

SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/272/"
    "skillcraft1+master+table+dataset.zip"
)


def _number(row, name):
    """SkillCraft marks missing values with '?', so parse rather than cast."""
    try:
        return float(row[name])
    except (ValueError, KeyError, TypeError):
        return None


def skillcraft_axes(path):
    """Rows in the common (speed, efficiency, skill) representation.

    speed       APM, actions per minute
    efficiency  hotkey selections per action - what share of the work was done
                the fast way. Normalising by APM is what makes it an efficiency
                rather than another measure of raw rate; the unnormalised field
                correlates 0.82 with APM and would simply be speed again.
    skill       LeagueIndex, from the ladder
    """
    rows = []
    with open(path, newline="") as fh:
        for record in csv.DictReader(fh):
            apm = _number(record, "APM")
            hotkeys = _number(record, "SelectByHotkeys")
            league = _number(record, "LeagueIndex")
            if not apm or hotkeys is None or league is None:
                continue
            actions_per_second = apm / 60.0
            rows.append({
                # Each row is one player observed once; the identifier exists so
                # grouped cross-validation has something to group on.
                "player": f"sc2-{record['GameID']}",
                "group": f"sc2-{record['GameID']}",
                "opponent": None,
                "index": 0,
                "speed": apm,
                "efficiency": hotkeys / actions_per_second,
                "skill": league,
                "rank_index": int(league) - 1,
            })
    return rows


def load(path="data/human/SkillCraft1_Dataset.csv"):
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download and unzip:\n  {SOURCE_URL}")
    return skillcraft_axes(path)
