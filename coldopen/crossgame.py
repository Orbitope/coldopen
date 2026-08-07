"""E1: does the speed/efficiency heuristic transfer between games?

Most real-time games log something like actions-per-minute and something like an
efficiency ratio. If the mapping from those two to skill has a shape that is
shared across games, it is a cold-start prior available to any game that records
them - no simulation, no per-game feature engineering, no training.

The test is deliberately the awkward one. Fitting a model on a game and scoring
it on held-out players of the *same* game measures nothing interesting: of course
faster players are better. The question is whether a model fitted on game A
predicts skill in game B, having never seen it. So the headline number is the
off-diagonal of a transfer matrix, and the diagonal is only there as the ceiling
to measure the off-diagonal against.

Two decisions keep that comparison honest:

**The model is deliberately weak.** Ridge regression on two standardised
inputs. A gradient-boosted model would score better on the diagonal and worse
off it, because the extra capacity goes into game-specific quirks - which is the
opposite of what is being tested. If a simple linear shape transfers, that is
the finding; if only a complicated one does, it is not a heuristic.

**Everything is standardised within game.** Three pieces per second and 160%
minesweeper efficiency are not comparable numbers. What is comparable is where a
player sits in their own game's distribution, so both axes become z-scores and
the skill label becomes a percentile. What survives that is shape.

    python -m coldopen.crossgame --out analysis/crossgame.json
"""

import argparse
import json
import pathlib

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

#: Observation budgets, in rounds/games seen per player. The cold-start question
#: is what can be said after a handful, so the curve matters more than any
#: single number.
BUDGETS = [1, 2, 3, 5, 10, None]

FEATURES = ("speed", "efficiency")


def design(rows, features=FEATURES):
    x = np.array([[row[f] for f in features] for row in rows], dtype=float)
    y = np.array([row["skill_pct"] for row in rows], dtype=float)
    return x, y


def spearman(a, b):
    """Rank correlation, without pulling in scipy for one function."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def within_game(rows, features=FEATURES, seed=0, splits=5):
    """Cross-validated fit on one game, grouped by player.

    Grouping matters: several rounds of one player's play are not independent
    observations, and a fold that trains and tests on the same person is scoring
    memorisation.
    """
    x, y = design(rows, features)
    groups = np.array([row["player"] for row in rows])
    unique = len(np.unique(groups))
    if unique < 2 or len(y) < 10:
        return None

    preds = np.zeros_like(y)
    splitter = GroupKFold(n_splits=min(splits, unique))
    for train, test in splitter.split(x, y, groups):
        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(x[train], y[train])
        preds[test] = model.predict(x[test])
    return {
        "n": int(len(y)),
        "players": int(unique),
        "spearman": spearman(preds, y),
        "mae_percentile": float(np.abs(preds - y).mean()),
    }


def transfer(train_rows, test_rows, features=FEATURES, seed=0):
    """Fit on one game, predict another. The number this experiment is about."""
    xt, yt = design(train_rows, features)
    xs, ys = design(test_rows, features)
    if len(yt) < 10 or len(ys) < 10:
        return None
    model = Ridge(alpha=1.0, random_state=seed)
    model.fit(xt, yt)
    preds = model.predict(xs)
    return {
        "n": int(len(ys)),
        "spearman": spearman(preds, ys),
        "mae_percentile": float(np.abs(preds - ys).mean()),
        "coefficients": {f: float(c) for f, c in zip(features, model.coef_)},
    }


#: Each axis alone, and the pair. The ablation is not optional colour: a
#: two-axis transfer score is only evidence of a transferable *shape* if it
#: beats what one axis achieves on its own. Standardised single-axis models have
#: almost nothing to transfer - the prediction is essentially the z-score - so
#: they set the floor that the pair has to clear.
ABLATIONS = {
    "speed_only": ("speed",),
    "efficiency_only": ("efficiency",),
    "both": ("speed", "efficiency"),
}


def ablate(games, seed=0):
    """Within-game and transfer scores for each axis alone and for the pair."""
    out = {}
    for name, features in ABLATIONS.items():
        entry = {"within": {}, "transfer": {}}
        for game, rows in games.items():
            scored = within_game(rows, features, seed)
            entry["within"][game] = scored["spearman"] if scored else None
        for source in games:
            for target in games:
                if source == target:
                    continue
                scored = transfer(games[source], games[target], features, seed)
                entry["transfer"][f"{source}->{target}"] = (
                    scored["spearman"] if scored else None)
        out[name] = entry
    return out


def run(games, features=FEATURES, seed=0):
    """Full transfer matrix over every ordered pair of games.

    ``games`` maps a name to already-standardised per-player rows.
    """
    names = sorted(games)
    result = {"games": names, "within": {}, "transfer": {}, "features": list(features)}
    for name in names:
        result["within"][name] = within_game(games[name], features, seed)
    for source in names:
        for target in names:
            if source == target:
                continue
            key = f"{source}->{target}"
            result["transfer"][key] = transfer(
                games[source], games[target], features, seed)
    return result


def budget_curve(prepared, features=FEATURES, seed=0):
    """Within-game quality against how many rounds of a player have been seen."""
    curves = {}
    for name, rows in prepared.items():
        curve = []
        for budget in BUDGETS:
            from coldopen.human.axes import per_player
            capped = per_player(rows, budget=budget)
            scored = within_game(capped, features, seed)
            if scored:
                curve.append({"budget": budget, **scored})
        curves[name] = curve
    return curves


def describe(result):
    print(f"\n{'':14}" + "".join(f"{g:>14}" for g in result["games"]))
    for source in result["games"]:
        row = f"{source:14}"
        for target in result["games"]:
            if source == target:
                w = result["within"].get(source)
                row += f"{(w['spearman'] if w else 0):>13.3f}*"
            else:
                t = result["transfer"].get(f"{source}->{target}")
                row += f"{(t['spearman'] if t else 0):>14.3f}"
        print(row)
    print("  rows are the game fitted on, columns the game predicted; "
          "* is within-game (the ceiling)")

    if "ablation" not in result:
        return
    print(f"\n{'feature set':18}" + "".join(
        f"{k:>22}" for k in sorted(result["ablation"]["both"]["transfer"])))
    for name in ("speed_only", "efficiency_only", "both"):
        entry = result["ablation"][name]
        row = f"{name:18}"
        for key in sorted(entry["transfer"]):
            row += f"{(entry['transfer'][key] or 0):>22.3f}"
        print(row)
    print("  a pair that does not beat its own best single axis has not shown "
          "a transferable shape")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="data/human",
                    help="directory of per-game ingested json")
    ap.add_argument("--out", default="analysis/crossgame.json")
    ap.add_argument("--budget", type=int, default=None,
                    help="rounds observed per player; default all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from coldopen.human import load_csv_game, load_game
    from coldopen.human.axes import per_player

    directory = pathlib.Path(args.human)
    prepared, rounds = {}, {}
    sources = [(p.stem, p, load_game) for p in sorted(directory.glob("*.json"))]
    # Not every source is an API response; SkillCraft ships as a flat file.
    sources += [("skillcraft", p, None)
                for p in sorted(directory.glob("SkillCraft*.csv"))]

    for name, path, loader in sources:
        if loader is None:
            rows = load_csv_game(name, path)
        else:
            rows = loader(name, json.loads(path.read_text()))
        if not rows:
            print(f"  ! {name}: no usable rows")
            continue
        rounds[name] = rows
        prepared[name] = per_player(rows, budget=args.budget)
        print(f"  {name}: {len(rows)} rows -> {len(prepared[name])} players")

    if not prepared:
        raise SystemExit(f"no ingested games found in {directory}")

    result = run(prepared, seed=args.seed)
    result["ablation"] = ablate(prepared, seed=args.seed)
    result["budget_curves"] = budget_curve(rounds, seed=args.seed)
    describe(result)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {path}")
    if len(prepared) < 2:
        print("note: transfer needs at least two games; only the diagonal is "
              "meaningful here.")


if __name__ == "__main__":
    main()
