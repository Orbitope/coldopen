"""Predict a player's skill tier from how they move, not from whether they win.

The cold-start problem this addresses: a brand new account has no results, so a
rating system has nothing to go on and starts everyone in the middle. That is
precisely the gap a shark walks through - a fresh account drops them into
beginner lobbies for however many matches it takes the rating to catch up.

If skill is legible in the moves themselves, it does not have to take that long.
Here the ladder from league.py provides players of known strength, telemetry.py
turns their games into features, and a gradient-boosted classifier tries to name
the tier from the first N moves alone.

    python -m coldopen.classifier --out analysis/telemetry.json
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from coldopen.league import load_checkpoints
from coldopen.telemetry import FEATURE_NAMES, TACTICAL_FEATURES, aggregate, collect_games

MOVE_BUDGETS = [1, 2, 3, 5, 8, 12, 16, 20]


def load_reference(directory, device="cpu"):
    """The strongest checkpoint of the *independent* run, used as the yardstick."""
    entries = load_checkpoints(directory, device)
    trained = [e for e in entries if e["net"] is not None]
    return trained[-1]["net"] if trained else None


def build_dataset(tiers, checkpoints, reference, games_per_tier=600, max_moves=20,
                  device="cpu", seed=0):
    """Play every tier against a spread of opponents and record their telemetry.

    Opponents are drawn from across the whole ladder rather than matched by
    strength, so the classifier cannot cheat by reading how strong the *other*
    player is - which is a real risk, since a weak opponent leaves more winning
    moves lying around and would otherwise make a strong player look sharper.
    """
    by_id = {e["id"]: e["net"] for e in checkpoints}
    opponents = [e["net"] for e in checkpoints]

    feats, counts, labels, groups = [], [], [], []
    per_opponent = max(1, games_per_tier // len(opponents))
    for tier_index, tier in enumerate(tiers):
        agent = by_id[tier["id"]]
        for opp_index, opponent in enumerate(opponents):
            f, c = collect_games(
                agent, opponent, per_opponent, max_moves=max_moves, device=device,
                seed=seed + tier_index * 1000 + opp_index, reference=reference,
            )
            feats.append(f)
            counts.append(c)
            labels.append(torch.full((per_opponent,), tier_index, dtype=torch.long))
            # Group by opponent so cross-validation never scores a fold against
            # games sharing an opponent with its training data.
            groups.append(torch.full((per_opponent,), opp_index, dtype=torch.long))
    return (
        torch.cat(feats).cpu().numpy(),
        torch.cat(counts).cpu().numpy(),
        torch.cat(labels).cpu().numpy(),
        torch.cat(groups).cpu().numpy(),
    )


def evaluate(features, counts, labels, groups, n_moves, feature_subset=None, seed=0):
    """Cross-validated tier prediction from the first ``n_moves`` moves."""
    idx = [FEATURE_NAMES.index(f) for f in (feature_subset or FEATURE_NAMES)]
    x_all, valid = aggregate(torch.tensor(features), torch.tensor(counts), n_moves)
    x = x_all.numpy()[:, idx]
    valid = valid.numpy()
    x, y, g = x[valid], labels[valid], groups[valid]
    if len(np.unique(y)) < 2:
        return None

    n_tiers = len(np.unique(labels))
    preds = np.zeros_like(y)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(g))))
    for train, test in splitter.split(x, y, g):
        model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.08, random_state=seed
        )
        model.fit(x[train], y[train])
        preds[test] = model.predict(x[test])

    exact = float((preds == y).mean())
    within_one = float((np.abs(preds - y) <= 1).mean())
    majority = float(np.bincount(y).max() / len(y))
    corr = float(np.corrcoef(preds, y)[0, 1]) if preds.std() > 0 else 0.0
    confusion = np.zeros((n_tiers, n_tiers), dtype=int)
    for t, p in zip(y, preds):
        confusion[t, p] += 1
    return {
        "n_moves": n_moves,
        "n_samples": int(len(y)),
        "exact_accuracy": exact,
        "within_one_tier": within_one,
        "majority_baseline": majority,
        "rank_correlation": corr,
        "mean_absolute_tier_error": float(np.abs(preds - y).mean()),
        "confusion": confusion.tolist(),
    }


def feature_means_by_tier(features, counts, labels, n_moves=12):
    """What each tier's play actually looks like. The interpretable half."""
    x, valid = aggregate(torch.tensor(features), torch.tensor(counts), n_moves)
    x, y = x.numpy()[valid.numpy()], labels[valid.numpy()]
    out = {}
    for tier in sorted(np.unique(y)):
        rows = x[y == tier]
        out[int(tier)] = {name: float(rows[:, i].mean()) for i, name in enumerate(FEATURE_NAMES)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="coldopen/checkpoints")
    ap.add_argument("--reference", default="coldopen/reference")
    ap.add_argument("--league", default="analysis/league.json")
    ap.add_argument("--out", default="analysis/telemetry.json")
    ap.add_argument("--games-per-tier", type=int, default=600)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    league = json.loads(pathlib.Path(args.league).read_text())
    tiers = league["tiers"]
    checkpoints = load_checkpoints(args.checkpoints, args.device)
    reference = load_reference(args.reference, args.device)
    print(f"{len(tiers)} tiers, reference {'loaded' if reference else 'MISSING'}", flush=True)

    features, counts, labels, groups = build_dataset(
        tiers, checkpoints, reference,
        games_per_tier=args.games_per_tier, device=args.device, seed=args.seed,
    )
    print(f"dataset {features.shape[0]} games x {features.shape[1]} moves "
          f"x {features.shape[2]} features", flush=True)

    curves = {"all": [], "tactical_only": []}
    for n in MOVE_BUDGETS:
        full = evaluate(features, counts, labels, groups, n, seed=args.seed)
        tac = evaluate(features, counts, labels, groups, n,
                       feature_subset=TACTICAL_FEATURES, seed=args.seed)
        if full:
            curves["all"].append(full)
            curves["tactical_only"].append(tac)
            print(f"  n={n:3}  exact {full['exact_accuracy']:.3f}  "
                  f"within-one {full['within_one_tier']:.3f}  "
                  f"(tactical only {tac['exact_accuracy']:.3f})  "
                  f"baseline {full['majority_baseline']:.3f}", flush=True)

    result = {
        "tiers": tiers,
        "feature_names": FEATURE_NAMES,
        "tactical_features": TACTICAL_FEATURES,
        "move_budgets": MOVE_BUDGETS,
        "curves": curves,
        "feature_means_by_tier": feature_means_by_tier(features, counts, labels),
        "games_per_tier": args.games_per_tier,
        "has_reference": reference is not None,
    }
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
