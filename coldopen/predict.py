"""Predict a player's skill tier from how they move, for any of the four games.

``classifier.py`` does this for the bespoke Connect Four pipeline; this is the
same experiment run through the game-agnostic stack, so the four games can be
compared with the method held fixed. That comparison is the point: the question
is not "can a classifier do this" but "what property of a game decides how early
it can". Connect Four is dense with forced tactics, Othello has almost none,
backgammon buries everything under dice, and Leduc hides half the state.

    python -m coldopen.predict --game othello

Two things here exist to stop the result being flattering:

**The reference network is not a contestant.** Features that score a move
against a strong network need that network trained independently of the agents
being profiled, or the top tier matches itself perfectly and the classifier
reads its own answer key.

**Cross-validation is grouped by opponent.** A weak opponent leaves more
mistakes lying around and makes whoever is playing them look sharper, so a fold
must never be scored against games sharing an opponent with its training data.
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from coldopen import features as feature_registry
from coldopen import games as game_mod
from coldopen.ladder import load_ladder
from coldopen.profile import aggregate, collect_sessions

MOVE_BUDGETS = [1, 2, 3, 5, 8, 12, 16, 20]


def load_reference(directory, info, device="cpu"):
    """The strongest checkpoint of the *independent* run, used as the yardstick."""
    entries = load_ladder(directory, info, device)
    trained = [e for e in entries if e["net"] is not None]
    return trained[-1]["net"] if trained else None


def build_dataset(game, tiers, checkpoints, reference, sessions_per_tier=600,
                  max_moves=20, seed=0):
    """Play every tier against a spread of opponents and record their telemetry.

    Opponents are drawn from across the whole ladder rather than matched by
    strength, so the classifier cannot cheat by reading how strong the *other*
    player is.
    """
    by_id = {e["id"]: e["net"] for e in checkpoints}
    opponents = [e["net"] for e in checkpoints]

    feats, counts, labels, groups = [], [], [], []
    per_opponent = max(1, sessions_per_tier // len(opponents))
    for tier_index, tier in enumerate(tiers):
        agent = by_id[tier["id"]]
        for opp_index, opponent in enumerate(opponents):
            f, c = collect_sessions(
                game, agent, opponent, per_opponent, max_moves=max_moves,
                seed=seed + tier_index * 1000 + opp_index, reference=reference,
            )
            feats.append(f)
            counts.append(c)
            labels.append(torch.full((per_opponent,), tier_index, dtype=torch.long))
            groups.append(torch.full((per_opponent,), opp_index, dtype=torch.long))
        print(f"  tier {tier_index} ({tier['tier']}) done", flush=True)
    return (
        torch.cat(feats).cpu().numpy(),
        torch.cat(counts).cpu().numpy(),
        torch.cat(labels).cpu().numpy(),
        torch.cat(groups).cpu().numpy(),
    )


def evaluate(all_names, features, counts, labels, groups, n_moves,
             feature_subset=None, seed=0):
    """Cross-validated tier prediction from the first ``n_moves`` moves."""
    idx = [all_names.index(f) for f in (feature_subset or all_names)]
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

    confusion = np.zeros((n_tiers, n_tiers), dtype=int)
    for t, p in zip(y, preds):
        confusion[t, p] += 1
    return {
        "n_moves": n_moves,
        "n_samples": int(len(y)),
        "exact_accuracy": float((preds == y).mean()),
        "within_one_tier": float((np.abs(preds - y) <= 1).mean()),
        "majority_baseline": float(np.bincount(y).max() / len(y)),
        "rank_correlation": float(np.corrcoef(preds, y)[0, 1]) if preds.std() > 0 else 0.0,
        "mean_absolute_tier_error": float(np.abs(preds - y).mean()),
        "confusion": confusion.tolist(),
    }


def feature_means_by_tier(all_names, features, counts, labels, n_moves=12):
    """What each tier's play actually looks like. The interpretable half."""
    x, valid = aggregate(torch.tensor(features), torch.tensor(counts), n_moves)
    x, y = x.numpy()[valid.numpy()], labels[valid.numpy()]
    return {
        int(tier): {name: float(x[y == tier][:, i].mean())
                    for i, name in enumerate(all_names)}
        for tier in sorted(np.unique(y))
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="othello", choices=list(game_mod.ALL_GAMES))
    ap.add_argument("--checkpoints")
    ap.add_argument("--reference")
    ap.add_argument("--league")
    ap.add_argument("--out")
    ap.add_argument("--sessions-per-tier", type=int, default=600)
    ap.add_argument("--max-moves", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt_dir = args.checkpoints or f"coldopen/ladders/{args.game}"
    ref_dir = args.reference or f"coldopen/ladders/{args.game}_reference"
    league_path = args.league or f"analysis/{args.game}/league.json"
    out = args.out or f"analysis/{args.game}/telemetry.json"

    game = game_mod.make(args.game, args.device)
    all_names = list(feature_registry.names(args.game))
    intrinsic = list(feature_registry.names(args.game, with_reference=False))

    league = json.loads(pathlib.Path(league_path).read_text())
    tiers = league["tiers"]
    checkpoints = load_ladder(ckpt_dir, game.info, args.device)
    reference = (load_reference(ref_dir, game.info, args.device)
                 if pathlib.Path(ref_dir).exists() else None)
    print(f"{args.game}: {len(tiers)} tiers, {len(all_names)} features, "
          f"reference {'loaded' if reference is not None else 'MISSING'}", flush=True)

    features, counts, labels, groups = build_dataset(
        game, tiers, checkpoints, reference,
        sessions_per_tier=args.sessions_per_tier, max_moves=args.max_moves,
        seed=args.seed,
    )
    print(f"dataset {features.shape[0]} sessions x {features.shape[1]} moves "
          f"x {features.shape[2]} features "
          f"(mean {counts.mean():.1f} moves recorded)", flush=True)

    curves = {"all": [], "intrinsic_only": []}
    for n in MOVE_BUDGETS:
        full = evaluate(all_names, features, counts, labels, groups, n, seed=args.seed)
        own = evaluate(all_names, features, counts, labels, groups, n,
                       feature_subset=intrinsic, seed=args.seed)
        if full:
            curves["all"].append(full)
            curves["intrinsic_only"].append(own)
            print(f"  n={n:3}  exact {full['exact_accuracy']:.3f}  "
                  f"within-one {full['within_one_tier']:.3f}  "
                  f"(game features only {own['exact_accuracy']:.3f})  "
                  f"baseline {full['majority_baseline']:.3f}", flush=True)

    result = {
        "game": args.game,
        "tiers": tiers,
        "feature_names": all_names,
        "intrinsic_features": intrinsic,
        "move_budgets": MOVE_BUDGETS,
        "curves": curves,
        "feature_means_by_tier": feature_means_by_tier(
            all_names, features, counts, labels),
        "sessions_per_tier": args.sessions_per_tier,
        "has_reference": reference is not None,
    }
    path = pathlib.Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
