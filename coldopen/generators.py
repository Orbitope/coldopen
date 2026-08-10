"""E2: does it matter how a skill ladder was generated?

Every result in this project so far rests on ladders built by *undertraining* -
snapshots taken during self-play, with the early ones standing in for weak
players. The alternative is to take the finished agent and cripple it. Both
produce players of graded strength; the question is whether they produce the
same *behaviour* at the same strength, because if they do not, then everything
measured about telemetry and skill is partly a property of the choice of
generator rather than of the game.

The comparison is made at **matched measured Elo**. Each handicapped ladder goes
through the same league as the trained one, so "equally strong" is established
by results rather than by the dial that produced it. Tiers are then chosen to sit
as close as possible to the trained ladder's tier ratings, and the two are
profiled with identical telemetry and an identical classifier.

The prediction, written down before running it: the handicapped ladders will show
*more structured* errors - blunders concentrated in particular kinds of position
rather than spread evenly - and a slower-rising accuracy-versus-N curve, because
the signal is in which mistakes are made rather than how many.

    python -m coldopen.generators --kind temperature
"""

import argparse
import json
import pathlib

import numpy as np
import torch

from coldopen import games as game_mod
from coldopen import handicap
from coldopen.ladder import load_ladder, run_league
from coldopen.league import monotonicity, pick_tiers
from coldopen.predict import build_dataset, evaluate, feature_means_by_tier, load_reference
from coldopen import features as feature_registry
from coldopen.predict import MOVE_BUDGETS

#: Strength grids that span each handicap's usable range, found by measuring
#: score against the random anchor. They differ by an order of magnitude because
#: the three quantities are not the same kind of number: a probability, a softmax
#: temperature, and a fraction of the board.
GRIDS = {
    "epsilon": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.85, 0.92, 0.97, 1.0],
    "temperature": [0.0, 0.15, 0.3, 0.6, 1.0, 2.0, 4.0, 8.0, 14.0, 20.0],
    "blindspot": [0.0, 0.15, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0],
}


#: Tier ratings every generator is compared at. Chosen inside the range all of
#: them reach, and *fixed across generators*, because tier spacing is the main
#: driver of how hard classification is - a first version of this experiment
#: matched only the total Elo range, which let the trained ladder's tiers sit
#: 1.4x further apart than the handicapped ones and made it look far more
#: legible for reasons that had nothing to do with how it was built.
MATCHED_ELOS = [0.0, 130.0, 260.0, 390.0, 520.0, 650.0]


#: Generators whose rungs are checkpoints on disk rather than wrappers around
#: one network. ``trained`` is the original undertrained ladder; ``distilled``
#: is a student snapshotted while learning to imitate it.
FROM_DISK = {"trained": "", "distilled": "_distilled"}


def build(game, strongest, kind, strengths=None, trained=None, disk=None):
    """A ladder plus the uniform-random anchor the league needs.

    Checkpoint-based generators and handicap-based ones go through one code path
    and one set of league settings. Comparing across two scripts is how tier
    spacing got away from this experiment the first time.
    """
    if kind in FROM_DISK:
        entries = list(trained) if kind == "trained" else list(disk)
        if not any(e["net"] is None for e in entries):
            entries = [{"id": "random", "plies": -1, "net": None}] + entries
        return entries
    strengths = strengths or GRIDS[kind]
    rungs = handicap.ladder(strongest, kind, strengths)
    return [{"id": "random", "plies": -1, "net": None}] + rungs


def pick_tiers_at(table, targets):
    """The rungs closest to a fixed set of ratings, one per target.

    Unlike ``pick_tiers``, which spreads tiers evenly across whatever range a
    ladder happens to span, this puts every generator's tiers at the *same*
    ratings so that classification difficulty is comparable between them.
    """
    rows = sorted(table, key=lambda r: r["elo"])
    chosen, used = [], set()
    for index, target in enumerate(targets):
        candidates = [r for r in rows if r["id"] not in used]
        if not candidates:
            break
        best = min(candidates, key=lambda r: abs(r["elo"] - target))
        used.add(best["id"])
        row = dict(best)
        row["tier"] = ["bronze", "silver", "gold", "platinum",
                       "diamond", "grandmaster"][index]
        row["tier_index"] = index
        row["target_elo"] = target
        chosen.append(row)
    chosen.sort(key=lambda r: r["elo"])
    for index, row in enumerate(chosen):
        row["tier_index"] = index
    return chosen


def measure(game, entries, n_games=300, seed=0, tiers=6, targets=None):
    result = run_league(game, entries, n_games=n_games, seed=seed)
    result["monotonicity"] = monotonicity(result["table"])
    result["tiers"] = (pick_tiers_at(result["table"], targets) if targets
                       else pick_tiers(result["table"], tiers))
    return result


def profile(game, tiers, entries, reference, sessions_per_tier=600, seed=0):
    """Telemetry and the accuracy-versus-N curve, identical to the trained run."""
    names = list(feature_registry.names(game.info.key))
    intrinsic = list(feature_registry.names(game.info.key, with_reference=False))
    featureset, counts, labels, groups = build_dataset(
        game, tiers, entries, reference,
        sessions_per_tier=sessions_per_tier, seed=seed)

    curves = {"all": [], "intrinsic_only": []}
    for budget in MOVE_BUDGETS:
        full = evaluate(names, featureset, counts, labels, groups, budget, seed=seed)
        own = evaluate(names, featureset, counts, labels, groups, budget,
                       feature_subset=intrinsic, seed=seed)
        if full:
            curves["all"].append(full)
            curves["intrinsic_only"].append(own)
            print(f"  n={budget:3}  exact {full['exact_accuracy']:.3f}  "
                  f"within-one {full['within_one_tier']:.3f}  "
                  f"(game features only {own['exact_accuracy']:.3f})", flush=True)
    return {
        "feature_names": names,
        "curves": curves,
        "feature_means_by_tier": feature_means_by_tier(
            names, featureset, counts, labels),
    }


def error_structure(means, feature_names):
    """How unevenly a tier's mistakes are distributed across error types.

    The prediction under test is that handicapping produces *structured* errors
    and undertraining produces uniform ones. This scores that directly: take the
    blunder-shaped features, normalise each tier's profile to a distribution
    over them, and report its concentration. A player who errs the same amount in
    every way scores near zero; one whose mistakes pile into a single kind scores
    near one.
    """
    blunders = [n for n in feature_names
                if n in ("missed_win", "missed_block", "gave_win")]
    if not blunders:
        return None
    out = {}
    for tier, values in means.items():
        profile = np.array([max(0.0, values[n]) for n in blunders], dtype=float)
        total = profile.sum()
        if total <= 0:
            out[int(tier)] = 0.0
            continue
        share = profile / total
        # Normalised deviation from a flat profile: 0 uniform, 1 concentrated.
        flat = np.full_like(share, 1.0 / len(share))
        out[int(tier)] = float(np.abs(share - flat).sum() / (2 * (1 - 1.0 / len(share))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="connect_four", choices=list(game_mod.ALL_GAMES))
    ap.add_argument("--kind", default="temperature",
                    choices=sorted(GRIDS) + sorted(FROM_DISK))
    ap.add_argument("--match-elo", action="store_true",
                    help="put tiers at MATCHED_ELOS so generators are comparable")
    ap.add_argument("--checkpoints")
    ap.add_argument("--reference")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--sessions-per-tier", type=int, default=600)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    ckpt_dir = args.checkpoints or f"coldopen/ladders/{args.game}"
    ref_dir = args.reference or f"coldopen/ladders/{args.game}_reference"
    out = args.out or f"analysis/{args.game}/gen_{args.kind}{'_matched' if args.match_elo else ''}.json"

    game = game_mod.make(args.game, args.device)
    trained = load_ladder(ckpt_dir, game.info, args.device)
    strongest = trained[-1]["net"]
    reference = (load_reference(ref_dir, game.info, args.device)
                 if pathlib.Path(ref_dir).exists() else None)
    print(f"{args.game}: handicapping {trained[-1]['id']} by {args.kind}, "
          f"reference {'loaded' if reference is not None else 'MISSING'}", flush=True)

    disk = None
    if args.kind in FROM_DISK and args.kind != "trained":
        disk = load_ladder(ckpt_dir + FROM_DISK[args.kind], game.info, args.device)
    entries = build(game, strongest, args.kind, trained=trained, disk=disk)
    league = measure(game, entries, n_games=args.games, seed=args.seed,
                     targets=MATCHED_ELOS if args.match_elo else None)
    m = league["monotonicity"]
    elos = [t["elo"] for t in league["tiers"]]
    gaps = [b - a for a, b in zip(elos, elos[1:])]
    print(f"\nladder: range {m['elo_range']:.0f} Elo  "
          f"concordance {m['rank_concordance']:.3f}  "
          f"mean tier gap {sum(gaps) / len(gaps):.0f}", flush=True)
    print("tiers: " + ", ".join(f"{t['tier']}={t['id']}({t['elo']:.0f})"
                               for t in league["tiers"]), flush=True)

    print("\nprofiling:", flush=True)
    telemetry = profile(game, league["tiers"], entries, reference,
                        sessions_per_tier=args.sessions_per_tier, seed=args.seed)
    telemetry["error_structure"] = error_structure(
        telemetry["feature_means_by_tier"], telemetry["feature_names"])

    result = {"game": args.game, "generator": args.kind,
              "league": league, "telemetry": telemetry}
    path = pathlib.Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
