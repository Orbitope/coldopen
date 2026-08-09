"""E6 for sprint: fit on agents, predict on people, and beat a real baseline.

Everything up to here established that the agent ladder *occupies* the human
telemetry manifold — coverage 0% off-manifold on four of five features, and
coherence 7.0 against a human noise floor of 8.0. That is a precondition, not
a result. Othello (E6a) failed it and so never got this far; passing it does
not by itself mean agent telemetry predicts anything.

This is the test. Fit a model on agent episodes only — labelled by each rung's
own virtual sprint time, which the game defines and no human contributes to —
then apply it once to 396 human records and measure Spearman against true
rank.

**The baseline is the whole point.** E1's most useful finding was that a
one-axis "transfer" score is numerically identical to just ranking players by
that axis: the fitted model was the target game's own correlation wearing a
costume. So every number here is reported beside `no_fit` — rank by a single
standardised feature, no model at all. An agent-fitted model that does not
beat its own no-fit baselines has added nothing, however good its correlation
looks in isolation.

**Where human data enters, precisely.** Only twice, and never in fitting:

* features are standardised *within* each population (agents by agent stats,
  humans by human stats), which uses the human feature distribution but no
  human labels. E1 set this convention. At an actual cold start this is
  available — a launch has a pool of unlabelled games from other new players,
  which is exactly what is being standardised against. What it does *not*
  need is any player's results.
* the final Spearman, which is the test.

    python -m coldopen.sprint_transfer --out analysis/tetris_sprint/transfer.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "envs"))

from coldopen.sprint_bot import SprintBot
from coldopen.sprint_ladders import SKILL_RUNGS, human_bands, measure
from coldopen.tetris import HUMAN_COMPARABLE

#: Higher is weaker; every other comparable feature is higher-is-stronger.
LOWER_IS_BETTER = {"inputs_per_piece"}


def spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def standardise(matrix):
    """Z-score each column within its own population.

    Within-population, so the comparison is of *shape* rather than units — and
    so nothing about the agents' absolute scale leaks into the human
    prediction. A column with no variance is left at zero instead of dividing
    by zero.
    """
    matrix = np.asarray(matrix, dtype=float)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (matrix - mean) / std


def agent_episodes(episodes=40, device="cpu"):
    """Per-episode agent rows, labelled by the rung's own sprint time.

    The label is the game's objective, measured on the agent's own play: the
    virtual milliseconds a rung takes over 40 lines. No human quantity is
    involved, so a model fitted here is cold-start legitimate. Episodes that
    do not finish are dropped — they have no sprint time, and there is no
    human counterpart for them either (all 396 records are finishes).
    """
    rows, labels = [], []
    for skill, ms in SKILL_RUNGS:
        result = measure(lambda env, s=skill: SprintBot(env, skill=s),
                         episodes=episodes, latency=ms, seed=21)
        if not result:
            continue
        for row in result["rows"]:
            if not row["finished"]:
                continue
            if any(row.get(f) is None for f in HUMAN_COMPARABLE):
                continue
            rows.append([row[f] for f in HUMAN_COMPARABLE])
            labels.append(-row["time_ms"])   # faster = stronger
        print(f"  scripted skill {skill:4.2f}: "
              f"{sum(r['finished'] for r in result['rows'])}/{len(result['rows'])}"
              f" finished", flush=True)
    return np.array(rows, dtype=float), np.array(labels, dtype=float)


def human_matrix(human_rows):
    keep = [r for r in human_rows
            if all(r.get(f) is not None for f in HUMAN_COMPARABLE)]
    features = np.array([[r[f] for f in HUMAN_COMPARABLE] for r in keep],
                        dtype=float)
    truth = np.array([r["rank_index"] for r in keep], dtype=float)
    return features, truth, keep


def ridge(x, y, alpha=1.0):
    x = np.hstack([x, np.ones((x.shape[0], 1))])
    gram = x.T @ x + alpha * np.eye(x.shape[1])
    gram[-1, -1] -= alpha            # do not penalise the intercept
    return np.linalg.solve(gram, x.T @ y)


def predict(weights, x):
    return np.hstack([x, np.ones((x.shape[0], 1))]) @ weights


def run(episodes=40, out=None):
    print("measuring agent episodes:", flush=True)
    agent_x, agent_y = agent_episodes(episodes=episodes)
    print(f"  {agent_x.shape[0]} finished agent episodes", flush=True)

    human_rows = human_bands()
    human_x, truth, kept = human_matrix(human_rows)
    print(f"  {human_x.shape[0]} human records", flush=True)

    agent_z = standardise(agent_x)
    human_z = standardise(human_x)

    weights = ridge(agent_z, standardise(agent_y.reshape(-1, 1)).ravel())
    fitted = spearman(predict(weights, human_z), truth)

    # The controls that E1 showed are indispensable: rank people by one
    # standardised feature, no model at all.
    no_fit = {}
    for i, name in enumerate(HUMAN_COMPARABLE):
        sign = -1.0 if name in LOWER_IS_BETTER else 1.0
        no_fit[name] = spearman(sign * human_z[:, i], truth)
    best_single = max(no_fit.items(), key=lambda kv: abs(kv[1]))

    # And a within-human ceiling: what a model fitted on HUMANS achieves,
    # cross-validated. Not cold-start legitimate - it is the upper bound the
    # agent-fitted model is being compared against.
    order = np.argsort(truth + 1e-6 * np.arange(len(truth)))
    folds, ceiling = 5, []
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        w = ridge(human_z[train], truth[train])
        ceiling.append(spearman(predict(w, human_z[test]), truth[test]))
    ceiling = float(np.mean(ceiling))

    # Ablation: pps is a speed axis and is trivially observable, so a model
    # that only rediscovers "faster is better" has not earned the simulator.
    # The question worth asking is whether the JUDGEMENT features - stacking,
    # planning, finesse - carry transferable signal on their own.
    subsets = {
        "all": list(range(len(HUMAN_COMPARABLE))),
        "no_pps": [i for i, f in enumerate(HUMAN_COMPARABLE) if f != "pps"],
        "pps_only": [i for i, f in enumerate(HUMAN_COMPARABLE) if f == "pps"],
        "judgement_only": [i for i, f in enumerate(HUMAN_COMPARABLE)
                           if f in ("quad_rate", "holds_per_piece", "max_b2b")],
    }
    ablation = {}
    for name, cols in subsets.items():
        if not cols:
            continue
        w = ridge(agent_z[:, cols],
                  standardise(agent_y.reshape(-1, 1)).ravel())
        ablation[name] = spearman(predict(w, human_z[:, cols]), truth)

    report = {
        "n_agent_episodes": int(agent_x.shape[0]),
        "n_human_records": int(human_x.shape[0]),
        "features": HUMAN_COMPARABLE,
        "agent_fitted_spearman": fitted,
        "no_fit_single_feature": no_fit,
        "best_single_feature": {"name": best_single[0], "spearman": best_single[1]},
        "human_fitted_ceiling": ceiling,
        "ablation": ablation,
        "beats_best_single": fitted > abs(best_single[1]),
        "agent_weights": dict(zip(HUMAN_COMPARABLE + ["intercept"],
                                  [float(w) for w in weights])),
    }

    print(f"\nfit on AGENTS, predict HUMANS   rho = {fitted:+.3f}")
    print("\nno-fit controls (rank by one standardised feature, no model):")
    for name, value in sorted(no_fit.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:20} {value:+.3f}")
    print(f"\nbest single feature             rho = {abs(best_single[1]):+.3f}"
          f"  ({best_single[0]})")
    print(f"human-fitted ceiling (5-fold)   rho = {ceiling:+.3f}")
    print("\nablation - which features carry the transfer?")
    for name in ("all", "no_pps", "pps_only", "judgement_only"):
        if name in ablation:
            print(f"  fit on agents, {name:16} rho = {ablation[name]:+.3f}")
    verdict = ("BEATS the best no-fit baseline" if report["beats_best_single"]
               else "does NOT beat the best no-fit baseline")
    print(f"\nverdict: the agent-fitted model {verdict}.")

    if out:
        path = pathlib.Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"wrote {path}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", default="analysis/tetris_sprint/transfer.json")
    args = ap.parse_args()
    run(episodes=args.episodes, out=args.out)


if __name__ == "__main__":
    main()
