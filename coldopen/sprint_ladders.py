"""Build and measure every sprint ladder, then compare them to real players.

Three generators, one game, and — for the first time in this project — human
data for that same game:

* **scripted** — the placement-search teacher on a coupled skill dial that
  degrades judgement and speed together. Latency alone turned out to be a pure
  speed dial (measured: identical 40-line finishes at every setting), so the
  coupling is explicit rather than emergent.
* **distilled** — a student snapshotted while learning to imitate the teacher.
* **trained** — DQN checkpoints. Kept although the run never learned to clear
  a line: "cannot clear" is the honest floor of the undertrained-RL ladder.

Skill needs no league here. Sprint is single-player, so a rung's ability is
just its mean result — lines cleared, and time when it finishes. Bradley-Terry,
round robins and the monotonicity gate all drop away.

The comparison that matters is run first and reported first: **overlap**. Do
the agents occupy the region of feature space that humans occupy? Othello
failed exactly there, invisibly, until it was measured.

    python -m coldopen.sprint_ladders --out analysis/tetris_sprint
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "envs"))

from coldopen.nets import BoardNet
from coldopen.sprint_bot import SprintBot
from coldopen.tetris import HUMAN_COMPARABLE, rollout
from coldopen.train_sprint import N_ACTIONS, OBS_SHAPE, action_mask
from tetris_sprint.fast import HORIZON, TetrisSprintBatched

#: (skill, latency) rungs for the scripted teacher.
#:
#: Latency alone was measured to be a PURE SPEED DIAL — at every setting from
#: 17ms to 1400ms the bot still cleared 40 lines with identical finesse,
#: because gravity only moves pieces vertically and a hard drop lands them at
#: the bottom regardless. That yields slow-but-flawless players. So the rungs
#: move `skill` too, which interpolates the bot's *objective* between expert
#: quad play and beginner take-any-clear (see `SprintBot`).
#:
#: The floor is skill 0.4 / 560ms, not lower, for a reason that comes from the
#: agent rather than from people: below it the bot stops finishing, and a
#: ladder whose lower rungs cannot complete the task measures survival instead
#: of skill. Latencies are spaced roughly geometrically. Measured span, with
#: no constant fitted to human data:
#:
#:     skill  time    in/pc  quad   hold   b2b   pps    finish
#:     1.00    15.4s   3.27  0.665  0.230  5.2   7.46   0.94
#:     0.70    63.3s   3.68  0.303  0.081  2.0   1.71   1.00
#:     0.40   272.3s   4.32  0.059  0.048  0.9   0.40   0.88
SKILL_RUNGS = [(1.0, 17), (0.9, 34), (0.8, 68), (0.7, 136),
               (0.6, 272), (0.5, 400), (0.4, 560)]


def net_policy(net, device, temperature=0.0, seed=0):
    """Roll out a net. `temperature > 0` samples instead of taking the argmax.

    Sampling is not a nicety for the distilled students, it is required. A
    deterministic argmax policy in this environment falls into **limit
    cycles**: it taps left, the tap moves the piece, the new state's argmax is
    tap right, and it oscillates until gravity locks the piece somewhere
    arbitrary. Same checkpoint, only the action rule changed:

        argmax      0.7 lines, 138.0 inputs per piece
        T = 1.0     5.1 lines,  13.7 inputs per piece

    138 keystrokes for a piece the teacher places in 3.3 is not a policy
    misplacing pieces, it is a policy stuck. Any deterministic policy over a
    state space where actions are reversible can do this; sampling breaks the
    cycle. DQN checkpoints keep `temperature = 0`, since greedy really is the
    policy Q-learning is trying to learn.
    """
    generator = torch.Generator().manual_seed(seed)

    def policy(obs, env):
        with torch.no_grad():
            q = net(obs.to(device)).cpu()
        q = q.masked_fill(~action_mask(env), -1e9)
        if temperature <= 0:
            return q.argmax(dim=1)
        probs = torch.softmax(q / temperature, dim=1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(1)
    return policy


def load_checkpoints(directory, device="cpu"):
    directory = pathlib.Path(directory)
    out = []
    for path in sorted(directory.glob("ckpt_*.pt")):
        blob = torch.load(path, map_location=device, weights_only=False)
        net = BoardNet(OBS_SHAPE, N_ACTIONS, blob.get("channels", 64)).to(device)
        net.load_state_dict(blob["state"])
        net.eval()
        out.append({"id": path.stem, "steps": int(blob.get("steps", 0)), "net": net})
    return out


def measure(policy_factory, episodes=24, latency=100, n_envs=24, seed=11):
    env = TetrisSprintBatched(min(n_envs, episodes), latency=latency)
    rows = rollout(env, policy_factory(env), total_episodes=episodes, seed=seed,
                   max_steps=episodes * HORIZON)
    if not rows:
        return None
    finished = [r for r in rows if r["finished"]]
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    return {
        "episodes": len(rows),
        "mean_lines": mean("lines"),
        "finish_rate": len(finished) / len(rows),
        "time_s": (float(np.mean([r["time_ms"] for r in finished])) / 1000.0
                   if finished else None),
        "inputs_per_piece": mean("inputs_per_piece"),
        "quad_rate": mean("quad_rate"),
        "holds_per_piece": mean("holds_per_piece"),
        "max_b2b": mean("max_b2b"),
        "pps": mean("pps"),
        "rows": rows,
    }


def build_ladders(episodes, device):
    ladders = {}

    ladders["scripted"] = []
    for skill, ms in SKILL_RUNGS:
        result = measure(lambda env, s=skill: SprintBot(env, skill=s),
                         episodes=episodes, latency=ms, seed=21)
        if result:
            result["rung"] = f"skill_{skill:g}"
            result["dial"] = skill
            ladders["scripted"].append(result)
            print(f"  scripted skill {skill:4.2f}  lines {result['mean_lines']:5.1f}  "
                  f"finish {result['finish_rate']:.2f}  "
                  f"in/pc {result['inputs_per_piece']:5.2f}  "
                  f"quads {result['quad_rate']:.2f}", flush=True)

    # Distilled students are classifiers and must be SAMPLED (see net_policy:
    # argmax deadlocks them into tap oscillations). DQN checkpoints stay
    # greedy, because greedy is the policy Q-learning is approximating.
    for name, directory, temp in (
            ("distilled", "coldopen/ladders/sprint_distilled", 1.0),
            ("trained", "coldopen/ladders/tetris_sprint", 0.0)):
        if not pathlib.Path(directory).exists():
            continue
        ladders[name] = []
        for entry in load_checkpoints(directory, device):
            result = measure(
                lambda env, n=entry["net"], t=temp: net_policy(n, device, t),
                episodes=episodes, latency=100, seed=31)
            if result:
                result["rung"] = entry["id"]
                result["dial"] = entry["steps"]
                ladders[name].append(result)
                print(f"  {name} {entry['steps']:>8,}  "
                      f"lines {result['mean_lines']:5.1f}  "
                      f"finish {result['finish_rate']:.2f}  "
                      f"in/pc {result['inputs_per_piece']:5.2f}  "
                      f"quads {result['quad_rate']:.2f}", flush=True)
    return ladders


def human_bands(path="data/human/tetrio_sprint.json"):
    rows = json.loads(pathlib.Path(path).read_text())["rows"]
    return rows


#: Features where a HIGHER value means a WEAKER player (all others: higher is
#: stronger). Needed to put every feature on one "which rank does this look
#: like" scale.
LOWER_IS_BETTER = {"inputs_per_piece"}


def rank_of(feature, value, human_rows):
    """The human rank whose mean `feature` is closest to `value`.

    Answers the only question that matters for transfer: if a person produced
    this number, roughly how good would they be? Returns a rank index in
    [0, 17], or None when the value is outside the human span entirely — which
    is itself the answer, and the one a range check hides.
    """
    means = {}
    for row in human_rows:
        if row.get(feature) is not None:
            means.setdefault(row["rank_index"], []).append(row[feature])
    table = {k: float(np.mean(v)) for k, v in means.items()}
    lo, hi = min(table.values()), max(table.values())
    if value < lo - 0.15 * abs(hi - lo) or value > hi + 0.15 * abs(hi - lo):
        return None
    return min(table, key=lambda k: abs(table[k] - value))


def overlap(ladders, human_rows):
    """Where each agent rung sits on the human ladder, feature by feature.

    A range check is not enough and was actively misleading: pooled human
    quad_rate spans 0.00 to 1.00, so an agent pinned at 0.00 scores "100%
    inside the human range" while in truth sitting below every human alive.

    So this maps each rung's value to the human rank that produces it, and
    reports two things: **coverage** (do the rungs traverse the human ladder?)
    and **coherence** (do a rung's features agree about which rank it is?). A
    coherent agent looks like one player; an incoherent one has the finesse of
    an expert and the stacking of a beginner, which is a shape no human has.
    """
    report = {}
    for feature in HUMAN_COMPARABLE:
        values = [r[feature] for r in human_rows if r.get(feature) is not None]
        if not values:
            continue
        by_rank = {}
        for row in human_rows:
            if row.get(feature) is not None:
                by_rank.setdefault(row["rank_index"], []).append(row[feature])
        entry = {
            "human_weakest_rank_mean": float(np.mean(by_rank[min(by_rank)])),
            "human_strongest_rank_mean": float(np.mean(by_rank[max(by_rank)])),
            "lower_is_better": feature in LOWER_IS_BETTER,
        }
        for name, rungs in ladders.items():
            agent = [r[feature] for r in rungs if r.get(feature) is not None]
            if not agent:
                continue
            ranks = [rank_of(feature, v, human_rows) for v in agent]
            placed = [r for r in ranks if r is not None]
            entry[name] = {
                "min": float(min(agent)), "max": float(max(agent)),
                "off_manifold_share": 1.0 - len(placed) / len(ranks),
                "rank_span": [min(placed), max(placed)] if placed else None,
            }
        report[feature] = entry
    return report


def coherence(ladders, human_rows):
    """Per rung: does every feature agree about which human rank this is?

    Reported as the spread (max - min) of the implied rank across features.
    Zero means the rung looks like a single coherent player; a wide spread
    means it looks like nobody, however well each feature scores alone.
    """
    out = {}
    for name, rungs in ladders.items():
        rows = []
        for rung in rungs:
            implied = {}
            for feature in HUMAN_COMPARABLE:
                value = rung.get(feature)
                if value is None:
                    continue
                rank = rank_of(feature, value, human_rows)
                if rank is not None:
                    implied[feature] = rank
            rows.append({
                "rung": rung["rung"],
                "implied": implied,
                "spread": (max(implied.values()) - min(implied.values())
                           if len(implied) > 1 else None),
            })
        out[name] = rows
    return out


def human_noise_floor(human_rows):
    """Run the coherence metric on the humans themselves, leave-one-out.

    Without this the coherence number is uninterpretable, and reading it alone
    is actively misleading: the agent ladder scored 8.0 ranks of median
    disagreement, which looked like a failure until a single *human* sprint
    record scored 8.0 as well. One 40-line run is a small sample and any real
    player's five features disagree about their rank by a median of eight
    ranks. 8.0 is the floor of the metric, not the agent's error.

    Each record is scored against rank means computed WITHOUT it, so nothing is
    compared to a table it helped build.

    Returns the spread distribution plus per-feature signed deviation from each
    record's own median implied rank — the second is what actually
    discriminates, because systematic bias survives a control that spread does
    not.
    """
    spreads, deviation, accuracy = [], {}, []
    for i, row in enumerate(human_rows):
        others = human_rows[:i] + human_rows[i + 1:]
        implied = {}
        for feature in HUMAN_COMPARABLE:
            if row.get(feature) is None:
                continue
            rank = rank_of(feature, row[feature], others)
            if rank is not None:
                implied[feature] = rank
        if len(implied) < 2:
            continue
        values = list(implied.values())
        spreads.append(max(values) - min(values))
        median = float(np.median(values))
        accuracy.append(median - row["rank_index"])
        for feature, rank in implied.items():
            deviation.setdefault(feature, []).append(rank - median)
    return {
        "n": len(spreads),
        "median_spread": float(np.median(spreads)),
        "spread_quartiles": [float(np.percentile(spreads, q)) for q in (25, 50, 75)],
        "mean_abs_rank_error": float(np.mean(np.abs(accuracy))),
        "feature_deviation": {k: float(np.mean(v)) for k, v in deviation.items()},
    }


def agent_deviation(ladders, human_rows):
    """Per-feature signed deviation, agent side, to compare with the floor."""
    out = {}
    for name, rungs in coherence(ladders, human_rows).items():
        per = {}
        for row in rungs:
            if len(row["implied"]) < 2:
                continue
            median = float(np.median(list(row["implied"].values())))
            for feature, rank in row["implied"].items():
                per.setdefault(feature, []).append(rank - median)
        out[name] = {k: float(np.mean(v)) for k, v in per.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--human", default="data/human/tetrio_sprint.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="analysis/tetris_sprint/ladders.json")
    args = ap.parse_args()

    print("measuring ladders:", flush=True)
    ladders = build_ladders(args.episodes, torch.device(args.device))
    human_rows = human_bands(args.human)
    report = overlap(ladders, human_rows)

    coh = coherence(ladders, human_rows)

    print(f"\noverlap with {len(human_rows)} human sprint records")
    print("  (rank span = which human ranks the rungs look like, 0=d .. 17=x+;"
          " off = share outside the human span entirely)")
    for feature, entry in report.items():
        direction = "lower better" if entry["lower_is_better"] else "higher better"
        print(f"\n  {feature}  [human d={entry['human_weakest_rank_mean']:.2f} "
              f"-> x+={entry['human_strongest_rank_mean']:.2f}, {direction}]")
        for name in ladders:
            got = entry.get(name)
            if not got:
                continue
            span = (f"ranks {got['rank_span'][0]:>2}-{got['rank_span'][1]:<2}"
                    if got["rank_span"] else "none placed")
            print(f"    {name:10} {got['min']:8.2f}-{got['max']:<8.2f} "
                  f"{span}   off {got['off_manifold_share']*100:3.0f}%")

    floor = human_noise_floor(human_rows)
    dev = agent_deviation(ladders, human_rows)

    print("\ncoherence — does a rung's features agree on which rank it is?")
    print(f"  {'HUMAN FLOOR':10} median rank disagreement "
          f"{floor['median_spread']:.1f} ranks  "
          f"(quartiles {'/'.join(f'{q:.0f}' for q in floor['spread_quartiles'])}, "
          f"n={floor['n']}) <- the metric's noise floor, not a target")
    for name, rows in coh.items():
        spreads = [r["spread"] for r in rows if r["spread"] is not None]
        if not spreads:
            print(f"  {name:10} no rung placed on 2+ features")
            continue
        print(f"  {name:10} median rank disagreement {np.median(spreads):.1f} "
              f"ranks (max {max(spreads)})")

    # Spread cannot see systematic bias, and systematic bias is what actually
    # separates an agent from a player. This is the discriminating table.
    print("\nper-feature signed deviation from the record's own median rank")
    print(f"  {'feature':18} {'human':>7} " +
          " ".join(f"{name:>10}" for name in dev))
    for feature in HUMAN_COMPARABLE:
        line = f"  {feature:18} {floor['feature_deviation'].get(feature, 0.0):+7.1f} "
        for name in dev:
            value = dev[name].get(feature)
            line += f"{value:+10.1f} " if value is not None else f"{'-':>10} "
        print(line)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = {name: [{k: v for k, v in r.items() if k != "rows"} for r in rungs]
            for name, rungs in ladders.items()}
    path.write_text(json.dumps(
        {"ladders": slim, "overlap": report, "coherence": coh,
         "human_noise_floor": floor, "feature_deviation": dev}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
