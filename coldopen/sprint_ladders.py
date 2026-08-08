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

#: (skill, latency) rungs for the scripted teacher. Latency alone was measured
#: to be a PURE SPEED DIAL — at every setting from 17ms to 1400ms the bot still
#: cleared 40 lines with identical finesse, because gravity only moves pieces
#: vertically and a hard drop lands them at the bottom regardless. That yields
#: slow-but-flawless players, which humans are not. The rungs therefore move
#: judgement and speed together (see SprintBot.skill).
SKILL_RUNGS = [(1.0, 17), (0.85, 60), (0.7, 140), (0.55, 260),
               (0.4, 420), (0.25, 650), (0.1, 900)]


def net_policy(net, device):
    def policy(obs, env):
        with torch.no_grad():
            q = net(obs.to(device)).cpu()
        return q.masked_fill(~action_mask(env), -1e9).argmax(dim=1)
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

    for name, directory in (("distilled", "coldopen/ladders/sprint_distilled"),
                            ("trained", "coldopen/ladders/tetris_sprint")):
        if not pathlib.Path(directory).exists():
            continue
        ladders[name] = []
        for entry in load_checkpoints(directory, device):
            result = measure(lambda env, n=entry["net"]: net_policy(n, device),
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


def overlap(ladders, human_rows):
    """Per-feature coverage: does the agent range sit inside the human range?

    Reported before any accuracy number. A generator whose features fall
    outside the human span cannot be tested for transfer no matter how well
    its own tiers separate — the Othello lesson, made routine.
    """
    report = {}
    for feature in HUMAN_COMPARABLE:
        values = [r[feature] for r in human_rows if r.get(feature) is not None]
        if not values:
            continue
        low, high = float(np.percentile(values, 2)), float(np.percentile(values, 98))
        entry = {"human_p2": low, "human_p98": high,
                 "human_median": float(np.median(values))}
        for name, rungs in ladders.items():
            agent = [r[feature] for r in rungs if r.get(feature) is not None]
            if not agent:
                continue
            inside = [v for v in agent if low <= v <= high]
            entry[name] = {
                "min": float(min(agent)), "max": float(max(agent)),
                "share_inside_human_range": len(inside) / len(agent),
            }
        report[feature] = entry
    return report


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

    print(f"\noverlap with {len(human_rows)} human sprint records:")
    print(f"  {'feature':18} {'human p2-p98':>20}   " +
          "  ".join(f"{name:>22}" for name in ladders))
    for feature, entry in report.items():
        line = (f"  {feature:18} "
                f"{entry['human_p2']:9.2f}-{entry['human_p98']:<9.2f} ")
        for name in ladders:
            got = entry.get(name)
            line += (f"  {got['min']:7.2f}-{got['max']:<7.2f}"
                     f"{got['share_inside_human_range']*100:4.0f}%"
                     if got else f"  {'-':>22}")
        print(line)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = {name: [{k: v for k, v in r.items() if k != "rows"} for r in rungs]
            for name, rungs in ladders.items()}
    path.write_text(json.dumps({"ladders": slim, "overlap": report}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
