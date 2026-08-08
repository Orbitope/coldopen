"""Distil the scripted sprint bot into a network, snapshotting as it learns.

E2 found this is a third kind of generator, distinct from undertrained RL and
from handicapping, and possibly the closest to how people are bad: a
part-trained imitator has the common patterns and not the rare ones. Here it
is also the only generator that spans the range — DQN from scratch never
clears a line (see `train_sprint.py`), while a student starts at chance and
ends near a teacher that finishes 40 lines in ~41s.

The teacher is `coldopen.sprint_bot`: placement search over hand-written
board heuristics. No human data touches it, so the ladder stays cold-start
legitimate.

Behavioural cloning's usual weakness applies — the student only sees states
the teacher's own play reaches, so it is fragile off-distribution. Two things
blunt it here: the teacher is stochastic through the piece bag, and the
collection runs with a little forced exploration so the student sees boards
its teacher would not have built. Both are about making a *graded* ladder
robust, not about making the top rung strong.

    python -m coldopen.distill_sprint --out coldopen/ladders/sprint_distilled
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "envs"))

from coldopen.distill import snapshot_schedule
from coldopen.nets import BoardNet
from coldopen.sprint_bot import SprintBot, plan_keystrokes
from coldopen.train_sprint import N_ACTIONS, OBS_SHAPE, action_mask, evaluate
from tetris_sprint.fast import TetrisSprintBatched


def collect(n_envs, steps, latency=100, seed=0, explore=0.05):
    """(observations, teacher actions) from the bot's own play.

    `explore` occasionally overrides the teacher's keystroke with a random
    legal one and forces a replan, which widens the state distribution the
    student learns on without changing what the teacher *would* do — the
    label always remains the teacher's choice for the state as seen.
    """
    env = TetrisSprintBatched(n_envs, latency=latency, emit_final_states=False)
    env.reset(torch.arange(n_envs, dtype=torch.int64) + seed * 7919)
    bot = SprintBot(env)
    generator = torch.Generator().manual_seed(seed)

    observations, labels = [], []
    obs = env.observe()
    for _ in range(steps):
        teacher = bot(obs, env)
        observations.append(obs.bool())
        labels.append(teacher.clone())

        played = teacher.clone()
        slip = torch.rand(n_envs, generator=generator) < explore
        if slip.any():
            mask = action_mask(env)
            noise = torch.rand(n_envs, N_ACTIONS, generator=generator)
            played[slip] = noise.masked_fill(~mask, -1).argmax(dim=1)[slip]
            for i in torch.nonzero(slip).flatten().tolist():
                bot.plans[i] = []  # the plan no longer matches the piece
        obs, _, _, _ = env.step(played)

    return torch.cat(observations), torch.cat(labels)


def distil(steps=6000, batch=512, lr=1e-3, channels=64, seed=0,
           latency=100, pool_envs=64, pool_steps=600, out=None, device="cpu"):
    device = torch.device(device)
    torch.manual_seed(seed)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"collecting {pool_envs * pool_steps:,} teacher decisions ...", flush=True)
    obs, labels = collect(pool_envs, pool_steps, latency=latency, seed=seed)
    print(f"  {obs.shape[0]:,} states; teacher action mix "
          f"{torch.bincount(labels, minlength=N_ACTIONS).tolist()}", flush=True)

    student = BoardNet(OBS_SHAPE, N_ACTIONS, channels).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    pending = snapshot_schedule(steps, count=14, lo=25)
    log, started = [], time.time()

    def save(tag):
        torch.save({"steps": tag, "channels": channels, "latency": latency,
                    "state": student.state_dict()}, out / f"ckpt_{tag:08d}.pt")
        index = torch.randint(0, obs.shape[0], (4096,))
        with torch.no_grad():
            predicted = student(obs[index].float().to(device)).argmax(dim=1).cpu()
        agreement = float((predicted == labels[index]).float().mean())
        report = evaluate(student, device, latency=latency)
        report.update({"steps": tag, "teacher_agreement": round(agreement, 4),
                       "elapsed_s": round(time.time() - started, 1)})
        log.append(report)
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        print(f"  [{tag:>6,}] agree {agreement:.3f}  "
              f"lines {report.get('mean_lines', 0):5.1f}  "
              f"finish {report.get('finish_rate', 0):.2f}  "
              f"in/pc {report.get('inputs_per_piece', 0):5.2f}  "
              f"quads {report.get('quad_rate', 0):.2f}", flush=True)

    while pending and pending[0] <= 0:
        save(pending.pop(0))
    for step in range(1, steps + 1):
        index = torch.randint(0, obs.shape[0], (batch,))
        logits = student(obs[index].float().to(device))
        loss = torch.nn.functional.cross_entropy(logits, labels[index].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        while pending and pending[0] <= step:
            save(pending.pop(0))
    return student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latency", type=int, default=100)
    ap.add_argument("--pool-envs", type=int, default=64)
    ap.add_argument("--pool-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="coldopen/ladders/sprint_distilled")
    args = ap.parse_args()
    distil(steps=args.steps, batch=args.batch, lr=args.lr, channels=args.channels,
           seed=args.seed, latency=args.latency, pool_envs=args.pool_envs,
           pool_steps=args.pool_steps, out=args.out, device=args.device)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
