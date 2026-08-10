"""Supervised mine-probability net: the conv-net payoff, without the RL wall.

The DQN's measured failure mode on minesweeper is instructive: by 116k steps
it had learned to avoid dying (deaths 43 -> 18 on curriculum probes) without
learning to win (0 wins), because with 99 hidden mines against a handful of
hidden safe cells, "never reveal" beats an uninformed reveal — and a sparse
1-in-1440 Q-target gives almost no gradient toward *locating* safety.

This module replaces the sparse signal with a dense one. The environment knows
where its mines are, so every visited state yields 480 labelled cells for
free:

    net : observation [11, H, W]  ->  logits [1, H, W]
    loss: BCE against the true mine mask, averaged over UNREVEALED cells only
    policy: reveal argmin P(mine) among unrevealed, unflagged cells

Three properties worth naming, each bought by an earlier failure:

* **Well-posed labels** (the tetris distillation lesson): the label is a pure
  function of (board, seed) — no teacher with hidden plan state, no DAgger
  ill-posedness. States are collected under the net's own policy, so there is
  no train/rollout distribution gap to close: the collector IS the policy.
* **Dense supervision** (the DQN lesson above): ~480 labels per state versus
  one, and every one of them is about exactly the question the policy asks.
* **No human data.** Boards, states and labels all come from the validated
  env. Checkpoints form a cold-start-legitimate judgement ladder for E4: the
  dial is how well the net reads boards, which is the skill axis 3BV/s does
  not carry.

The honest limit is the receptive field: a conv sees local constraint
patterns, not global mine-count parity, so some endgame guesses stay guesses.
That is fine — humans mostly play locally too, and the telemetry oracle (an
exact solver, separate module) judges what was *provably* safe regardless of
what the net believed.

    python -m coldopen.mineprob --out coldopen/ladders/minesweeper_prob
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
from coldopen.nets import FullyConvNet
from minesweeper.fast import MinesweeperBatched

OBS_CHANNELS = 11

#: The three ranked board sizes on minesweeper.online. The FCN is fully
#: convolutional, so ONE architecture handles every size unchanged - another
#: payoff of the spatial head. Beginner and intermediate exist because of the
#: sprint lesson: every human leaderboard record is a completed game, so a
#: comparable agent must actually WIN games, and Expert wins are still out of
#: reach. On Beginner the same net should win outright.
BOARDS = {"beginner": (9, 9, 10),
          "intermediate": (16, 16, 40),
          "expert": (16, 30, 99)}


def mine_probs(net, obs):
    """P(mine) per cell, [N, K]."""
    with torch.no_grad():
        return torch.sigmoid(net(obs).flatten(1))


def prob_policy(net, epsilon=0.0, generator=None):
    """Reveal the least-likely mine among unrevealed, unflagged cells.

    `epsilon` occasionally reveals a random hidden cell instead — used during
    collection so the pool keeps containing states a slightly-worse policy
    reaches, and as the *skill dial* when a ladder is built from a finished
    net: an agent that sometimes clicks an un-reasoned cell is exactly the
    "lapse" degradation E2 found most human-like, applied at the judgement
    level rather than the motor level.
    """
    def policy(obs, env):
        probs = mine_probs(net, obs)
        hidden = ~env.revealed & ~env.flagged
        scores = probs.masked_fill(~hidden, 2.0)      # never pick non-hidden
        choice = scores.argmin(dim=1)
        if epsilon > 0 and generator is not None:
            noise = torch.rand_like(probs).masked_fill(~hidden, 2.0)
            rand_choice = noise.argmin(dim=1)
            flip = torch.rand(env.n, generator=generator) < epsilon
            choice = torch.where(flip, rand_choice, choice)
        return choice                                  # kind 0 = reveal, so
    return policy                                      # action id == cell id


def collect(net, n_envs, steps, epsilon, seed, device, board=(16, 30, 99)):
    """(observation, mine mask, hidden mask) under the net's own play."""
    H, W, M = board
    env = MinesweeperBatched(n_envs, H=H, W=W, M=M, emit_final_states=False)
    env.reset(torch.arange(n_envs, dtype=torch.int64) + seed * 9973)
    policy = prob_policy(net, epsilon,
                         torch.Generator().manual_seed(seed))
    observations, labels, masks = [], [], []
    obs = env.observe()
    for _ in range(steps):
        acts = policy(obs.to(device), env)
        # label AFTER generation only: an ungenerated board has no mines yet
        keep = env.generated
        if bool(keep.any()):
            observations.append(obs[keep].bool())
            labels.append(env.mines[keep])
            masks.append((~env.revealed & keep.unsqueeze(1))[keep])
        obs, _, _, _ = env.step(acts)
    return (torch.cat(observations), torch.cat(labels), torch.cat(masks))


def evaluate(net, device, episodes=64, seed=123, board=(16, 30, 99)):
    """Win rate, progress and time of the argmin policy on full boards."""
    H, W, M = board
    env = MinesweeperBatched(min(episodes, 64), H=H, W=W, M=M,
                             emit_final_states=False)
    env.reset(torch.arange(env.n, dtype=torch.int64) + seed)
    policy = prob_policy(net)
    done = torch.zeros(env.n, dtype=torch.bool)
    wins = torch.zeros(env.n, dtype=torch.bool)
    progress = torch.zeros(env.n, dtype=torch.int64)
    times = torch.zeros(env.n, dtype=torch.int64)
    obs = env.observe()
    for _ in range(env.K):
        acts = policy(obs.to(device), env)
        safe_before = (env.revealed & ~env.mines).sum(dim=1)
        time_before = env.time_ms.clone()
        obs, reward, term, _ = env.step(acts)
        newly = term & ~done
        won_now = newly & (reward > 500_000)
        wins = wins | won_now
        progress = torch.where(newly, safe_before, progress)
        times = torch.where(won_now, time_before + 230, times)
        done = done | term
        if bool(done.all()):
            break
    progress = torch.where(done, progress,
                           (env.revealed & ~env.mines).sum(dim=1))
    return {"episodes": int(env.n),
            "total_safe": int(env.K - env.M),
            "win_rate": round(float(wins.float().mean()), 4),
            "mean_safe_revealed": round(float(progress.float().mean()), 1),
            "time_s_on_win": (round(float(times[wins].float().mean()) / 1000, 1)
                              if bool(wins.any()) else None)}


def train(args):
    board = BOARDS[args.board]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    net = FullyConvNet(OBS_CHANNELS, 1, args.channels, depth=args.depth).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    pending = snapshot_schedule(args.steps, count=16, lo=50)
    log, started = [], time.time()

    def save(tag):
        torch.save({"steps": tag, "channels": args.channels,
                    "depth": args.depth, "arch": "mineprob",
                    "board": board, "state": net.state_dict()},
                   out / f"ckpt_{tag:08d}.pt")
        report = evaluate(net, device, board=board)
        report.update({"steps": tag,
                       "elapsed_s": round(time.time() - started, 1)})
        log.append(report)
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        print(f"  [{tag:>7,}] win {report['win_rate']:.3f}  "
              f"safe {report['mean_safe_revealed']:6.1f}/{report['total_safe']}  "
              f"time {report['time_s_on_win']}", flush=True)

    obs = labels = mask = None
    step = 0
    while pending and pending[0] <= 0:
        save(pending.pop(0))
    round_id = 0
    while step < args.steps:
        # fresh states under the CURRENT policy, high epsilon early so the
        # very first pool is not just one deterministic trajectory
        eps = max(0.05, 0.5 * (0.5 ** round_id))
        new_obs, new_labels, new_mask = collect(
            net, args.pool_envs, args.pool_steps, eps,
            seed=args.seed + round_id, device=device, board=board)
        if obs is None:
            obs, labels, mask = new_obs, new_labels, new_mask
        else:
            cap = args.pool_cap
            obs = torch.cat([obs, new_obs])[-cap:]
            labels = torch.cat([labels, new_labels])[-cap:]
            mask = torch.cat([mask, new_mask])[-cap:]
        round_id += 1
        print(f"  pool: {obs.shape[0]:,} states (round {round_id}, eps {eps:.2f})",
              flush=True)

        for _ in range(args.steps_per_round):
            i = torch.randint(0, obs.shape[0], (args.batch,))
            logits = net(obs[i].float().to(device)).flatten(1)
            y = labels[i].float().to(device)
            m = mask[i].to(device)
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, reduction="none")
            loss = (raw * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            while pending and pending[0] <= step:
                save(pending.pop(0))
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--board", choices=sorted(BOARDS), default="expert")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pool-envs", type=int, default=48)
    ap.add_argument("--pool-steps", type=int, default=120)
    ap.add_argument("--pool-cap", type=int, default=60_000)
    ap.add_argument("--steps-per-round", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="coldopen/ladders/minesweeper_prob")
    args = ap.parse_args()
    train(args)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
