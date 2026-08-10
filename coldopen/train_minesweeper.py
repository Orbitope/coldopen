"""DQN on minesweeper, checkpointed into a ladder.

Single-player, like sprint: no league, a checkpoint's skill is its mean result
(win rate, and time when it wins). The trainer mirrors `train_sprint.py` where
the games agree and differs where they measurably differ:

* **Action space** is 3 kinds x 480 cells = 1440. A mask cheaply removes the
  actions the spec defines as no-ops (revealing a revealed/flagged cell,
  flagging a revealed cell, chording anything that does not chord): they cost
  time and do nothing, so exploration that wastes its budget on them learns
  slower. The mask changes exploration only - no-ops remain legal in the env,
  and telemetry still counts any the policy takes.
* **Expectation, set honestly by sprint.** Random exploration in sprint never
  assembled a 40-line clear and DQN learned nothing in 3M steps. A minesweeper
  win needs ~381 correct reveals with no fatal one; random play dies in a
  handful of clicks, so the same wall is likely. The run is still worth
  making: its checkpoints are the undertrained-RL ladder either way, and
  "cannot win" is a measured floor rather than an assumption. The strong
  generator, as in sprint, will be a scripted solver-teacher.
* **Reward shaping** on progress (potential = number of revealed safe cells)
  is potential-based, training-only, and never reported: without it the only
  signals are the per-click cost and a terminal bonus random play rarely sees.

    python -m coldopen.train_minesweeper --out coldopen/ladders/minesweeper
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "envs"))

from coldopen.nets import BoardNet
from coldopen.selfplay import NStep, checkpoint_schedule
from minesweeper.fast import CHORD, FLAG, REVEAL, MinesweeperBatched

OBS_SHAPE = (11, 16, 30)
K = 16 * 30
N_ACTIONS = 3 * K


def action_mask(env):
    """True where an action is not a spec-defined no-op.

    Exploration-only: the env accepts everything, this just stops epsilon from
    spending its budget clicking revealed cells. Chord legality is the full
    precondition (revealed, count > 0, flags == count), computed batched.
    """
    counts = env._counts()
    flags = env._neighbour_sum(env.flagged)
    reveal_ok = ~env.revealed & ~env.flagged
    flag_ok = ~env.revealed
    chord_ok = env.revealed & (counts > 0) & (flags == counts)
    return torch.cat([reveal_ok, flag_ok, chord_ok], dim=1)


def potential(env):
    """Φ(s) = revealed safe cells. Progress toward the win condition.

    Potential-based shaping (Ng et al. 1999) preserves the optimal policy;
    nothing outside training sees the shaped numbers.
    """
    return (env.revealed & ~env.mines).sum(dim=1).to(torch.float32)


def evaluate(net, device, episodes=32, latency=200, seed=123):
    """Greedy rollout; wins, mean revealed, and time on wins. Unshaped."""
    env = MinesweeperBatched(min(episodes, 32), latency=latency,
                             emit_final_states=False)
    env.reset(torch.arange(env.n, dtype=torch.int64) + seed)
    done = torch.zeros(env.n, dtype=torch.bool)
    wins = torch.zeros(env.n, dtype=torch.bool)
    revealed_at_end = torch.zeros(env.n, dtype=torch.int64)
    time_at_end = torch.zeros(env.n, dtype=torch.int64)
    steps_cap = 800
    obs = env.observe()
    for _ in range(steps_cap):
        with torch.no_grad():
            q = net(obs.to(device)).cpu()
        acts = q.masked_fill(~action_mask(env), -1e9).argmax(dim=1)
        prev_rev = (env.revealed & ~env.mines).sum(dim=1)
        prev_time = env.time_ms.clone()
        obs, _, term, _ = env.step(acts)
        newly = term & ~done
        wins = wins | (newly & env.won) | (newly & (prev_rev == (K - env.M)))
        revealed_at_end = torch.where(newly, prev_rev, revealed_at_end)
        time_at_end = torch.where(newly, prev_time, time_at_end)
        done = done | term
        if bool(done.all()):
            break
    revealed_at_end = torch.where(done, revealed_at_end,
                                  (env.revealed & ~env.mines).sum(dim=1))
    win_rate = float(wins.float().mean())
    won_times = time_at_end[wins]
    return {
        "episodes": int(env.n),
        "win_rate": round(win_rate, 4),
        "mean_safe_revealed": round(float(revealed_at_end.float().mean()), 1),
        "time_s_on_win": (round(float(won_times.float().mean()) / 1000, 1)
                          if wins.any() else None),
    }


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    net = BoardNet(OBS_SHAPE, N_ACTIONS, args.channels).to(device)
    target = BoardNet(OBS_SHAPE, N_ACTIONS, args.channels).to(device)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    env = MinesweeperBatched(args.envs, latency=args.latency,
                             emit_final_states=False)
    env.reset(torch.arange(args.envs, dtype=torch.int64) + args.seed)

    from coldopen.train_sprint import BoolReplay
    buf = BoolReplay(args.buffer)
    nstep = NStep(args.n_step)
    ones = torch.ones(args.envs)

    obs = env.observe()
    phi = potential(env)
    pending = checkpoint_schedule(args.steps, lo=2_000)
    log, updates, steps = [], 0, 0
    started = time.time()

    def save(tag):
        torch.save({"steps": tag, "channels": args.channels,
                    "latency": args.latency, "state": net.state_dict()},
                   out / f"ckpt_{tag:08d}.pt")
        report = evaluate(net, device, latency=args.latency)
        report.update({"steps": tag,
                       "elapsed_s": round(time.time() - started, 1)})
        log.append(report)
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        print(f"  [{tag:>9,}] win {report['win_rate']:.3f}  "
              f"safe {report['mean_safe_revealed']:6.1f}/{K - env.M}  "
              f"time {report['time_s_on_win']}", flush=True)

    while steps < args.steps:
        frac = min(1.0, steps / args.eps_decay)
        eps = args.eps_start + (args.eps_end - args.eps_start) * frac

        with torch.no_grad():
            q = net(obs.to(device)).cpu()
        mask = action_mask(env)
        greedy = q.masked_fill(~mask, -1e9).argmax(dim=1)
        noise = torch.rand(args.envs, N_ACTIONS)
        random_legal = noise.masked_fill(~mask, -1.0).argmax(dim=1)
        explore = torch.rand(args.envs) < eps
        acts = torch.where(explore, random_legal, greedy)

        next_obs, reward, term, _ = env.step(acts)
        # potential-based shaping on revealed-safe progress, training-only
        phi_next = potential(env)
        phi_next = torch.where(term, torch.zeros_like(phi_next), phi_next)
        shaped = reward + args.gamma * phi_next - phi
        phi = potential(env)

        next_mask = action_mask(env)
        for sample in nstep.push(obs.bool(), acts, shaped / 1000.0, ones,
                                 term, next_obs.bool(), next_mask):
            buf.push(**sample)
        obs = next_obs
        steps += args.envs

        if len(buf) >= args.learn_start:
            for _ in range(args.updates_per_step):
                o, a, r, s, d, no, nm = buf.sample(args.batch, device)
                with torch.no_grad():
                    nq = target(no)
                    nq = nq.masked_fill(~nm, -1e9)
                    online = net(no).masked_fill(~nm, -1e9).argmax(
                        dim=1, keepdim=True)
                    bootstrap = nq.gather(1, online).squeeze(1)
                    y = r + (~d).float() * s * (args.gamma ** args.n_step) * bootstrap
                pred = net(o).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = torch.nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                updates += 1
                if updates % args.target_sync == 0:
                    target.load_state_dict(net.state_dict())

        while pending and pending[0] <= steps:
            save(pending.pop(0))

    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--latency", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--n-step", type=int, default=8)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=int, default=600_000)
    p.add_argument("--learn-start", type=int, default=10_000)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--target-sync", type=int, default=2_000)
    p.add_argument("--device",
                   default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="coldopen/ladders/minesweeper")
    args = p.parse_args()
    train(args)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
