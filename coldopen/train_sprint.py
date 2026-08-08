"""DQN on tetris_sprint, checkpointed into a ladder.

Single-player, so this is plainer than `selfplay.py`: no negamax sign, no
opponent, and no league afterwards — a checkpoint's skill is just its mean
sprint result, measured directly. The parts that carry over are the ones the
backgammon failure taught: n-step returns (the clear bonus is many steps
downstream of the stacking decisions that earn it), log-spaced checkpoints,
and measuring strength instead of assuming training produced it.

One addition, standard for Tetris and used only during training:
**potential-based reward shaping** on holes and stack height,
``r' = r + γ·Φ(s') − Φ(s)`` with Φ = −(w_h·holes + w_s·height). Random play
essentially never clears a line, so the unshaped signal is a −cost drip plus a
bonus the agent may never see — the same sparse-reward wall the two-player
Tetris literature warns about. Potential-based shaping provably preserves the
optimal policy (Ng et al. 1999); nothing outside training ever sees the shaped
numbers, and every reported reward or telemetry figure is unshaped.

    python -m coldopen.train_sprint --out coldopen/ladders/tetris_sprint
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
from coldopen.tetris import rollout
from tetris_sprint.fast import HORIZON, TetrisSprintBatched

OBS_SHAPE = (9, 24, 10)
N_ACTIONS = 10
HOLD = 9
#: Raw rewards run to ±2·10^5; the Q head wants O(1) targets.
REWARD_SCALE = 10_000.0


def action_mask(env):
    mask = torch.ones(env.n, N_ACTIONS, dtype=torch.bool)
    mask[:, HOLD] = env.hold_used == 0
    return mask


def potential(board):
    """Φ(s) from the locked board [N, 24, 10]: minus holes, minus height.

    A hole is an empty cell with any filled cell above it in the same column;
    height is the tallest column. Both are the standard Tetris quality signals
    and both are computed from the board plane alone.
    """
    filled = board != 0                                  # [N, 24, 10]
    above = filled.flip(1).cummax(dim=1).values.flip(1)  # filled at or above
    holes = ((~filled) & above).sum(dim=(1, 2)).float()
    heights = (filled.any(dim=2).float()
               * torch.arange(1, 25, device=board.device)).amax(dim=1)
    return -(0.30 * holes + 0.05 * heights)


class BoolReplay:
    """Ring buffer with observations stored as bools (planes are 0/1)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.obs = torch.zeros(capacity, *OBS_SHAPE, dtype=torch.bool)
        self.next_obs = torch.zeros_like(self.obs)
        self.action = torch.zeros(capacity, dtype=torch.long)
        self.reward = torch.zeros(capacity)
        self.done = torch.zeros(capacity, dtype=torch.bool)
        self.next_mask = torch.zeros(capacity, N_ACTIONS, dtype=torch.bool)
        self.pos, self.full = 0, False

    def add(self, obs, action, reward, sign, done, next_obs, next_legal):
        del sign  # single-player: NStep carries +1 signs, nothing to store
        n = obs.shape[0]
        idx = (torch.arange(n) + self.pos) % self.capacity
        self.obs[idx] = obs.bool()
        self.next_obs[idx] = next_obs.bool()
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done
        self.next_mask[idx] = next_legal
        self.full = self.full or self.pos + n >= self.capacity
        self.pos = (self.pos + n) % self.capacity

    def __len__(self):
        return self.capacity if self.full else self.pos

    def sample(self, batch, device):
        i = torch.randint(0, len(self), (batch,))
        return (self.obs[i].float().to(device),
                self.action[i].to(device),
                self.reward[i].to(device),
                self.next_obs[i].float().to(device),
                self.done[i].to(device),
                self.next_mask[i].to(device))


def evaluate(net, device, episodes=24, latency=100, seed=123):
    """Greedy telemetry rollout — the unshaped truth about the policy."""
    env = TetrisSprintBatched(min(episodes, 32), latency=latency)

    def greedy(obs, env):
        with torch.no_grad():
            q = net(obs.to(device)).cpu()
        q = q.masked_fill(~action_mask(env), -1e9)
        return q.argmax(dim=1)

    # A stalling policy runs every episode to the horizon; cap the eval
    # budget so a bad checkpoint costs seconds, not minutes.
    rows = rollout(env, greedy, total_episodes=episodes, seed=seed,
                   max_steps=episodes * HORIZON)
    if not rows:
        return {"episodes": 0}
    mean = lambda k: sum(r[k] for r in rows) / len(rows)
    finished = [r for r in rows if r["finished"]]
    return {
        "episodes": len(rows),
        "mean_lines": round(mean("lines"), 2),
        "finish_rate": round(len(finished) / len(rows), 3),
        "mean_time_s": round(sum(r["time_ms"] for r in finished) / 1000.0
                             / max(len(finished), 1), 1),
        "inputs_per_piece": round(mean("inputs_per_piece"), 3),
        "quad_rate": round(mean("quad_rate"), 3),
    }


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    net = BoardNet(OBS_SHAPE, N_ACTIONS, args.channels).to(device)
    target = BoardNet(OBS_SHAPE, N_ACTIONS, args.channels).to(device)
    resumed_from = 0
    if args.resume:
        # Continue from the newest checkpoint in the output directory. The
        # replay buffer and epsilon restart cold - a brief re-exploration
        # bump, visible in the log, cheaper than serializing a 600MB buffer.
        ckpts = sorted(out.glob("ckpt_*.pt"))
        if ckpts:
            blob = torch.load(ckpts[-1], map_location=device)
            net.load_state_dict(blob["state"])
            resumed_from = int(blob["steps"])
            print(f"resumed from {ckpts[-1].name} ({resumed_from:,} steps)",
                  flush=True)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    env = TetrisSprintBatched(args.envs, latency=args.latency,
                              emit_final_states=False)
    env.reset(torch.arange(args.envs, dtype=torch.int64) + args.seed)
    buf = BoolReplay(args.buffer)
    nstep = NStep(args.n_step)
    ones = torch.ones(args.envs)

    obs = env.observe()
    phi = potential(env.board)
    pending = [c for c in checkpoint_schedule(args.steps, lo=2_000)
               if c > resumed_from]
    log, updates, steps = [], 0, resumed_from
    if resumed_from and (out / "train_log.json").exists():
        log = json.loads((out / "train_log.json").read_text())
    started = time.time()

    def save(tag):
        torch.save({"steps": tag, "channels": args.channels,
                    "latency": args.latency, "state": net.state_dict()},
                   out / f"ckpt_{tag:08d}.pt")
        report = evaluate(net, device, latency=args.latency)
        report.update({"steps": tag, "updates": updates,
                       "elapsed_s": round(time.time() - started, 1)})
        log.append(report)
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        print(f"[sprint ckpt {tag:>9,}] lines {report.get('mean_lines', 0):6.1f} "
              f"finish {report.get('finish_rate', 0):.2f} "
              f"in/pc {report.get('inputs_per_piece', 0):5.2f} "
              f"quads {report.get('quad_rate', 0):.2f} "
              f"{report['elapsed_s']:.0f}s", flush=True)

    while pending and pending[0] <= steps:
        save(pending.pop(0))

    while steps < args.steps:
        frac = min(1.0, (steps - resumed_from) / args.eps_decay)
        eps = args.eps_start + frac * (args.eps_end - args.eps_start)

        mask = action_mask(env)
        with torch.no_grad():
            q = net(obs.to(device)).cpu()
        acts = q.masked_fill(~mask, -1e9).argmax(dim=1)
        explore = torch.rand(args.envs) < eps
        if explore.any():
            noise = torch.rand(args.envs, N_ACTIONS).masked_fill(~mask, -1)
            acts[explore] = noise.argmax(dim=1)[explore]

        next_obs, reward, done, _ = env.step(acts)
        # Potential-based shaping (training-only). A terminal step hands back
        # the whole potential rather than the post-reset board's: γΦ(s')−Φ(s)
        # with Φ(terminal)=0, which keeps the telescoped sum exact.
        next_phi = potential(env.board)
        shaped = (reward / REWARD_SCALE
                  + torch.where(done, torch.zeros_like(phi), next_phi) - phi)
        phi = next_phi

        ready = nstep.push(obs=obs, action=acts, reward=shaped, sign=ones,
                           done=done, next_obs=next_obs,
                           next_legal=action_mask(env))
        if ready is not None:
            buf.add(**ready)
        obs = next_obs
        steps += args.envs

        if len(buf) >= args.learn_start:
            for _ in range(args.updates_per_step):
                o, a, r, no, d, nm = buf.sample(args.batch, device)
                with torch.no_grad():
                    pick = target(no).masked_fill(~nm, -1e9)
                    online = net(no).masked_fill(~nm, -1e9).argmax(dim=1, keepdim=True)
                    nq = pick.gather(1, online).squeeze(1)
                    y = torch.where(d, r, r + args.gamma * nq)
                pred = net(o).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = torch.nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                updates += 1
                if updates % args.target_sync == 0:
                    target.load_state_dict(net.state_dict())

        while pending and pending[0] <= steps:
            save(pending.pop(0))

    print("done", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--buffer", type=int, default=150_000)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--latency", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--n-step", type=int, default=8)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=int, default=1_000_000)
    p.add_argument("--learn-start", type=int, default=20_000)
    p.add_argument("--updates-per-step", type=int, default=2)
    p.add_argument("--target-sync", type=int, default=2_000)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="continue from the newest checkpoint in --out")
    p.add_argument("--out", default="coldopen/ladders/tetris_sprint")
    args = p.parse_args()
    print(f"tetris_sprint DQN: {args.steps:,} env steps on {args.device}, "
          f"latency {args.latency}", flush=True)
    train(args)


if __name__ == "__main__":
    main()
