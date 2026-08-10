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

**v2 changes, priced by v1's measured failure.** v1 ran 2M steps to a 0.000
win rate at every checkpoint, ending in a flag-spin - and the arithmetic says
that was the optimum of the reward it was given: an uninformed reveal is a
mine with prior ~21%, expected value ~-210,000, against flagging forever at
-230 a step. The shaping bonus for a correct reveal (+1) was 0.4% of the step
cost. Three training-only changes (env and spec untouched, telemetry always
reports env truth):

* `--shaping-weight` multiplies the potential term so correct reveals are
  materially positive rather than rounding error;
* `--death-penalty` rescales the terminal -1M during training. -1M targets do
  not just discourage death - through generalization they smear massive
  negatives across ALL reveal Q-values, burying the safe-vs-unsafe spread the
  agent has to learn. The trained objective becomes mildly death-tolerant
  rather than pure time-minimization, which a ladder generator can afford;
* an **auxiliary mine-prediction loss** on a 4th head channel, BCE against
  the env's own mines masked to unrevealed cells, computed on the live batch
  each loop. The A/B against mineprob showed the bottleneck is gradient
  density, not reward shape - same net and env, 480 labels/state reached 81.7
  safe cells in 7k updates while 1 scalar/action reached 0 in 2M steps. The
  aux head gives the shared conv body that dense signal while the Q-head
  learns values on top.

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

from coldopen.ms_curriculum import (CurriculumMinesweeper, Pacer,
                                    SyntheticStates)
from coldopen.nets import FullyConvNet
from coldopen.selfplay import NStep, checkpoint_schedule
from minesweeper.fast import CHORD, FLAG, REVEAL

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
    env = CurriculumMinesweeper(min(episodes, 32), source=None,
                                latency=latency, emit_final_states=False)
    env.reset(torch.arange(env.n, dtype=torch.int64) + seed)
    done = torch.zeros(env.n, dtype=torch.bool)
    wins = torch.zeros(env.n, dtype=torch.bool)
    revealed_at_end = torch.zeros(env.n, dtype=torch.int64)
    time_at_end = torch.zeros(env.n, dtype=torch.int64)
    steps_cap = 800
    obs = env.observe()
    for _ in range(steps_cap):
        with torch.no_grad():
            q = net.q_values(obs.to(device)).cpu()
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


class MsReplay:
    """Ring buffer with bool observation planes, sized for THIS game.

    Not shared with train_sprint's BoolReplay deliberately: that class binds
    its module's OBS_SHAPE and N_ACTIONS at import, so importing it here would
    silently allocate sprint-shaped tensors. Ten lines of duplication beats a
    shape bug that only explodes at the first add().
    """

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
        del sign  # single-player: signs are all +1
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


class MsNet(FullyConvNet):
    """FullyConvNet with 3 Q channels and 1 auxiliary mine-logit channel."""

    def __init__(self, in_channels, channels, depth):
        super().__init__(in_channels, 4, channels, depth)

    def q_values(self, x):
        return self.forward(x)[:, :3].flatten(1)

    def mine_logits(self, x):
        return self.forward(x)[:, 3].flatten(1)


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Fully convolutional: the [3, H, W] head IS the kind-major action space,
    # so a local deduction pattern is learned once and applied at all 480
    # positions instead of once per location through a flat bottleneck.
    # 4 output channels: 3 action kinds for the Q-head + 1 mine-logit channel
    # for the auxiliary supervised loss. One body, two signals.
    net = MsNet(OBS_SHAPE[0], args.channels, args.depth).to(device)
    target = MsNet(OBS_SHAPE[0], args.channels, args.depth).to(device)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Reverse curriculum: start near the goal, walk backwards as win rate
    # allows. Synthetic source only here - cold-start legitimate. The pacer
    # widens k_max whenever the recent win rate clears its threshold.
    source = SyntheticStates(k_max=args.k_start, seed=args.seed) \
        if args.curriculum else None
    pacer = Pacer(source, threshold=0.5, window=256) if source else None
    env = CurriculumMinesweeper(args.envs, source=source, latency=args.latency,
                                emit_final_states=False)
    env.reset(torch.arange(args.envs, dtype=torch.int64) + args.seed)

    buf = MsReplay(args.buffer)
    nstep = NStep(args.n_step)
    ones = torch.ones(args.envs)

    obs = env.observe()
    phi = potential(env)
    pending = checkpoint_schedule(args.steps, lo=2_000)
    log, updates, steps = [], 0, 0
    started = time.time()

    def save(tag):
        torch.save({"steps": tag, "channels": args.channels,
                    "depth": args.depth, "arch": "fcn",
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
            q = net.q_values(obs.to(device)).cpu()
        mask = action_mask(env)
        greedy = q.masked_fill(~mask, -1e9).argmax(dim=1)
        noise = torch.rand(args.envs, N_ACTIONS)
        random_legal = noise.masked_fill(~mask, -1.0).argmax(dim=1)
        explore = torch.rand(args.envs) < eps
        acts = torch.where(explore, random_legal, greedy)

        won_before = env.won  # cleared by auto-reset; capture via term+dead
        dead_snapshot = env.dead
        # auxiliary dense loss on the LIVE batch: the env's own mines are the
        # labels, masked to unrevealed cells of generated boards. No replay
        # plumbing needed, and the body sees a per-cell gradient every loop.
        if args.aux_weight > 0:
            gen = env.generated
            if bool(gen.any()):
                logits = net.mine_logits(obs[gen].to(device))
                y = env.mines[gen].float().to(device)
                m = (~env.revealed[gen]).to(device)
                raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y, reduction="none")
                aux = (raw * m).sum() / m.sum().clamp(min=1)
                opt.zero_grad(set_to_none=True)
                (args.aux_weight * aux).backward()
                opt.step()

        next_obs, reward, term, _ = env.step(acts)
        if pacer is not None and bool(term.any()):
            # a terminated instance won iff it terminated without dying and
            # before the horizon; env.won is already cleared by auto-reset,
            # but reward carries the win bonus exactly once
            won_flags = (reward > 500_000)[term].tolist()
            if pacer.update(won_flags):
                print(f"  curriculum -> k_max {source.k_max} "
                      f"(step {steps:,})", flush=True)
        # training-only reward surgery; env truth is never altered:
        # 1. rescale the -1M death term so generalization does not smear
        #    catastrophic negatives across every reveal Q-value;
        # 2. weight the potential term so a correct reveal is material.
        died = reward < -500_000
        reward_train = torch.where(
            died, reward + 1_000_000 - args.death_penalty, reward)
        phi_next = potential(env)
        phi_next = torch.where(term, torch.zeros_like(phi_next), phi_next)
        shaped = reward_train + args.shaping_weight * (
            args.gamma * phi_next - phi)
        phi = potential(env)

        ready = nstep.push(obs=obs.bool(), action=acts,
                           reward=shaped / 1000.0, sign=ones, done=term,
                           next_obs=next_obs.bool(),
                           next_legal=action_mask(env))
        if ready is not None:
            buf.add(**ready)
        obs = next_obs
        steps += args.envs

        if len(buf) >= args.learn_start:
            for _ in range(args.updates_per_step):
                o, a, r, no, d, nm = buf.sample(args.batch, device)
                with torch.no_grad():
                    pick = target.q_values(no).masked_fill(~nm, -1e9)
                    online = net.q_values(no).masked_fill(~nm, -1e9).argmax(
                        dim=1, keepdim=True)
                    nq = pick.gather(1, online).squeeze(1)
                    # n-step return, so the bootstrap is discounted by gamma^n
                    y = torch.where(d, r, r + (args.gamma ** args.n_step) * nq)
                pred = net.q_values(o).gather(1, a.unsqueeze(1)).squeeze(1)
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

    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--depth", type=int, default=8,
                   help="conv layers; receptive field is ~2*depth+1 cells")
    p.add_argument("--curriculum", action="store_true", default=True)
    p.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    p.add_argument("--k-start", type=int, default=3)
    p.add_argument("--shaping-weight", type=float, default=300.0)
    p.add_argument("--death-penalty", type=float, default=20_000.0,
                   help="training-time replacement for the env's 1M death term")
    p.add_argument("--aux-weight", type=float, default=1.0,
                   help="auxiliary mine-prediction loss weight; 0 disables")
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
