"""Self-play DQN on Connect Four, checkpointed into a skill ladder.

The point of this script is not a strong bot. It is a *graded* set of bots: we
save the network at log-spaced points during training so that later analysis has
six agents of genuinely different strength, whose relative skill we then measure
directly with a round robin (see league.py) rather than assuming it.

Values are negamax: Q(s, a) is the result from the point of view of whoever is
to move at s, so the bootstrap target for a non-terminal move is the negation of
the opponent's best reply.
"""

import argparse
import json
import pathlib
import time

import torch

from coldopen.c4 import COLS, BatchedC4
from coldopen.net import C4Net, masked_q

def checkpoint_schedule(total, count=18):
    """Log-spaced ply counts at which to snapshot the network.

    Deliberately denser than the six tiers the article ends up using: a random
    initialisation already beats a uniform-random opponent about 85% of the time
    (any *consistent* policy does), so training step is a poor proxy for
    strength. We save a lot of rungs and let the measured league Elo decide
    which six are actually far enough apart to call tiers.
    """
    pts = {0}
    lo = 5_000
    for i in range(count):
        f = i / (count - 1)
        pts.add(int(round(lo * (total / lo) ** f)))
    pts.add(total)
    return sorted(pts)


class Replay:
    """Flat ring buffer of negamax transitions, held as int8 boards."""

    def __init__(self, capacity, device):
        self.capacity, self.device = capacity, device
        self.obs = torch.zeros(capacity, 2, 6, COLS, dtype=torch.int8, device=device)
        self.next_obs = torch.zeros_like(self.obs)
        self.action = torch.zeros(capacity, dtype=torch.long, device=device)
        self.reward = torch.zeros(capacity, device=device)
        self.done = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.next_legal = torch.zeros(capacity, COLS, dtype=torch.bool, device=device)
        self.pos, self.full = 0, False

    def add(self, obs, action, reward, next_obs, done, next_legal):
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.device) + self.pos) % self.capacity
        self.obs[idx] = obs.to(torch.int8)
        self.next_obs[idx] = next_obs.to(torch.int8)
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done
        self.next_legal[idx] = next_legal
        self.full = self.full or self.pos + n >= self.capacity
        self.pos = (self.pos + n) % self.capacity

    def __len__(self):
        return self.capacity if self.full else self.pos

    def sample(self, batch):
        i = torch.randint(0, len(self), (batch,), device=self.device)
        return (
            self.obs[i].float(),
            self.action[i],
            self.reward[i],
            self.next_obs[i].float(),
            self.done[i],
            self.next_legal[i],
        )


def evaluate_vs_random(net, device, games=200, seed=0):
    """Win rate of the greedy net against uniform-random play, seats alternating."""
    torch.manual_seed(seed)
    env = BatchedC4(games, device)
    net_seat = torch.arange(games, device=device) % 2
    while not env.done.all():
        legal = env.legal_mask()
        live = ~env.done
        acts = torch.zeros(games, dtype=torch.long, device=device)

        net_turn = live & (env.to_move == net_seat)
        if net_turn.any():
            with torch.no_grad():
                q = masked_q(net, env.observe()[net_turn], legal[net_turn])
            acts[net_turn] = q.argmax(dim=1)
        rand_turn = live & ~net_turn
        if rand_turn.any():
            noise = torch.rand(int(rand_turn.sum()), COLS, device=device)
            acts[rand_turn] = noise.masked_fill(~legal[rand_turn], -1).argmax(dim=1)
        env.step(acts)

    wins = (env.winner == net_seat).float().mean().item()
    draws = (env.winner < 0).float().mean().item()
    return wins, draws


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    net = C4Net(args.channels).to(device)
    target = C4Net(args.channels).to(device)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    env = BatchedC4(args.envs, device)
    buf = Replay(args.buffer, device)

    pending = checkpoint_schedule(args.plies)
    log = []
    plies = 0
    updates = 0
    started = time.time()

    def save(tag):
        path = out / f"ckpt_{tag:08d}.pt"
        torch.save({"plies": tag, "channels": args.channels, "state": net.state_dict()}, path)
        wins, draws = evaluate_vs_random(net, device, games=args.eval_games)
        rec = {
            "plies": tag,
            "updates": updates,
            "win_vs_random": round(wins, 4),
            "draw_vs_random": round(draws, 4),
            "elapsed_s": round(time.time() - started, 1),
        }
        log.append(rec)
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        print(f"[ckpt {tag:>9,}] vs-random win {wins:.3f} draw {draws:.3f} "
              f"updates {updates:,} {rec['elapsed_s']:.0f}s", flush=True)

    while pending and pending[0] <= plies:
        save(pending.pop(0))

    while plies < args.plies:
        frac = min(1.0, plies / (args.eps_decay_plies))
        eps = args.eps_start + frac * (args.eps_end - args.eps_start)

        legal = env.legal_mask()
        obs = env.observe()
        with torch.no_grad():
            q = masked_q(net, obs, legal)
        acts = q.argmax(dim=1)
        explore = torch.rand(env.n, device=device) < eps
        if explore.any():
            noise = torch.rand(env.n, COLS, device=device)
            acts[explore] = noise.masked_fill(~legal, -1).argmax(dim=1)[explore]

        reward, done = env.step(acts)
        next_obs = env.observe()
        next_legal = env.legal_mask()
        # A finished game's legal mask is all False; the target ignores it, but
        # keep at least one bit set so a downstream max() never sees an empty row.
        next_legal = next_legal | done.unsqueeze(1) * torch.nn.functional.one_hot(
            torch.zeros(env.n, dtype=torch.long, device=device), COLS
        ).bool()
        buf.add(obs, acts, reward, next_obs, done, next_legal)
        plies += env.n
        env.reset_done()

        if len(buf) >= args.learn_start:
            for _ in range(args.updates_per_step):
                o, a, r, no, d, nl = buf.sample(args.batch)
                with torch.no_grad():
                    # Double DQN: online net picks the reply, target net prices it.
                    pick = masked_q(net, no, nl).argmax(dim=1, keepdim=True)
                    nq = masked_q(target, no, nl).gather(1, pick).squeeze(1)
                    # Negamax: the opponent's best reply is our loss.
                    y = torch.where(d, r, -args.gamma * nq)
                pred = net(o).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = torch.nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                updates += 1
                if updates % args.target_sync == 0:
                    target.load_state_dict(net.state_dict())

        while pending and pending[0] <= plies:
            save(pending.pop(0))

    print("done", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plies", type=int, default=6_000_000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--buffer", type=int, default=500_000)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.08)
    p.add_argument("--eps-decay-plies", type=int, default=1_500_000)
    p.add_argument("--learn-start", type=int, default=20_000)
    p.add_argument("--updates-per-step", type=int, default=6)
    p.add_argument("--target-sync", type=int, default=2000)
    p.add_argument("--eval-games", type=int, default=400)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="coldopen/checkpoints")
    train(p.parse_args())


if __name__ == "__main__":
    main()
