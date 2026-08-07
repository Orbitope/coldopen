"""Self-play DQN over any of the four games, checkpointed into a skill ladder.

This is ``train.py`` with the Connect Four assumptions taken out. Two of them
were load-bearing and neither survives contact with the other three games:

**"The other player moves next."** Connect Four alternates strictly, so the
bootstrap target is always the negation of the next state's value. Othello makes
a player pass when they have no legal move, and Pgx backgammon spends one action
per die, so a turn is two to four consecutive actions by the same player. Both
break a hardcoded negation. The fix is to read who is actually to move next and
negate only when it changed - which reduces to the Connect Four rule wherever
the game does alternate, so nothing is special-cased.

**"A win is worth 1."** Backgammon pays 1, 2 or 3 depending on how badly the
loser lost, and Leduc pays the pot. Both are divided down to [-1, 1] by the game
adapter so the value head stays in range; the league scores wins by sign, so
measured strength is unaffected.

A third assumption was not visible in Connect Four at all. These games pay out
only at the end, so a one-step target has to walk the result back one ply per
sweep. Over a 42-ply Connect Four game that is affordable; over a backgammon
game, which runs to a couple of hundred plies, it is not - the first attempt
here produced a ladder with a 32 Elo spread in which the untrained network
placed third, because no checkpoint had learned anything. The targets are
therefore n-step, which walks the result back n plies per sweep instead, with n
set per game against how long its episodes are.

    python -m coldopen.selfplay --game othello --out coldopen/ladders/othello
"""

import argparse
import json
import pathlib
import time

import jax
import torch

from coldopen import games
from coldopen.nets import epsilon_actions, make_net, masked_q, save_checkpoint

#: Per-game training budgets, in plies. A ply is not a comparable unit across
#: these games - a Connect Four game is 42 of them and a backgammon game runs to
#: several hundred - so backgammon gets the larger budget to see a comparable
#: number of *games*, and Leduc, which is two decisions over three actions, gets
#: much less because it saturates almost immediately.
#:
#: ``n_step`` is set against episode length, since the only reward any of these
#: games pays arrives at the end: roughly a tenth of a typical game, so the
#: result walks back in ten sweeps rather than one per ply. Leduc's episodes are
#: about four plies long, so anything above that is simply Monte Carlo.
PROFILES = {
    "connect_four": dict(plies=6_000_000, envs=256, channels=64,
                         eps_decay_plies=1_500_000, n_step=4),
    "othello": dict(plies=6_000_000, envs=256, channels=64,
                    eps_decay_plies=1_500_000, n_step=6),
    "backgammon": dict(plies=12_000_000, envs=256, channels=256,
                       eps_decay_plies=3_000_000, n_step=24),
    "leduc_holdem": dict(plies=2_000_000, envs=256, channels=128,
                         eps_decay_plies=400_000, n_step=4),
}


def default_device(info):
    """MPS pays for itself on the conv nets and not on the MLPs.

    Measured on this machine: a batch-512 update costs 107 ms on CPU and 6.5 ms
    on MPS for the Othello conv stack, but 1.8 ms either way for the backgammon
    MLP, where the transfers cancel the arithmetic. So pick per game rather than
    globally.
    """
    if info.spatial and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def checkpoint_schedule(total, count=18, lo=5_000):
    """Log-spaced ply counts at which to snapshot the network.

    Deliberately denser than the six tiers the analysis ends up using: a random
    initialisation already beats a uniform-random opponent most of the time
    (any *consistent* policy does), so training step is a poor proxy for
    strength. Save a lot of rungs and let the measured league Elo decide which
    six are actually far enough apart to call tiers.
    """
    pts = {0}
    for i in range(count):
        f = i / (count - 1)
        pts.add(int(round(lo * (total / lo) ** f)))
    pts.add(total)
    return sorted(p for p in pts if p <= total)


class Replay:
    """Flat ring buffer of transitions, with the negamax sign carried along.

    ``sign`` is +1 when the same player is still to move in the next state and
    -1 when the turn passed. Storing it per transition is what lets one buffer
    hold Othello passes and backgammon's multi-action turns without the learner
    knowing which game it is looking at.
    """

    def __init__(self, capacity, obs_shape, n_actions, device):
        self.capacity, self.device = capacity, device
        self.obs = torch.zeros(capacity, *obs_shape, dtype=torch.float16, device=device)
        self.next_obs = torch.zeros_like(self.obs)
        self.action = torch.zeros(capacity, dtype=torch.long, device=device)
        self.reward = torch.zeros(capacity, device=device)
        self.sign = torch.zeros(capacity, device=device)
        self.done = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.next_legal = torch.zeros(capacity, n_actions, dtype=torch.bool, device=device)
        self.pos, self.full = 0, False

    def add(self, obs, action, reward, sign, next_obs, done, next_legal):
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.device) + self.pos) % self.capacity
        self.obs[idx] = obs.to(torch.float16)
        self.next_obs[idx] = next_obs.to(torch.float16)
        self.action[idx] = action
        self.reward[idx] = reward
        self.sign[idx] = sign
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
            self.sign[i],
            self.next_obs[i].float(),
            self.done[i],
            self.next_legal[i],
        )


class NStep:
    """Turns single plies into n-step transitions, one window per batch slot.

    These games pay nothing until somebody wins, which makes the n-step return
    unusually simple: with no intermediate rewards it is the terminal result
    seen through the product of the negamax signs between here and there, or -
    if the episode has not ended within n plies - the bootstrap from the state n
    plies ahead through that same product.

    Slots run their own episodes, so the walk stops at the *first* termination
    inside each slot's window. Anything after that belongs to the next episode
    and is picked up by later windows.
    """

    def __init__(self, n):
        self.n = n
        self.window = []

    def push(self, **step):
        self.window.append(step)
        if len(self.window) < self.n:
            return None
        emitted = self._emit()
        self.window.pop(0)
        return emitted

    def _emit(self):
        head = self.window[0]
        cum = torch.ones_like(head["sign"])
        reward = torch.zeros_like(head["reward"])
        settled = torch.zeros_like(head["done"])
        for entry in self.window:
            ends = entry["done"] & ~settled
            reward = torch.where(ends, cum * entry["reward"], reward)
            settled = settled | ends
            cum = cum * entry["sign"]
        tail = self.window[-1]
        return dict(
            obs=head["obs"],
            action=head["action"],
            reward=reward,
            sign=cum,  # only read where the window did not terminate
            done=settled,
            next_obs=tail["next_obs"],
            next_legal=tail["next_legal"],
        )


def evaluate_vs_random(game, net, batch=256, seed=0):
    """Progress check against uniform-random play, seats alternating.

    Returns ``(win_rate, draw_rate, score)``, where score is the mean of
    ``(r + 1) / 2``. Watch the score, not the win rate: in Leduc a policy that
    folds everything wins the majority of *hands* and loses the money, so its
    win rate goes up as it gets worse. Pgx picks which seat acts first at deal
    time, so alternating seats here is not the same as alternating who is
    "player 0".
    """
    device = game.device
    gen = torch.Generator().manual_seed(seed)
    state = game.init(batch, seed=seed)
    net_seat = (torch.arange(batch, device=device) % 2).long()
    result = torch.zeros(batch, device=device)
    settled = torch.zeros(batch, dtype=torch.bool, device=device)
    key = jax.random.PRNGKey(seed + 1)

    for _ in range(game.info.max_plies):
        if bool(settled.all()):
            break
        obs, legal = game.observe(state), game.legal_mask(state)
        mover = game.current_player(state)
        net_turn = mover == net_seat
        acts = torch.zeros(batch, dtype=torch.long, device=device)
        if net_turn.any():
            acts[net_turn] = epsilon_actions(
                net, obs[net_turn], legal[net_turn], 0.0, gen, device)
        if (~net_turn).any():
            acts[~net_turn] = epsilon_actions(
                None, obs[~net_turn], legal[~net_turn], 0.0, gen, device)
        key, sub = jax.random.split(key)
        state = game.step(state, acts, sub)
        just = game.finished(state) & ~settled
        if just.any():
            result[just] = game.reward_for(state, net_seat)[just]
            settled |= just

    wins = float((result > 0).float().mean())
    draws = float((result == 0).float().mean())
    score = float(((result.clamp(-1.0, 1.0) + 1.0) / 2.0).mean())
    return wins, draws, score


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    game = games.make(args.game, device)
    info = game.info
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    net = make_net(info, args.channels).to(device)
    target = make_net(info, args.channels).to(device)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    buf = Replay(args.buffer, info.obs_shape, info.n_actions, device)
    nstep = NStep(args.n_step)
    state = game.init(args.envs, seed=args.seed)
    key = jax.random.PRNGKey(args.seed + 7)

    pending = checkpoint_schedule(args.plies)
    log, plies, updates = [], 0, 0
    started = time.time()

    def save(tag):
        save_checkpoint(out / f"ckpt_{tag:08d}.pt", net, info, tag, args.channels)
        wins, draws, score = evaluate_vs_random(
            game, net, batch=args.eval_games, seed=args.seed)
        rec = {
            "plies": tag,
            "updates": updates,
            "win_vs_random": round(wins, 4),
            "draw_vs_random": round(draws, 4),
            "score_vs_random": round(score, 4),
            "elapsed_s": round(time.time() - started, 1),
        }
        log.append(rec)
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        print(f"[{info.key} ckpt {tag:>9,}] vs-random score {score:.3f} "
              f"(win {wins:.3f} draw {draws:.3f}) "
              f"updates {updates:,} {rec['elapsed_s']:.0f}s", flush=True)

    while pending and pending[0] <= plies:
        save(pending.pop(0))

    while plies < args.plies:
        frac = min(1.0, plies / args.eps_decay_plies)
        eps = args.eps_start + frac * (args.eps_end - args.eps_start)

        obs, legal = game.observe(state), game.legal_mask(state)
        mover = game.current_player(state)
        with torch.no_grad():
            acts = masked_q(net, obs, legal).argmax(dim=1)
        explore = torch.rand(args.envs, device=device) < eps
        if explore.any():
            noise = torch.rand(args.envs, info.n_actions, device=device)
            acts[explore] = noise.masked_fill(~legal, -1).argmax(dim=1)[explore]

        key, sub = jax.random.split(key)
        nxt = game.step(state, acts, sub)

        done = game.finished(nxt)
        reward = game.reward_for(nxt, mover)
        next_obs = game.observe(nxt)
        next_legal = games.ensure_nonempty(game.legal_mask(nxt))
        # +1 when the same player still has the move (an Othello pass by the
        # opponent, or the second die of a backgammon turn), -1 when it passed.
        sign = torch.where(game.current_player(nxt) == mover, 1.0, -1.0)

        ready = nstep.push(
            obs=obs, action=acts, reward=reward, sign=sign,
            done=done, next_obs=next_obs, next_legal=next_legal,
        )
        if ready is not None:
            buf.add(**ready)
        plies += args.envs

        key, sub = jax.random.split(key)
        state = game.restart(nxt, done, sub)

        if len(buf) >= args.learn_start:
            for _ in range(args.updates_per_step):
                o, a, r, sg, no, d, nl = buf.sample(args.batch)
                with torch.no_grad():
                    # Double DQN: online net picks the reply, target net prices it.
                    pick = masked_q(net, no, nl).argmax(dim=1, keepdim=True)
                    nq = masked_q(target, no, nl).gather(1, pick).squeeze(1)
                    y = torch.where(d, r, sg * args.gamma * nq)
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
    p.add_argument("--game", default="othello", choices=list(games.ALL_GAMES))
    p.add_argument("--plies", type=int)
    p.add_argument("--envs", type=int)
    p.add_argument("--channels", type=int)
    p.add_argument("--eps-decay-plies", type=int)
    p.add_argument("--n-step", type=int)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--buffer", type=int, default=400_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.08)
    p.add_argument("--learn-start", type=int, default=20_000)
    p.add_argument("--updates-per-step", type=int, default=4)
    p.add_argument("--target-sync", type=int, default=2000)
    p.add_argument("--eval-games", type=int, default=256)
    p.add_argument("--device")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    args = p.parse_args()

    for name, value in PROFILES[args.game].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.device is None:
        args.device = default_device(games.Game(args.game).info)
    if args.out is None:
        args.out = f"coldopen/ladders/{args.game}"
    print(f"{args.game}: {args.plies:,} plies on {args.device}", flush=True)
    train(args)


if __name__ == "__main__":
    main()
