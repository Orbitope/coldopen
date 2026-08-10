"""A third way to build a ladder: snapshot a student while it learns to imitate.

E2 compared two generators. **Undertraining** catches a network partway through
reinforcement learning, and produces agents that are uniformly bad - their value
function is noisy everywhere. **Handicapping** takes a finished agent and adds
noise or blindness, and produces agents whose errors are unbiased but spread
evenly across every kind of mistake. Neither is obviously how people are bad.

This is the third option, and it may be the closest of the three. Train a
student to copy a strong teacher, and snapshot it as it learns. A
partially-trained imitator has picked up the *common* patterns and not the rare
ones, so its mistakes concentrate in the situations it has seen least - which is
what "strong at what you have drilled, weak elsewhere" means, and is roughly how
people learn games. It is also how a beginner differs from an expert in a way
that pure noise cannot reproduce.

Practically it has a second advantage, which is what made it worth building.
A ladder must span the range the humans occupy, and imitation gives that for
free: the student starts at random and ends near the teacher, so choosing a
teacher who sits at the top of the human range makes the ladder cover the whole
of it. Handicapping can only reach downward from wherever the finished agent
already is, and undertraining only reaches upward from random to wherever
training got to.

The student is trained on positions from the teacher's own play, which is the
standard behavioural-cloning setup and its standard weakness: the student never
sees the positions its own mistakes lead to. That is a real limitation for
building a *strong* agent and an irrelevance for building a *graded* one, which
is all this is for.

    python -m coldopen.distill --game connect_four
"""

import argparse
import json
import pathlib
import time

import jax
import torch

from coldopen import games as game_mod
from coldopen.ladder import load_ladder
from coldopen.nets import make_net, masked_q, save_checkpoint

#: Log-spaced snapshots, dense early because that is where the student changes
#: fastest - the first few hundred steps take it from random to roughly
#: competent, and a ladder wants rungs there rather than a cluster at the top.
def snapshot_schedule(total, count=16, lo=50):
    points = {0}
    for i in range(count):
        fraction = i / (count - 1)
        points.add(int(round(lo * (total / lo) ** fraction)))
    points.add(total)
    return sorted(p for p in points if p <= total)


def teacher_positions(game, teacher, batch, steps, seed=0, epsilon=0.25):
    """Positions from the teacher's own play, with its preferred action.

    Exploration is deliberately high. A teacher playing itself greedily visits a
    narrow band of positions, and a student trained only there is helpless the
    moment it deviates - which for a *ladder* matters less than usual, but a
    ladder made of agents that collapse off-distribution would grade on
    fragility rather than on skill.
    """
    device = game.device
    generator = torch.Generator().manual_seed(seed)
    state = game.init(batch, seed=seed)
    key = jax.random.PRNGKey(seed + 5)

    observations, actions, masks = [], [], []
    for _ in range(steps):
        obs, legal = game.observe(state), game.legal_mask(state)
        with torch.no_grad():
            best = masked_q(teacher, obs, legal).argmax(dim=1)
        observations.append(obs.cpu())
        actions.append(best.cpu())
        masks.append(legal.cpu())

        played = best.clone()
        explore = torch.rand(batch, generator=generator).to(device) < epsilon
        if explore.any():
            noise = torch.rand(batch, game.info.n_actions, generator=generator).to(device)
            played[explore] = noise.masked_fill(~legal, -1).argmax(dim=1)[explore]

        key, sub = jax.random.split(key)
        state = game.step(state, played, sub)
        key, sub = jax.random.split(key)
        state = game.restart(state, game.finished(state), sub)

    return (torch.cat(observations), torch.cat(actions), torch.cat(masks))


def distill(game, teacher, steps=4000, batch=512, lr=1e-3, channels=64,
            seed=0, out=None, pool_steps=400, pool_batch=256):
    """Train a student to copy the teacher, snapshotting as it goes."""
    device = game.device
    torch.manual_seed(seed)
    student = make_net(game.info, channels).to(device)
    optimiser = torch.optim.Adam(student.parameters(), lr=lr)

    print(f"collecting {pool_steps * pool_batch:,} teacher decisions ...", flush=True)
    obs, target, legal = teacher_positions(
        game, teacher, pool_batch, pool_steps, seed=seed)
    obs, target, legal = obs.to(device), target.to(device), legal.to(device)
    print(f"  {obs.shape[0]:,} positions", flush=True)

    pending = snapshot_schedule(steps)
    log, started = [], time.time()
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def save(tag):
        save_checkpoint(out / f"ckpt_{tag:08d}.pt", student, game.info, tag, channels)
        with torch.no_grad():
            agree = float((masked_q(student, obs[:4096], legal[:4096]).argmax(dim=1)
                           == target[:4096]).float().mean())
        log.append({"steps": tag, "teacher_agreement": round(agree, 4),
                    "elapsed_s": round(time.time() - started, 1)})
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        print(f"  [{tag:>7,}] agrees with teacher {agree:.3f}", flush=True)

    while pending and pending[0] <= 0:
        save(pending.pop(0))

    for step in range(1, steps + 1):
        index = torch.randint(0, obs.shape[0], (batch,), device=device)
        logits = masked_q(student, obs[index], legal[index])
        loss = torch.nn.functional.cross_entropy(logits, target[index])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        while pending and pending[0] <= step:
            save(pending.pop(0))
    return student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="connect_four", choices=list(game_mod.ALL_GAMES))
    ap.add_argument("--checkpoints")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    ckpt_dir = args.checkpoints or f"coldopen/ladders/{args.game}"
    out = args.out or f"coldopen/ladders/{args.game}_distilled"

    game = game_mod.make(args.game, args.device)
    trained = load_ladder(ckpt_dir, game.info, args.device)
    teacher = trained[-1]["net"]
    print(f"{args.game}: distilling {trained[-1]['id']} into a fresh student", flush=True)
    distill(game, teacher, steps=args.steps, batch=args.batch, lr=args.lr,
            channels=args.channels, seed=args.seed, out=out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
