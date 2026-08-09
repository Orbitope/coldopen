"""Distil the sprint teacher at the PLACEMENT level, not the keystroke level.

Keystroke-level cloning does not work here, and the reason is structural rather
than a tuning miss. Asking a conv net for the next keystroke asks it to do two
jobs at once: run a 44-way placement search *and* track whether the piece has
already arrived at a column the net must itself have computed. Getting the
second job slightly wrong is unrecoverable — the piece never reaches the exact
state where the teacher says "hard drop", so the student taps forever. Measured
on the keystroke student: 98.6% agreement with the teacher, 25.8% agreement on
its own states, 1.8 lines, 60 inputs per piece.

This module splits the two jobs, which is also how they come apart in people:

* **placement judgement** — where should this piece go? Learned here, as a
  44-way classification over (rotation, column). Conv nets are good at this;
  it is a spatial question about a board picture.
* **motor execution** — how many keystrokes to get it there? Handled by the
  same emitter the teacher uses (`markov_action`), with a `fumble` rate as an
  explicit dial.

That split is a claim about what the ladder measures, so it is worth stating
plainly: the rungs vary in *judgement*, and finesse is imposed rather than
learned. That is a real limitation compared with the scripted ladder, where
both fall out of one `skill` number — but it is the honest way to get a
learned generator at all, and the E2 question ("is a part-trained imitator a
distinct kind of bad?") is a question about judgement.

One label per piece instead of one per keystroke, so a pool of the same size
carries roughly 3-4x fewer examples but every one of them is clean.

    python -m coldopen.distill_placement --out coldopen/ladders/sprint_placement
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "envs"))

from coldopen.distill import snapshot_schedule
from coldopen.nets import BoardNet
from coldopen.sprint_bot import HOLD, SprintBot, best_placement, markov_action
from coldopen.tetris import rollout
from coldopen.train_sprint import OBS_SHAPE
from tetris_sprint.fast import HORIZON, TetrisSprintBatched

#: The action space of this student: 4 rotations x the reachable column range.
#: `x` is the left edge of the piece's bounding box, and the largest legal
#: value is 8 (a vertical I there occupies column 9), so the sweep is -2..8.
X_LO, X_HI = -2, 9
N_X = X_HI - X_LO
N_PLACEMENTS = 4 * N_X


def encode(rot, x):
    return rot * N_X + (x - X_LO)


def decode(index):
    return int(index) // N_X, int(index) % N_X + X_LO


def collect(n_envs=64, pieces=400, latency=100, seed=0):
    """(observation, target placement) for every step of every piece's fall.

    Deliberately *not* sampled only at piece boundaries. The same (board,
    piece) recurs with the active piece drawn at a different height each step,
    so those rows are distinct observations carrying the same label — which is
    exactly the invariance the policy needs, because `placement_policy`
    re-predicts the target on every step rather than committing once. A student
    trained only on spawn-height boards would be asked at rollout about boards
    it had never seen.

    Hold steps are skipped: the label would describe a piece on its way out.
    """
    env = TetrisSprintBatched(n_envs, latency=latency, emit_final_states=False)
    env.reset(torch.arange(n_envs, dtype=torch.int64) + seed * 7919)
    bot = SprintBot(env, markov=True)

    observations, labels = [], []
    obs = env.observe()
    seen = 0
    while seen < pieces * n_envs:
        boards = env.board.cpu().numpy()
        active = env.piece.cpu().tolist()
        actions = bot(obs, env)
        # A hold swaps the piece, so the label for THIS observation would be
        # about a piece that is on its way out. Skip those rows.
        keep = (actions != HOLD).nonzero().flatten().tolist()
        if keep:
            targets = []
            for i in keep:
                _, rot, x = best_placement(boards[i], active[i])
                targets.append(encode(rot, x) if rot is not None else 0)
            observations.append(obs[keep].bool())
            labels.append(torch.tensor(targets, dtype=torch.int64))
            seen += len(keep)
        obs, _, _, _ = env.step(actions)
    return torch.cat(observations), torch.cat(labels)


def collect_student(net, device, n_envs=64, steps=250, latency=100, seed=0):
    """Boards the STUDENT reaches, labelled with the teacher's placement.

    A DAgger round, and at this level it is well posed in a way the
    keystroke version never was: the teacher's placement is a pure function of
    (board, piece), with no plan state to be out of sync with and no way for
    one disagreement to flood the labels.

    It is also necessary. Plain cloning reached 91.5% agreement on the
    teacher's own pool and **21.5% on the boards the student actually built** —
    the student's boards are messy in ways the teacher's never are, so nothing
    in the training set describes them.
    """
    env = TetrisSprintBatched(n_envs, latency=latency, emit_final_states=False)
    env.reset(torch.arange(n_envs, dtype=torch.int64) + seed * 104_729)
    observations, labels = [], []
    obs = env.observe()
    policy = placement_policy(net, device)
    for _ in range(steps):
        boards = env.board.cpu().numpy()
        pieces = env.piece.cpu().tolist()
        targets = []
        for i in range(env.n):
            _, rot, x = best_placement(boards[i], pieces[i])
            targets.append(encode(rot, x) if rot is not None else 0)
        observations.append(obs.bool())
        labels.append(torch.tensor(targets, dtype=torch.int64))
        obs, _, _, _ = env.step(policy(obs, env))
    return torch.cat(observations), torch.cat(labels)


def own_state_agreement(net, device, n_envs=12, steps=180, latency=100, seed=7):
    """Agreement with the teacher on the student's OWN boards.

    The number that matters, and the one that was missing every time this
    project mistook a well-fitted student for a working one. Teacher-pool
    agreement said 91.5% while this said 21.5%.
    """
    env = TetrisSprintBatched(n_envs, latency=latency, emit_final_states=False)
    env.reset(torch.arange(n_envs, dtype=torch.int64) + seed)
    policy = placement_policy(net, device)
    obs = env.observe()
    hits = total = 0
    for _ in range(steps):
        boards = env.board.cpu().numpy()
        pieces = env.piece.cpu().tolist()
        with torch.no_grad():
            choice = net(obs.to(device)).cpu().argmax(dim=1)
        for i in range(env.n):
            _, rot, x = best_placement(boards[i], pieces[i])
            total += 1
            if rot is not None and decode(choice[i]) == (rot, x):
                hits += 1
        obs, _, _, _ = env.step(policy(obs, env))
    return hits / max(total, 1)


def placement_policy(net, device, fumble=0.0, seed=0):
    """Roll out a placement student: predict a target, emit toward it.

    The target is recomputed every step from the board, exactly as the Markov
    teacher does, so a mispredicted piece simply gets re-aimed rather than
    stranding the policy — which is the failure mode that killed the
    keystroke student.
    """
    rng = np.random.default_rng(seed)

    def policy(obs, env):
        with torch.no_grad():
            logits = net(obs.to(device)).cpu()
        choice = logits.argmax(dim=1)
        boards = env.board.cpu().numpy()
        pieces = env.piece.cpu().tolist()
        rots = env.rot.cpu().tolist()
        xs = env.x.cpu().tolist()
        actions = torch.zeros(env.n, dtype=torch.int64)
        for i in range(env.n):
            rot, x = decode(choice[i])
            if fumble and rng.random() < fumble:
                x = int(np.clip(x + rng.choice([-1, 1]), X_LO, X_HI - 1))
            actions[i] = markov_action(boards[i], pieces[i], rots[i], xs[i],
                                       rot, x)
        return actions
    return policy


def evaluate(net, device, episodes=12, latency=100, fumble=0.0, seed=123):
    env = TetrisSprintBatched(min(episodes, 24), latency=latency)
    rows = rollout(env, placement_policy(net, device, fumble), seed=seed,
                   total_episodes=episodes, max_steps=episodes * HORIZON)
    if not rows:
        return {"episodes": 0}
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    finished = [r for r in rows if r["finished"]]
    return {
        "episodes": len(rows),
        "mean_lines": round(mean("lines"), 2),
        "finish_rate": round(len(finished) / len(rows), 3),
        "time_s": (round(float(np.mean([r["time_ms"] for r in finished])) / 1000, 1)
                   if finished else None),
        "inputs_per_piece": round(mean("inputs_per_piece"), 3),
        "quad_rate": round(mean("quad_rate"), 3),
        "holds_per_piece": round(mean("holds_per_piece"), 3),
        "pps": round(mean("pps"), 3),
    }


def distil(steps=8000, batch=256, lr=1e-3, channels=64, seed=0, latency=100,
           pool_envs=64, pool_pieces=120, rounds=5, out=None, device="cpu"):
    device = torch.device(device)
    torch.manual_seed(seed)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)

    print("collecting teacher placements ...", flush=True)
    obs, labels = collect(pool_envs, pool_pieces, latency=latency, seed=seed)
    print(f"  {obs.shape[0]:,} placement decisions, "
          f"{len(torch.unique(labels))} distinct targets used", flush=True)

    student = BoardNet(OBS_SHAPE, N_PLACEMENTS, channels).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    pending = snapshot_schedule(steps, count=14, lo=25)
    per_round = max(1, steps // rounds)
    log, started = [], time.time()

    def save(tag):
        torch.save({"steps": tag, "channels": channels, "latency": latency,
                    "placement": True, "state": student.state_dict()},
                   out / f"ckpt_{tag:08d}.pt")
        index = torch.randint(0, obs.shape[0], (4096,))
        with torch.no_grad():
            predicted = student(obs[index].float().to(device)).argmax(dim=1).cpu()
        pool = float((predicted == labels[index]).float().mean())
        # Both numbers, always, and `own` first when reading: pool agreement
        # said 91.5% while own-state agreement said 21.5%, and the second was
        # the one describing the policy.
        own = own_state_agreement(student, device, latency=latency)
        report = evaluate(student, device, latency=latency)
        report.update({"steps": tag, "pool_agreement": round(pool, 4),
                       "own_state_agreement": round(own, 4),
                       "elapsed_s": round(time.time() - started, 1)})
        log.append(report)
        (out / "train_log.json").write_text(json.dumps(log, indent=2))
        print(f"  [{tag:>6,}] own {own:.3f} (pool {pool:.3f})  "
              f"lines {report.get('mean_lines', 0):5.1f}  "
              f"finish {report.get('finish_rate', 0):.2f}  "
              f"in/pc {report.get('inputs_per_piece', 0):5.2f}  "
              f"quad {report.get('quad_rate', 0):.3f}", flush=True)

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
        if rounds > 1 and step % per_round == 0 and step < steps:
            extra_obs, extra_labels = collect_student(
                student, device, pool_envs, max(60, pool_pieces // 3),
                latency=latency, seed=seed + step)
            obs = torch.cat([obs, extra_obs])
            labels = torch.cat([labels, extra_labels])
            print(f"  dagger round at {step:,}: +{extra_obs.shape[0]:,} boards "
                  f"(total {obs.shape[0]:,})", flush=True)
    return student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latency", type=int, default=100)
    ap.add_argument("--pool-envs", type=int, default=64)
    ap.add_argument("--pool-pieces", type=int, default=120)
    ap.add_argument("--rounds", type=int, default=5,
                    help="DAgger rounds; 1 reproduces plain cloning")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="coldopen/ladders/sprint_placement")
    args = ap.parse_args()
    distil(steps=args.steps, batch=args.batch, lr=args.lr, channels=args.channels,
           seed=args.seed, latency=args.latency, pool_envs=args.pool_envs,
           pool_pieces=args.pool_pieces, rounds=args.rounds, out=args.out,
           device=args.device)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
