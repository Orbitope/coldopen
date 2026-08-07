"""Measure how strong each checkpoint actually is, instead of assuming.

Training step is a poor proxy for skill - a randomly initialised network already
beats a uniform-random opponent about 85% of the time, because any *consistent*
policy does. So every checkpoint plays a round robin, and the results are fitted
with Bradley-Terry to put them on an Elo scale anchored by a uniform-random
reference player.

Two deterministic networks playing each other produce the same game every time,
which would make a round robin one game per pair dressed up as many. Each game
therefore opens with a few uniformly random plies before either policy takes
over, the way engine testing normally handles this.
"""

import argparse
import json
import math
import pathlib
import re

import torch

from coldopen.c4 import COLS, BatchedC4
from coldopen.net import C4Net, masked_q

RANDOM_PLAYER = "random"


def load_checkpoints(directory, device="cpu"):
    """Every saved network, ordered by training plies, plus a random baseline."""
    directory = pathlib.Path(directory)
    entries = []
    for path in sorted(directory.glob("ckpt_*.pt")):
        plies = int(re.search(r"ckpt_(\d+)\.pt", path.name).group(1))
        blob = torch.load(path, map_location=device)
        net = C4Net(blob.get("channels", 64)).to(device)
        net.load_state_dict(blob["state"])
        net.eval()
        entries.append({"id": f"ckpt_{plies}", "plies": plies, "net": net})
    entries.sort(key=lambda e: e["plies"])
    return [{"id": RANDOM_PLAYER, "plies": -1, "net": None}] + entries


def _act(net, obs, legal, epsilon, generator, device):
    """Greedy under the net, or uniform over legal columns if there is no net."""
    n = obs.shape[0]
    if net is None:
        noise = torch.rand(n, COLS, device=device, generator=generator)
        return noise.masked_fill(~legal, -1).argmax(dim=1)
    with torch.no_grad():
        acts = masked_q(net, obs, legal).argmax(dim=1)
    if epsilon > 0:
        explore = torch.rand(n, device=device, generator=generator) < epsilon
        if explore.any():
            noise = torch.rand(n, COLS, device=device, generator=generator)
            acts[explore] = noise.masked_fill(~legal, -1).argmax(dim=1)[explore]
    return acts


def play_pair(net_a, net_b, games, device="cpu", seed=0, epsilon=0.03, opening_plies=2):
    """Play ``games`` between two policies, seats alternating. Returns (wins_a, wins_b, draws)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    env = BatchedC4(games, device)
    seat_a = (torch.arange(games, device=device) % 2).long()

    for _ in range(opening_plies):
        legal = env.legal_mask()
        noise = torch.rand(games, COLS, device=device, generator=gen)
        env.step(noise.masked_fill(~legal, -1).argmax(dim=1))

    while not env.done.all():
        legal, obs = env.legal_mask(), env.observe()
        acts = torch.zeros(games, dtype=torch.long, device=device)
        a_turn = (~env.done) & (env.to_move == seat_a)
        b_turn = (~env.done) & ~a_turn
        if a_turn.any():
            acts[a_turn] = _act(net_a, obs[a_turn], legal[a_turn], epsilon, gen, device)
        if b_turn.any():
            acts[b_turn] = _act(net_b, obs[b_turn], legal[b_turn], epsilon, gen, device)
        env.step(acts)

    wins_a = int((env.winner == seat_a).sum())
    wins_b = int(((env.winner >= 0) & (env.winner != seat_a)).sum())
    return wins_a, wins_b, games - wins_a - wins_b


def bradley_terry(win_matrix, iterations=1000, tol=1e-10):
    """MLE strengths from pairwise results, by the standard MM iteration.

    ``win_matrix[i][j]`` is how often i beat j (draws counted as half to each).
    Returns strengths normalised to a geometric mean of 1.
    """
    n = len(win_matrix)
    wins = [sum(win_matrix[i]) for i in range(n)]
    games = [[win_matrix[i][j] + win_matrix[j][i] for j in range(n)] for i in range(n)]
    p = [1.0] * n
    for _ in range(iterations):
        new = list(p)
        for i in range(n):
            denom = sum(
                games[i][j] / (p[i] + p[j]) for j in range(n) if j != i and games[i][j] > 0
            )
            # A player who never won gets a floor rather than a zero, so the log
            # below stays finite and the ordering is still reported honestly.
            new[i] = (wins[i] / denom) if denom > 0 and wins[i] > 0 else 1e-6
        gm = math.exp(sum(math.log(max(x, 1e-12)) for x in new) / n)
        new = [x / gm for x in new]
        if max(abs(a - b) for a, b in zip(new, p)) < tol:
            p = new
            break
        p = new
    return p


def to_elo(strengths, anchor_index=0, anchor_rating=0.0):
    """Bradley-Terry strengths on an Elo scale, anchored on the random player."""
    raw = [400.0 * math.log10(max(s, 1e-12)) for s in strengths]
    shift = anchor_rating - raw[anchor_index]
    return [r + shift for r in raw]


def run_league(entries, games=200, device="cpu", seed=0, epsilon=0.03):
    n = len(entries)
    matrix = [[0.0] * n for _ in range(n)]
    records = []
    for i in range(n):
        for j in range(i + 1, n):
            wa, wb, dr = play_pair(
                entries[i]["net"], entries[j]["net"], games,
                device=device, seed=seed + i * 97 + j, epsilon=epsilon,
            )
            matrix[i][j] += wa + dr / 2.0
            matrix[j][i] += wb + dr / 2.0
            records.append({
                "a": entries[i]["id"], "b": entries[j]["id"],
                "wins_a": wa, "wins_b": wb, "draws": dr,
                "score_a": (wa + dr / 2.0) / games,
            })
            print(f"  {entries[i]['id']:>16} vs {entries[j]['id']:<16} "
                  f"{wa:3}-{wb:3}-{dr:3}", flush=True)

    strengths = bradley_terry(matrix)
    elos = to_elo(strengths)
    table = [
        {"id": e["id"], "plies": e["plies"], "elo": round(elos[k], 1),
         "strength": strengths[k]}
        for k, e in enumerate(entries)
    ]
    return {"table": table, "pairs": records, "games_per_pair": games}


def monotonicity(table):
    """Does measured strength actually increase with training?

    Reported, not enforced elsewhere: if it fails, the ladder is not a ladder and
    the telemetry section has nothing to classify.
    """
    trained = [r for r in table if r["id"] != RANDOM_PLAYER]
    trained.sort(key=lambda r: r["plies"])
    elos = [r["elo"] for r in trained]
    inversions = sum(1 for a, b in zip(elos, elos[1:]) if b < a)
    pairs = [(a, b) for i, a in enumerate(elos) for b in elos[i + 1:]]
    concordant = sum(1 for a, b in pairs if b > a)

    # Checkpoints saved before the replay buffer reaches its learning threshold
    # are all the same untrained network, so their ordering is noise. Report
    # where the ladder actually starts behaving like one.
    start = 0
    for i in range(len(elos)):
        tail = elos[i:]
        if all(b >= a for a, b in zip(tail, tail[1:])):
            start = i
            break
    return {
        "adjacent_inversions": inversions,
        "adjacent_pairs": len(elos) - 1,
        "rank_concordance": concordant / len(pairs) if pairs else None,
        "elo_range": max(elos) - min(elos) if elos else 0.0,
        "strictly_monotone": inversions == 0,
        "monotone_from_plies": trained[start]["plies"] if trained else None,
        "monotone_suffix_length": len(elos) - start,
    }


def pick_tiers(table, count=6):
    """Choose evenly spaced rungs by *measured* Elo, not by training step."""
    rows = sorted(table, key=lambda r: r["elo"])
    lo, hi = rows[0]["elo"], rows[-1]["elo"]
    targets = [lo + (hi - lo) * i / (count - 1) for i in range(count)]
    chosen, used = [], set()
    for t in targets:
        best = min(
            (r for r in rows if r["id"] not in used),
            key=lambda r: abs(r["elo"] - t),
        )
        used.add(best["id"])
        chosen.append(best)
    chosen.sort(key=lambda r: r["elo"])
    names = ["bronze", "silver", "gold", "platinum", "diamond", "grandmaster"]
    for i, row in enumerate(chosen):
        row["tier"] = names[i] if count == len(names) else f"tier{i}"
        row["tier_index"] = i
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="coldopen/checkpoints")
    ap.add_argument("--out", default="analysis/league.json")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--tiers", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    entries = load_checkpoints(args.checkpoints, args.device)
    print(f"{len(entries)} players (including the random anchor)", flush=True)
    result = run_league(entries, games=args.games, device=args.device, seed=args.seed)
    result["monotonicity"] = monotonicity(result["table"])
    result["tiers"] = pick_tiers(result["table"], args.tiers)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))

    print(f"\n{'id':>18} {'plies':>10} {'elo':>8}")
    for row in sorted(result["table"], key=lambda r: r["elo"]):
        print(f"{row['id']:>18} {row['plies']:>10} {row['elo']:>8.1f}")
    m = result["monotonicity"]
    print(f"\nmonotone: {m['strictly_monotone']}  inversions "
          f"{m['adjacent_inversions']}/{m['adjacent_pairs']}  "
          f"concordance {m['rank_concordance']:.3f}  range {m['elo_range']:.0f} Elo")
    print("\ntiers: " + ", ".join(f"{t['tier']}={t['id']}({t['elo']:.0f})"
                                 for t in result["tiers"]))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
