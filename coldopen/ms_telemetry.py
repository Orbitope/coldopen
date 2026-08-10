"""Minesweeper telemetry: the E4 feature set, extracted from played episodes.

Two families, deliberately:

* **Leaderboard-comparable** — the numbers a human record carries: win, time,
  3BV, 3BV/s, efficiency (3BV per click). These come from the game itself.
* **Judgement** — the numbers only the oracle can produce: how often a click
  was provably safe, an avoidable guess, a forced guess, or a proven-mine
  blunder. This is the axis E6 showed sprint never had: it is about *reading
  the board*, not about speed, and it is computable for humans and agents
  alike from nothing but what was on screen.

3BV (Bechtel's Board Benchmark Value): the minimum number of left clicks to
clear the board — one per opening (a connected region of zero-cells plus its
numbered border opens with a single click) plus one per remaining safe cell.
Efficiency is 3BV / clicks-used; over 100% is possible for humans via chords.

The oracle is consulted BEFORE each click on the position the player saw, so
its verdict is exactly the information that was available at decision time.
Solver calls are per-step Python; telemetry runs use small batches and that
is fine — measurement is not training.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "envs"))

from coldopen.ms_solver import judge_click, neighbours, solve
from minesweeper.fast import MinesweeperBatched

#: Feature names, families kept adjacent for the export layer.
FEATURES = [
    # leaderboard-comparable
    "won", "time_s", "three_bv", "three_bv_per_s", "efficiency",
    "clicks", "safe_revealed", "progress",
    # judgement (oracle)
    "proven_safe_rate", "guess_avoidable_rate", "guess_forced_rate",
    "blunder_rate", "flag_rate", "chord_rate", "oracle_complete",
]


def three_bv(mines, H, W):
    """Bechtel's Board Benchmark Value of a layout."""
    mine_set = set(int(j) for j in mines)
    K = H * W
    counts = [sum(1 for j in neighbours(i, H, W) if j in mine_set)
              for i in range(K)]
    zero = [i for i in range(K) if i not in mine_set and counts[i] == 0]
    seen, openings = set(), 0
    for start in zero:
        if start in seen:
            continue
        openings += 1
        stack = [start]
        seen.add(start)
        while stack:
            i = stack.pop()
            for j in neighbours(i, H, W):
                if j in mine_set or j in seen:
                    continue
                seen.add(j)              # the numbered border joins the opening
                if counts[j] == 0:
                    stack.append(j)
    isolated = sum(1 for i in range(K)
                   if i not in mine_set and i not in seen)
    return openings + isolated


def play_episodes(policy, episodes=16, board=(9, 9, 10), latency=200,
                  seed=11, judge=True, max_steps=None):
    """Roll `policy` and extract one telemetry row per finished episode."""
    H, W, M = board
    K = H * W
    n = min(episodes, 16)
    env = MinesweeperBatched(n, H=H, W=W, M=M, latency=latency,
                            emit_final_states=False)
    env.reset(torch.arange(n, dtype=torch.int64) + seed)
    max_steps = max_steps or 6 * K

    live = [dict(clicks=0, labels=[], flags=0, chords=0, incomplete=False)
            for _ in range(n)]
    rows = []
    obs = env.observe()
    for _ in range(max_steps):
        acts = policy(obs, env)
        kinds = (acts // K).tolist()
        cells = (acts % K).tolist()

        if judge:
            counts_all = env._counts()
            for i in range(n):
                if kinds[i] != 0 or not bool(env.generated[i]):
                    continue          # oracle judges reveals on live boards
                verdict = solve(env.revealed[i].tolist(),
                                counts_all[i].tolist(), H, W)
                live[i]["labels"].append(judge_click(verdict, cells[i], 0))
                live[i]["incomplete"] |= not verdict.complete

        before_time = env.time_ms.clone()
        before_mines = env.mines.clone()
        before_safe = (env.revealed & ~env.mines).sum(dim=1)
        obs, reward, term, _ = env.step(acts)
        for i in range(n):
            live[i]["clicks"] += 1
            if kinds[i] == 1:
                live[i]["flags"] += 1
            if kinds[i] == 2:
                live[i]["chords"] += 1
            if bool(term[i]):
                rows.append(_finish(live[i], bool(reward[i] > 500_000),
                                    int(before_time[i]) + 230,
                                    before_mines[i], int(before_safe[i]),
                                    H, W, M))
                live[i] = dict(clicks=0, labels=[], flags=0, chords=0,
                               incomplete=False)
        if len(rows) >= episodes:
            break
    return rows[:episodes]


def _finish(state, won, time_ms, mines, safe_revealed, H, W, M):
    mine_cells = torch.nonzero(mines).flatten().tolist()
    bv = three_bv(mine_cells, H, W)
    labels = state["labels"]
    reveals = max(len(labels), 1)
    time_s = time_ms / 1000.0
    total_safe = H * W - M
    row = {
        "won": int(won),
        "time_s": round(time_s, 2),
        "three_bv": bv,
        "three_bv_per_s": round(bv / time_s, 3) if won else None,
        "efficiency": round(bv / max(state["clicks"], 1), 3) if won else None,
        "clicks": state["clicks"],
        "safe_revealed": safe_revealed,
        "progress": round(safe_revealed / total_safe, 3),
        "proven_safe_rate": round(
            labels.count("proven_safe") / reveals, 3),
        "guess_avoidable_rate": round(
            labels.count("guess_avoidable") / reveals, 3),
        "guess_forced_rate": round(
            labels.count("guess_forced") / reveals, 3),
        "blunder_rate": round(labels.count("blunder") / reveals, 3),
        "flag_rate": round(state["flags"] / max(state["clicks"], 1), 3),
        "chord_rate": round(state["chords"] / max(state["clicks"], 1), 3),
        "oracle_complete": int(not state["incomplete"]),
    }
    return row


def summarise(rows):
    """Mean of each feature over episodes; None-aware for win-only fields."""
    out = {"episodes": len(rows)}
    for name in FEATURES:
        values = [r[name] for r in rows if r.get(name) is not None]
        out[name] = round(float(np.mean(values)), 3) if values else None
    return out
