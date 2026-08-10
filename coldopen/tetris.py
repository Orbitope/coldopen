"""Sprint telemetry: what a 40 LINES run says about the player who ran it.

The features here are chosen to line up with what TETR.IO's public API
reports for a human 40L record — `inputs`, `piecesplaced`, `holds`, the
clears breakdown, `topbtb`, `pps`, `time` — so the agent ladder and the human
records can be compared column for column. Two caveats travel with that
comparison and are repeated wherever the numbers go:

* our input counting is not TETR.IO's (spec: fidelity note 6), so absolute
  finesse is not comparable — standardise within population first;
* agent PPS is downstream of the LATENCY dial, so it is a chosen quantity,
  not an emergent one.

The extractor watches a batched env from the outside. Locks are not state
(the spec keeps state minimal), so they are *detected*: a queue shift means a
piece was consumed, and a consumed piece is a lock unless the action was the
hold that swapped it in. Terminal steps need care because a finishing or
topping-out lock does not spawn — those corrections are in `_terminal_locks`.
The rare double-lock step (an action lock followed by an instant auto-lock of
the next piece, needing a 19-high stack and a slow player) is counted as one
lock; accepted and documented rather than chased.
"""

from __future__ import annotations

import torch

HOLD_ACTION = 9
HARD_DROP = 8
DAS_ACTIONS = (2, 3)
SOFT_ACTION = 7

#: Feature order for the per-episode rows this module emits.
FEATURE_NAMES = [
    "inputs",
    "pieces",
    "inputs_per_piece",
    "holds",
    "holds_per_piece",
    "singles",
    "doubles",
    "triples",
    "quads",
    "quad_rate",
    "max_b2b",
    "lines",
    "time_ms",
    "pps",
    "finished",
    "pieces_per_line",
    "das_share",
    "soft_share",
]

#: The subset that has a direct counterpart in a TETR.IO 40L record.
HUMAN_COMPARABLE = [
    "inputs_per_piece",
    "quad_rate",
    "holds_per_piece",
    "max_b2b",
    "pps",
]


class SprintTelemetry:
    """Accumulates per-instance counters across steps; emits rows at episode end."""

    def __init__(self, env):
        self.env = env
        n, device = env.n, env.device
        zero = torch.zeros(n, dtype=torch.int64, device=device)
        self.steps = zero.clone()
        self.locks = zero.clone()
        self.holds = zero.clone()
        self.das = zero.clone()
        self.soft = zero.clone()
        self.clears = torch.zeros(n, 5, dtype=torch.int64, device=device)  # by size 0..4
        self.b2b_cur = zero.clone()
        self.b2b_max = zero.clone()
        self.lines_prev = env.lines.clone()
        self.queue_prev = env.queue.clone()
        self.hold_used_prev = env.hold_used.clone()
        self.rows: list[dict] = []

    def update(self, actions, terminated, info):
        """Call after every env.step with that step's inputs and outputs."""
        env = self.env
        a = actions.to(torch.int64)
        self.steps = self.steps + 1

        # A queue shift means a piece was consumed this step. Five identical
        # queue entries are impossible under a 7-bag, so no false positives.
        final = info.get("final_state_json", {})
        queue_now = env.queue.clone()
        for i in final:  # live tensors already hold the next episode
            queue_now[i] = torch.tensor(final[i]["queue"], device=env.device)
        consumed = (queue_now != self.queue_prev).any(dim=1)

        # spec: Hold sequence — a first hold consumes without locking.
        hold_taken = (a == HOLD_ACTION) & (env.hold_used == 1) | (
            (a == HOLD_ACTION) & terminated)
        locks_now = (consumed & (a != HOLD_ACTION)).long()
        locks_now = locks_now + self._terminal_locks(a, terminated, consumed, final)
        self.locks = self.locks + locks_now
        self.holds = self.holds + ((a == HOLD_ACTION) & (self.hold_used_prev == 0)).long()
        self.das = self.das + ((a == DAS_ACTIONS[0]) | (a == DAS_ACTIONS[1])).long()
        self.soft = self.soft + (a == SOFT_ACTION).long()

        # Clears from the lines delta; terminal lines come from the snapshot
        # because the live tensor is already reset.
        lines_now = env.lines.clone()
        time_now = env.time_ms.clone()
        for i in final:
            lines_now[i] = int(final[i]["lines"])
            time_now[i] = int(final[i]["time_ms"])
        delta = (lines_now - self.lines_prev).clamp(min=0, max=4)
        size_hit = torch.nn.functional.one_hot(delta, 5)
        self.clears = self.clears + size_hit

        # B2B: consecutive clearing locks of size 4.
        cleared = delta > 0
        quad = delta == 4
        self.b2b_cur = torch.where(quad, self.b2b_cur + 1,
                                   torch.where(cleared, torch.zeros_like(self.b2b_cur),
                                               self.b2b_cur))
        self.b2b_max = torch.maximum(self.b2b_max, self.b2b_cur)

        # Emit rows for finished episodes, then zero their counters.
        for i in sorted(final):
            self.rows.append(self._row(i, int(lines_now[i]), int(time_now[i])))
        done = terminated
        zero = torch.zeros_like(self.steps)
        for field in ("steps", "locks", "holds", "das", "soft", "b2b_cur", "b2b_max"):
            setattr(self, field, torch.where(done, zero, getattr(self, field)))
        self.clears = torch.where(done.unsqueeze(1), torch.zeros_like(self.clears),
                                  self.clears)
        self.lines_prev = env.lines.clone()   # already 0 for reset instances
        self.queue_prev = env.queue.clone()
        self.hold_used_prev = env.hold_used.clone()
        del hold_taken

    def _terminal_locks(self, a, terminated, consumed, final):
        """Locks on terminal steps that never spawned (T1 success, T3 lock-out).

        T2 block-out locks DID consume (the spawn draw happened) and are
        already counted; a hold-swap block-out neither consumed nor locked.
        """
        add = torch.zeros_like(self.locks)
        for i in final:
            if terminated[i] and not consumed[i] and int(a[i]) != HOLD_ACTION:
                add[i] = 1
        return add

    def _row(self, i, lines, time_ms):
        pieces = max(int(self.locks[i]), 1)
        total_clears = int(self.clears[i, 1:].sum())
        seconds = max(time_ms, 1) / 1000.0
        return {
            "inputs": int(self.steps[i]),
            "pieces": int(self.locks[i]),
            "inputs_per_piece": int(self.steps[i]) / pieces,
            "holds": int(self.holds[i]),
            "holds_per_piece": int(self.holds[i]) / pieces,
            "singles": int(self.clears[i, 1]),
            "doubles": int(self.clears[i, 2]),
            "triples": int(self.clears[i, 3]),
            "quads": int(self.clears[i, 4]),
            "quad_rate": int(self.clears[i, 4]) / max(total_clears, 1),
            "max_b2b": int(self.b2b_max[i]),
            "lines": lines,
            "time_ms": time_ms,
            "pps": int(self.locks[i]) / seconds,
            "finished": int(lines >= 40),
            "pieces_per_line": int(self.locks[i]) / max(lines, 1),
            "das_share": int(self.das[i]) / max(int(self.steps[i]), 1),
            "soft_share": int(self.soft[i]) / max(int(self.steps[i]), 1),
        }


def rollout(env, policy, total_episodes, seed=0, max_steps=2_000_000):
    """Collect telemetry rows by running `policy` until enough episodes end.

    `policy(obs, env) -> actions [N]`. The env must have been constructed with
    `emit_final_states=True` (the default) — terminal snapshots are how the
    extractor sees final lines and clock values.
    """
    env.reset(torch.arange(env.n, dtype=torch.int64) + seed * 100_003)
    telemetry = SprintTelemetry(env)
    obs = env.observe()
    steps = 0
    while len(telemetry.rows) < total_episodes and steps < max_steps:
        actions = policy(obs, env)
        obs, _, terminated, info = env.step(actions)
        telemetry.update(actions, terminated, info)
        steps += env.n
    return telemetry.rows[:total_episodes]
