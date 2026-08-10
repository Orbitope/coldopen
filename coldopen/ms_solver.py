"""The deduction oracle: which hidden cells are provably safe, provably mines?

This is E4's yardstick. The headline judgement feature — *was a provably safe
cell available, and did the player click one?* — is decided by logic from the
rules, not by a model's opinion, which is why minesweeper was chosen: no
yardstick-independence argument is ever needed.

The oracle sees exactly what the player sees: revealed cells with their
counts, and the total mine count. Flags are ignored (a flag is a claim, not
information), and the true mine layout is never consulted — the oracle must
be honest enough to run on scraped human games where we know only what was on
screen.

Three layers, each subsuming the last:

1. **Unary fixpoint**: a revealed `n` with `n` hidden neighbours makes them
   all mines; with `n` already-proven mines among its neighbours, its other
   hidden neighbours are safe. Iterated to fixpoint.
2. **Pairwise subset**: constraints A ⊆ B with |B∖A| = c(B) − c(A) prove
   B∖A all mines; c(B) = c(A) proves B∖A all safe. (The 1-2 and 1-1 patterns.)
3. **Exact enumeration** on small frontier components (≤ `enum_limit` cells):
   split the frontier into independent components, enumerate assignments
   consistent with the constraints, and mark cells whose value is the same in
   every solution. This is complete for the component, so within components of
   that size "ambiguous" genuinely means *a guess was forced*.

Global mine-count reasoning across components is NOT implemented (it matters
mostly in endgames); `complete=False` on a verdict says enumeration was
skipped somewhere, so "no safe cell" is then a lower bound rather than a
proof. The telemetry layer records that distinction instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations


def neighbours(cell, H, W):
    r, c = divmod(cell, W)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                out.append(rr * W + cc)
    return out


@dataclass
class Verdict:
    """The oracle's reading of one position."""

    safe: set = field(default_factory=set)     # provably NOT mines
    mines: set = field(default_factory=set)    # provably mines
    ambiguous: set = field(default_factory=set)  # frontier cells, undecided
    interior: set = field(default_factory=set)   # hidden, no adjacent info
    complete: bool = True   # every frontier component fully enumerated?


def solve(revealed, counts, H, W, enum_limit=16):
    """Classify every hidden cell of a position.

    `revealed`: iterable of 0/1 per cell. `counts`: adjacent-mine count per
    cell (only read where revealed). Flags deliberately not an input.
    """
    K = H * W
    revealed = [int(v) for v in revealed]
    hidden = [i for i in range(K) if not revealed[i]]
    known_mine, known_safe = set(), set()

    def build_constraints():
        out = []
        for i in range(K):
            if not revealed[i]:
                continue
            cells = frozenset(j for j in neighbours(i, H, W)
                              if not revealed[j] and j not in known_mine
                              and j not in known_safe)
            need = int(counts[i]) - sum(1 for j in neighbours(i, H, W)
                                        if j in known_mine)
            if cells:
                out.append((cells, need))
        return out

    # --- layers 1 + 2 to fixpoint --------------------------------------------
    changed = True
    while changed:
        changed = False
        constraints = build_constraints()
        for cells, need in constraints:
            if need == 0:
                for j in cells:
                    if j not in known_safe:
                        known_safe.add(j); changed = True
            elif need == len(cells):
                for j in cells:
                    if j not in known_mine:
                        known_mine.add(j); changed = True
        if changed:
            continue
        for (a, na), (b, nb) in combinations(constraints, 2):
            small, ns, big, nb_ = (a, na, b, nb) if len(a) <= len(b) else (b, nb, a, na)
            if not small < big:
                continue
            rest = big - small
            if nb_ - ns == len(rest):
                for j in rest:
                    if j not in known_mine:
                        known_mine.add(j); changed = True
            elif nb_ == ns:
                for j in rest:
                    if j not in known_safe:
                        known_safe.add(j); changed = True

    # --- layer 3: exact enumeration per frontier component -------------------
    constraints = build_constraints()
    frontier = set().union(*[c for c, _ in constraints]) if constraints else set()
    complete = True

    # union-find the frontier into independent components
    parent = {j: j for j in frontier}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for cells, _ in constraints:
        cells = list(cells)
        for j in cells[1:]:
            parent[find(j)] = find(cells[0])
    components = {}
    for j in frontier:
        components.setdefault(find(j), []).append(j)

    for members in components.values():
        member_set = set(members)
        relevant = [(list(c), n) for c, n in constraints if c & member_set]
        if len(members) > enum_limit:
            complete = False
            continue
        order = sorted(members)
        index = {j: k for k, j in enumerate(order)}
        always_mine = [True] * len(order)
        always_safe = [True] * len(order)
        found = 0

        def consistent(assign):
            for cells, need in relevant:
                got = sum(assign[index[j]] for j in cells)
                if got != need:
                    return False
            return True

        for bits in range(1 << len(order)):
            assign = [(bits >> k) & 1 for k in range(len(order))]
            if not consistent(assign):
                continue
            found += 1
            for k, v in enumerate(assign):
                if v:
                    always_safe[k] = False
                else:
                    always_mine[k] = False
        if found:
            for k, j in enumerate(order):
                if always_mine[k]:
                    known_mine.add(j)
                elif always_safe[k]:
                    known_safe.add(j)

    verdict = Verdict(complete=complete)
    frontier_all = frontier | known_mine | known_safe
    for j in hidden:
        if j in known_safe:
            verdict.safe.add(j)
        elif j in known_mine:
            verdict.mines.add(j)
        elif j in frontier_all:
            verdict.ambiguous.add(j)
        else:
            verdict.interior.add(j)
    return verdict


def judge_click(verdict, cell, kind):
    """One clicked cell against the oracle's verdict. The E4 feature.

    Returns one label:
      'proven_safe'   — revealed a cell the oracle had proven safe
      'blunder'       — revealed a cell the oracle had proven a MINE
      'guess_avoidable' — revealed an unproven cell while a proven-safe one existed
      'guess_forced'  — revealed an unproven cell and no safe cell was proven
      'other'         — flags, chords, and clicks on revealed cells
    """
    if kind != 0:
        return "other"
    if cell in verdict.safe:
        return "proven_safe"
    if cell in verdict.mines:
        return "blunder"
    if verdict.safe:
        return "guess_avoidable"
    return "guess_forced"
