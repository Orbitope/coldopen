"""Leduc Hold'em telemetry: what the action meant given the hand actually held.

Leduc is the hidden-information axis, and it is where the premise of this
project should be hardest to satisfy. There are three actions, two betting
rounds and at most a couple of decisions per hand, so a single move carries very
little - which is exactly the case worth measuring.

The raw actions - raised, called, folded - are close to worthless on their own,
because with three actions every tier plays all of them. What separates good
from bad is the *pairing* of the action with the hand: raising a pair is value,
calling a pair is money left behind, folding a pair is a blunder, and raising
the worst card is a bluff, which is not a mistake at all. So the features are
mostly those conjunctions.

This is the one game where the extractor uses information the player does not
have: it reads both hole cards and the board. That is not leakage of the kind
this project worries about - a real poker site knows the cards too, and it is
the only way to tell a bluff from a value bet. The leakage that would matter is
scoring a player against a network that is itself one of the profiled agents,
which is handled in ``profile.py`` by keeping the reference independent.

Pgx encodes actions as 0 call, 1 raise, 2 fold; cards as ranks 0, 1, 2 with two
of each; and ``_cards`` as ``[player 0, player 1, public]``. The public card is
dealt at the start but only in play from round 1.
"""

import numpy as np
import torch

CALL, RAISE, FOLD = 0, 1, 2
TOP_RANK = 2


def _int(x, device):
    return torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.int64))).to(device)


class LeducFeatures:
    names = (
        "raised",
        "called",
        "folded",
        "has_pair",
        "raise_with_pair",
        "passive_with_pair",
        "fold_with_pair",
        "bluff_raise",
        "fold_to_raise",
        "chips_committed",
        "second_round",
    )

    def extract(self, game, state, actions, key):
        device = game.device
        cards = _int(state._cards, device)  # [B, 3]
        chips = _int(state._chips, device)  # [B, 2]
        rnd = _int(state._round, device)
        last = _int(state._last_action, device)
        mover = game.current_player(state)

        own = cards.gather(1, mover.view(-1, 1)).squeeze(1)
        public = cards[:, 2]
        # The public card exists from the deal but is only live in round 1, so a
        # pair does not count before then.
        has_pair = (rnd >= 1) & (own == public)
        weakest = own == 0

        raised = actions == RAISE
        called = actions == CALL
        folded = actions == FOLD
        facing_raise = last == RAISE

        own_chips = chips.gather(1, mover.view(-1, 1)).squeeze(1).float()

        return {
            "raised": raised.float(),
            "called": called.float(),
            "folded": folded.float(),
            "has_pair": has_pair.float(),
            # Betting the hand you should be betting.
            "raise_with_pair": (raised & has_pair).float(),
            # Holding the best hand and only calling with it.
            "passive_with_pair": (called & has_pair).float(),
            # Folding it outright, which is the outright blunder.
            "fold_with_pair": (folded & has_pair).float(),
            # Raising the worst card with no pair: a bluff, and a sign of
            # strength rather than weakness.
            "bluff_raise": (raised & weakest & ~has_pair).float(),
            "fold_to_raise": (folded & facing_raise).float(),
            "chips_committed": own_chips,
            "second_round": (rnd >= 1).float(),
        }


FEATURES = LeducFeatures()
