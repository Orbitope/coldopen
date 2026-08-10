"""Behavioural features, one set per game.

The premise being tested is that *how* somebody plays gives away how good they
are long before their win/loss record does. To test it fairly the features have
to be things a real game client could log - properties of the moves themselves -
and they have to be specific enough to the game to be worth logging. "Distance
from the centre column" means something in Connect Four and nothing in poker.

Each game therefore gets its own feature set, and each set is built from that
game's own vocabulary of good and bad play:

    connect_four   did you take the win, did you block the threat, did you hand
                   one over - tactics that are exact and settled by the rules
    othello        corners, X-squares, the opponent's mobility, your frontier -
                   the positional ideas that separate a beginner from a club
                   player, in a game where almost no position contains a forced
                   win
    backgammon     blots left exposed, points made, hitting, over-stacking -
                   risk management under dice rather than calculation
    leduc_holdem   raising, folding, and specifically the pairing of those with
                   the hand actually held: value-betting, slow-playing, bluffing
                   and folding the best hand

Two features are common to every game and live in ``profile.py`` instead:
agreement with a separately trained reference network, and the value that
network thinks the move gave up. Those are the ones that need care - the
reference must not be one of the agents being profiled, or the top tier scores a
perfect match with itself and the classifier reads its own answer key.

Every extractor takes a batch of Pgx states and the actions about to be played
in them, and returns a dict of [B] float tensors. Pgx states are immutable, so
lookahead is just calling ``step`` and looking at what comes back; nothing here
can disturb the game in progress.

Note that these are computed by the *operator*, not the player. In Leduc that
means the extractor can see both hole cards, which is the only way to tell a
bluff from a value bet - and is exactly what a real poker site can see.
"""

from coldopen.features import backgammon, connect_four, leduc_holdem, othello

#: Features every game shares, appended by the profiler once a reference
#: network is available. Kept out of the per-game sets so that "tactical only"
#: means "needs nothing but the rules".
REFERENCE_FEATURES = ("ref_agreement", "ref_regret")

_SETS = {
    "connect_four": connect_four.FEATURES,
    "othello": othello.FEATURES,
    "backgammon": backgammon.FEATURES,
    "leduc_holdem": leduc_holdem.FEATURES,
}


def for_game(key):
    if key not in _SETS:
        raise KeyError(f"no feature set for {key!r}; have {sorted(_SETS)}")
    return _SETS[key]


def names(key, with_reference=True):
    """Full feature vector layout for a game, reference features last."""
    intrinsic = tuple(for_game(key).names)
    return intrinsic + REFERENCE_FEATURES if with_reference else intrinsic
