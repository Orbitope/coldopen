"""The common representation has to mean the same thing for every game.

E1 compares the *shape* of the skill/speed/efficiency relationship across games
that measure in incompatible units. That comparison is only valid if the
standardisation is doing what it claims, and if the observation budget really
truncates to the first N rather than the best N - a "first three rounds" number
that quietly picked the best three would be worthless and would look excellent.
"""

import numpy as np
import pytest

from coldopen.human.axes import per_player, standardise, to_percentile, tetrio_axes
from coldopen.human.client import pseudonymise


def rounds(player, values, skill):
    """Rounds for one player: values are (pps, apm) pairs."""
    return [
        {"player": player, "match": f"m{i}", "opponent": "opp", "round": i,
         "pps": pps, "apm": apm, "vsscore": apm * 2}
        for i, (pps, apm) in enumerate(values)
    ], {"player": player, "rank": "c", "rank_index": 3, "tr": skill}


def test_efficiency_is_attack_per_piece():
    """The derived axis, and the one that carries most of the signal."""
    rows, label = rounds("p1", [(2.0, 60.0)], skill=1000)
    out = tetrio_axes(rows, [label])
    assert len(out) == 1
    # 60 attack/min over 2 pieces/sec = 120 pieces/min -> 0.5 attack per piece.
    assert out[0]["efficiency"] == pytest.approx(0.5)
    assert out[0]["speed"] == pytest.approx(2.0)


def test_rounds_with_no_pieces_placed_are_dropped():
    """Every rate statistic is undefined in a round the player never started."""
    rows, label = rounds("p1", [(0.0, 0.0), (1.5, 30.0)], skill=1000)
    out = tetrio_axes(rows, [label])
    assert len(out) == 1
    assert out[0]["speed"] == pytest.approx(1.5)


def test_a_player_with_no_rating_is_dropped_rather_than_guessed():
    rows, _ = rounds("ghost", [(1.0, 10.0)], skill=1)
    assert tetrio_axes(rows, []) == []


def test_standardisation_makes_games_comparable():
    """Both axes become z-scores; the label becomes a within-game percentile."""
    rows = []
    for i, player in enumerate("abcdefgh"):
        made, label = rounds(player, [(1.0 + i, 10.0 * (i + 1))], skill=100 * i)
        rows.extend(tetrio_axes(made, [label]))

    out = standardise(rows)
    speeds = np.array([r["speed"] for r in out])
    assert speeds.mean() == pytest.approx(0.0, abs=1e-9)
    assert speeds.std() == pytest.approx(1.0, abs=1e-9)

    pcts = [r["skill_pct"] for r in out]
    assert min(pcts) == pytest.approx(0.0)
    assert max(pcts) == pytest.approx(1.0)
    # Percentile must preserve the rating ordering.
    assert pcts == sorted(pcts)


def test_standardisation_clips_a_single_absurd_round():
    """One outlier must not set the scale for everyone else."""
    rows = []
    for i, player in enumerate("abcdefgh"):
        made, label = rounds(player, [(1.0 + i * 0.1, 10.0)], skill=100 * i)
        rows.extend(tetrio_axes(made, [label]))
    made, label = rounds("z", [(500.0, 10.0)], skill=800)
    rows.extend(tetrio_axes(made, [label]))

    out = standardise(rows, clip=4.0)
    assert max(r["speed"] for r in out) <= 4.0


def test_budget_takes_the_first_rounds_not_the_best():
    """A cold-start number that silently picked a player's best games is a lie."""
    made, label = rounds("p1", [(1.0, 10.0), (9.0, 90.0), (9.0, 90.0)], skill=500)
    rows = standardise(tetrio_axes(made, [label]))

    first = per_player(rows, budget=1)[0]
    assert first["n"] == 1
    everything = per_player(rows, budget=None)[0]
    assert everything["n"] == 3
    # The first round is the slow one, so a budget of 1 must be worse-looking.
    assert first["speed"] < everything["speed"]


def test_percentile_shares_ties():
    out = to_percentile([5.0, 5.0, 1.0, 9.0])
    assert out[0] == out[1], "tied ratings must not get an arbitrary order"
    assert out[2] == pytest.approx(0.0)
    assert out[3] == pytest.approx(1.0)


def test_pseudonyms_are_stable_and_do_not_carry_the_name():
    assert pseudonymise("player-one") == pseudonymise("player-one")
    assert pseudonymise("player-one") != pseudonymise("player-two")
    assert "player-one" not in pseudonymise("player-one")
