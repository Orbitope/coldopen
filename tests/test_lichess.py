"""Lichess PGN parsing, where a sign error would be invisible and fatal.

Evaluations are always reported from White's point of view, so Black's mistakes
look like improvements unless the sign is flipped. Getting that backwards
produces an efficiency feature that ranks players upside down for half the
dataset while every summary statistic still looks reasonable.
"""

import pytest

from coldopen.human.lichess import game_rows, parse_eval, parse_game

GAME = """[Event "Rated Blitz game"]
[Site "https://lichess.org/test01"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "180+0"]

1. e4 { [%eval 0.2] [%clk 0:02:58] } 1... e5 { [%eval 0.2] [%clk 0:02:56] } 2. Nf3 { [%eval 0.3] [%clk 0:02:56] } 2... Nc6 { [%eval 0.3] [%clk 0:02:52] } 3. Bb5 { [%eval 0.3] [%clk 0:02:54] } 3... a6 { [%eval 0.4] [%clk 0:02:48] } 4. Ba4 { [%eval 0.3] [%clk 0:02:52] } 4... Nf6 { [%eval 0.3] [%clk 0:02:44] } 5. O-O { [%eval 0.2] [%clk 0:02:50] } 5... Be7 { [%eval 0.3] [%clk 0:02:40] } 6. Re1 { [%eval 0.3] [%clk 0:02:48] } 6... b5 { [%eval 0.3] [%clk 0:02:36] } 1-0
"""


def parsed():
    headers, moves = parse_game("\n" + GAME.split("[Event ", 1)[1])
    return headers, moves


def test_headers_and_move_annotations_are_read():
    headers, moves = parsed()
    assert headers["WhiteElo"] == "1500"
    assert headers["TimeControl"] == "180+0"
    annotated = [m for m in moves if m is not None]
    assert len(annotated) == 12
    assert annotated[0] == (pytest.approx(0.2), 178)


def test_both_players_get_a_row_with_their_own_rating():
    headers, moves = parsed()
    rows = game_rows(headers, moves, index=0)
    assert len(rows) == 2
    assert {row["skill"] for row in rows} == {1500.0, 1600.0}


def test_black_spent_twice_as_long_per_move():
    """White drops 2 seconds a move, Black 4. Speed is moves per minute."""
    headers, moves = parsed()
    rows = game_rows(headers, moves, index=0)
    white, black = rows[0], rows[1]
    assert white["speed"] == pytest.approx(30.0, rel=0.05)   # 60 / 2s
    assert black["speed"] == pytest.approx(15.0, rel=0.05)   # 60 / 4s


def test_a_black_blunder_is_charged_to_black_not_white():
    """The sign test. Evaluation rising is good for White and bad for Black."""
    text = GAME.replace(
        "6... b5 { [%eval 0.3] [%clk 0:02:36] }",
        "6... b5 { [%eval 4.3] [%clk 0:02:36] }")
    headers, moves = parse_game("\n" + text.split("[Event ", 1)[1])
    rows = game_rows(headers, moves, index=0)
    by_rating = {row["skill"]: row for row in rows}
    black = by_rating[1600.0]
    white = by_rating[1500.0]
    # Efficiency is negative centipawn loss, so the blunderer must be lower.
    assert black["efficiency"] < white["efficiency"]
    assert black["efficiency"] < -50


def test_a_white_blunder_is_charged_to_white():
    """The mirror of the test above, and it has to be set up more carefully.

    Dropping the evaluation on White's move and then letting the next move
    restore it charges *both* players: White for the blunder and Black for
    handing the advantage straight back. That is correct accounting and a
    useless test, so Black keeps the won position here and only White pays.
    """
    text = GAME.replace(
        "6. Re1 { [%eval 0.3] [%clk 0:02:48] }",
        "6. Re1 { [%eval -4.0] [%clk 0:02:48] }").replace(
        "6... b5 { [%eval 0.3] [%clk 0:02:36] }",
        "6... b5 { [%eval -4.2] [%clk 0:02:36] }")
    headers, moves = parse_game("\n" + text.split("[Event ", 1)[1])
    rows = game_rows(headers, moves, index=0)
    by_rating = {row["skill"]: row for row in rows}
    assert by_rating[1500.0]["efficiency"] < by_rating[1600.0]["efficiency"]
    assert by_rating[1500.0]["efficiency"] < -50


def test_giving_back_an_advantage_charges_both_players():
    """A blunder answered by a blunder is two mistakes, not one."""
    text = GAME.replace(
        "6. Re1 { [%eval 0.3] [%clk 0:02:48] }",
        "6. Re1 { [%eval -4.0] [%clk 0:02:48] }")
    headers, moves = parse_game("\n" + text.split("[Event ", 1)[1])
    rows = game_rows(headers, moves, index=0)
    assert all(row["efficiency"] < -50 for row in rows)


def test_games_without_annotations_are_skipped():
    text = GAME.replace(" { [%eval 0.2] [%clk 0:02:58] }", "")
    headers, moves = parse_game("\n" + text.split("[Event ", 1)[1])
    # Too few annotated moves to say anything about a player.
    assert game_rows(headers, [None] * len(moves), index=0) == []


def test_mate_scores_and_wild_evaluations_are_bounded():
    """One lost position must not dominate a player's mean."""
    assert parse_eval("#3") == pytest.approx(10.0)
    assert parse_eval("#-3") == pytest.approx(-10.0)
    assert parse_eval("57.2") == pytest.approx(8.0)
    assert parse_eval("-57.2") == pytest.approx(-8.0)
    assert parse_eval("nonsense") is None
