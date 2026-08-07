"""Chess players from the Lichess open database.

The third game in the E1 panel, and the cleanest instance of the schema so far.
Lichess publishes every rated game it has ever hosted as monthly PGN dumps -
no key, no scraping, no rate limit, explicitly for research - and since 2017 the
move text carries a clock reading per move. Roughly a tenth of games also carry
Stockfish evaluations, because a user asked for analysis.

That gives all three quantities from one file, and gives them cleanly:

    speed        moves per minute, from the clock deltas
    efficiency   negative mean centipawn loss - quality obtained per move, the
                 direct analogue of attack per piece
    skill        Elo, which is computed from results and from nothing else

Chess being turn-based is the point rather than a problem. If the speed and
efficiency axes behave the same way here as in a real-time block-stacking game
and a real-time strategy game, the claim is about skill; if they only work in
action games, that is a much narrower finding and worth knowing.

**One time control at a time.** Mixing bullet with classical would manufacture a
spurious result: within a person, playing faster costs accuracy, so a pooled
sample contains a strong speed/accuracy tradeoff that has nothing to do with
skill and would swamp the cross-person effect being measured. Every ingestion
here is filtered to a single time control.

Streaming rather than downloading: a monthly file is tens of gigabytes, but the
games are independent, so the reader stops as soon as it has enough. Collecting
tens of thousands of annotated games costs a few hundred megabytes of transfer.
"""

import argparse
import json
import pathlib
import re
import subprocess

from coldopen.human.client import USER_AGENT, pseudonymise

LIST_URL = "https://database.lichess.org/standard/list.txt"

HEADER = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
#: Move text: an optional move number, the move itself, then the annotation
#: comment carrying evaluation and clock.
MOVE = re.compile(
    r"(?:\d+\.+\s*)?(\S+)\s*\{([^}]*)\}"
)
EVAL = re.compile(r"%eval\s+(#?-?[\d.]+)")
CLOCK = re.compile(r"%clk\s+(\d+):(\d+):(\d+)")

#: Evaluations beyond this are already decisive, and the difference between +8
#: and +14 pawns is not a measure of anything a player did. Clipping stops one
#: blunder in a won position from dominating a player's mean.
EVAL_CLIP = 8.0
#: Mate scores become a large but bounded advantage rather than infinity.
MATE_VALUE = 10.0


def stream_games(url, max_bytes):
    """Yield raw PGN blocks from a remote .zst without downloading it whole."""
    curl = subprocess.Popen(
        ["curl", "-sL", "-A", USER_AGENT, url],
        stdout=subprocess.PIPE)
    unzip = subprocess.Popen(
        ["zstd", "-dc"], stdin=curl.stdout, stdout=subprocess.PIPE)
    curl.stdout.close()

    read, buffer = 0, b""
    try:
        while read < max_bytes:
            chunk = unzip.stdout.read(1 << 20)
            if not chunk:
                break
            read += len(chunk)
            buffer += chunk
            # A game ends where the next one's headers begin.
            parts = buffer.split(b"\n\n[Event ")
            buffer = parts.pop()
            for part in parts:
                yield part.decode("utf-8", "replace")
    finally:
        for process in (unzip, curl):
            try:
                process.kill()
            except OSError:
                pass


def parse_eval(text):
    if text.startswith("#"):
        sign = -1.0 if text[1:].startswith("-") else 1.0
        return sign * MATE_VALUE
    try:
        return max(-EVAL_CLIP, min(EVAL_CLIP, float(text)))
    except ValueError:
        return None


def parse_game(block):
    """(headers, moves) where each move is (evaluation, clock seconds)."""
    headers = dict(HEADER.findall(block))
    body = block.split("]\n\n", 1)
    if len(body) < 2:
        return headers, []
    moves = []
    for _, comment in MOVE.findall(body[1]):
        ev = EVAL.search(comment)
        ck = CLOCK.search(comment)
        if not ev or not ck:
            moves.append(None)
            continue
        hours, minutes, seconds = (int(g) for g in ck.groups())
        moves.append((parse_eval(ev.group(1)), hours * 3600 + minutes * 60 + seconds))
    return headers, moves


def game_rows(headers, moves, index):
    """One row per player in this game: their speed, efficiency and rating.

    Centipawn loss is measured from the mover's own point of view. Lichess
    reports evaluation from White's, so Black's loss is the negation - getting
    that backwards would produce a feature that ranks players upside down while
    looking entirely plausible.
    """
    control = headers.get("TimeControl", "")
    if "+" not in control:
        return []
    base, increment = (int(part) for part in control.split("+"))

    usable = [(i, m) for i, m in enumerate(moves) if m is not None]
    if len(usable) < 12:
        return []

    per_side = {0: {"loss": [], "time": []}, 1: {"loss": [], "time": []}}
    previous_clock = {0: base, 1: base}
    previous_eval = 0.0
    for ply, (evaluation, clock) in usable:
        side = ply % 2
        if evaluation is None:
            continue
        swing = evaluation - previous_eval
        # White wants the evaluation to go up, Black wants it to go down.
        loss = -swing if side == 0 else swing
        per_side[side]["loss"].append(max(0.0, loss) * 100.0)
        spent = previous_clock[side] - clock + increment
        if 0 <= spent <= base + increment:
            per_side[side]["time"].append(spent)
        previous_clock[side] = clock
        previous_eval = evaluation

    rows = []
    for side, key in ((0, "White"), (1, "Black")):
        data = per_side[side]
        if len(data["loss"]) < 6 or not data["time"]:
            continue
        seconds = sum(data["time"]) / len(data["time"])
        if seconds <= 0:
            continue
        try:
            elo = float(headers[f"{key}Elo"])
        except (KeyError, ValueError):
            continue
        centipawn_loss = sum(data["loss"]) / len(data["loss"])
        rows.append({
            "player": pseudonymise(headers.get(key, "?")),
            "group": headers.get("Site", str(index)),
            "opponent": pseudonymise(headers.get("Black" if side == 0 else "White", "?")),
            "index": index,
            "speed": 60.0 / seconds,          # moves per minute
            "efficiency": -centipawn_loss,     # higher is better, as elsewhere
            "skill": elo,
            "rank_index": int(min(9, max(0, (elo - 800) // 200))),
        })
    return rows


def ingest(url=None, time_control="180+0", target=20000, max_mb=600):
    """Rows for one time control, streaming until ``target`` games are parsed."""
    if url is None:
        listing = subprocess.run(
            ["curl", "-sL", "-A", USER_AGENT, LIST_URL],
            capture_output=True, text=True, check=True).stdout.split()
        url = listing[1]  # the most recent complete month
    print(f"streaming {url}\n  time control {time_control}, target {target} games",
          flush=True)

    rows, kept, seen = [], 0, 0
    for block in stream_games(url, max_mb * (1 << 20)):
        seen += 1
        headers, moves = parse_game(block)
        if headers.get("TimeControl") != time_control:
            continue
        if not any(m is not None for m in moves):
            continue
        made = game_rows(headers, moves, kept)
        if made:
            rows.extend(made)
            kept += 1
            if kept % 2500 == 0:
                print(f"  {kept} games, {len(rows)} player-rows "
                      f"(from {seen} scanned)", flush=True)
        if kept >= target:
            break
    return {"game": "lichess", "url": url, "time_control": time_control,
            "games": kept, "scanned": seen, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-control", default="180+0",
                    help="a single control; mixing them manufactures a tradeoff")
    ap.add_argument("--target", type=int, default=20000)
    ap.add_argument("--max-mb", type=int, default=600)
    ap.add_argument("--url", default=None)
    ap.add_argument("--out", default="data/human/lichess.json")
    args = ap.parse_args()

    result = ingest(url=args.url, time_control=args.time_control,
                    target=args.target, max_mb=args.max_mb)
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1))
    print(f"\n{result['games']} games -> {len(result['rows'])} player-rows "
          f"from {result['scanned']} scanned")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
