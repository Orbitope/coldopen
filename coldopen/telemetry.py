"""Turn played games into behavioural features, the way a game client would.

The premise of the last section of the article is that how somebody plays gives
away how good they are long before their win/loss record does. To test it we
need features an actual game could log: properties of the moves themselves, not
of the network that produced them.

Two families, kept separate on purpose:

**Tactical.** Exact, objective facts about each move, computed from the board
alone - did you have a win available and decline it, did you fail to block one,
did you hand the opponent a win, did you create a threat. These depend on
nothing but the rules, so no amount of information about the agents can leak
into them.

**Reference-relative.** How the move compares to a *separately trained* network's
opinion: does it match the reference's choice, and how much value does the
reference think it gave up. Scoring against an engine is standard practice, but
the reference has to be independent of the agents being profiled. If the same
network were both the strongest tier and the yardstick, that tier would score a
perfect match by construction and the classifier would be reading its own
answer key. ``coldopen/reference`` is trained from a different seed for exactly this
reason.
"""

import torch

from coldopen.c4 import COLS, BatchedC4, blocking_moves, winning_moves
from coldopen.net import masked_q

FEATURE_NAMES = [
    "missed_win",
    "missed_block",
    "gave_win",
    "made_threat",
    "took_win",
    "center_distance",
    "ref_agreement",
    "ref_regret",
]

TACTICAL_FEATURES = FEATURE_NAMES[:6]


def _epsilon_actions(net, obs, legal, epsilon, generator, device):
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


def move_features(env, actions, reference=None):
    """Per-move features for the side to move, evaluated before the move lands.

    Returns a dict of [B] tensors. Callers mask out slots that are finished or
    where the profiled player is not the one moving.
    """
    wins = winning_moves(env)
    blocks = blocking_moves(env)
    chose = torch.nn.functional.one_hot(actions, COLS).bool()

    had_win = wins.any(dim=1)
    took_win = (wins & chose).any(dim=1)
    had_threat_against = blocks.any(dim=1)
    blocked = (blocks & chose).any(dim=1)

    feats = {
        "missed_win": (had_win & ~took_win).float(),
        # Failing to block only counts when you had no winning move of your own:
        # taking the win instead of blocking is correct play, not a blunder.
        "missed_block": (had_threat_against & ~blocked & ~took_win & ~had_win).float(),
        "took_win": (had_win & took_win).float(),
        "center_distance": (actions - COLS // 2).abs().float(),
    }

    if reference is not None:
        with torch.no_grad():
            q = masked_q(reference, env.observe(), env.legal_mask())
        best = q.max(dim=1).values
        taken = q.gather(1, actions.unsqueeze(1)).squeeze(1)
        feats["ref_agreement"] = (q.argmax(dim=1) == actions).float()
        feats["ref_regret"] = (best - taken).clamp(min=0.0)
    else:
        feats["ref_agreement"] = torch.zeros_like(feats["missed_win"])
        feats["ref_regret"] = torch.zeros_like(feats["missed_win"])

    # The remaining two need the position *after* the move, so look ahead on a
    # copy rather than disturbing the live game.
    state = env.clone_state()
    env.step(actions)
    # Now the opponent is to move: their winning moves are ones we just handed
    # them, and our own threats are what we just created.
    feats["gave_win"] = winning_moves(env).any(dim=1).float()
    feats["made_threat"] = blocking_moves(env).any(dim=1).float()
    env.load_state(state)
    return feats


def collect_games(agent_net, opponent_net, games, max_moves=20, device="cpu",
                  seed=0, epsilon=0.05, reference=None, opening_plies=1):
    """Play ``games`` and return the profiled agent's per-move features.

    ``agent_net`` is the player under observation; it takes seat 0 in half the
    games and seat 1 in the other half so nothing keys off move parity. Returns
    ``(features, counts)`` where features is [games, max_moves, F] and counts is
    how many of the agent's moves were actually recorded per game.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    env = BatchedC4(games, device)
    agent_seat = (torch.arange(games, device=device) % 2).long()

    for _ in range(opening_plies):
        legal = env.legal_mask()
        noise = torch.rand(games, COLS, device=device, generator=gen)
        env.step(noise.masked_fill(~legal, -1).argmax(dim=1))

    n_feat = len(FEATURE_NAMES)
    out = torch.zeros(games, max_moves, n_feat, device=device)
    counts = torch.zeros(games, dtype=torch.long, device=device)

    while not env.done.all():
        legal, obs = env.legal_mask(), env.observe()
        live = ~env.done
        acts = torch.zeros(games, dtype=torch.long, device=device)
        a_turn = live & (env.to_move == agent_seat)
        o_turn = live & ~a_turn
        if a_turn.any():
            acts[a_turn] = _epsilon_actions(
                agent_net, obs[a_turn], legal[a_turn], epsilon, gen, device)
        if o_turn.any():
            acts[o_turn] = _epsilon_actions(
                opponent_net, obs[o_turn], legal[o_turn], epsilon, gen, device)

        record = a_turn & (counts < max_moves)
        if record.any():
            feats = move_features(env, acts, reference)
            slot = counts.clamp(max=max_moves - 1)
            stacked = torch.stack([feats[name] for name in FEATURE_NAMES], dim=1)
            idx = torch.nonzero(record, as_tuple=True)[0]
            out[idx, slot[idx]] = stacked[idx]
            counts[idx] += 1
        env.step(acts)

    return out, counts


def aggregate(features, counts, n_moves):
    """Mean of each feature over however many of a player's first ``n_moves``
    moves the game actually contained.

    Returns ``(X, valid)``: a [games, F] design matrix and a mask of games with
    at least one recorded move.

    Averaging over the moves available, rather than dropping games too short to
    fill the window, is what keeps an accuracy-versus-N curve interpretable. The
    alternative silently changes the sample at every N - long games are the ones
    between evenly matched players - so accuracy moves for two reasons at once
    and the curve stops answering "how many moves do we need?". It is also what
    a deployed system would do: score what you have.
    """
    n_moves = min(n_moves, features.shape[1])
    window = features[:, :n_moves, :]
    take = counts.clamp(max=n_moves)
    steps = torch.arange(n_moves, device=features.device)
    mask = (steps.unsqueeze(0) < take.unsqueeze(1)).float().unsqueeze(-1)
    total = (window * mask).sum(dim=1)
    return total / take.clamp(min=1).unsqueeze(1).float(), counts >= 1
