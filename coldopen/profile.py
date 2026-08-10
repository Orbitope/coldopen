"""Watch one player for N moves and write down what they did.

This is ``telemetry.py`` generalised to all four games. The per-game feature
sets live in ``coldopen/features``; what is here is the part that is the same
everywhere: sit a player down against an opponent, record their own moves as
they make them, and stop at N.

The unit of observation is a **session**, not a game. Connect Four made those
the same thing, because twenty moves is most of a game. Leduc is two decisions
per hand, so twenty moves is a dozen hands, and a per-game unit would cap the
telemetry at almost nothing. A session therefore runs until the player has made
N moves, restarting the game as often as that takes - which is also what a game
client would do, since it observes an *account*, not a match.

Two features are added here rather than per game: agreement with a reference
network, and the value that network thinks the move gave up. Scoring against an
engine is standard, but the reference must be independent of the agents being
profiled. If the same network were both the strongest tier and the yardstick,
that tier would score a perfect match by construction and the classifier would
be reading its own answer key, so the reference for each game is trained from a
different seed.
"""

import jax
import torch

from coldopen import features as feature_registry
from coldopen.nets import epsilon_actions, masked_q


def _reference_features(reference, obs, legal, actions):
    if reference is None:
        zeros = torch.zeros(obs.shape[0], device=obs.device)
        return {"ref_agreement": zeros, "ref_regret": zeros.clone()}
    with torch.no_grad():
        q = masked_q(reference, obs, legal)
    best = q.max(dim=1).values
    taken = q.gather(1, actions.unsqueeze(1)).squeeze(1)
    return {
        "ref_agreement": (q.argmax(dim=1) == actions).float(),
        "ref_regret": (best - taken).clamp(min=0.0),
    }


def collect_sessions(game, agent_net, opponent_net, n_sessions, max_moves=20,
                     seed=0, epsilon=0.05, reference=None, opening_plies=None,
                     max_iterations=None):
    """Record the profiled agent's first ``max_moves`` moves in each session.

    The agent takes the first seat in half the sessions and the second in the
    other half, so nothing can key off move parity. Returns
    ``(features, counts)`` with features [n_sessions, max_moves, F] laid out in
    the order of ``features.names(game.info.key)``, and counts giving how many
    of the agent's moves each session actually got.
    """
    info = game.info
    device = game.device
    extractor = feature_registry.for_game(info.key)
    names = feature_registry.names(info.key)
    if opening_plies is None:
        # Two deterministic policies replay one another exactly; the games with
        # chance in them deal their own variety and need no help.
        opening_plies = 0 if info.chance else 1
    if max_iterations is None:
        max_iterations = max_moves * 8 + info.max_plies

    gen = torch.Generator().manual_seed(seed)
    state = game.init(n_sessions, seed=seed)
    key = jax.random.PRNGKey(seed + 11)
    agent_seat = (torch.arange(n_sessions, device=device) % 2).long()

    for _ in range(opening_plies):
        acts = epsilon_actions(
            None, game.observe(state), game.legal_mask(state), 0.0, gen, device)
        key, sub = jax.random.split(key)
        state = game.step(state, acts, sub)

    out = torch.zeros(n_sessions, max_moves, len(names), device=device)
    counts = torch.zeros(n_sessions, dtype=torch.long, device=device)

    for _ in range(max_iterations):
        if bool((counts >= max_moves).all()):
            break
        obs, legal = game.observe(state), game.legal_mask(state)
        a_turn = game.current_player(state) == agent_seat
        acts = torch.zeros(n_sessions, dtype=torch.long, device=device)
        if a_turn.any():
            acts[a_turn] = epsilon_actions(
                agent_net, obs[a_turn], legal[a_turn], epsilon, gen, device)
        if (~a_turn).any():
            acts[~a_turn] = epsilon_actions(
                opponent_net, obs[~a_turn], legal[~a_turn], epsilon, gen, device)

        record = a_turn & (counts < max_moves)
        if record.any():
            key, sub = jax.random.split(key)
            feats = extractor.extract(game, state, acts, sub)
            feats.update(_reference_features(reference, obs, legal, acts))
            stacked = torch.stack([feats[name] for name in names], dim=1)
            idx = torch.nonzero(record, as_tuple=True)[0]
            out[idx, counts[idx]] = stacked[idx]
            counts[idx] += 1

        key, sub = jax.random.split(key)
        state = game.step(state, acts, sub)
        key, sub = jax.random.split(key)
        # A session outlives the games inside it: keep dealing until the player
        # has been watched for long enough.
        state = game.restart(state, game.finished(state), sub)

    return out, counts


def aggregate(features, counts, n_moves):
    """Mean of each feature over however many of the first ``n_moves`` exist.

    Returns ``(X, valid)``: an [n, F] design matrix and a mask of sessions with
    at least one recorded move.

    Averaging over the moves available, rather than dropping sessions too short
    to fill the window, is what keeps an accuracy-versus-N curve interpretable.
    The alternative silently changes the sample at every N, so accuracy would
    move for two reasons at once and the curve would stop answering "how many
    moves do we need?". It is also what a deployed system would do: score what
    you have.
    """
    n_moves = min(n_moves, features.shape[1])
    window = features[:, :n_moves, :]
    take = counts.clamp(max=n_moves)
    steps = torch.arange(n_moves, device=features.device)
    mask = (steps.unsqueeze(0) < take.unsqueeze(1)).float().unsqueeze(-1)
    total = (window * mask).sum(dim=1)
    return total / take.clamp(min=1).unsqueeze(1).float(), counts >= 1
