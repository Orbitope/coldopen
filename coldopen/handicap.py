"""Make one strong agent play badly, in the ways people play badly.

Every ladder in this project so far is built by *undertraining*: catch the
network at points during self-play and let the early snapshots be the weak
players. That is convenient and it may be wrong, because an undertrained network
is uniformly bad. Its value function is noisy everywhere, so it errs at roughly
the same rate in every kind of position. People are not like that. A weak human
is close to expert at the patterns they have drilled and hopeless elsewhere, and
their mistakes cluster rather than spread.

If that difference matters, a classifier trained on an undertrained ladder has
learned a one-dimensional notion of badness that human play will not match. E2
exists to find out, by building a second ladder the other way - take the
strongest checkpoint and degrade it - and comparing the two at *matched measured
Elo*, so that "equally strong" is established by the league rather than assumed.

Three handicaps, chosen because they fail differently:

**Epsilon.** With some probability, throw the move away and play at random. This
is the naive degradation and it is included as the control that should look
least human: it is known from computer chess that a strong engine plus random
blunders feels nothing like a player of the resulting rating, because the errors
are uncorrelated with the position.

**Temperature.** Sample from a softmax over the action values instead of taking
the best. Errors stay *ordered* - a slightly worse move is much likelier than a
disastrous one - so mistakes concentrate where the position is close, which is
where people actually go wrong.

**Blindspot.** Hide part of the board from the network before it looks. This is
the only one of the three that models attention rather than decision noise: the
agent plays well on what it sees and misses what it does not, so its blunders
cluster by region and by threat type, which is the shape of a beginner missing a
diagonal.

All three wrap a network and expose ``act``, so the league, the profiler and the
classifier treat them as players and need no changes.
"""

import torch

from coldopen.nets import masked_q

NEG = -1e9


class Handicapped:
    """Base for policies that degrade a network in a specific way."""

    #: Set by subclasses; used for labelling ladders and results.
    kind = "none"

    def __init__(self, net, strength):
        self.net = net
        self.strength = float(strength)

    @property
    def label(self):
        return f"{self.kind}_{self.strength:g}"

    def values(self, obs, legal):
        with torch.no_grad():
            return masked_q(self.net, obs, legal)

    def act(self, obs, legal, epsilon, generator, device):
        raise NotImplementedError


class EpsilonHandicap(Handicapped):
    """Play the best move, except when throwing it away entirely.

    The control. Its errors are independent of the position, which is exactly
    what makes it the least plausible model of a weak player.
    """

    kind = "epsilon"

    def act(self, obs, legal, epsilon, generator, device):
        n, a = obs.shape[0], legal.shape[1]
        acts = self.values(obs, legal).argmax(dim=1)
        # The handicap and any exploration the caller wants are the same kind of
        # noise, so take whichever is larger rather than compounding them.
        rate = max(self.strength, epsilon)
        if rate > 0:
            slip = torch.rand(n, generator=generator).to(device) < rate
            if slip.any():
                noise = torch.rand(n, a, generator=generator).to(device)
                acts[slip] = noise.masked_fill(~legal, -1).argmax(dim=1)[slip]
        return acts


class TemperatureHandicap(Handicapped):
    """Sample from the action values, so near-ties become coin flips.

    At temperature zero this is greedy play; as it rises the policy starts
    preferring good moves only on average. Errors stay ordered by how bad they
    are, which is the property epsilon noise throws away.
    """

    kind = "temperature"

    def act(self, obs, legal, epsilon, generator, device):
        values = self.values(obs, legal)
        if self.strength <= 0:
            return values.argmax(dim=1)
        logits = (values / self.strength).masked_fill(~legal, NEG)
        probabilities = torch.softmax(logits, dim=1)
        # Sample on the CPU so the stream is reproducible on any device.
        picked = torch.multinomial(probabilities.cpu(), 1, generator=generator)
        return picked.squeeze(1).to(values.device)


class BlindspotHandicap(Handicapped):
    """Hide a band of the board before the network sees it.

    ``strength`` is the *fraction of the board* the player fails to attend to,
    always applied, at a position that moves from move to move. Parameterising
    it as how often the player is blind instead of how much they miss turned out
    not to degrade the agent at all - hiding two of seven columns some of the
    time still leaves enough of a Connect Four board to play well on. How much
    is unseen is the quantity that matters, and it is also the one that
    corresponds to attention.

    The agent then plays well on a position that is not the real one, which
    produces confident mistakes about the part it could not see - the closest of
    the three to how a beginner misses a threat.
    """

    kind = "blindspot"

    def act(self, obs, legal, epsilon, generator, device):
        n = obs.shape[0]
        width = obs.shape[-1]
        span = int(round(self.strength * width))
        if span <= 0:
            return self.values(obs, legal).argmax(dim=1)

        view = obs.clone()
        starts = torch.randint(0, width, (n,), generator=generator).tolist()
        for row in range(n):
            # Contiguous, and wrapping: a band is an unattended region, whereas
            # scattered missing cells would just be noise on the input.
            columns = [(starts[row] + k) % width for k in range(min(span, width))]
            view[row, ..., :, columns] = 0.0
        return self.values(view, legal).argmax(dim=1)


KINDS = {
    "epsilon": EpsilonHandicap,
    "temperature": TemperatureHandicap,
    "blindspot": BlindspotHandicap,
}


def make(kind, net, strength):
    if kind not in KINDS:
        raise KeyError(f"unknown handicap {kind!r}; have {sorted(KINDS)}")
    return KINDS[kind](net, strength)


def ladder(net, kind, strengths):
    """Players of graded strength from one network, weakest first.

    The ordering here is *intended*, not measured. Which of these rungs are
    actually far enough apart to be tiers is the league's job, exactly as it is
    for the undertrained ladder.
    """
    return [
        {"id": f"{kind}_{strength:g}", "plies": index, "strength": strength,
         "net": make(kind, net, strength)}
        for index, strength in enumerate(sorted(strengths, reverse=True))
    ]
