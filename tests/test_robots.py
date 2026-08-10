"""robots.txt must be enforced correctly, not approximately.

Written because the stdlib got this wrong on a real policy we were about to
crawl. `urllib.robotparser` matches rules in file order and returns the first
hit, so against

    User-agent: *
    Allow: /
    Disallow: /chat

it reports `/chat` as crawlable. An enforcement layer that says yes to a
forbidden path is worse than none, because it gets believed.

RFC 9309 section 2.2.2: longest matching pattern wins, `Allow` breaks ties.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coldopen.human.client import Robots

#: The live policy at the time of writing. Kept verbatim so a change on their
#: side shows up as a test failure rather than as silent over-crawling.
MINESWEEPER = """User-agent: *
Allow: /
Disallow: /chat
Disallow: /chat-history
Disallow: */password-reset/*
Disallow: */invoice/*
Sitemap: https://minesweeper.online/sitemap.xml
"""

AGENT = "coldopen-research/0.1"


@pytest.mark.parametrize("path,allowed", [
    ("/", True),
    ("/player/12345", True),
    ("/ranking", True),
    ("/sitemap.xml", True),
    # The case the stdlib gets wrong: a broad Allow listed before a specific
    # Disallow must not win, because it is shorter.
    ("/chat", False),
    ("/chat-history", False),
    ("/chatroom/5", False),
    ("/en/invoice/9", False),
    ("/en/password-reset/token", False),
])
def test_minesweeper_policy(path, allowed):
    assert Robots(MINESWEEPER).allowed(AGENT, path) is allowed


def test_longest_match_wins_not_first_match():
    """The precedence rule itself, isolated from any real site."""
    robots = Robots("User-agent: *\nAllow: /\nDisallow: /private\n")
    assert robots.allowed(AGENT, "/public") is True
    assert robots.allowed(AGENT, "/private") is False
    # Order must not matter: same rules, reversed.
    reversed_order = Robots("User-agent: *\nDisallow: /private\nAllow: /\n")
    assert reversed_order.allowed(AGENT, "/private") is False


def test_allow_beats_disallow_on_an_exact_tie():
    robots = Robots("User-agent: *\nDisallow: /a\nAllow: /a\n")
    assert robots.allowed(AGENT, "/a") is True


def test_more_specific_allow_overrides_broader_disallow():
    robots = Robots("User-agent: *\nDisallow: /api\nAllow: /api/public\n")
    assert robots.allowed(AGENT, "/api/private") is False
    assert robots.allowed(AGENT, "/api/public/thing") is True


def test_wildcards_and_end_anchor():
    robots = Robots("User-agent: *\nDisallow: /*.pdf$\n")
    assert robots.allowed(AGENT, "/docs/manual.pdf") is False
    assert robots.allowed(AGENT, "/docs/manual.pdf.html") is True


def test_named_agent_group_beats_the_wildcard_group():
    robots = Robots(
        "User-agent: *\nDisallow: /\n\n"
        "User-agent: coldopen-research\nAllow: /\nDisallow: /secret\n")
    assert robots.allowed(AGENT, "/anything") is True
    assert robots.allowed(AGENT, "/secret") is False
    assert robots.allowed("SomeOtherBot", "/anything") is False


def test_empty_disallow_means_everything_is_allowed():
    robots = Robots("User-agent: *\nDisallow:\n")
    assert robots.allowed(AGENT, "/anything") is True


def test_default_is_permissive_when_there_are_no_rules():
    assert Robots("").allowed(AGENT, "/anything") is True


def test_client_refuses_a_disallowed_path_without_fetching(monkeypatch):
    """The guard has to sit in front of the network, not beside it."""
    from coldopen.human import client as mod

    c = mod.PoliteClient("https://example.test", cache_dir="/tmp/robots_test",
                         min_interval=0.0)
    c._robots = Robots(MINESWEEPER)
    c._robots_loaded = True

    def explode(url):
        raise AssertionError(f"network was touched for a disallowed URL: {url}")

    monkeypatch.setattr(c, "_fetch", explode)
    with pytest.raises(mod.Disallowed):
        c.get_json("/chat")
    assert c.stats["disallowed"] == 1
