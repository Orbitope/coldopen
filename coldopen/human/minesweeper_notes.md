# minesweeper.online ingestion: investigation, and why it stopped

## What the data would be worth

The all-time ranking pages carry exactly the two axes E1 needs, per game:

| Rank | Player | Time | NF | 3BV | 3BV/s | Eff | Date |
|---|---|---|---|---|---|---|---|
| 1 | *(name)* | 0.132 | NF | 5 | 38 | 71% | 16 March 2026 |

`3BV/s` is the speed axis and `Eff` the efficiency axis, with a leaderboard
position as the skill label — the same shape as the TETR.IO ingestion. Player
profiles live at `/player/<id>`.

Minesweeper matters to E1 specifically because it is **not** a real-time
versus game, so a relationship that holds there is a statement about skill
rather than about action games. It is also the best remaining candidate after
E6 showed sprint has no headroom above `pps` (ρ 0.93 from one observable).

## robots.txt: permissive

```
User-agent: *
Allow: /
Disallow: /chat
Disallow: /chat-history
Disallow: */password-reset/*
Disallow: */invoice/*
```

`/ranking`, `/best-players` and `/player/*` are all allowed, and no
`Crawl-delay` is declared. Checking this turned up a real bug in our client —
see `tests/test_robots.py`: Python's `urllib.robotparser` matches rules in file
order, so `Allow: /` won for every path and it reported `/chat` as crawlable.
Replaced with an RFC 9309 longest-match matcher.

## Why the scrape stopped anyway

**The data is not reachable over plain HTTP.** Every variant returns a
byte-identical 29,946-byte SPA shell containing the string "Loading data...":

* our identifying User-Agent, curl's default, and no User-Agent — same MD5;
* with and without a `connect.sid` session cookie — same MD5;
* with standard `Accept` / `Accept-Language` headers — same MD5;
* `?page=`, `?level=`, `/ranking/1` — same MD5.

There is no bot mitigation in front of it (plain nginx, no challenge headers),
and the JS bundle references no data API. In a real browser the DOM grows from
~30 KB to ~49 KB with the table present, while the only same-origin XHR is
`/authorize`. So the table is assembled client-side from something we could not
identify without deeper reverse engineering.

**And `/authorize` has a write side effect.** Each call returns:

```json
{"userId": 56457114, "isNewUser": true,
 "user": {"createdAt": "2026-08-10T05:23:21.316Z", ...}}
```

Consecutive calls returned `56457114` then `56457116` — **every request mints a
new anonymous user row**. Roughly four were created during this investigation.
That is a write against somebody else's production database, performed by
accident while diagnosing a read-only question.

## The judgement call

robots.txt permitting a path is not the whole of the question. Here:

* the operator serves this data only to their own front end;
* the one endpoint the front end calls **creates records**;
* any bulk collection would therefore either drive a real browser thousands of
  times, or pollute their user table, or both;
* a support enquiry is outstanding and unanswered.

"Allowed by robots.txt" and "not overwhelming the server" were the conditions
set for this work, and the second one is not really the binding constraint —
the binding constraint is that the only available route writes to their
database. So this stops here pending a reply from
`support@minesweeper.online`, or an official API.

## If it resumes

* Ask specifically for a read-only endpoint or a data dump; the site has a
  `/statistics` page, so aggregates may already exist in a serveable form.
* Any browser-driven collection should reuse **one** session rather than
  re-running the `/authorize` handshake per page.
* Pseudonymise at ingest with `client.pseudonymise` — the leaderboard shows
  display names, and nothing downstream needs them.
* Treat scraped display names as untrusted text. They are user-authored and
  must never be interpreted as instructions.
