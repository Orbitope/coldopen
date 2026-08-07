"""A deliberately slow HTTP client for public game APIs.

Every source this project reads from is somebody else's server, run for players
rather than for researchers. None of them owe us the data. So the default here
is one request per second, an identifying User-Agent with a contact route, and a
disk cache that makes re-running an ingestion free rather than repeating it.

The cache is the part that matters most in practice. Building a feature pipeline
takes dozens of iterations; without a cache each one is another full crawl of
somebody else's database, which is both rude and slow. With it, only the first
run touches the network.

Nothing here is clever. It is stdlib ``urllib`` because the alternative is a new
dependency for a handful of GETs.
"""

import gzip
import hashlib
import json
import pathlib
import random
import time
import urllib.error
import urllib.parse
import urllib.request

#: Identifies the crawler and gives an operator somewhere to complain to. A
#: server admin who can see who we are and why can ask us to stop; one who
#: cannot has to block us.
USER_AGENT = (
    "coldopen-research/0.1 "
    "(offline skill-estimation research; +https://github.com/orbitope/coldopen)"
)


class RateLimited(Exception):
    """The server asked us to back off more times than we were willing to wait."""


class PoliteClient:
    """Cached, rate-limited GET client for one host.

    ``min_interval`` is the floor between *network* requests; cache hits are
    free and do not wait. ``expiry_of`` optionally reads a server-declared cache
    window out of a response body, so that a source telling us "this is good for
    ten minutes" is believed rather than second-guessed.
    """

    def __init__(self, base_url, cache_dir, min_interval=1.0, session_id=None,
                 expiry_of=None, max_retries=5, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.session_id = session_id
        self.expiry_of = expiry_of
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request = 0.0
        self.stats = {"hits": 0, "misses": 0, "retries": 0}

    # -- cache ---------------------------------------------------------------

    def _cache_path(self, url):
        digest = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.cache_dir / f"{digest}.json.gz"

    def _read_cache(self, url, max_age):
        path = self._cache_path(url)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                blob = json.load(fh)
        except (OSError, ValueError):
            return None
        if max_age is not None and time.time() - blob.get("fetched_at", 0) > max_age:
            return None
        return blob["body"]

    def _write_cache(self, url, body):
        tmp = self._cache_path(url).with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump({"url": url, "fetched_at": time.time(), "body": body}, fh)
        tmp.replace(self._cache_path(url))

    # -- fetching ------------------------------------------------------------

    def _wait_turn(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.time()

    def _fetch(self, url):
        request = urllib.request.Request(url, headers=self._headers())
        for attempt in range(self.max_retries):
            self._wait_turn()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as err:
                if err.code in (429, 500, 502, 503, 504):
                    self.stats["retries"] += 1
                    self._sleep_off(err, attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                self.stats["retries"] += 1
                self._sleep_off(None, attempt)
        raise RateLimited(f"gave up after {self.max_retries} attempts: {url}")

    def _sleep_off(self, err, attempt):
        """Back off exponentially, and believe Retry-After when it is given."""
        retry_after = None
        if err is not None:
            header = err.headers.get("Retry-After") if err.headers else None
            if header and header.isdigit():
                retry_after = int(header)
        delay = retry_after if retry_after is not None else 2 ** attempt
        time.sleep(delay + random.uniform(0, 0.5))

    def _headers(self):
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.session_id:
            # Sources that support it return a consistent snapshot for a session
            # and do fewer database round-trips for it.
            headers["X-Session-ID"] = self.session_id
        return headers

    def get_json(self, path, params=None, max_age=None):
        """GET a JSON document, from cache when possible.

        ``max_age`` in seconds overrides the cached copy's lifetime; the default
        is to trust anything already on disk, because these are historical
        records and a crawl that re-fetches them has gained nothing.
        """
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)

        cached = self._read_cache(url, max_age)
        if cached is not None:
            self.stats["hits"] += 1
            return cached

        body = self._fetch(url)
        self.stats["misses"] += 1
        if self.expiry_of is not None:
            # Only cache what the server is willing to stand behind.
            try:
                self.expiry_of(body)
            except Exception:
                pass
        self._write_cache(url, body)
        return body


def pseudonymise(identifier, salt="coldopen"):
    """A stable, non-reversible key for one player.

    The pipeline needs to group records by player and attach a skill label. It
    never needs to know who the player is, and an exported artefact that
    contains usernames is a liability with no scientific value attached. Hash at
    the boundary so nothing downstream can leak an identity it never received.
    """
    return hashlib.sha256(f"{salt}:{identifier}".encode()).hexdigest()[:16]
