"""Download high-elo TFT matches from the official Riot API (PC Teamfight Tactics only).

    python -m imitation.riot_fetch --platform na1 --tiers challenger,grandmaster --players 30 --matches-per-player 15

Endpoints: tft-league-v1 (top ladder players) -> tft-match-v1 (match ids, match detail).
Raw match JSON is saved under imitation_data/raw/ and re-runs skip matches already on disk.
The API key is read from $RIOT_API_KEY or a git-ignored .env file and is never printed or stored.
For local, non-commercial use only; respect Riot's developer terms and rate limits.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

RAW_DIR = Path("imitation_data/raw")
REGION_OF = {
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas", "oc1": "sea",
    "kr": "asia", "jp1": "asia",
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe",
    "sg2": "sea", "tw2": "sea", "vn2": "sea", "th2": "sea", "ph2": "sea",
}
TIERS = ("challenger", "grandmaster", "master")


class RiotError(RuntimeError):
    pass


def load_key(env_file=".env"):
    key = os.environ.get("RIOT_API_KEY")
    if not key and Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            line = line.strip()
            if line.startswith("RIOT_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise RiotError("RIOT_API_KEY not set (export it or put it in a git-ignored .env file)")
    return key


class RateLimiter:
    """Sliding-window limiter; defaults match a personal dev key (20/s and 100/2min)."""

    def __init__(self, limits=((20, 1.0), (100, 120.0)), clock=time.monotonic, sleep=time.sleep):
        self.limits, self.clock, self.sleep = limits, clock, sleep
        self.stamps = {w: deque() for _, w in limits}

    def wait(self):
        while True:
            now = self.clock()
            delay = 0.0
            for n, w in self.limits:
                q = self.stamps[w]
                while q and now - q[0] >= w:
                    q.popleft()
                if len(q) >= n:
                    delay = max(delay, w - (now - q[0]))
            if delay <= 0:
                break
            self.sleep(delay)
        now = self.clock()
        for _, w in self.limits:
            self.stamps[w].append(now)


def _urllib_get(url, headers):
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, None, dict(e.headers or {})


class RiotClient:
    def __init__(self, key, platform="na1", limiter=None, http_get=_urllib_get, sleep=time.sleep, retries=5):
        if platform not in REGION_OF:
            raise RiotError(f"unknown platform {platform!r}; choose from {sorted(REGION_OF)}")
        self.key, self.platform, self.region = key, platform, REGION_OF[platform]
        self.limiter = limiter or RateLimiter()
        self.http_get, self.sleep, self.retries = http_get, sleep, retries

    def _get(self, host, path):
        url = f"https://{host}.api.riotgames.com{path}"
        for _ in range(self.retries):
            self.limiter.wait()
            status, body, headers = self.http_get(url, {"X-Riot-Token": self.key})
            if status == 200:
                return body
            if status == 404:
                return None
            if status == 429:
                self.sleep(float({k.lower(): v for k, v in headers.items()}.get("retry-after", 10)) + 0.5)
                continue
            if status in (401, 403):
                raise RiotError(f"Riot API rejected the key (HTTP {status}); dev keys expire every 24h")
            if status >= 500:
                self.sleep(2)
                continue
            raise RiotError(f"unexpected HTTP {status} for {path}")
        raise RiotError(f"gave up after {self.retries} attempts: {path}")

    def league_entries(self, tier):
        body = self._get(self.platform, f"/tft/league/v1/{tier}") or {}
        return body.get("entries", [])

    def puuid_of(self, entry):
        if entry.get("puuid"):
            return entry["puuid"]
        body = self._get(self.platform, f"/tft/summoner/v1/summoners/{entry['summonerId']}")
        return body and body.get("puuid")

    def match_ids(self, puuid, count):
        return self._get(self.region, f"/tft/match/v1/matches/by-puuid/{puuid}/ids?start=0&count={count}") or []

    def match(self, match_id):
        return self._get(self.region, f"/tft/match/v1/matches/{match_id}")


def fetch(client, tiers, players, matches_per_player, raw_dir=RAW_DIR, log=print):
    """Returns the number of newly saved matches."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for tier in tiers:
        entries += sorted(client.league_entries(tier), key=lambda e: -e.get("leaguePoints", 0))
    new = 0
    for entry in entries[:players]:
        puuid = client.puuid_of(entry)
        if not puuid:
            continue
        for mid in client.match_ids(puuid, matches_per_player):
            path = raw_dir / f"{mid}.json"
            if path.exists():
                continue
            data = client.match(mid)
            if data:
                path.write_text(json.dumps(data))
                new += 1
        log(f"saved {new} new matches so far")
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--platform", default="na1", choices=sorted(REGION_OF))
    ap.add_argument("--tiers", default="challenger,grandmaster")
    ap.add_argument("--players", type=int, default=30)
    ap.add_argument("--matches-per-player", type=int, default=15)
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    args = ap.parse_args()
    tiers = [t.strip() for t in args.tiers.split(",")]
    bad = [t for t in tiers if t not in TIERS]
    if bad:
        ap.error(f"unknown tiers {bad}; choose from {TIERS}")
    client = RiotClient(load_key(), args.platform)
    n = fetch(client, tiers, args.players, args.matches_per_player, args.raw_dir)
    print(f"done: {n} new matches in {args.raw_dir}")


if __name__ == "__main__":
    main()
