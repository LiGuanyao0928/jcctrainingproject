import io
import json
import random

import numpy as np
import pytest

from imitation import dataset as ds
from imitation import riot_fetch as rf
from imitation.sample import fake_match
from webui.app import create_app


def matches(n=12, seed=0):
    rng = random.Random(seed)
    return [fake_match(rng, i) for i in range(n)]


# ---------- Riot client ----------
class FakeHttp:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url, headers):
        self.calls.append((url, headers))
        for frag, resp in self.routes:
            if frag in url:
                r = resp.pop(0) if isinstance(resp, list) else resp
                return r if isinstance(r, tuple) else (200, r, {})
        return 404, None, {}


def client(routes, **kw):
    return rf.RiotClient("KEY", "na1", limiter=rf.RateLimiter(limits=((1000, 1.0),)), http_get=FakeHttp(routes),
                         sleep=lambda s: None, **kw)


def test_rate_limiter_waits_when_window_full():
    t = [0.0]
    slept = []
    lim = rf.RateLimiter(limits=((2, 1.0),), clock=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
    for _ in range(3):
        lim.wait()
    assert len(slept) == 1 and 0.99 <= slept[0] <= 1.0


def test_fetch_saves_new_matches_and_skips_existing(tmp_path):
    m = matches(2)
    routes = [("/tft/league/v1/challenger", {"entries": [{"puuid": "P1", "leaguePoints": 900}, {"puuid": "P2", "leaguePoints": 100}]}),
              ("by-puuid/P1", (200, ["M1", "M2"], {})), ("matches/M1", m[0]), ("matches/M2", m[1])]
    c = client(routes)
    assert rf.fetch(c, ["challenger"], 1, 5, raw_dir=tmp_path, log=lambda *_: None) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["M1.json", "M2.json"]
    assert all(h["X-Riot-Token"] == "KEY" for _, h in c.http_get.calls)
    assert not any("P2" in u for u, _ in c.http_get.calls)  # only top player requested
    c2 = client([("/tft/league/v1/challenger", {"entries": [{"puuid": "P1"}]}), ("by-puuid/P1", (200, ["M1", "M2"], {}))])
    assert rf.fetch(c2, ["challenger"], 1, 5, raw_dir=tmp_path, log=lambda *_: None) == 0  # resume: nothing re-downloaded


def test_retry_after_429_and_key_rejected():
    slept = []
    c = client([("matches/M", [(429, None, {"Retry-After": "3"}), (200, {"ok": 1}, {})])])
    c.sleep = slept.append
    assert c.match("M") == {"ok": 1} and slept == [3.5]
    with pytest.raises(rf.RiotError, match="rejected the key"):
        client([("matches/M", (403, None, {}))]).match("M")


def test_summoner_id_fallback_and_key_loading(tmp_path, monkeypatch):
    c = client([("summoners/S1", {"puuid": "PX"})])
    assert c.puuid_of({"summonerId": "S1"}) == "PX"
    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('RIOT_API_KEY="RGAPI-abc"\n')
    assert rf.load_key(env) == "RGAPI-abc"
    with pytest.raises(rf.RiotError):
        rf.load_key(tmp_path / "missing")
    with pytest.raises(rf.RiotError):
        rf.RiotClient("k", "mars")


# ---------- normalisation / filtering / stats ----------
def test_normalize_riot_match_and_manual_records():
    recs, errs = ds.ingest(matches(1)[0])
    assert not errs and len(recs) == 8
    assert sorted(r["placement"] for r in recs) == list(range(1, 9))
    assert recs[0]["match_id"] == "SYN_00000" and recs[0]["source"] == "riot"
    manual, errs = ds.ingest([{"placement": 2, "units": ["A", {"id": "B", "tier": 2, "items": ["I"]}], "augments": ["X"]}])
    assert not errs and manual[0]["level"] == 2 and manual[0]["match_id"].startswith("manual-")
    assert manual[0]["units"][0] == {"id": "A", "tier": 1, "items": []}


def test_bad_records_reported_not_fatal():
    recs, errs = ds.ingest([{"placement": 9, "units": ["A"]}, {"placement": 1, "units": []},
                            {"placement": 1, "units": [{"id": "A", "tier": 7}]}, {"placement": 3, "units": ["A"]}, "junk"])
    assert len(recs) == 1 and len(errs) == 4
    assert ds.ingest(42)[1]


def test_store_dedupes_and_ingests_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i, m in enumerate(matches(3)):
        (raw / f"{i}.json").write_text(json.dumps(m))
    (raw / "bad.json").write_text(json.dumps([{"placement": 0}]))
    store = ds.RecordStore(tmp_path / "r.jsonl")
    added, dup, errors = store.ingest_dir(raw)
    assert (added, dup, len(errors)) == (24, 0, 1)
    assert store.ingest_dir(raw)[:2] == (0, 24)
    assert len(store.load()) == 24


def test_filters_and_stats():
    recs = [r for m in matches(20) for r in ds.ingest(m)[0]]
    top = ds.filter_records(recs, placement_max=2)
    assert top and all(r["placement"] <= 2 for r in top)
    one = ds.filter_records(recs, units=["unit3"], augments=["aug1"])
    assert all(any("unit3" in u["id"].lower() for u in r["units"]) and any("aug1" in a.lower() for a in r["augments"]) for r in one)
    assert ds.filter_records(recs, units=["nope"]) == []
    s = ds.stats(recs, min_n=5)
    assert s["n_records"] == 160 and s["n_matches"] == 20 and abs(s["avg_placement"] - 4.5) < 1e-9
    avgs = [r["avg_placement"] for r in s["units"]]
    assert avgs == sorted(avgs) and all(r["n"] >= 5 for r in s["units"])
    assert ds.strong_comps(recs, min_n=1)


def test_build_dataset_arrays(tmp_path):
    recs = [r for m in matches(10) for r in ds.ingest(m)[0]]
    meta = ds.build_dataset(recs, out_dir=tmp_path, name="d", id_map={"SYN_Unit1": "Aria", "SYN_Unit2": "Brute"},
                            sim_names=["Aria", "Brute", "Cinder"])
    z = np.load(tmp_path / "d.npz")
    vocab = json.loads((tmp_path / "d.vocab.json").read_text())
    n = len(recs)
    assert z["unit_presence"].shape == (n, len(vocab["units"])) and z["augments"].shape == (n, len(vocab["augments"]))
    assert (z["augments"].sum(1) == 3).all()
    assert np.allclose(z["weight"], (4.5 - z["placement"]) / 3.5)
    assert z["sim_units"].shape == (n, 3) and 0 < meta["sim_unit_coverage"] < 1
    assert z["unit_presence"].sum(1).min() >= 6
    with pytest.raises(ValueError):
        ds.build_dataset([], out_dir=tmp_path)


# ---------- web UI ----------
@pytest.fixture
def cl(tmp_path):
    return create_app(tmp_path).test_client()


def upload(cl, payload, name="m.json"):
    return cl.post("/api/import", data={"files": (io.BytesIO(json.dumps(payload).encode()), name)},
                   content_type="multipart/form-data")


def test_web_import_browse_build_download(cl):
    assert cl.get("/").status_code == 200 and b"tft-sim" in cl.get("/").data
    r = upload(cl, matches(6)).json
    assert r["added"] == 48 and r["n_errors"] == 0
    assert upload(cl, matches(6)).json["duplicates"] == 48
    assert cl.get("/api/summary").json["n_matches"] == 6
    m = cl.get("/api/matches?placement_max=1&limit=3").json
    assert m["total"] == 6 and len(m["rows"]) == 3 and all(x["placement"] == 1 for x in m["rows"])
    assert cl.get("/api/stats?placement_max=4&min_n=1").json["units"]
    b = cl.post("/api/build", json={"name": "top4", "placement_max": 4})
    assert b.status_code == 200 and b.json["n_records"] == 24
    assert "top4.npz" in cl.get("/api/datasets").json
    d = cl.get("/api/download/top4.npz")
    assert d.status_code == 200 and d.data[:2] == b"PK"


def test_web_validation_and_paste_and_traversal(cl):
    assert cl.post("/api/import", data={"text": "{not json"}).json["n_errors"] == 1
    assert cl.post("/api/import", data={"text": json.dumps([{"placement": 1, "units": ["A"]}])}).json["added"] == 1
    assert upload(cl, [{"placement": 12, "units": ["A"]}]).json["n_errors"] == 1
    assert cl.post("/api/build", json={"name": "../evil"}).status_code == 400
    assert cl.post("/api/build", json={"name": "ok", "unit": "zzz"}).status_code == 400
    assert cl.get("/api/matches?placement_max=abc").status_code == 400
    for bad in ("..%2Frecords.jsonl", "records.jsonl", "x.npz"):
        assert cl.get(f"/api/download/{bad}").status_code == 404
