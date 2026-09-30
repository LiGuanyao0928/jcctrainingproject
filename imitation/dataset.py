"""Normalise TFT match data (Riot API JSON or hand-written records), filter it, summarise it,
and build training arrays.

Record schema (one per player per match):
  {match_id, puuid, source, set, placement 1-8, level, augments: [str],
   units: [{id, tier 1-4, items: [str]}], traits: [{name, num_units, tier}]}

What the API gives: end-of-game board, stars, items, augments, level, placement.
What it does NOT give: per-round actions, nor which augments were offered, so the augment data is a
popularity prior (chosen augment vs placement), not a choice-among-options dataset.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path("imitation_data")


# ---------- normalisation ----------
def _as_str_list(x, field):
    if x is None:
        return []
    if not isinstance(x, list) or not all(isinstance(i, str) for i in x):
        raise ValueError(f"{field} must be a list of strings")
    return x


def normalize_record(r, defaults=None):
    """Validate one record (hand-written or already normalised). Raises ValueError with a readable reason."""
    if not isinstance(r, dict):
        raise ValueError("record must be an object")
    r = {**(defaults or {}), **r}
    placement = r.get("placement")
    if not isinstance(placement, int) or isinstance(placement, bool) or not 1 <= placement <= 8:
        raise ValueError("placement must be an integer 1-8")
    units = r.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("units must be a non-empty list")
    clean_units = []
    for u in units:
        if isinstance(u, str):
            u = {"id": u}
        if not isinstance(u, dict) or not isinstance(u.get("id"), str) or not u["id"]:
            raise ValueError("each unit needs a string id")
        tier = u.get("tier", 1)
        if not isinstance(tier, int) or not 1 <= tier <= 4:
            raise ValueError(f"unit {u['id']}: tier must be 1-4")
        clean_units.append({"id": u["id"], "tier": tier, "items": _as_str_list(u.get("items"), "items")})
    traits = []
    for t in r.get("traits") or []:
        if not isinstance(t, dict) or not isinstance(t.get("name"), str):
            raise ValueError("each trait needs a string name")
        traits.append({"name": t["name"], "num_units": int(t.get("num_units", 0)), "tier": int(t.get("tier", 0))})
    level = r.get("level", len(clean_units))
    if not isinstance(level, int) or not 1 <= level <= 12:
        raise ValueError("level must be an integer 1-12")
    rec = {
        "match_id": str(r.get("match_id") or ""), "puuid": str(r.get("puuid") or ""),
        "source": str(r.get("source", "manual")), "set": r.get("set"),
        "placement": placement, "level": level,
        "augments": _as_str_list(r.get("augments"), "augments"),
        "units": clean_units, "traits": traits,
    }
    if not rec["match_id"]:
        body = json.dumps([rec["placement"], rec["augments"], clean_units], sort_keys=True)
        rec["match_id"] = "manual-" + hashlib.sha1(body.encode()).hexdigest()[:12]
    return rec


def normalize_riot_match(raw):
    info = raw.get("info") or {}
    match_id = (raw.get("metadata") or {}).get("match_id", "")
    out = []
    for p in info.get("participants", []):
        out.append(normalize_record({
            "match_id": match_id, "puuid": p.get("puuid", ""), "source": "riot",
            "set": info.get("tft_set_number"), "placement": p.get("placement"), "level": p.get("level"),
            "augments": p.get("augments", []),
            "units": [{"id": u.get("character_id"), "tier": u.get("tier", 1), "items": u.get("itemNames", [])}
                      for u in p.get("units", [])],
            "traits": [{"name": t.get("name"), "num_units": t.get("num_units", 0), "tier": t.get("tier_current", 0)}
                       for t in p.get("traits", [])],
        }))
    return out


def ingest(obj):
    """Accepts a Riot match dict, a list of those, a list of records, or {"records": [...]}.
    Returns (records, errors); a bad record is reported, never fatal."""
    records, errors = [], []
    items = obj.get("records") if isinstance(obj, dict) and "records" in obj else obj
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return [], ["expected a JSON object or list"]
    for n, it in enumerate(items):
        try:
            if isinstance(it, dict) and "info" in it:
                records += normalize_riot_match(it)
            else:
                records.append(normalize_record(it))
        except (ValueError, TypeError, KeyError) as e:
            errors.append(f"item {n}: {e}")
    return records, errors


class RecordStore:
    """Append-only JSONL store, de-duplicated on (match_id, puuid, placement)."""

    def __init__(self, path=DATA_DIR / "records.jsonl"):
        self.path = Path(path)

    @staticmethod
    def _key(r):
        return (r["match_id"], r["puuid"], r["placement"])

    def load(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def add(self, records):
        seen = {self._key(r) for r in self.load()}
        new = []
        for r in records:
            k = self._key(r)
            if k not in seen:
                seen.add(k)
                new.append(r)
        if new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                for r in new:
                    f.write(json.dumps(r) + "\n")
        return len(new), len(records) - len(new)

    def ingest_dir(self, directory):
        added = dup = 0
        errors = []
        for p in sorted(Path(directory).glob("*.json")):
            recs, errs = ingest(json.loads(p.read_text()))
            a, d = self.add(recs)
            added, dup = added + a, dup + d
            errors += [f"{p.name}: {e}" for e in errs]
        return added, dup, errors


# ---------- filtering & stats ----------
def filter_records(records, placement_max=8, units=(), traits=(), augments=(), min_level=0, set_number=None):
    """Substring match, case-insensitive; every given term must be present."""
    def has(terms, values):
        vals = [v.lower() for v in values]
        return all(any(t.lower() in v for v in vals) for t in terms)

    out = []
    for r in records:
        if r["placement"] > placement_max or r["level"] < min_level:
            continue
        if set_number is not None and r.get("set") != set_number:
            continue
        if not has(units, [u["id"] for u in r["units"]]):
            continue
        if not has(traits, [t["name"] for t in r["traits"] if t["tier"] > 0]):
            continue
        if not has(augments, r["augments"]):
            continue
        out.append(r)
    return out


def _table(groups, min_n, top):
    rows = []
    for name, places in groups.items():
        if len(places) >= min_n:
            rows.append({"name": name, "n": len(places), "avg_placement": round(sum(places) / len(places), 3),
                         "top4": round(sum(p <= 4 for p in places) / len(places), 3),
                         "win": round(sum(p == 1 for p in places) / len(places), 3)})
    rows.sort(key=lambda r: (r["avg_placement"], -r["n"]))
    return rows[:top]


def stats(records, min_n=3, top=15):
    units, augs, traits = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        for uid in {u["id"] for u in r["units"]}:
            units[uid].append(r["placement"])
        for a in set(r["augments"]):
            augs[a].append(r["placement"])
        for t in r["traits"]:
            if t["tier"] > 0:
                traits[f"{t['name']} {t['num_units']}"].append(r["placement"])
    n = len(records)
    return {
        "n_records": n, "n_matches": len({r["match_id"] for r in records}),
        "avg_placement": round(sum(r["placement"] for r in records) / n, 3) if n else None,
        "units": _table(units, min_n, top), "augments": _table(augs, min_n, top),
        "traits": _table(traits, min_n, top),
    }


def comp_signature(r, k=3):
    active = sorted((t for t in r["traits"] if t["tier"] > 0), key=lambda t: (-t["tier"], -t["num_units"], t["name"]))
    return " / ".join(f"{t['name']} {t['num_units']}" for t in active[:k]) or "(no traits)"


def strong_comps(records, top4_only=True, min_n=2, top=20):
    groups = defaultdict(list)
    for r in records:
        if not top4_only or r["placement"] <= 4:
            groups[comp_signature(r)].append(r["placement"])
    return _table(groups, min_n, top)


# ---------- training arrays ----------
def _vocab(values):
    return sorted(set(values))


def build_dataset(records, out_dir=DATA_DIR / "datasets", name=None, id_map=None, sim_names=None):
    """Write <name>.npz (+ .vocab.json, .meta.json) and return the meta dict.

    Arrays (N = records): unit_presence/unit_tier [N,U], item_counts [N,I], augments [N,A],
    trait_tier [N,T], level/placement/weight [N] where weight = (4.5 - placement) / 3.5.
    With id_map {riot_unit_id: sim champion name} and sim_names [list], also sim_units [N,S]
    so the data can be used against the simulator's own (replaceable) champion set.
    """
    if not records:
        raise ValueError("no records to build a dataset from")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or "dataset"
    vocab = {
        "units": _vocab(u["id"] for r in records for u in r["units"]),
        "items": _vocab(i for r in records for u in r["units"] for i in u["items"]),
        "augments": _vocab(a for r in records for a in r["augments"]),
        "traits": _vocab(t["name"] for r in records for t in r["traits"] if t["tier"] > 0),
    }
    idx = {k: {v: i for i, v in enumerate(vs)} for k, vs in vocab.items()}
    N = len(records)
    arr = {
        "unit_presence": np.zeros((N, len(vocab["units"])), np.float32),
        "unit_tier": np.zeros((N, len(vocab["units"])), np.float32),
        "item_counts": np.zeros((N, len(vocab["items"])), np.float32),
        "augments": np.zeros((N, len(vocab["augments"])), np.float32),
        "trait_tier": np.zeros((N, len(vocab["traits"])), np.float32),
        "level": np.array([r["level"] for r in records], np.float32),
        "placement": np.array([r["placement"] for r in records], np.int64),
    }
    arr["weight"] = ((4.5 - arr["placement"]) / 3.5).astype(np.float32)
    meta = {"name": name, "n_records": N, "n_matches": len({r["match_id"] for r in records}),
            "placement_mean": float(arr["placement"].mean())}
    if id_map and sim_names:
        sim_idx = {n: i for i, n in enumerate(sim_names)}
        arr["sim_units"] = np.zeros((N, len(sim_names)), np.float32)
        total = mapped = 0
    for n, r in enumerate(records):
        for u in r["units"]:
            j = idx["units"][u["id"]]
            arr["unit_presence"][n, j] += 1
            arr["unit_tier"][n, j] = max(arr["unit_tier"][n, j], u["tier"])
            for it in u["items"]:
                arr["item_counts"][n, idx["items"][it]] += 1
            if id_map and sim_names:
                total += 1
                sim = id_map.get(u["id"])
                if sim in sim_idx:
                    arr["sim_units"][n, sim_idx[sim]] = max(arr["sim_units"][n, sim_idx[sim]], u["tier"])
                    mapped += 1
        for a in r["augments"]:
            arr["augments"][n, idx["augments"][a]] = 1
        for t in r["traits"]:
            if t["tier"] > 0:
                arr["trait_tier"][n, idx["traits"][t["name"]]] = t["tier"]
    if id_map and sim_names:
        meta["sim_unit_coverage"] = round(mapped / total, 3) if total else 0.0
    np.savez_compressed(out_dir / f"{name}.npz", **arr)
    (out_dir / f"{name}.vocab.json").write_text(json.dumps(vocab, indent=1))
    meta["strong_comps"] = strong_comps(records)
    meta["files"] = [f"{name}.npz", f"{name}.vocab.json", f"{name}.meta.json"]
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main():
    import argparse
    ap = argparse.ArgumentParser(description="ingest raw Riot matches / build a dataset")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest")
    a.add_argument("directory", nargs="?", default="imitation_data/raw")
    b = sub.add_parser("build")
    b.add_argument("--placement-max", type=int, default=4)
    b.add_argument("--name", default="dataset")
    b.add_argument("--id-map", default=None, help="JSON {riot unit id: simulator champion name}")
    args = ap.parse_args()
    store = RecordStore()
    if args.cmd == "ingest":
        added, dup, errors = store.ingest_dir(args.directory)
        print(f"added {added}, duplicates {dup}, errors {len(errors)}")
        for e in errors[:10]:
            print(" ", e)
    else:
        from tft_sim.data import GameData
        recs = filter_records(store.load(), placement_max=args.placement_max)
        id_map = json.loads(Path(args.id_map).read_text()) if args.id_map else None
        meta = build_dataset(recs, name=args.name, id_map=id_map, sim_names=sorted(GameData().champions))
        print(json.dumps({k: v for k, v in meta.items() if k != "strong_comps"}, indent=1))


if __name__ == "__main__":
    main()
