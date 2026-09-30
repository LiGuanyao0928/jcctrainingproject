"""Local import/browse page for TFT match data.   python -m webui.app   ->  http://127.0.0.1:5000
Bound to localhost only: it reads and writes files under imitation_data/ and has no authentication."""
import argparse
import json
import re
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from imitation import dataset as ds

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
FILE_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}\.(npz|vocab\.json|meta\.json)$")
MAX_ROWS = 200


def _filters(args):
    def terms(key):
        return [t.strip() for t in args.get(key, "").split(",") if t.strip()]
    try:
        return dict(placement_max=min(8, max(1, int(args.get("placement_max", 8)))),
                    units=terms("unit"), traits=terms("trait"), augments=terms("augment"),
                    min_level=int(args.get("min_level", 0) or 0))
    except ValueError:
        abort(400, "numeric filter expected")


def create_app(data_dir="imitation_data"):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
    root = Path(data_dir)
    store = ds.RecordStore(root / "records.jsonl")
    out_dir = root / "datasets"

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/favicon.ico")
    def favicon():
        return "", 204

    @app.get("/api/summary")
    def summary():
        recs = store.load()
        return jsonify({"n_records": len(recs), "n_matches": len({r["match_id"] for r in recs}),
                        "sources": sorted({r["source"] for r in recs})})

    @app.get("/api/stats")
    def stats():
        recs = ds.filter_records(store.load(), **_filters(request.args))
        out = ds.stats(recs, min_n=int(request.args.get("min_n", 3)))
        out["strong_comps"] = ds.strong_comps(recs, top4_only=False, min_n=int(request.args.get("min_n", 3)))
        return jsonify(out)

    @app.get("/api/matches")
    def matches():
        recs = ds.filter_records(store.load(), **_filters(request.args))
        recs.sort(key=lambda r: (r["match_id"], r["placement"]))
        limit = min(MAX_ROWS, max(1, int(request.args.get("limit", 50))))
        offset = max(0, int(request.args.get("offset", 0)))
        rows = [dict(r, comp=ds.comp_signature(r)) for r in recs[offset:offset + limit]]
        return jsonify({"total": len(recs), "rows": rows})

    @app.post("/api/import")
    def import_():
        added = dup = 0
        errors = []
        payloads = []
        for f in request.files.getlist("files"):
            try:
                payloads.append((f.filename, json.loads(f.read())))
            except (ValueError, UnicodeDecodeError):
                errors.append(f"{f.filename}: not valid JSON")
        if not request.files and request.is_json:
            payloads.append(("pasted JSON", request.get_json(silent=True)))
        elif not request.files and request.form.get("text"):
            try:
                payloads.append(("pasted JSON", json.loads(request.form["text"])))
            except ValueError:
                errors.append("pasted text: not valid JSON")
        for name, obj in payloads:
            recs, errs = ds.ingest(obj)
            a, d = store.add(recs)
            added, dup = added + a, dup + d
            errors += [f"{name}: {e}" for e in errs]
        return jsonify({"added": added, "duplicates": dup, "errors": errors[:50], "n_errors": len(errors)})

    @app.post("/api/build")
    def build():
        body = request.get_json(silent=True) or {}
        name = body.get("name") or "dataset"
        if not NAME_RE.match(name):
            abort(400, "name must be 1-40 characters of letters, digits, _ or -")
        recs = ds.filter_records(store.load(), **_filters({k: str(v) for k, v in body.items() if v is not None}))
        if not recs:
            abort(400, "no records match the current filters")
        id_map = None
        if (root / "id_map.json").exists():
            id_map = json.loads((root / "id_map.json").read_text())
        from tft_sim.data import GameData
        meta = ds.build_dataset(recs, out_dir=out_dir, name=name, id_map=id_map, sim_names=sorted(GameData().champions))
        return jsonify(meta)

    @app.get("/api/datasets")
    def datasets():
        files = sorted(p.name for p in out_dir.glob("*") if FILE_RE.match(p.name)) if out_dir.exists() else []
        return jsonify(files)

    @app.get("/api/download/<filename>")
    def download(filename):
        if not FILE_RE.match(filename) or not (out_dir / filename).exists():
            abort(404)
        return send_from_directory(out_dir.resolve(), filename, as_attachment=True)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--data-dir", default="imitation_data")
    args = ap.parse_args()
    create_app(args.data_dir).run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
