"""Convert a Community Dragon TFT dump into a simulator "set dir" (champions/traits/items JSON + report).

    python -m tft_sim.season_import en_us.json --set 13 --out tft_sim/data/sets/set13

Input: https://raw.communitydragon.org/latest/cdragon/tft/en_us.json  (download it yourself if the sandbox
can't reach the host).  NOTE: this reads the schema as I understand it (top-level "items", "sets"/"setData";
champion {apiName,name,cost,traits,stats,ability.variables}; trait {apiName,effects[{minUnits,variables}]};
item {apiName,composition,effects}). It has only been tested against a hand-written fixture in that shape,
not a real dump, so expect to adjust field names on first contact with real data.

Only generic mechanics are translated (see IMPORT_REPORT.md in the output for what was and wasn't):
  champions  hp/attack/armor/MR/attack speed/range/mana/crit + one generic skill (damage|shield|heal)
  traits     breakpoints whose effects are plain stat bonuses (AD, AS, armor, MR, AP, HP, mana, crit)
  items      components + two-component completed items with plain stat effects
Unit ids are Riot's apiName (e.g. TFT13_Jinx), the same ids the match API returns, so imported match data
lines up with the simulator without an id map.
"""
import argparse
import json
import re
from pathlib import Path

SKIP_NAME = re.compile(r"dummy|minion|summon|target|tower|turret|voidspawn", re.I)

# variable/effect name -> (sim stat key, kind); kind "pct" = fraction of base, "flat" = absolute
STAT_RULES = [
    (re.compile(r"^(bonus)?(ad|attackdamage)$", re.I), "atk_pct"),
    (re.compile(r"^(bonus)?(as|attackspeed)$", re.I), "aspd_pct"),
    (re.compile(r"^(bonus)?(ap|abilitypower)$", re.I), "ap"),
    (re.compile(r"^(bonus)?armor$", re.I), "armor"),
    (re.compile(r"^(bonus)?(mr|magicresist(ance)?)$", re.I), "mr"),
    (re.compile(r"^(bonus)?(hp|health)$", re.I), "hp"),
    (re.compile(r"^(bonus)?crit(ical)?(strike)?(chance)?$", re.I), "crit_chance"),
    (re.compile(r"^(bonus)?(starting|initial)?mana$", re.I), "mana_start"),
]


def map_stat(name, value):
    """-> (key, value) or None when the variable isn't a plain stat we model."""
    if not isinstance(value, (int, float)) or value == 0:
        return None
    for rx, key in STAT_RULES:
        if rx.match(name):
            if key == "hp":  # fractions are % of base hp, bigger numbers are flat hp
                return ("hp_pct", value) if abs(value) <= 2 else ("hp", value)
            if key == "atk_pct" and abs(value) > 3:  # flat attack damage
                return ("atk", value)
            if key == "aspd_pct" and abs(value) > 3:
                return None
            return key, value
    return None


def pick_set(data, number=None):
    """Return (set_number, champions, traits)."""
    sets = {}
    if isinstance(data.get("sets"), dict):
        for k, v in data["sets"].items():
            sets[int(k)] = v
    for v in data.get("setData", []) or []:
        if isinstance(v.get("number"), int) and v.get("champions"):
            sets.setdefault(v["number"], v)
    if not sets:
        raise ValueError("no sets found (expected top-level 'sets' or 'setData')")
    number = number or max(sets)
    if number not in sets:
        raise ValueError(f"set {number} not in dump; available: {sorted(sets)}")
    s = sets[number]
    return number, s.get("champions", []), s.get("traits", [])


def _first_star(values):
    """CDragon ability arrays are per star with a dummy slot 0; take the 1-star value."""
    if isinstance(values, list):
        nz = [v for v in values if isinstance(v, (int, float))]
        return nz[1] if len(nz) > 1 and nz[0] == 0 else (nz[0] if nz else 0)
    return values if isinstance(values, (int, float)) else 0


def convert_skill(ability, atk, report, who):
    vars_ = {v.get("name", ""): _first_star(v.get("value")) for v in (ability or {}).get("variables", [])}
    best = None
    for kind, rx, dtype in (("damage", re.compile(r"(magic|physical|true)?damage", re.I), None),
                            ("shield", re.compile(r"shield", re.I), None),
                            ("heal", re.compile(r"heal", re.I), None)):
        cands = [(n, v) for n, v in vars_.items() if rx.search(n) and v and "ratio" not in n.lower()
                 and "duration" not in n.lower() and "percent" not in n.lower()]
        if cands:
            n, v = max(cands, key=lambda x: x[1])
            dtype = "physical" if re.search("physical", n, re.I) else "true" if re.search("true", n, re.I) else "magic"
            best = (kind, v, dtype)
            break
    if not best:
        report["skills_default"].append(who)
        return {"type": "damage", "value": round(atk * 2.5, 1), "damage_type": "magic"}
    kind, v, dtype = best
    if v < 10:  # looks like a ratio of AD/AP, not an absolute amount
        v = v * atk
        report["skills_ratio_guess"].append(who)
    return {"type": kind, "value": round(float(v), 1), "damage_type": dtype}


def convert_champions(raw, set_no, report):
    out = []
    prefix = f"TFT{set_no}_"
    for c in raw:
        api, cost, st = c.get("apiName", ""), c.get("cost"), c.get("stats") or {}
        if not api.startswith(prefix) or SKIP_NAME.search(api) or cost not in (1, 2, 3, 4, 5) \
                or not c.get("traits") or not st.get("hp"):
            report["champions_skipped"].append(api or c.get("name", "?"))
            continue
        atk = st.get("damage", 0) or 0
        out.append({
            "name": api, "cost": cost, "traits": list(c["traits"]), "hp": st["hp"], "atk": atk,
            "armor": st.get("armor", 0), "mr": st.get("magicResist", st.get("armor", 0)),
            "aspd": st.get("attackSpeed", 0.7), "range": max(1, int(st.get("range", 1))),
            "mana": int(st.get("mana") or 100), "mana_start": int(st.get("initialMana") or 0),
            "crit_chance": st.get("critChance", 0.25), "crit_mult": st.get("critMultiplier", 1.4),
            "skill": convert_skill(c.get("ability"), atk, report, api),
            "display_name": c.get("name", api)})
    return out


def convert_traits(raw, champions, report):
    used = {t for c in champions for t in c["traits"]}
    out = {}
    for t in raw:
        name = t.get("name") or t.get("apiName")
        if name not in used:
            continue
        bps = {}
        for e in t.get("effects", []):
            eff, unmapped = {}, []
            for vn, vv in (e.get("variables") or {}).items():
                m = map_stat(vn, vv)
                if m:
                    eff[m[0]] = eff.get(m[0], 0) + m[1]
                elif isinstance(vv, (int, float)) and vv:
                    unmapped.append(vn)
            minu = e.get("minUnits")
            if minu:
                bps[str(minu)] = eff
                if unmapped:
                    report["trait_unmapped"].setdefault(name, set()).update(unmapped)
                if not eff:
                    report["trait_no_effect"].append(f"{name} {minu}")
        if bps:
            out[name] = bps
    for c in champions:  # traits with no JSON entry still need to exist so champions validate
        for tn in c["traits"]:
            out.setdefault(tn, {})
    return out


def convert_items(raw, report):
    by_api = {i.get("apiName"): i for i in raw if i.get("apiName")}
    parts = {p for i in raw for p in (i.get("composition") or [])}
    comps, completed = {}, {}

    def stats_of(item):
        eff = {}
        for vn, vv in (item.get("effects") or {}).items():
            m = map_stat(vn, vv)
            if m:
                eff[m[0]] = eff.get(m[0], 0) + m[1]
        return eff

    for api in sorted(parts):
        it = by_api.get(api)
        if not it or it.get("composition"):
            continue
        s = stats_of(it)
        if s:
            comps[api] = s
        else:
            report["items_skipped"].append(api)
    for api, it in by_api.items():
        comp = it.get("composition") or []
        if len(comp) == 2 and all(p in comps for p in comp):
            s = stats_of(it)
            if s:
                completed[api] = {"from": comp, "stats": s}
            else:
                report["items_special"].append(api)
    return {"components": comps, "completed": completed}


def write_report(path, set_no, champs, traits, items, report):
    lines = [f"# Import report: set {set_no}", "",
             f"- champions imported: {len(champs)} (skipped {len(report['champions_skipped'])})",
             f"- traits: {len(traits)}; breakpoints with no modelled effect: {len(report['trait_no_effect'])}",
             f"- item components: {len(items['components'])}; completed items with stat effects: {len(items['completed'])};"
             f" completed items with only special effects (not modelled): {len(report['items_special'])}",
             f"- skills: {len(champs) - len(report['skills_default'])} mapped to damage/shield/heal, "
             f"{len(report['skills_default'])} fell back to a generic damage skill, "
             f"{len(report['skills_ratio_guess'])} used an AD/AP-ratio guess", "",
             "## Not translated (need code or manual curation)",
             "- unique champion abilities (summons, displacement, multi-hit, conditional effects); every champion gets one generic skill",
             "- trait effects that aren't plain stats; augments (augments.json is left at the generic default set)",
             "- season rules (pool sizes, shop odds, XP table, level cap): rules.json is left at the defaults, check it against the patch",
             "- round schedule / carousel / PvE rounds are fixed in tft_sim/game.py", ""]
    for title, key in (("Champions skipped", "champions_skipped"), ("Skills using the default fallback", "skills_default"),
                       ("Trait breakpoints without modelled effect", "trait_no_effect"),
                       ("Items with special effects only", "items_special")):
        vals = report[key]
        if vals:
            lines += [f"## {title}", *[f"- {v}" for v in sorted(vals)[:200]], ""]
    if report["trait_unmapped"]:
        lines += ["## Unmapped trait variables", *[f"- {t}: {', '.join(sorted(v))}" for t, v in sorted(report["trait_unmapped"].items())], ""]
    Path(path).write_text("\n".join(lines))


def convert(dump, set_no=None, out_dir="tft_sim/data/sets/new"):
    set_no, raw_champs, raw_traits = pick_set(dump, set_no)
    report = {"champions_skipped": [], "skills_default": [], "skills_ratio_guess": [], "trait_no_effect": [],
              "trait_unmapped": {}, "items_skipped": [], "items_special": []}
    champs = convert_champions(raw_champs, set_no, report)
    if not champs:
        raise ValueError(f"no usable champions found for set {set_no}")
    traits = convert_traits(raw_traits, champs, report)
    items = convert_items(dump.get("items", []), report)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "champions.json").write_text(json.dumps(champs, indent=1, ensure_ascii=False))
    (out / "traits.json").write_text(json.dumps(traits, indent=1, ensure_ascii=False))
    (out / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    write_report(out / "IMPORT_REPORT.md", set_no, champs, traits, items, report)
    return set_no, champs, traits, items, report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump", help="path to Community Dragon en_us.json")
    ap.add_argument("--set", type=int, default=None, help="set number (default: newest in the dump)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dump = json.loads(Path(args.dump).read_text())
    no, champs, traits, items, _ = convert(dump, args.set, args.out or f"tft_sim/data/sets/set{args.set or 'new'}")
    print(f"set {no}: {len(champs)} champions, {len(traits)} traits, {len(items['components'])} components, "
          f"{len(items['completed'])} completed items -> see IMPORT_REPORT.md in the output dir")


if __name__ == "__main__":
    main()
