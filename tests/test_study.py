"""Human-study pipeline without images: assignment rules, the left/right mapping and the analysis on simulated raters."""
import collections, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "study"))
import analyze, make_raters  # noqa: E402
from server import Study  # noqa: E402

COMPS = {"4k_cfg16": 4096, "4k_esrgan": 4096, "4k_hires": 4096, "6k_cfg16": 6144}


def build(tmp: Path, n_raters=9, per=40, p_reg=None):
    pairs = [{"pair": f"{c}-{i}", "comparison": c, "size": s, "prompt_id": f"p{i:02d}", "prompt": f"prompt {i}", "seed": i,
              "reg": {"hash": f"r-{c}-{i}"}, "other": {"hash": f"o-{c}-{i}"}} for c, s in COMPS.items() for i in range(30)]
    (tmp / "manifest.json").write_text(json.dumps({"salt": "x", "comparisons": {c: {} for c in COMPS}, "pairs": pairs}))
    rng = random.Random(0); counts = collections.Counter(); raters = {}
    for k in range(n_raters):
        raters[f"tok{k}"] = {"label": f"name {k}", "pilot": False, "items": make_raters.assign(pairs, counts, per, 4, rng)}
    (tmp / "raters.json").write_text(json.dumps(raters))
    st = Study(tmp); by = {p["pair"]: p for p in pairs}
    for tok, r in raters.items():
        for i, it in enumerate(r["items"]):
            want = "reg" if rng.random() < (p_reg or {}).get(by[it["pair"]]["comparison"], 1.0) else "other"
            side = "left" if (want == "reg") != it["flip"] else "right"   # REG is on the left unless flipped
            shown = st.item(r["items"], i)
            assert shown["left"]["fit"].startswith("/img/" + ("o-" if it["flip"] else "r-"))
            assert st.answer(tok, {"idx": i, "overall": side, "detail": side, "ms": 1000, "panned": True})
    return pairs, raters, counts


def test_assignment_rules(tmp_path):
    pairs, raters, counts = build(tmp_path)
    by = {p["pair"]: p for p in pairs}
    for r in raters.values():
        main = [it for it in r["items"] if not it["repeat"]]; rep = [it for it in r["items"] if it["repeat"]]
        assert len(main) == 40 and len(rep) == 4 and len({it["pair"] for it in main}) == 40
        p4k = [by[it["pair"]]["prompt_id"] for it in main if by[it["pair"]]["size"] == 4096]
        assert len(p4k) == len(set(p4k))                       # no 4K prompt twice: the REG image never repeats
        first = {it["pair"]: (i, it["flip"]) for i, it in enumerate(r["items"]) if not it["repeat"]}
        for i, it in enumerate(r["items"]):
            if it["repeat"]:
                assert i > first[it["pair"]][0] and it["flip"] != first[it["pair"]][1]
    cov = [counts[p["pair"]] for p in pairs]
    assert max(cov) - min(cov) <= 2 and sum(cov) == 9 * 40    # near-even coverage around 3 ratings per pair (the 4K prompt rule costs a little)


def test_analysis_recovers_preference(tmp_path):
    build(tmp_path, p_reg={"4k_cfg16": 1.0, "4k_esrgan": 0.0, "4k_hires": 0.7, "6k_cfg16": 0.5})
    man, rows, repeats, info = analyze.load(tmp_path)
    assert all(x["rater"].startswith("R") for x in info) and len(rows) == 9 * 40
    table, extra = analyze.summarise(man, rows, repeats, nboot=500)
    s = {(t["comparison"], t["question"]): t for t in table}
    assert s[("4k_cfg16", "overall")]["score"] == 1.0 and s[("4k_esrgan", "detail")]["score"] == 0.0
    t = s[("4k_hires", "overall")]
    assert 0.55 < t["score"] < 0.85 and t["lo"] <= t["score"] <= t["hi"]
    assert abs(s[("6k_cfg16", "overall")]["score"] - 0.5) < 0.15
