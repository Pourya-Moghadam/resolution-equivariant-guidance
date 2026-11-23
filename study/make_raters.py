#!/usr/bin/env python
"""Human study, step 2: create raters and assign each a subset of the pairs.

Each rater gets --per-rater pairs, spread over the comparisons in proportion to their size, always taking the pairs with
the fewest assignments so far (random ties), so coverage is even and every pair reaches --min-ratings once enough raters
exist. A rater never sees the same 4K prompt in two comparisons (the REG image would repeat). --repeats of a rater's pairs
come back later with the sides swapped (self-consistency). Left/right and order are random per rater.
Running it again appends raters and keeps existing links valid.

  python study/make_raters.py --labels "Rater 1" "Rater 2" ... [--per-rater 60] [--repeats 4] [--url http://host:8347]
  python study/make_raters.py --n 10         (labels R01..R10)
Labels stay in raters.json under $REG_DATA and are never written to the repository; analysis output uses R01, R02, ...
"""
from __future__ import annotations

import argparse, collections, json, os, random, secrets
from pathlib import Path


def assign(pairs: list[dict], counts: collections.Counter, k: int, repeats: int, rng: random.Random) -> list[dict]:
    by = collections.defaultdict(list)
    for p in pairs:
        by[p["comparison"]].append(p)
    # quotas proportional to comparison size (largest remainders), then one global greedy pass over all pairs, fewest
    # assignments first: a pair is taken if its comparison still has quota and its 4K prompt is not used by this rater
    raw = {c: k * len(v) / len(pairs) for c, v in by.items()}
    quota = {c: int(x) for c, x in raw.items()}
    for c in sorted(raw, key=lambda c: raw[c] - quota[c], reverse=True)[:k - sum(quota.values())]:
        quota[c] += 1
    chosen, used4k, got = [], set(), collections.Counter()
    for p in sorted(pairs, key=lambda p: (counts[p["pair"]], rng.random())):
        c = p["comparison"]
        if got[c] >= quota[c] or (p["size"] == 4096 and p["prompt_id"] in used4k):
            continue
        chosen.append(p); got[c] += 1
        if p["size"] == 4096:
            used4k.add(p["prompt_id"])
    rng.shuffle(chosen)
    items = [{"pair": p["pair"], "flip": rng.random() < 0.5, "repeat": False} for p in chosen]
    half = len(items) // 2
    for it in rng.sample(items[:half], min(repeats, half)):
        pos = rng.randrange(max(half + 1, len(items) * 2 // 3), len(items) + 1)
        items.insert(pos, {"pair": it["pair"], "flip": not it["flip"], "repeat": True})
    for p in chosen:
        counts[p["pair"]] += 1
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data"))
    ap.add_argument("--dir", default=str(data / "human_study" / "v1"))
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--per-rater", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--pilot", action="store_true", help="mark these raters as pilot (excluded from the analysis)")
    ap.add_argument("--url", default="http://localhost:8347")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    d = Path(a.dir); pairs = json.loads((d / "manifest.json").read_text())["pairs"]
    rf = d / "raters.json"; raters = json.loads(rf.read_text()) if rf.exists() else {}
    counts = collections.Counter(it["pair"] for r in raters.values() if not r.get("pilot") for it in r["items"] if not it["repeat"])
    rng = random.Random(a.seed)
    labels = a.labels or [f"R{len(raters) + i + 1:02d}" for i in range(a.n)]
    for lab in labels:
        tok = secrets.token_urlsafe(9)
        items = assign(pairs, collections.Counter() if a.pilot else counts, min(a.per_rater, len(pairs)), a.repeats, rng)
        raters[tok] = {"label": lab, "pilot": a.pilot, "items": items}
        print(f"{lab:>12}  {len(items)} items  {a.url}/?t={tok}")
    rf.write_text(json.dumps(raters, indent=1))
    cov = [counts[p["pair"]] for p in pairs]
    print(f"{len(raters)} raters; ratings per pair (non-pilot): min {min(cov)}, mean {sum(cov) / len(cov):.2f}, max {max(cov)}")


if __name__ == "__main__":
    main()
