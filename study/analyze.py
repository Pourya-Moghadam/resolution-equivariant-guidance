#!/usr/bin/env python
"""Human study, step 4: turn the responses into preference rates for REG with confidence intervals.

Per comparison and question (overall preference, fine detail):
  reg / other / same   share of ratings that chose REG, the other method, or "no difference"
  score                (REG + half of "no difference") / ratings; 0.5 means no preference
  CI95                 bootstrap over raters and prompts together (both resampled with replacement, 10,000 draws)
  prompts              share of prompts on which REG has the majority of the non-tied ratings (ties between methods count half)
Also: agreement (two raters on the same pair give the same answer), self-consistency on the repeated pairs (same answer
after the sides were swapped), and how many ratings were given without panning the detail view.
Repeated presentations and pilot raters are left out of the rates. Raters appear as R01, R02, ... in the output.

  source env.sh
  python study/analyze.py [--dir $REG_DATA/human_study/v1] [--csv results/human_study_v1.csv] [--min-done 0.5]
"""
from __future__ import annotations

import argparse, collections, csv, itertools, json, os
from pathlib import Path

import numpy as np

QUESTIONS = ("overall", "detail")


def load(d: Path, min_done: float = 0.0, include_pilot: bool = False):
    man = json.loads((d / "manifest.json").read_text()); pairs = {p["pair"]: p for p in man["pairs"]}
    raters = json.loads((d / "raters.json").read_text())
    rows, repeats, info = [], [], []
    for k, (tok, r) in enumerate(raters.items(), 1):
        f = d / "responses" / f"{tok}.jsonl"
        resp = {}
        if f.exists():
            for line in f.read_text().splitlines():
                if line.strip():
                    x = json.loads(line); resp[x["idx"]] = x  # a re-sent answer replaces the earlier one
        frac = len(resp) / max(1, len(r["items"]))
        keep = (include_pilot or not r.get("pilot")) and frac >= min_done and resp
        info.append({"rater": f"R{k:02d}", "pilot": bool(r.get("pilot")), "done": len(resp), "items": len(r["items"]), "used": bool(keep)})
        if not keep:
            continue
        first = {}
        for i in sorted(resp):
            x = resp[i]; p = pairs[x["pair"]]
            who = lambda v: "same" if v == "same" else ("reg" if (v == "left") != x["flip"] else "other")
            rec = {"rater": f"R{k:02d}", "pair": x["pair"], "comparison": p["comparison"], "prompt_id": p["prompt_id"],
                   "overall": who(x["overall"]), "detail": who(x["detail"]), "ms": x.get("ms", 0), "panned": x.get("panned", False)}
            if x["repeat"]:
                if x["pair"] in first:
                    repeats.append((first[x["pair"]], rec))
            else:
                rows.append(rec); first[x["pair"]] = rec
    return man, rows, repeats, info


def score(vals) -> float:
    vals = list(vals)
    return (sum(v == "reg" for v in vals) + 0.5 * sum(v == "same" for v in vals)) / len(vals) if vals else float("nan")


def bootstrap(rows: list[dict], q: str, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    raters = sorted({r["rater"] for r in rows}); prompts = sorted({r["prompt_id"] for r in rows})
    ri = {x: i for i, x in enumerate(raters)}; pi = {x: i for i, x in enumerate(prompts)}
    num = np.zeros((len(raters), len(prompts))); den = np.zeros_like(num)
    for r in rows:
        num[ri[r["rater"]], pi[r["prompt_id"]]] += {"reg": 1.0, "same": 0.5, "other": 0.0}[r[q]]; den[ri[r["rater"]], pi[r["prompt_id"]]] += 1
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        a = rng.integers(0, len(raters), len(raters)); b = rng.integers(0, len(prompts), len(prompts))
        dsum = den[np.ix_(a, b)].sum()
        if dsum:
            out.append(num[np.ix_(a, b)].sum() / dsum)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (float("nan"), float("nan"))


def summarise(man: dict, rows: list[dict], repeats: list, nboot: int = 10000) -> tuple[list[dict], dict]:
    table = []
    for comp in man["comparisons"]:
        sub = [r for r in rows if r["comparison"] == comp]
        if not sub:
            continue
        for q in QUESTIONS:
            c = collections.Counter(r[q] for r in sub); n = len(sub); lo, hi = bootstrap(sub, q, nboot)
            byp = collections.defaultdict(list)
            for r in sub:
                byp[r["prompt_id"]].append(r[q])
            maj = [1.0 if v.count("reg") > v.count("other") else 0.5 if v.count("reg") == v.count("other") else 0.0 for v in byp.values()]
            table.append({"comparison": comp, "question": q, "ratings": n, "raters": len({r["rater"] for r in sub}), "prompts": len(byp),
                          "reg": c["reg"] / n, "other": c["other"] / n, "same": c["same"] / n, "score": score(r[q] for r in sub),
                          "lo": lo, "hi": hi, "prompt_majority": sum(maj) / len(maj)})
    extra = {}
    for q in QUESTIONS:
        byp = collections.defaultdict(list)
        for r in rows:
            byp[r["pair"]].append(r[q])
        agree = [a == b for v in byp.values() for a, b in itertools.combinations(v, 2)]
        extra[f"agreement_{q}"] = (sum(agree) / len(agree), len(agree)) if agree else (float("nan"), 0)
        same = [a[q] == b[q] for a, b in repeats]
        extra[f"consistency_{q}"] = (sum(same) / len(same), len(same)) if same else (float("nan"), 0)
    extra["not_panned"] = (sum(not r["panned"] for r in rows) / len(rows), len(rows)) if rows else (float("nan"), 0)
    extra["median_seconds"] = float(np.median([r["ms"] for r in rows]) / 1000) if rows else float("nan")
    return table, extra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data"))
    ap.add_argument("--dir", default=str(data / "human_study" / "v1"))
    ap.add_argument("--csv")
    ap.add_argument("--min-done", type=float, default=0.0, help="leave out raters who finished less than this fraction of their items")
    ap.add_argument("--include-pilot", action="store_true")
    a = ap.parse_args()
    man, rows, repeats, info = load(Path(a.dir), a.min_done, a.include_pilot)
    for x in info:
        print(f"{x['rater']}  {'pilot' if x['pilot'] else '     '}  {x['done']:>3} / {x['items']:<3}  {'used' if x['used'] else 'left out'}")
    if not rows:
        return print("no ratings yet")
    table, extra = summarise(man, rows, repeats)
    print(f"\n{'comparison':<11} {'question':<8} {'n':>5} {'REG':>6} {'other':>6} {'same':>6} {'score':>6}  {'CI95':<14} {'prompts':>7}")
    for t in table:
        print(f"{t['comparison']:<11} {t['question']:<8} {t['ratings']:>5} {t['reg']:>6.1%} {t['other']:>6.1%} {t['same']:>6.1%} {t['score']:>6.3f}  "
              f"[{t['lo']:.3f}, {t['hi']:.3f}] {t['prompt_majority']:>7.1%}")
    print()
    for k, v in extra.items():
        print(f"{k:<22} {v[0]:.3f} (n = {v[1]})" if isinstance(v, tuple) else f"{k:<22} {v:.1f}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
