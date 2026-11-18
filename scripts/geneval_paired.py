#!/usr/bin/env python3
"""Paired GenEval accuracy: per-prompt correct/incorrect for two methods from run_geneval.py's <stem>_items.csv.
Reports accuracy per tag and overall (mean of tags, as GenEval does) for each method, the paired difference with a
percentile-bootstrap 95% CI over prompts (10k resamples, stratified by tag for the overall), and the discordant
counts (prompts only REG gets right / only the reference gets right).
  python scripts/geneval_paired.py --items results/geneval_4k_geneval_items.csv --ref cfg --cmp anchor_base_pyr_s12 \
      --csv results/geneval_4k_paired.csv
"""
import argparse, csv, collections, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--items", required=True); ap.add_argument("--ref", default="cfg")
ap.add_argument("--cmp", default="anchor_base_pyr_s12"); ap.add_argument("--csv", required=True); ap.add_argument("--boot", type=int, default=10000)
a = ap.parse_args(); rng = np.random.default_rng(0)
rows = list(csv.DictReader(open(a.items)))
by = collections.defaultdict(dict)   # (tag, index) -> method -> correct
for r in rows: by[(r["tag"], int(r["index"]))][r["method"]] = int(r["correct"])
tags = sorted({t for t, _ in by})
pairs = {t: np.array([(by[k][a.cmp], by[k][a.ref]) for k in by if k[0] == t and a.cmp in by[k] and a.ref in by[k]]) for t in tags}
out = []
def boot_tag(p):
    idx = rng.integers(0, len(p), (a.boot, len(p))); d = p[:, 0] - p[:, 1]
    return d[idx].mean(1)
overall_c = overall_r = 0.0; overall_d = np.zeros(a.boot)
for t in tags:
    p = pairs[t]; c, r = p[:, 0].mean(), p[:, 1].mean(); bd = boot_tag(p)
    only_c = int(((p[:, 0] == 1) & (p[:, 1] == 0)).sum()); only_r = int(((p[:, 0] == 0) & (p[:, 1] == 1)).sum())
    out.append(dict(tag=t, n=len(p), acc_cmp=c, acc_ref=r, diff=c - r, lo=np.percentile(bd, 2.5), hi=np.percentile(bd, 97.5), only_cmp=only_c, only_ref=only_r))
    overall_c += c / len(tags); overall_r += r / len(tags); overall_d += bd / len(tags)
n_all = sum(len(p) for p in pairs.values())
out.append(dict(tag="overall", n=n_all, acc_cmp=overall_c, acc_ref=overall_r, diff=overall_c - overall_r, lo=np.percentile(overall_d, 2.5), hi=np.percentile(overall_d, 97.5),
                only_cmp=sum(o["only_cmp"] for o in out), only_ref=sum(o["only_ref"] for o in out)))
with open(a.csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for o in out: print(f"{o['tag']:>14} n={o['n']:4d}  {a.cmp[:10]} {o['acc_cmp']:.3f}  {a.ref} {o['acc_ref']:.3f}  diff {o['diff']:+.3f} [{o['lo']:+.3f}, {o['hi']:+.3f}]  only-{a.cmp[:3]} {o['only_cmp']:3d} only-{a.ref} {o['only_ref']:3d}")
print("wrote", a.csv)
