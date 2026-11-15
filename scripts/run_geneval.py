#!/usr/bin/env python3
"""Score one of our output directories with the official GenEval evaluator (Mask2Former + CLIP colour check).

For every (tags.method, tags.cfg) present in --out, the records that carry tags.geneval_index are laid out the way
geneval/evaluation/evaluate_images.py reads them:

    <tmp>/<method>_w<cfg>/<index:05d>/metadata.jsonl   (the matching line of evaluation_metadata.jsonl)
    <tmp>/<method>_w<cfg>/<index:05d>/samples/0000.png (the image, downscaled to --size px on the long side, LANCZOS)

then scripts/geneval_evaluate_images.py (our mmdet-3.x copy of the official script) is run in the GenEval venv on
each method directory, and the results.jsonl files are summarised into

    <csv>              method,cfg,tag,n,accuracy      one row per GenEval tag, plus
                       tag=overall  (official GenEval score: mean of the six per-tag accuracies) and
                       tag=images   (fraction of all images judged correct)
    <csv stem>_items.csv  method,cfg,index,tag,correct,reason   one row per image, for paired comparisons.

Typical call (reg1):
    scripts/run_geneval.py --out $REG_DATA/outputs/geneval_4k --metadata $REG_DATA/datasets/geneval/evaluation_metadata.jsonl \
        --venv-python /data1/moghadamp/reg_geneval_venv/bin/python --csv results/geneval_4k_geneval.csv
The GenEval layout is kept under --tmp (default $REG_DATA/tmp/geneval_eval/<out dir name>) so re-runs skip the downscale;
--redo rebuilds it. --device cpu runs the detector on CPU (slow; for smoke tests while the GPU is busy).
"""
import argparse, csv, json, os, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

TAGS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
HERE = Path(__file__).resolve().parent
REG_DATA = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data"))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="our output dir with <id>.png + <id>.json records")
    ap.add_argument("--metadata", default=str(REG_DATA / "datasets/geneval/evaluation_metadata.jsonl"))
    ap.add_argument("--venv-python", default="/data1/moghadamp/reg_geneval_venv/bin/python")
    ap.add_argument("--csv", required=True, help="summary csv; per-image rows go to <stem>_items.csv")
    ap.add_argument("--tmp", default=None, help="GenEval layout dir (default $REG_DATA/tmp/geneval_eval/<out name>)")
    ap.add_argument("--evaluator", default=str(HERE / "geneval_evaluate_images.py"))
    ap.add_argument("--model-path", default=str(REG_DATA / "weights/geneval"), help="dir holding the Mask2Former .pth")
    ap.add_argument("--size", type=int, default=1024, help="long side of the downscaled image fed to GenEval")
    ap.add_argument("--workers", type=int, default=8, help="processes for the downscale")
    ap.add_argument("--device", default=None, help="cuda (default when available) or cpu")
    ap.add_argument("--methods", nargs="*", default=None, help="only these tags.method values")
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompt indices per method (smoke tests)")
    ap.add_argument("--redo", action="store_true", help="rebuild the downscaled images even if present")
    ap.add_argument("--options", nargs="*", default=[], help="passed through to the evaluator (key=value)")
    return ap.parse_args()


def load_records(out_dir, methods, limit):
    groups = defaultdict(dict)  # (method, cfg) -> {index: record}
    skipped = Counter()
    for jp in sorted(Path(out_dir).glob("*.json")):
        try:
            rec = json.loads(jp.read_text())
        except json.JSONDecodeError:
            skipped["bad json"] += 1
            continue
        tags = rec.get("tags", {})
        if "geneval_index" not in tags:
            skipped["no geneval_index"] += 1
            continue
        if rec.get("status", "ok") != "ok":
            skipped["status != ok"] += 1
            continue
        img = Path(rec.get("image") or jp.with_suffix(".png"))
        if not img.exists():
            img = jp.with_suffix(".png")
        if not img.exists():
            skipped["missing png"] += 1
            continue
        method = tags.get("method", "unknown")
        if methods and method not in methods:
            skipped["method filtered"] += 1
            continue
        idx = int(tags["geneval_index"])
        if limit is not None and idx >= limit:
            skipped["beyond --limit"] += 1
            continue
        key = (method, float(tags.get("cfg", float("nan"))))
        if idx in groups[key]:
            skipped["duplicate (method,cfg,index)"] += 1
            continue
        rec["_image"] = str(img)
        groups[key][idx] = rec
    return groups, skipped


def group_name(method, cfg):
    return f"{method}_w{cfg:g}"


def downscale(job):
    src, dst, size = job
    if os.path.exists(dst):
        return "kept"
    im = Image.open(src).convert("RGB")
    w, h = im.size
    s = size / max(w, h)
    if s < 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    tmp = dst + ".part.png"
    im.save(tmp, format="PNG")
    os.replace(tmp, dst)
    return "made"


def layout(group_dir, records, meta_lines, size, workers, redo):
    jobs = []
    mismatched = 0
    for idx, rec in sorted(records.items()):
        if idx >= len(meta_lines):
            raise SystemExit(f"geneval_index {idx} beyond metadata file ({len(meta_lines)} lines)")
        meta = json.loads(meta_lines[idx])
        if rec.get("prompt") and rec["prompt"].strip() != meta["prompt"].strip():
            mismatched += 1
            print(f"  WARNING index {idx}: record prompt {rec['prompt']!r} != metadata {meta['prompt']!r}", file=sys.stderr)
        d = group_dir / f"{idx:05d}"
        (d / "samples").mkdir(parents=True, exist_ok=True)
        (d / "metadata.jsonl").write_text(json.dumps(meta) + "\n")
        dst = d / "samples" / "0000.png"
        if redo and dst.exists():
            dst.unlink()
        jobs.append((rec["_image"], str(dst), size))
    # drop stale prompt dirs from an earlier, larger run so GenEval only sees what we laid out
    wanted = {f"{i:05d}" for i in records}
    for d in group_dir.iterdir():
        if d.is_dir() and d.name.isdigit() and d.name not in wanted:
            shutil.rmtree(d)
    t0 = time.time()
    with Pool(min(workers, max(1, len(jobs)))) as pool:
        counts = Counter(pool.imap_unordered(downscale, jobs, chunksize=4))
    print(f"  downscale: {dict(counts)} in {time.time() - t0:.0f}s" + (f", {mismatched} prompt mismatches" if mismatched else ""))


def run_evaluator(args, group_dir):
    results = group_dir / "results.jsonl"
    cmd = [args.venv_python, "-I", args.evaluator, str(group_dir), "--outfile", str(results), "--model-path", args.model_path]
    if args.options:
        cmd += ["--options", *args.options]
    env = dict(os.environ)
    env.setdefault("GENEVAL_ROOT", str(REG_DATA / "tools/geneval"))
    env["HF_HOME"] = str(REG_DATA / "weights/hf")  # where open_clip fetches/keeps the ViT-L-14 (openai) weights, as in env.sh
    if args.device:
        env["GENEVAL_DEVICE"] = args.device
    print("  $ " + " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    print(f"  evaluator done in {time.time() - t0:.0f}s", flush=True)
    rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
    out = {}
    for r in rows:
        idx = int(Path(r["filename"]).parts[-3])
        out[idx] = (r["tag"], bool(r["correct"]), r.get("reason") or "")
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out)
    meta_lines = [l for l in Path(args.metadata).read_text().splitlines() if l.strip()]
    groups, skipped = load_records(out_dir, args.methods, args.limit)
    if skipped:
        print(f"skipped records: {dict(skipped)}")
    if not groups:
        raise SystemExit(f"no scorable records in {out_dir}")
    tmp = Path(args.tmp) if args.tmp else REG_DATA / "tmp/geneval_eval" / out_dir.name
    tmp.mkdir(parents=True, exist_ok=True)

    summary, items = [], []
    for (method, cfg), records in sorted(groups.items()):
        print(f"== {method} cfg={cfg:g}: {len(records)} images")
        gdir = tmp / group_name(method, cfg)
        gdir.mkdir(parents=True, exist_ok=True)
        layout(gdir, records, meta_lines, args.size, args.workers, args.redo)
        res = run_evaluator(args, gdir)
        by_tag = defaultdict(list)
        for idx in sorted(records):
            if idx not in res:
                print(f"  WARNING no result for index {idx}", file=sys.stderr)
                continue
            tag, ok, reason = res[idx]
            by_tag[tag].append(ok)
            items.append(dict(method=method, cfg=cfg, index=idx, tag=tag, correct=int(ok), reason=reason.replace("\n", " | ")))
        accs = []
        for tag in TAGS + sorted(set(by_tag) - set(TAGS)):
            if tag not in by_tag:
                continue
            v = by_tag[tag]
            acc = sum(v) / len(v)
            accs.append(acc)
            summary.append(dict(method=method, cfg=cfg, tag=tag, n=len(v), accuracy=round(acc, 4)))
            print(f"  {tag:<14} {acc:7.2%}  ({sum(v)}/{len(v)})")
        n_all = sum(len(v) for v in by_tag.values())
        n_ok = sum(sum(v) for v in by_tag.values())
        overall = sum(accs) / len(accs) if accs else float("nan")
        summary.append(dict(method=method, cfg=cfg, tag="overall", n=n_all, accuracy=round(overall, 4)))
        summary.append(dict(method=method, cfg=cfg, tag="images", n=n_all, accuracy=round(n_ok / n_all, 4) if n_all else float("nan")))
        print(f"  overall (mean over {len(accs)} tags) {overall:7.2%}   images {n_ok}/{n_all}")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "cfg", "tag", "n", "accuracy"])
        w.writeheader()
        w.writerows(summary)
    items_path = csv_path.with_name(csv_path.stem + "_items.csv")
    with open(items_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "cfg", "index", "tag", "correct", "reason"])
        w.writeheader()
        w.writerows(items)
    print(f"wrote {csv_path} ({len(summary)} rows) and {items_path} ({len(items)} rows); GenEval layout kept in {tmp}")


if __name__ == "__main__":
    main()
