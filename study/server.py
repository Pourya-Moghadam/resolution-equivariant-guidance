#!/usr/bin/env python
"""Human study, step 3: the rating server (Python standard library only).

  source env.sh
  python study/server.py [--dir $REG_DATA/human_study/v1] [--host 0.0.0.0] [--port 8347]

A rater opens  http://<host>:<port>/?t=<token>  (tokens from study/make_raters.py). The page gets one item at a time: the
prompt and two image names that carry no method information. Each answer is appended to responses/<token>.jsonl at once, so
a rater can close the page and continue later. Progress of all raters:  /admin?key=<admin key printed at start>.
"""
from __future__ import annotations

import argparse, json, os, re, secrets, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
LOCK = threading.Lock()
CHOICES = {"left", "right", "same"}


class Study:
    def __init__(self, d: Path):
        self.d = d; self.img = d / "img"; self.resp = d / "responses"; self.resp.mkdir(exist_ok=True)
        self.pairs = {p["pair"]: p for p in json.loads((d / "manifest.json").read_text())["pairs"]}
        kf = d / "admin_key.txt"
        if not kf.exists():
            kf.write_text(secrets.token_urlsafe(12))
        self.admin = kf.read_text().strip()

    def raters(self) -> dict:
        return json.loads((self.d / "raters.json").read_text())  # re-read: raters can be added while the server runs

    def answered(self, tok: str) -> dict[int, dict]:
        f = self.resp / f"{tok}.jsonl"
        out = {}
        if f.exists():
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line); out[r["idx"]] = r
        return out

    def state(self, tok: str) -> dict | None:
        r = self.raters().get(tok)
        if r is None:
            return None
        done = self.answered(tok); items = r["items"]
        nxt = next((i for i in range(len(items)) if i not in done), None)
        s = {"total": len(items), "done": len(done), "item": None}
        if nxt is not None:
            s["item"] = self.item(items, nxt)
            if nxt + 1 < len(items):
                s["preload"] = [v["fit"] for v in (self.item(items, nxt + 1)["left"], self.item(items, nxt + 1)["right"])]
        return s

    def item(self, items: list[dict], i: int) -> dict:
        it = items[i]; p = self.pairs[it["pair"]]
        a, b = (p["other"], p["reg"]) if it["flip"] else (p["reg"], p["other"])
        side = lambda x: {"fit": f"/img/{x['hash']}_fit.jpg", "full": f"/img/{x['hash']}_full.jpg"}
        return {"idx": i, "prompt": p["prompt"], "size": p["size"], "left": side(a), "right": side(b)}

    def answer(self, tok: str, body: dict) -> bool:
        r = self.raters().get(tok)
        if r is None:
            return False
        i = body.get("idx")
        if not isinstance(i, int) or not 0 <= i < len(r["items"]) or body.get("overall") not in CHOICES or body.get("detail") not in CHOICES:
            return False
        it = r["items"][i]
        rec = {"idx": i, "pair": it["pair"], "flip": it["flip"], "repeat": it["repeat"], "overall": body["overall"], "detail": body["detail"],
               "ms": int(body.get("ms", 0)), "saw_detail": bool(body.get("saw_detail")), "panned": bool(body.get("panned")),
               "dpr": float(body.get("dpr", 1)), "screen": str(body.get("screen", ""))[:20], "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with LOCK, open(self.resp / f"{tok}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        return True


class Handler(BaseHTTPRequestHandler):
    study: Study

    def log_message(self, fmt, *args):  # keep the log free of tokens
        pass

    def send(self, code: int, body: bytes, ctype: str, cache: str = "no-store"):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache); self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code: int = 200):
        self.send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query); s = self.study
        if u.path == "/":
            return self.send(200, (HERE / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            st = s.state(q.get("t", [""])[0])
            return self.json(st) if st else self.json({"error": "unknown link"}, 403)
        m = re.fullmatch(r"/img/([0-9a-f]{16}_(?:fit|full)\.jpg)", u.path)
        if m and (s.img / m.group(1)).exists():
            f = s.img / m.group(1)
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(f.stat().st_size))
            self.send_header("Cache-Control", "private, max-age=86400"); self.end_headers()
            with open(f, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    self.wfile.write(chunk)
            return
        if u.path == "/admin" and secrets.compare_digest(q.get("key", [""])[0], s.admin):
            rows = []
            for i, (tok, r) in enumerate(s.raters().items(), 1):
                d = s.answered(tok); last = max((x["time"] for x in d.values()), default="-")
                rows.append(f"{r['label']:<24} {'pilot' if r.get('pilot') else '':<6} {len(d):>4} / {len(r['items']):<4} last {last}")
            return self.send(200, ("\n".join(rows) + "\n").encode(), "text/plain; charset=utf-8")
        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/answer":
            return self.send(404, b"not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n)) if 0 < n < 10000 else {}
        except (ValueError, json.JSONDecodeError):
            body = {}
        tok = str(body.get("t", ""))
        if not self.study.answer(tok, body):
            return self.json({"error": "bad request"}, 400)
        self.json(self.study.state(tok))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = Path(os.environ.get("REG_DATA", "/data1/moghadamp/reg_data"))
    ap.add_argument("--dir", default=str(data / "human_study" / "v1"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8347)
    a = ap.parse_args()
    Handler.study = Study(Path(a.dir))
    print(f"serving {a.dir} on http://{a.host}:{a.port}  (progress: /admin?key={Handler.study.admin})", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
