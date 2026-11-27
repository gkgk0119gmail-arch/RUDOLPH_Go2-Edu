"""Clue parser as a small HTTP service (standard library only), for the robot's search node.

    python -m clue.serve_parse --port 8765                       # on the GPU PC (lab 5090)
    curl -s localhost:8765/parse -d '{"text": "오른쪽으로 5미터쯤 한 명 있어요"}'
    -> {"parsed": {...schema JSON...}, "raw": "...", "latency_s": 1.4, "prompt": "v3f", "problems": []}
    GET /health -> {"ok": true, "model": ..., "prompt": ...}
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import parse as P
from . import schema as S
from .prompts import PROMPTS

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--prompt", default="v3f", choices=sorted(PROMPTS))
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--port", type=int, default=8765)
a = ap.parse_args()

P.PROMPT = PROMPTS[a.prompt]
parser = P.Parser(a.model, a.device)
lock = threading.Lock()


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "model": a.model, "prompt": a.prompt})
        else:
            self._send(404, {"error": "use POST /parse or GET /health"})

    def do_POST(self):
        if self.path != "/parse":
            self._send(404, {"error": "use POST /parse"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            text = str(req["text"])[:500]
        except (ValueError, KeyError):
            self._send(400, {"error": 'body must be {"text": "..."}'})
            return
        t = time.time()
        with lock:                                   # one generation at a time on the GPU
            raw, _ = parser(text)
        pred, bad = S.normalize(S.extract_json(raw))
        self._send(200, {"parsed": pred, "raw": raw, "problems": bad, "latency_s": round(time.time() - t, 2),
                         "prompt": a.prompt, "text": text})

    def log_message(self, fmt, *args):
        print(time.strftime("%H:%M:%S"), self.address_string(), fmt % args, flush=True)


print(f"clue parser ({a.model}, prompt {a.prompt}) on http://{a.host}:{a.port}", flush=True)
ThreadingHTTPServer((a.host, a.port), H).serve_forever()
