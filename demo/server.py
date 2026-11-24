"""Phase 4 demo server: one endpoint, /suggest, that applies the frozen invocation policy.

Needs a running llama.cpp server (see PHASE3_RUNTIME.md):
    llama-server -m <path>/Qwen_Qwen3-0.6B-Q4_K_M.gguf --port 8080 -c 2048
Then, from the repo root:
    uv run python -m demo.server
and open http://127.0.0.1:8000 in the browser.

POST /suggest  {"text": "...", "session": "abc", "mode": "policy"}
  -> {"ghost": " suggestion", "source": "model" | "reuse" | "skip", "granularity": ..., "latency_ms": ..., "stats": {...}}

What happens after every key press (one request per change of the text):
  1. reuse   if the last suggestion still matches what was typed since, its rest is returned; no model call
  2. policy  src.policy.decide(text) says whether to call the model and for which granularity
  3. model   the llama.cpp server is called with the Phase 1 sampling settings and stopping rules

Modes, to compare in the page:
  policy           steps 1-3 (the default)
  policy_no_reuse  steps 2-3 only (the stateless rule that Phase 2a evaluated)
  always           a model call at every key press (the always-invoke baseline)
  always_reuse     a model call at every key press unless the last suggestion still matches
                   (the fair baseline for 'policy': same reuse, no policy)

The counters are per session and mode: key presses, model calls, reused, skipped, model time.
It uses only the Python standard library.
Environment: LLAMA_URL (default http://127.0.0.1:8080), PORT (default 8000)
"""
import http.client
import json
import os
import threading
import time
from urllib.parse import urlparse

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src import policy
from src.bench_llama import MAX_TOKEN_CEILING, should_stop

LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
MODES = ("policy", "policy_no_reuse", "always", "always_reuse")

PORT = int(os.environ.get("PORT", "8000"))
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

_lock = threading.Lock()
_last = {}      # (session, mode) -> {"text": text at the call, "ghost": suggestion returned then}
_stats = {}     # (session, mode) -> counters


def choose_granularity(text, mode):
    """Which single suggestion to generate after this key press (None = do not call)."""
    if mode in ("always", "always_reuse"):
        if not text:
            return None
        if text[-1].isspace():
            return "sentence" if policy.words_typed(text) >= policy.SENTENCE_MIN_WORDS else "next_word"
        return "partial_word"
    calls = policy.decide(text)
    if not calls:
        return None
    return "sentence" if "sentence" in calls else calls[0]      # the longest one the policy allows


def trim(raw, granularity):
    """Text to show after the cursor. Leading spaces are kept, so it can be appended as typed."""
    if granularity in ("partial_word", "next_word"):
        body = raw.lstrip()
        lead = raw[:len(raw) - len(body)]
        first = body.split()[0] if body.split() else ""
        return (lead + first) if first else ""
    return raw.rstrip()


def generate(text, granularity):
    """One suggestion from the llama.cpp server. Returns (ghost text, milliseconds)."""
    u = urlparse(LLAMA_URL)
    body = json.dumps({
        "prompt": text, "n_predict": MAX_TOKEN_CEILING[granularity], "stream": True,
        "temperature": 0.8, "top_k": 50, "top_p": 0.95, "min_p": 0.0, "cache_prompt": True,
    })
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=60)
    t0 = time.perf_counter()
    raw = ""
    try:
        conn.request("POST", "/completion", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"llama.cpp server answered {resp.status}")
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            d = json.loads(line[5:])
            raw += d.get("content", "")
            if d.get("stop") or should_stop(raw, granularity):
                break
    finally:
        conn.close()
    ghost = trim(raw, granularity)
    if text and text[-1].isspace():
        ghost = ghost.lstrip()                 # no double space at a word start
    return ghost, (time.perf_counter() - t0) * 1000


def still_valid(last, text):
    """Rest of the last suggestion if what was typed since then matches its start, else None."""
    if not last or not last["ghost"] or not text.startswith(last["text"]):
        return None
    extra = text[len(last["text"]):]
    if last["ghost"].startswith(extra) and len(extra) < len(last["ghost"]):
        return last["ghost"][len(extra):]
    return None


def new_stats():
    return {"key_presses": 0, "model_calls": 0, "reused": 0, "skipped": 0, "model_ms": 0.0, "shown": 0}


def public(s):
    k = max(1, s["key_presses"])
    return dict(s, model_ms=round(s["model_ms"], 1),
                calls_pct=round(100 * s["model_calls"] / k, 1),
                shown_pct=round(100 * s["shown"] / k, 1))


def suggest(text, session="default", mode="policy"):
    """The whole decision for one key press. Returns the answer as a dict."""
    mode = mode if mode in MODES else "policy"
    key = (session, mode)
    with _lock:
        stats = _stats.setdefault(key, new_stats())
        stats["key_presses"] += 1
        rest = still_valid(_last.get(key), text) if mode in ("policy", "always_reuse") else None
        if rest is not None:
            stats["reused"] += 1
            stats["shown"] += 1
            return {"text": text, "ghost": rest, "source": "reuse", "granularity": None, "latency_ms": 0, "stats": public(stats)}
        granularity = choose_granularity(text, mode)
        if granularity is None:
            stats["skipped"] += 1
            _last.pop(key, None)
            return {"text": text, "ghost": "", "source": "skip", "granularity": None, "latency_ms": 0, "stats": public(stats)}
    try:
        ghost, ms = generate(text, granularity)          # outside the lock: this is the slow part
    except (OSError, RuntimeError) as e:
        return {"text": text, "ghost": "", "source": "error", "granularity": granularity, "latency_ms": 0,
                "error": f"cannot reach the llama.cpp server at {LLAMA_URL}: {e}", "stats": public(stats)}
    with _lock:
        stats["model_calls"] += 1
        stats["model_ms"] += ms
        if ghost:
            stats["shown"] += 1
        _last[key] = {"text": text, "ghost": ghost}
        return {"text": text, "ghost": ghost, "source": "model", "granularity": granularity,
                "latency_ms": round(ms, 1), "stats": public(stats)}


def reset(session="default"):
    with _lock:
        for m in MODES:
            _stats.pop((session, m), None)
            _last.pop((session, m), None)
    return {"ok": True}


def health():
    return {"status": "ok", "llama_url": LLAMA_URL, "min_letters": policy.MIN_LETTERS,
            "sentence_min_words": policy.SENTENCE_MIN_WORDS}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):          # keep the terminal quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(PAGE, "rb") as f:
                self._send(200, f.read(), "text/html")
        elif self.path == "/health":
            self._send(200, health())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            text = str(req.get("text", ""))
            session = str(req.get("session", "default"))
        except (ValueError, AttributeError):
            return self._send(400, {"error": "send JSON like {\"text\": \"...\"}"})
        if self.path == "/suggest":
            self._send(200, suggest(text, session, str(req.get("mode", "policy"))))
        elif self.path == "/reset":
            self._send(200, reset(session))
        else:
            self._send(404, {"error": "not found"})


def main():
    print(f"Demo on http://127.0.0.1:{PORT}   (llama.cpp server expected at {LLAMA_URL}; stop with Ctrl+C)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
