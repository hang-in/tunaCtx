#!/usr/bin/env python3
"""
vec-daemon.py — Unix socket daemon for fast vector queries.
Loads multilingual-e5-small once, then serves embedding requests
via a Unix domain socket. chat-memory.py connects with 0.3s timeout.

Protocol (line-oriented JSON over Unix socket):
  Request:  {"q": "query text"}\n
  Response: {"ok": true, "emb": [0.1, 0.2, ...]}\n  (384 floats)
  Response: {"ok": false, "error": "..."}\n

Usage:
  python3 vec-daemon.py &          # start in background
  python3 vec-daemon.py --stop     # create stop file
"""
import sys, os, json, socket, time, struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

SOCKET_PATH = Path.home() / ".local/share/claude-vault/vec-daemon.sock"
PID_FILE    = Path.home() / ".local/share/claude-vault/vec-daemon.pid"
STOP_FILE   = Path.home() / ".local/share/claude-vault/vec-daemon.stop"
MODEL_NAME  = "intfloat/multilingual-e5-small"

# Windows fallback: AF_UNIX is not exposed by MSVC-built CPython. Use TCP loopback.
USE_TCP = not hasattr(socket, "AF_UNIX")
VEC_PORT = int(os.environ.get("CTX_VEC_PORT", "29501"))

if "--stop" in sys.argv:
    STOP_FILE.write_text("stop")
    print("Stop file written.")
    sys.exit(0)

# Guard: exit if already running
if PID_FILE.exists():
    try:
        existing_pid = int(PID_FILE.read_text().strip())
        os.kill(existing_pid, 0)
        print(f"[vec-daemon] Already running (PID {existing_pid}). Exiting.")
        sys.exit(0)
    except (ProcessLookupError, ValueError, OSError):
        pass  # stale PID file (or Windows os.kill not supported), continue

def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def handle_client(conn: socket.socket, model):
    """Handle one client connection."""
    try:
        conn.settimeout(5.0)
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                return
            buf += chunk
            if len(buf) > 10_000_000:   # 10MB request cap (matches bge-daemon)
                raise ValueError("request too large")
        line = buf.split(b"\n")[0]
        req = json.loads(line.decode("utf-8"))

        # Batch embedding: one request → many embeddings in a single encode call.
        # Keeps the line protocol backward-compatible with single {"q": ...}.
        batch = req.get("batch")
        if batch:
            if not isinstance(batch, list) or not batch:
                raise ValueError("empty batch")
            if len(batch) > 64:
                raise ValueError("batch too large (max 64)")
            texts = ["query: " + str(t)[:1000] for t in batch]
            embs = model.encode(texts, normalize_embeddings=True)
            resp = json.dumps({"ok": True, "embs": embs.tolist()}) + "\n"
            conn.sendall(resp.encode("utf-8"))
            return

        q = req.get("q", "")
        if not q:
            raise ValueError("empty query")

        # Add query prefix for asymmetric embedding
        text = "query: " + q[:1000]
        emb = model.encode([text], normalize_embeddings=True)[0]
        resp = json.dumps({"ok": True, "emb": emb.tolist()}) + "\n"
        conn.sendall(resp.encode("utf-8"))
    except Exception as e:
        try:
            resp = json.dumps({"ok": False, "error": str(e)}) + "\n"
            conn.sendall(resp.encode("utf-8"))
        except Exception:
            pass
    finally:
        conn.close()


def main():
    t0 = time.time()
    print(f"[vec-daemon] Loading {MODEL_NAME}...", flush=True)
    model = load_model()
    print(f"[vec-daemon] Model ready in {time.time()-t0:.1f}s", flush=True)

    # Write PID
    PID_FILE.write_text(str(os.getpid()))

    if USE_TCP:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Skip SO_REUSEADDR on Windows: semantics differ (allows multiple
        # bind to same port → port hijacking risk). Linux/macOS keep TIME_WAIT
        # rebinding behavior.
        if sys.platform != "win32":
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", VEC_PORT))
        listen_target = f"127.0.0.1:{VEC_PORT}"
    else:
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(SOCKET_PATH))
        listen_target = str(SOCKET_PATH)
    srv.listen(5)
    srv.settimeout(1.0)  # 1s accept timeout for stop-check loop
    print(f"[vec-daemon] Listening on {listen_target}", flush=True)

    # Bounded worker pool: replaces unbounded threading.Thread-per-connection
    # so a burst of clients can't spawn unlimited concurrent model.encode() calls.
    MAX_WORKERS = 16
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="vec")

    while True:
        if STOP_FILE.exists():
            print("[vec-daemon] Stop file detected. Shutting down.")
            STOP_FILE.unlink()
            break
        try:
            conn, _ = srv.accept()
            executor.submit(handle_client, conn, model)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[vec-daemon] Accept error: {e}", flush=True)

    executor.shutdown(wait=False)
    srv.close()
    if not USE_TCP:
        SOCKET_PATH.unlink(missing_ok=True)
    PID_FILE.unlink(missing_ok=True)
    print("[vec-daemon] Stopped.")


if __name__ == "__main__":
    main()
