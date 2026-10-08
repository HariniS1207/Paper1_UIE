"""Local, read-only browser for paired EUVP Underwater ImageNet samples."""
from __future__ import annotations

import argparse
import json
import mimetypes
import random
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "EUVP" / "EUVP-Dataset" / "EUVP" / "Paired" / "underwater_imagenet"
WEB = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def inventory() -> tuple[list[str], list[str], list[str]]:
    """Return valid common filenames and unmatched names from each folder."""
    def names(folder: str) -> set[str]:
        path = DATASET / folder
        if not path.is_dir():
            return set()
        return {p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}

    a, b = names("trainA"), names("trainB")
    return sorted(a & b, key=str.casefold), sorted(a - b, key=str.casefold), sorted(b - a, key=str.casefold)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def send_json(self, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/api/pairs":
            valid, missing_a, missing_b = inventory()
            query = parse_qs(url.query)
            seed = query.get("seed", [None])[0]
            rng = random.Random(seed) if seed is not None else random.SystemRandom()
            shuffled = valid[:]
            rng.shuffle(shuffled)
            self.send_json({"pairs": shuffled, "validCount": len(valid), "missingFromTrainA": missing_a, "missingFromTrainB": missing_b})
            return
        if url.path.startswith("/dataset/"):
            parts = url.path.strip("/").split("/", 2)
            if len(parts) != 3 or parts[0] != "dataset" or parts[1] not in {"trainA", "trainB"}:
                self.send_error(404)
                return
            from urllib.parse import unquote
            filename = unquote(parts[2])
            if Path(filename).name != filename or Path(filename).suffix.lower() not in EXTENSIONS:
                self.send_error(404)
                return
            valid, _, _ = inventory()
            if filename not in set(valid):
                self.send_error(404, "Unmatched dataset filename")
                return
            path = DATASET / parts[1] / filename
            if not path.is_file():
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 128):
                    self.wfile.write(chunk)
            return
        super().do_GET()

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not DATASET.is_dir():
        raise SystemExit(f"Dataset directory not found: {DATASET}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dataset viewer: http://{args.host}:{args.port}")
    print(f"Reading originals from: {DATASET}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
