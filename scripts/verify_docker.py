"""Build and start the Docker stack, then check that API and UI really answer. Standard library only.

Run from the repo root:  python scripts/verify_docker.py          (leaves the stack running)
                         python scripts/verify_docker.py --down   (stops it after the checks)
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request

API, UI = "http://localhost:8000", "http://localhost:8501"


def sh(*args: str) -> None:
    """Run a docker compose command, streaming its output; stop the script if it fails."""
    print("$", " ".join(args), flush=True)
    if subprocess.run(args).returncode != 0:
        sys.exit(f"FAILED: {' '.join(args)}")


def get(url: str, timeout: float = 30.0) -> bytes:
    """HTTP GET -> body (raises on non-2xx or connection error)."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def wait_for(url: str, seconds: int = 120) -> None:
    """Poll url until it answers 2xx or the time is up."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            get(url, timeout=5)
            return
        except OSError:
            time.sleep(2)
    sys.exit(f"FAILED: {url} did not answer within {seconds}s (see: docker compose logs)")


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Print one PASS/FAIL line."""
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--down", action="store_true", help="stop the containers after the checks")
    args = p.parse_args()

    sh("docker", "compose", "build")
    sh("docker", "compose", "up", "-d")   # blocks until init has finished and api is healthy enough to start ui
    wait_for(f"{API}/health")
    wait_for(f"{UI}/_stcore/health")

    health = json.loads(get(f"{API}/health"))
    due = json.loads(get(f"{API}/customers/due?limit=3", timeout=120))   # first call scores every customer
    results = [
        check("api /health", health.get("status") == "ok", f"as_of={health.get('as_of')}"),
        check("api /customers/due (data + model loaded)", len(due.get("customers", [])) > 0,
              f"{len(due.get('customers', []))} customers"),
        check("ui /_stcore/health", get(f"{UI}/_stcore/health").strip() == b"ok"),
    ]
    if args.down:
        sh("docker", "compose", "down")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
