"""Provenance stamping. Every results file carries one of these blocks.

A number is quotable in the technical report only if it came out of a results file with a
provenance block. Nothing is typed in by hand.
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn

from config import RESULTS, ROOT, SEED


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git():
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return None
    return {"sha": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def stamp(inputs, started):
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(time.time() - started, 1),
        "git": git(),
        "seed": SEED,
        "python": sys.version.split()[0],
        "packages": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__},
        "platform": platform.platform(),
        "inputs": {str(Path(p).name): {"path": str(p), "sha256": sha256(p),
                                       "bytes": Path(p).stat().st_size}
                   for p in inputs},
    }


def write(name, payload, inputs, started):
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    with open(out, "w") as f:
        json.dump({"diagnostic": name, "result": payload,
                   "provenance": stamp(inputs, started)}, f, indent=2, default=float)
    return out
