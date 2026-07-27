"""Runner and provenance verifier for the diagnostic suite.

    python paper1/runAll.py                 status of all nine
    python paper1/runAll.py d1 d2           run those, in order
    python paper1/runAll.py --all           run everything
    python paper1/runAll.py --figures d1    re-render figures only
    python paper1/runAll.py --verify        re-hash every input and check stored provenance
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import config as C

DIAGNOSTICS = {
    "d0": ("d0DatasetAudit", "Dataset and substrate audit", "reference"),
    "d1": ("d1AngleReadout", "Angle readout characterization", "descriptive"),
    "d2": ("d2Equivariance", "O(2) equivariance", "input-space causal"),
    "d3": ("d3Chirality", "Chirality", "input-space causal"),
    "d4": ("d4NuisanceLeakage", "Nuisance decodability and leakage", "descriptive"),
    "d5": ("d5Degradation", "Degradation response", "input-space causal"),
    "d6": ("d6Decodability", "Decodability battery", "descriptive"),
    "d7": ("d7ConceptGeometry", "Concept geometry under a calibrated null", "calibrated null"),
    "d8": ("d8StructuredRelations", "Structured relations", "descriptive"),
    "d9": ("d9ArtificialRedshift", "Artificial redshifting", "input-space causal"),
}


sys.path.insert(0, str(C.PAPER1 / "diagnostics"))


def load(module):
    return importlib.import_module(module)


def status():
    print(f"{'key':4} {'diagnostic':44} {'tier':20} status")
    for key, (mod, title, tier) in DIAGNOSTICS.items():
        p = C.RESULTS / f"{mod}.json"
        if p.exists():
            pv = json.loads(p.read_text())["provenance"]
            s = f"run {pv['utc']}  {pv['wall_seconds']}s  sha {(pv['git']['sha'] or '')[:7]}"
        else:
            s = "not run"
        print(f"{key:4} {title:44} {tier:20} {s}")


def verify():
    import provenance
    bad = 0
    for key, (mod, title, _) in DIAGNOSTICS.items():
        p = C.RESULTS / f"{mod}.json"
        if not p.exists():
            continue
        pv = json.loads(p.read_text())["provenance"]
        for name, rec in pv["inputs"].items():
            f = Path(rec["path"])
            if not f.exists():
                print(f"  {key} {name}: MISSING")
                bad += 1
            elif provenance.sha256(f) != rec["sha256"]:
                print(f"  {key} {name}: HASH CHANGED since the run")
                bad += 1
        print(f"{key}: {len(pv['inputs'])} inputs checked")
    print("provenance verification:", "clean" if bad == 0 else f"{bad} problem(s)")
    return bad


def run(keys, figures_only=False):
    manifest = []
    for k in keys:
        mod, title, tier = DIAGNOSTICS[k]
        for suffix in (["Figures"] if figures_only else ["", "Figures"]):
            name = mod + suffix
            if not (C.PAPER1 / "diagnostics" / f"{name}.py").exists():
                continue
            t0 = time.time()
            print(f"\n=== {k} {title} :: {name} ===", flush=True)
            try:
                module = load(name)
                if hasattr(module, "main"):
                    module.main()
                ok = True
            except NotImplementedError as e:
                print(f"  not implemented: {e}")
                ok = False
            manifest.append({"key": k, "module": name, "ok": ok,
                             "seconds": round(time.time() - t0, 1)})
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "runManifest.json").write_text(json.dumps(
        {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": manifest}, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", metavar="d1..d9")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    bad = [k for k in a.keys if k not in DIAGNOSTICS]
    if bad:
        ap.error(f"unknown diagnostic(s) {bad}; choose from {list(DIAGNOSTICS)}")
    if a.verify:
        sys.exit(1 if verify() else 0)
    keys = list(DIAGNOSTICS) if a.all else (a.keys or [])
    if not keys:
        status()
        return
    run(keys, figures_only=a.figures)


if __name__ == "__main__":
    main()
