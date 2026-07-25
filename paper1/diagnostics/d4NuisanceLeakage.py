"""Diagnostic 4 - Nuisance decodability and leakage

Tier: descriptive

Science question
    How strongly does the embedding encode the observation rather than the object, and can it recover information that should be unavailable from the image entirely?

Procedure (from paperScopingV1.md, to implement)
    - Part A conditions: ridge-probe seeing, point-source depth and Galactic extinction from E_img, in every band, with intervals and n
    - Part A identity: hemisphere and per-band coverage/blankness probed from E_img (feature-correlation counts are a different statistic and do not substitute)
    - Part B impossible-quantity leakage: probe sky coordinates, which a north-oriented cutout with no compass should not carry
    - narrow the route: residualize each coordinate against the Part A covariates, then re-probe the residual, with the residualizing fit held out rather than in-sample

Nulls
    - shuffled labels on every condition probe
    - the drop under residualization compared against the drop from residualizing on RANDOM covariates of matched dimension

Status at scaffold time (2026-07-25)
    PARTIAL. Present: seeing, depth and extinction in the r band only, and the RA leak with its residualization. Missing: other bands, intervals, Dec from the image substrate, hemisphere and band-blankness probes, both nulls. Stated limitation to carry verbatim: the residualization drop compares R2 on targets of different variance, so it is a crude apportionment and not a mediation decomposition.

Run
    python paper1/diagnostics/d4NuisanceLeakage.py

Writes
    paper1/results/d4NuisanceLeakage.json   (with a provenance block)
    paper1/figures/d4NuisanceLeakage*.png

Every number this script emits must land in the results file. Nothing is quoted into the
technical report that did not come from here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    raise NotImplementedError("scaffold only; implement when this diagnostic is scheduled")


if __name__ == "__main__":
    main()
