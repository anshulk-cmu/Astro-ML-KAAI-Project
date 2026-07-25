"""Diagnostic 6 - Decodability battery

Tier: descriptive

Science question
    Which physical quantities are linearly readable from pixels alone, particularly quantities the model never received as input?

Procedure (from paperScopingV1.md, to implement)
    - ridge probes on E_img across colours, morphology votes, redshift, stellar mass, star formation rate and structural concentration
    - held-out R2 with bootstrap intervals, and the sparse-label sample size reported beside EVERY entry
    - control arm A, leakage ablation: repeat on the multimodal substrate that ingests scalar photometry and catalog redshift; the gap is the leakage correction and belongs in the main table, not an appendix
    - control arm B, selection versus physics: re-probe within apparent-magnitude bins, and regress out apparent magnitude, angular size and colours before re-probing the residual

Nulls
    - shuffled labels
    - a raw-pixel-PCA floor requiring no model weights; any entry where the model ties the floor is non-discriminating and must be marked as such

Status at scaffold time (2026-07-25)
    PARTIAL. The main table exists with intervals and n. Missing: the leakage ablation for the morphology votes, control arm B for anything except redshift, shuffled-label nulls, and the pixel-PCA floor, which does not exist anywhere in the project. The frozen slide deck currently displays a pixel-PCA row in the same format as measured results; it is a prediction and this diagnostic is what settles it.

Run
    python paper1/diagnostics/d6Decodability.py

Writes
    paper1/results/d6Decodability.json   (with a provenance block)
    paper1/figures/d6Decodability*.png

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
