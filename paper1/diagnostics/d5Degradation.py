"""Diagnostic 5 - Degradation response

Tier: causal

Science question
    Does the representation respond to controlled degradation the way the physics requires, and where does each physical readout break?

Procedure (from paperScopingV1.md, to implement)
    - apply known perturbations to inputs, re-encode, and read BOTH the condition probes and the physical probes with probes trained on unperturbed data
    - PSF blur: convolve to worse seeing; the seeing probe should track the applied blur
    - noise injection to a target depth: graceful degradation, or a cliff
    - band ablation: zero each band in turn to attribute each physical readout to bands
    - synthetic band-blanking: blank a band on objects that HAVE it; a mechanism test, since migration toward the population that natively lacks the band shows the model uses blankness as a survey token
    - zeropoint offset: scale all fluxes
    - reddening: per-band multiplicative from a standard extinction curve; must move colour and NOT morphology
    - background pedestal: additive constant; should damage morphology and size and leave colour
    - sweep each perturbation over several severity levels and report the half-accuracy severity as an operating limit

Carry-forward from Diagnostic 3
    None of these perturbations moves light onto a new pixel grid, so the resampling result of
    standing caveat 6 does not apply directly. The weaker lesson does: this encoder travels a
    long way under input changes that leave the physics almost untouched, so severity should be
    read from the targeted probes rather than from whole-vector distances.

Nulls
    - unperturbed baseline
    - perturbations applied to a matched random subset of pixels rather than physically

Status at scaffold time (2026-07-25)
    NOT STARTED. Rotation and mirror flip are the only perturbations ever pushed through the frozen encoder on real galaxies. The Phase 2 realism code that implements Moffat blur and depth-matched noise is on the Vera cluster, not on this machine. Compute note from the document: use a fixed stratified subsample held identical across all conditions and severities.

Run
    python paper1/diagnostics/d5Degradation.py

Writes
    paper1/results/d5Degradation.json   (with a provenance block)
    paper1/figures/d5Degradation*.png

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
