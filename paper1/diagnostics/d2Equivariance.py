"""Diagnostic 2 - O(2) equivariance

Tier: causal

Science question
    Does the internal angle coordinate transform correctly under known transformations of the input, and do the transformations compose the way they compose in the plane?

Procedure (from paperScopingV1.md, to implement)
    - fit the probe ONCE on untransformed embeddings, freeze it, then transform inputs, re-encode, read with the unmodified probe
    - rotation over a grid of angles: measured shift tracks applied shift, slope of unit magnitude
    - self-map: rotation by 180 deg returns the readout to zero shift
    - reflection: mirror flip sends the readout to -theta, with fixed points at 0/90 and antinodes at 45/135
    - composition: flip then rotate matches the composed prediction
    - INVARIANCE COMPLEMENT: physical readouts (mass, colour, morphology) must NOT move under any of the above
    - report all-population and held-out-only summaries; held-out is the leakage-free one

Nulls
    - untransformed baseline error, which sets the noise floor the transformed errors should match
    - matched random directions

Status at scaffold time (2026-07-25)
    MOSTLY DONE in the pre-harness record, except the invariance complement, which has never been run and needs no re-encoding: the eight transformed embedding files already exist and the physical labels are on disk. Sign bookkeeping was verified before the original run but only as prose in the log; it becomes a test here.

Run
    python paper1/diagnostics/d2Equivariance.py

Writes
    paper1/results/d2Equivariance.json   (with a provenance block)
    paper1/figures/d2Equivariance*.png

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
