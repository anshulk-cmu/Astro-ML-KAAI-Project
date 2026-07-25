"""Diagnostic 3 - Chirality

Tier: causal

Science question
    Does the model encode parity-odd structure, spiral arm handedness, or has it discarded a real physical observable?

Procedure (from paperScopingV1.md, to implement)
    - paired design: each object against its OWN reflection. A pool-versus-pool comparison measures nothing because handedness is roughly 50/50
    - the isolating operator is a flip about the object's own major axis, which leaves position angle, ellipticity and the whole elliptical envelope unchanged and inverts only chirality
    - difference vector d_i = E(x_i) - E(majorflip(x_i)), antisymmetric by construction
    - is chirality encoded: compare ||d_i|| for spiral-armed objects against smooth/elliptical objects, whose magnitude is the interpolation-noise floor
    - is it a single axis: PCA on the difference vectors; one encoded feature means the d_i lie along +/- c
    - self-supervised label: the sign of the projection onto c is a handedness label at arbitrary sample size
    - optional external validation on whatever handedness catalog overlaps

Nulls
    - the achiral (elliptical) population
    - the same procedure with a rotation instead of a flip, which should produce no antisymmetric structure

Status at scaffold time (2026-07-25)
    NOT STARTED, nothing exists. Two facts to carry: the existing flip in the repo is a plain column mirror, which is exactly the orientation-dominated operator this diagnostic says fails; and no handedness label is held locally, since GZ DESI asks arm tightness and count, both mirror-symmetric, while the clockwise/anticlockwise question belongs to GZ1/GZ2 and carries a documented S-wise reporting bias.

Run
    python paper1/diagnostics/d3Chirality.py

Writes
    paper1/results/d3Chirality.json   (with a provenance block)
    paper1/figures/d3Chirality*.png

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
