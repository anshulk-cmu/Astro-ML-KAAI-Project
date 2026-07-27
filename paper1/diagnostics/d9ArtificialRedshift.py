"""Diagnostic 9 - Artificial redshifting

Tier: causal

Science question
    When an object is transformed to LOOK like it sits at a higher redshift, does the embedding move along the empirically derived redshift direction, by the right magnitude, and without disturbing intrinsic properties?

Procedure (from paperScopingV1.md, to implement)
    - apply an established artificial-redshifting transformation (FERENGI-style, Barden, Jahnke and Haeussler 2008) to low-redshift objects, targeting a higher redshift
    - 1 angular size: rescale by the angular diameter distance ratio
    - 2 flux: dim by the luminosity distance ratio
    - 3 surface brightness: the (1+z)^-4 dimming follows automatically from 1 and 2, no separate step
    - 4 k-correction: the only genuinely hard step, since it requires an SED; band interpolation suffices for modest steps
    - 5 PSF: reconvolve to target seeing
    - 6 noise: add to match target depth
    - keep the redshift step modest to limit k-correction difficulty and stay within the sample's density
    - direction and magnitude: does the recovered-to-intended slope approach unity
    - selectivity: do intrinsic quantities hold still while observed quantities move
    - on-manifold validation: are transformed objects embedded near REAL objects at that redshift

Carry-forward from Diagnostic 3, standing caveat 6
    Artificial redshifting rescales angular size, so it RESAMPLES. Diagnostic 3 measured the
    encoder's response to resampling directly and found it large: four bilinear rotations that
    return an image almost exactly to itself still move the embedding by 23.3 per cent of the
    distance between two unrelated galaxies. A design that compares a redshifted condition
    against a non-redshifted one on whole-vector distances will be dominated by that, not by
    redshift. Read targeted probes, or compare conditions that carry the same resampling.

Nulls
    - matched-norm random directions
    - the untransformed baseline
    - real objects at the target redshift as the reference population

Status at scaffold time (2026-07-25)
    NOT STARTED. The existing embedding-level translation test adds a mean shift vector and inspects neighbours; the artifact itself states it is not an input-space intervention, and this diagnostic is what upgrades it. One project-specific advantage worth using: the anchor has spectra for about a third of its galaxies, which can supply real SEDs for the k-correction instead of templates.

Run
    python paper1/diagnostics/d9ArtificialRedshift.py

Writes
    paper1/results/d9ArtificialRedshift.json   (with a provenance block)
    paper1/figures/d9ArtificialRedshift*.png

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
