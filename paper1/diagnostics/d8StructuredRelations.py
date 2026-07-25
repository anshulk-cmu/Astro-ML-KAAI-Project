"""Diagnostic 8 - Structured relations

Tier: descriptive

Science question
    Does the model organize physical relationships the way nature does, preserving ordering, regime-dependence, curvature and conditional structure, rather than merely encoding the marginal quantities?

Procedure (from paperScopingV1.md, to implement)
    - 8a the handle: fit the sequence axis on the featured/smooth vote fraction; report Pearson AND Spearman against the fitted label, its complement, and an independently measured concentration index; carry the redshift-entanglement caveat alongside, not separately
    - 8b the fork: spread along the bar direction should grow from confident ellipticals to confident disks, with BOTH circularity defences (shuffled-vote refits, and directions fit on half the branch evaluated only on held-out disks) and a non-branching control property that must stay flat
    - 8c the shape: bin along the sequence, connect centroids, compare path to chord against a MATCHED null simulating a straight sequence at the real bin spacing and noise; report qualitatively, never as a sigma count
    - the bar axis independence from the sequence axis, scored with Diagnostic 7's machinery but belonging to the fork's structural claims
    - 8d residual: fit a scaling relation in label space, take the residual, ask whether the embedding predicts it
    - 8d conditional: does the model know the relation has different slopes in populations it was never told about, for example size-mass differing between ellipticals and disks
    - 8e observation-physics mixtures: inclination-reddening in disks, and the colour-redshift track
    - INSTANCE GATE: before running any 8d or 8e instance, verify the relation is visible in LABEL space with the sample actually in hand. Stated in advance so that dropping an instance is distinguishable from selecting the ones that worked

Nulls
    - shuffled labels for EVERY fitted direction, not only the bar direction
    - matched noise null for the curvature test
    - held-out evaluation for the fork test
    - a non-branching control property for 8b

Status at scaffold time (2026-07-25)
    PARTIAL. 8a to 8c are essentially complete with both circularity defences and a matched curvature null. 8d, 8e and the instance gate itself do not exist. Feasibility already checked: size-mass is constructible from catalog shape radii plus either mass catalog, and inclination-colour from the shape catalog plus magnitudes; the joint age and metallicity sample caps any stellar-population instance at about eleven percent of the anchor.

Run
    python paper1/diagnostics/d8StructuredRelations.py

Writes
    paper1/results/d8StructuredRelations.json   (with a provenance block)
    paper1/figures/d8StructuredRelations*.png

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
