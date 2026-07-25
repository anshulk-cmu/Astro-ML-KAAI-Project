"""Diagnostic 7 - Concept geometry under a calibrated null

Tier: descriptive+ (methods contribution)

Science question
    Do the directions encoding distinct physical properties sit further apart than the properties' own correlations require?

Procedure (from paperScopingV1.md, to implement)
    - 7a the statistic: angle between two unit ridge directions at fixed regularization, against arccos of the labels' correlation; the difference is the excess
    - 7b why it needs a null: the identity holds only in an idealized whitened, unregularized, infinite-sample setting; the covariance is anisotropic, ridge shrinkage rotates directions, and finite-sample noise biases estimated directions toward 90 deg, worst for weak or sparse labels
    - 7c the null: per pair, draw directions in whitened space with separation exactly arccos(rho), map back through the inverse square root of the covariance, add noise tuned so a probe recovers each at the OBSERVED held-out R2 of the real label, subsample to match each label's coverage pattern, run the identical downstream pipeline, repeat 500 to 1000 times
    - 7d sanity anchors: a label against itself on two disjoint halves, a label against a monotone transform of itself, the same physical quantity from two independent catalogs; any significant excess there means the pipeline is broken
    - 7e application: full pairwise sweep with per-pair p-values, FDR-controlled
    - 7f concept arithmetic: whether vector composition of concept directions lands where the corresponding population sits
    - robustness column: angles computed directly in a whitened metric with shrinkage

Nulls
    - the calibrated generative null of 7c, which is the point of the diagnostic
    - the sanity anchors of 7d, which gate whether any 7c output can be trusted

Status at scaffold time (2026-07-25)
    PARTIAL. The statistic exists on 44 pairs with two interval constructions, and the reasons it needs calibrating are already written into the code and the old report. Everything from 7c onward is new. Until 7c runs, every disentanglement claim in the project stays descriptive. Stated assumption to carry: the null assumes the model's linearly-decodable structure is adequately described by a linear-Gaussian generative model in the embedding.

Run
    python paper1/diagnostics/d7ConceptGeometry.py

Writes
    paper1/results/d7ConceptGeometry.json   (with a provenance block)
    paper1/figures/d7ConceptGeometry*.png

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
