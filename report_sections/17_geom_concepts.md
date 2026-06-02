## 17. The geometry of concepts: an honest null

Every other arm of this report measured something positive. This one did not, and we report it anyway, because a negative result you actually ran is more informative than a positive result you wished into existence.

The question is about how concepts sit inside the embedding. Throughout the report we have treated a "concept" as a direction in the 1024-dimensional space: a single unit vector such that moving along it changes one human-named property (Section 10 found these as ridge-probe weight vectors, Section 13 as sparse-autoencoder decoder vectors). A natural next idea, from Park et al. and the broader work on the linear geometry of concepts in large models, is that concepts might be organised in a hierarchy that you can read off the directions themselves.

### The hierarchy idea, in plain words

Some concepts are children of others. "Spiral" is a kind of "featured" galaxy: a galaxy cannot be spiral without first being featured (having visible structure rather than being a smooth blob). Featured is the parent, spiral is the child. The Park-style claim is that this parent-child relationship should show up geometrically in a specific way: the child's direction should equal the parent's direction plus an extra piece that is orthogonal to it and carries only the child-specific information.

Write $\mathbf{u}_{\text{featured}}$ for the parent direction and $\mathbf{u}_{\text{spiral}}$ for the child direction, each a unit vector. The clean-hierarchy prediction is

$$\mathbf{u}_{\text{spiral}} = \alpha\,\mathbf{u}_{\text{featured}} + \mathbf{r},\qquad \mathbf{r}\perp\mathbf{u}_{\text{featured}},$$

where $\alpha$ is some positive weight and $\mathbf{r}$ is the spiral-specific residual that is at right angles to the featured direction. In words: to become "spiral" you first move in the "featured" direction (because spirals are featured), then add a sideways step that is pure spiral and has nothing to do with featuredness. If this holds, the embedding has literally encoded the taxonomy in its geometry: the child is built out of the parent plus an independent correction.

The thing you measure to test this is the alignment between the parent direction and the spiral residual, written as the cosine of the angle between them. Cosine of an angle between two unit vectors runs from $-1$ (pointing opposite) through $0$ (exactly perpendicular) to $+1$ (pointing the same way). Under a clean hierarchy you want the residual, after you have already accounted for the parent, to still carry a leftover trace of the parent that you can detect, which shows up as a small but real positive cosine between the featured direction and the parent-component recovered from the spiral direction. A cosine indistinguishable from what you would get by chance means the geometry does not encode the hierarchy.

### Fixing a test that was cheating

Our first version of this test was tautological, and we corrected it. The earlier construction effectively built the spiral residual out of the featured direction and then measured how much of the featured direction was in it, which is circular: by construction the answer had to be positive, so it proved nothing. We caught this and replaced the baseline.

The fix is a permutation null. A null is the distribution of the test statistic you would see if there were no real effect, and a permutation null builds that distribution by destroying the structure you are testing for while keeping everything else fixed. Here we shuffle the labels that define the concepts (so the "spiral" and "featured" assignments are scrambled relative to the galaxies), recompute the directions and the parent-residual cosine from scratch many times, and collect the cosines those scrambled runs produce. That gives us the range of cosine values consistent with no hierarchy at all. The real, unshuffled cosine is significant only if it lands above almost all of the shuffled values. We use the 95th percentile of the null as the threshold: the real value must beat the cosine that 95% of shuffled runs fall below.

This is the same null logic we used for the sparse-autoencoder alignment thresholds in Section 13, and for the same reason: a test of "is this direction real" is only honest against a baseline of "what would a fake direction score".

### The result

The measured alignment is

$$\cos\big(\mathbf{u}_{\text{featured}},\,\text{spiral-residual}\big) = 0.072,$$

and the permutation null's 95th percentile is $0.084$ (both measured). Since $0.072 < 0.084$, the observed value falls below the threshold. It is not significant. The hierarchy signal we measured is weaker than what shuffled, structure-free data produces 5% of the time by chance.

So the honest finding is a null: our data does not establish a clean linear featured-to-spiral hierarchy in the AION-1 embedding. We do not say the hierarchy is absent. We say we could not detect it, which is a different and weaker claim. The value $0.072$ is not zero, and it is not far below the $0.084$ line, so the test is consistent with a small real effect that we simply lack the resolution to confirm, and it is equally consistent with no effect at all. The data cannot decide between those, and we will not pretend it can.

### Why this null is fine to report, and what limits it

Two reasons this belongs in the report rather than in a drawer. First, negative results matter: a reader deciding whether to build on the linear-concept-hierarchy idea in galaxy embeddings should know that our one direct test of it came back empty. Reporting only the arms that worked would give a falsely tidy picture of a model that is, in most respects, very legible.

Second, the test is genuinely exploratory and its weakness is partly a sample problem we can name. The spiral and bar labels exist only for featured, non-edge-on disk galaxies, because that is where the Galaxy-Zoo decision tree even asks the spiral-arm question (Section 4 explained this conditionality). That restricts the spiral sample to a few thousand galaxies (the spiral-fraction probe in Section 10 ran on $n = 3{,}034$), and the spiral probe itself was the weakest in the whole probe battery ($R^2 = 0.252$). A concept direction estimated from a small, noisy label is itself noisy, and a noisy child direction makes the residual geometry hard to read. With a sharper spiral label on more galaxies, the cosine might rise above the null, or it might not. This test does not settle it.

There is a cleaner way to read this null against the rest of the report. The model clearly carries strong, separable concept directions: colours and morphology decode with high $R^2$ (Section 10), the probe directions are more orthogonal than the label correlations force (Section 11, disentanglement), and the sparse autoencoder finds hundreds of seed-stable, physics-aligned features (Section 13). What this section adds is a boundary on those claims. The model encoding a concept as a usable direction is one thing, and we have strong evidence for it. The model arranging its concepts into a specific linear parent-child algebra is a stronger, more particular claim, and for the one hierarchy we tested, the evidence is not there. Strong concept directions, no demonstrated concept hierarchy. That is the honest line, and we carry it forward as a limit on how far the geometry-of-concepts reading should be pushed.
