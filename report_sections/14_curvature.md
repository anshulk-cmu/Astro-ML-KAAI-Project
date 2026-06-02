## 14. Curvature and tree-likeness

We now ask a different question about the shape of the embedding. Section 6 and Section 7 (intrinsic dimension) counted how many independent knobs the manifold uses, about 10 to 12. Section 8 (diffusion maps) showed the cloud is one continuous body with no spectral gap. But knowing the dimension and the connectivity does not tell you the *bend* of the surface. A flat sheet and a sphere can both be two-dimensional. A flat sheet and a branching tree can both be connected. Curvature is the missing piece: it says how the manifold turns as you move across it, and whether it pinches into branch points. This section measures that.

The reason curvature matters here is physical, not just geometric. Section 2 laid out two competing expectations for how galaxies should organise. One is the smooth morphological continuum (ellipticals shade into lenticulars shade into spirals, with no hard wall between them), which would show up as a gently curved continuous surface. The other is the idea of distinct quenching channels (a fast disk-to-spheroid route versus a slow one), which, if the model encoded it as a real bifurcation, would show up as branch points: places where one population splits into two and the geometry locally looks like a fork in a tree. So the curvature measurement is a direct test of whether AION-1's geometry is a continuum, a tree, or a continuum with a few weak forks in it.

### 14.1 What curvature and tree-likeness mean, in plain words

Start with curvature on an ordinary surface, then carry it to a manifold of points.

**Positive curvature** is the curvature of a sphere or the top of a hill. If you stand at a point and walk out in every direction, the surface curves the same way under your feet, and the area around you closes in on itself. In a point cloud, positive curvature shows up as *clustering*: the neighbours of a point are also neighbours of each other, so a small ball around the point is densely interconnected. Triangles close up. Local neighbourhoods look like tight, well-connected blobs.

**Negative curvature** is the curvature of a saddle, a mountain pass, or a Pringle chip. Walk out in one direction and the surface falls away; walk out at right angles and it rises. Space spreads apart faster than flat space does, so there is "more room" as you move out. In a point cloud, negative curvature shows up as a *bridge* or *bottleneck*: a point whose neighbours fan out into two or more groups that are not neighbours of each other. The point sits at a saddle between regions. If you imagine two dense blobs joined by a thin neck, the points in the neck have negative curvature. This is exactly what a branch point in a tree looks like locally.

**Zero curvature** is flat space, like a plane or a sheet of paper rolled into a cylinder (a cylinder is flat in the intrinsic sense, because you can unroll it without stretching).

**Tree-likeness** is a global cousin of negative curvature. A tree is a graph with no loops, where every pair of points is joined by exactly one path, and the whole thing branches outward from a root. Trees are the extreme case of negative curvature everywhere: at every junction the space forks, and distances grow the way they do in a saddle. The question "is this manifold tree-like?" is the question "does it branch repeatedly, the way an evolutionary tree or a river delta branches, rather than forming a single connected sheet?" A continuum is the opposite of a tree: it is one body you can slide across smoothly, with neighbourhoods that reconnect rather than fork.

We measure two things, at two scales. A **global** tree-likeness score (delta-hyperbolicity) asks whether the whole cloud, taken together, branches like a tree. A **local** signed curvature (Ollivier-Ricci) asks, edge by edge, whether each small neighbourhood clusters (positive) or bridges (negative). They answer different questions, and we report both.

### 14.2 Delta-hyperbolicity: a global tree-likeness score

The global measure is **Gromov's four-point delta**, also called delta-hyperbolicity. The intuition: in a tree, distances satisfy a very strict relationship, and the more a space behaves like a tree, the closer it comes to satisfying that relationship exactly. Delta measures how far the space is from perfect tree behaviour. **Lower delta means more tree-like. Delta = 0 means a perfect tree.**

Here is the rule it checks. Pick any four points $w, x, y, z$. You can pair them up three ways and add the distances within each pairing:

$$S_1 = d(w,x) + d(y,z), \quad S_2 = d(w,y) + d(x,z), \quad S_3 = d(w,z) + d(x,y).$$

Sort these three sums so that the largest is $L$ and the second-largest is $M$. The four-point delta for this quadruple is half the gap between them:

$$\delta = \frac{L - M}{2}.$$

In a perfect tree, the two largest of those three sums are always equal, so $L = M$ and $\delta = 0$ for every quadruple. The more the largest sum exceeds the second, the further the four points are from sitting on a tree, and the larger $\delta$. We compute this for a huge number of random quadruples (one million of them, sampled with a fixed seed) and take the median over all of them. We divide by the diameter of the cloud (the largest pairwise distance) so the number is dimensionless and comparable across datasets of different scale. So the reported $\delta$ is a *diameter-normalised median four-point delta*: a pure shape number between 0 (perfect tree) and roughly 0.25 (the value you get from a maximally non-tree-like, sphere-like arrangement).

All of this runs on the z-scored embedding under Euclidean distance, with cosine distance reported as a secondary check.

#### Why the absolute number is useless without anchors

Here is the honest catch, and it is the reason this measure is built around comparison rather than a single readout. A raw delta of, say, 0.0136 means nothing on its own. Is that tree-like? Compared to what? Any finite cloud of points in high dimension has *some* nonzero delta just from sampling and from the geometry of high-dimensional space, even if there is no real branching at all. The number only becomes interpretable when you bracket it between two reference clouds whose tree-likeness you already know. We use two anchors, both computed with the identical procedure (same quadruple count, same normalisation, same distance):

1. A **matched-covariance Gaussian cloud**. We draw points from a multivariate normal distribution with the same mean and covariance as the real embedding, the same number of points. This is "what the delta would be if the cloud had AION's overall spread and correlations but *no* real branching structure, just a smooth random blob." It is the null: a featureless ellipsoidal cloud. Its median delta is **0.0177** (measured).

2. A **synthetic tree**. We build a point cloud that genuinely branches like a tree and run the same measurement. This is the floor: "what does a real tree score under this estimator at this sample size?" Its median delta is **0.0079** (measured).

These two anchors turn the abstract delta into a ruler. Below the Gaussian (0.0177) means "more tree-like than a structureless matched blob." Down near the tree (0.0079) would mean "genuinely tree-like." The real embedding's number lives somewhere on that ruler, and where it falls is the result.

### 14.3 Ollivier-Ricci curvature: a trustworthy signed local measure

The global delta gives one number for the whole cloud. To find *where* the geometry branches, and to get a curvature whose sign we can trust, we use **Ollivier-Ricci curvature** on the neighbour graph.

The idea uses optimal transport, which is worth a plain-words definition. **Optimal transport** asks: if you have a pile of sand shaped one way and you want to reshape it into a pile shaped another way, what is the least total "sand times distance moved" to get from the first shape to the second? That minimum total cost is the **earth-mover distance** (also called the Wasserstein distance) between the two shapes. It is a way of measuring how far apart two *distributions* (two clouds of mass) are, accounting for how far each bit of mass has to travel.

Ollivier-Ricci applies this to a graph edge. Take two points $A$ and $B$ that are linked by an edge. Around $A$, spread a little probability mass over $A$ and its nearest neighbours; do the same around $B$. (We keep a fraction $\alpha = 0.5$ of the mass on the centre point and spread the rest over its $k = 10$ neighbours.) Now ask: what is the earth-mover distance $W$ between $A$'s neighbour cloud and $B$'s neighbour cloud, compared to the plain distance $d(A,B)$ between the two centres? The Ollivier-Ricci curvature of the edge is

$$\kappa(A, B) = 1 - \frac{W\big(m_A, m_B\big)}{d(A, B)},$$

where $m_A$ and $m_B$ are the neighbour distributions around $A$ and $B$, and $W$ is the earth-mover distance between them.

The sign of $\kappa$ is what we read, and it is meaningful:

- **Positive $\kappa$ (clustered).** It costs *less* to move $A$'s neighbourhood onto $B$'s than the centre distance, because the two neighbourhoods overlap heavily. The neighbours of $A$ are also near the neighbours of $B$. This is the clustered, positively curved case: a tight, well-connected local blob. Triangles close up, mass is shared.

- **Negative $\kappa$ (bridge).** It costs *more* to move $A$'s neighbourhood onto $B$'s than the centre distance, because the two neighbourhoods point away from each other into different regions. The edge spans a bottleneck. The neighbours fan apart. This is the saddle, the branch point, the bridge between two otherwise separate parts of the cloud.

So Ollivier-Ricci gives us a signed number per edge, and its sign directly answers "is this spot clustered or is it a bridge?" We compute it on a 2,000-point subsample (the optimal-transport solve is expensive) with $k = 10$ neighbours, and summarise the distribution of edge curvatures: the mean, the fraction negative, and the 5th and 95th percentiles. This is the *trustworthy signed measure* in our battery.

#### Why Forman-Ricci is only a ranking tool, not a sign

There is a second, much cheaper curvature called **Forman-Ricci** that we computed on the full neighbour graph (k = 15, augmented with triangle counts). We mention it only to be complete and to say plainly why we do *not* read its sign. Forman-Ricci is a combinatorial formula that mostly counts node degrees (how many edges meet at each end of an edge). On a k-nearest-neighbour graph, every node has roughly the same high degree by construction, and that degree term dominates the formula. The result is that Forman-Ricci comes out negative on essentially every edge no matter what the real geometry is: in our run, 99.7% of edges are "negative" with a mean of -32.5 (measured). That is a structural artefact of the kNN graph, not a finding about the manifold. So we use Forman-Ricci for one narrow job only: as a fast score to *rank* which edges are the most bridge-like candidates, to be confirmed by Ollivier. We never claim "the manifold is negatively curved" from Forman. Ollivier carries all the sign interpretation.

To keep the three measures and their jobs straight:

| Measure | Scale | What it answers | Sign trustworthy? | Role here |
|---|---|---|---|---|
| Gromov delta-hyperbolicity | global | Is the whole cloud tree-like? | n/a (lower = more tree-like) | headline, read against anchors |
| Ollivier-Ricci | local, per edge | Clustered or bridge, and where? | yes | trustworthy signed measure |
| Forman-Ricci | local, per edge | (degree-dominated) | no | rank bridge candidates only |

### 14.4 What the numbers say

Figure 14 shows both measures.

![Figure 14. Curvature of the AION-1 galaxy embedding: delta-hyperbolicity against anchors, and Ollivier-Ricci local curvature.](figures/14_curvature.png)

Figure 14. Curvature of the AION-1 galaxy-embedding geometry. **Panel A (left): Gromov delta-hyperbolicity, lower = more tree-like.** The vertical axis is the diameter-normalised median four-point delta (dimensionless, 0 = a perfect tree, larger = less tree-like). Each bar is one cloud. The solid dark portion of each bar is the median delta (the headline statistic), printed above the bar; the lighter portion stacked on top shows the 95th-percentile upper spread of the per-quadruple deltas, so the full bar height conveys how heavy the upper tail is. The horizontal dashed line marks the Gaussian-anchor median, 0.0177, which is the null reference. Left to right: the synthetic tree (validation floor) at 0.0079; AION E_full under Euclidean distance at 0.0136; AION E_img (image-only) under Euclidean at 0.0138; the matched-covariance Gaussian anchor at 0.0177; and AION E_full under cosine distance at 0.0273. What to look for: both AION Euclidean bars (0.0136, 0.0138) sit just *below* the dashed Gaussian line and well *above* the synthetic-tree bar. Measured fact: AION is slightly more tree-like than a structureless matched random cloud, but nowhere near a real tree. The cosine bar (0.0273) is higher because cosine distance reshapes the cloud; we report Euclidean as primary. **Panel B (right): Ollivier-Ricci edge curvature on a 2,000-point subsample with k = 10.** The vertical axis is the signed edge curvature $\kappa$ (dimensionless; positive = locally clustered, negative = a bridge/saddle). The filled circle is the mean over edges, +0.155, with a whisker spanning the 5th percentile (+0.006) to the 95th percentile (+0.319) of the per-edge distribution. The horizontal red line at $\kappa = 0$ is the sign boundary: above it is positive (clustered), below it is negative (bridge). The annotation reports that 4.2% of edges fall below the red line (the negative-curvature bridge edges). What to look for: the mean and almost the entire 5th-to-95th-percentile band sit *above* the red zero line. Measured fact: local curvature is overwhelmingly positive (clustered neighbourhoods), with only a small minority of edges (about one in twenty-four) being negative bridges.

Read the two panels together.

**Globally (Panel A), the embedding is barely tree-like.** AION's diameter-normalised median delta is **0.0136** for the full multimodal embedding E_full and **0.0138** for the image-only embedding E_img, both under Euclidean distance (measured). The two embeddings agree to within 1.5%, which is reassuring: the global shape does not depend on whether redshift and flux were fed in as inputs. Both numbers fall just below the matched-covariance Gaussian anchor at **0.0177** (measured), so AION is *mildly* more tree-like than a featureless random cloud with the same spread and correlations. But both numbers are nearly double the synthetic-tree floor at **0.0079** (measured). On the ruler from "structureless blob" (0.0177) to "real tree" (0.0079), AION sits close to the blob end and far from the tree end. The honest reading: there is a faint global tree tendency, just enough to clear the random null, and nothing like the signature of an actual branching tree.

**Locally (Panel B), the embedding is mostly clustered, with a few bridges.** The Ollivier-Ricci mean edge curvature is **+0.155** (measured), firmly positive. The middle of the distribution, from the 5th percentile (**+0.006**) to the 95th percentile (**+0.319**), stays on the positive side of zero (measured). Only **4.2%** of edges have negative curvature (measured): a small population of genuine bridge edges, the places where the local geometry forks instead of clustering. So the typical neighbourhood is a tight, positively curved blob, and branching is the exception, concentrated in a thin set of bridge edges rather than spread everywhere.

### 14.5 What this means, and the honest caveats

Putting the global and local pictures together gives a clean, consistent statement, and we are careful about which parts are measured and which are interpreted.

**Measured.** Median delta-hyperbolicity 0.0136 (E_full) and 0.0138 (E_img), below the Gaussian anchor 0.0177 and above the tree floor 0.0079. Ollivier-Ricci mean +0.155, with 4.2% of edges negative and the 5th-to-95th-percentile band running +0.006 to +0.319. Forman-Ricci negative on 99.7% of edges (a kNN-graph artefact we do not interpret as curvature).

**Interpreted.** The manifold is mostly *locally positively curved*, meaning neighbourhoods cluster the way they would on a curved continuous surface rather than fanning out the way they would on a tree. Branching is real but **weak and localised**: it lives in the roughly 4% of edges that are negative bridges, not in the bulk. Globally the cloud is only *mildly* more tree-like than a matched random blob and is nowhere near an actual tree. So the one-line verdict for this section: **AION-1's galaxy embedding is a mostly-clustered continuum with weak, localised branching, not a tree.**

This lines up with the physics from Section 2 without overreaching it. A smooth morphological continuum and a smooth colour-mass sequence would produce exactly this: a continuous, positively curved body. The small set of negative-curvature bridge edges is *consistent with* the idea of a few distinct quenching channels meeting at narrow necks, but consistency is not proof. We measured where the geometry forks; we did not test what physical process put a fork there. With only about 4% of edges negative, and with the global delta only barely below the random null, the branching signal is weak enough that we will not hang any strong evolutionary claim on it. The data shows a continuum with hints of forks, and that is exactly as far as we take it.

There is one important caveat that this section owns. The Ollivier-Ricci result is computed on a 2,000-point subsample, because solving an optimal-transport problem at every edge of the full 48,398-point graph is expensive. A subsample can smear out the rarest, sharpest bridges (a real bottleneck that only a handful of galaxies pass through might be undersampled and read as merely flat rather than negative). So the 4.2% negative-edge fraction is best treated as an estimate of the *typical* prevalence of bridges, not a complete census of every branch point. The delta-hyperbolicity numbers, by contrast, use the full sample via a million random quadruples, so the global statement is on firmer ground than the precise local fraction.

Finally, this curvature result connects directly to the intrinsic-dimension finding in Section 7, and the two reinforce each other. There we noted a *small global-minus-local dimension gap*: the nonlinear intrinsic dimension measured at large scale (about 10) sits close to the linear PCA participation ratio (about 11), and we read that small gap as evidence of *weak curvature at manifold scale*. The logic is that strong curvature would make the nonlinear (manifold-following) dimension fall well below the linear (straight-line) one, because a tightly curled surface needs many linear directions to contain it but few intrinsic ones to move along it. A small gap means the surface does not curl much over the scale of the whole cloud. The curvature measurement here is the independent confirmation of that same conclusion from a completely different method: mostly positive but *gentle* local curvature (mean +0.155, not a large value), a barely-tree-like global delta, and only a thin set of negative bridges. Two unrelated estimators, the ID gap and the direct curvature, both say the manifold is curved but only weakly so. That is the kind of cross-method agreement that turns a single measurement into a finding.

The next section moves from how the manifold *bends* to whether it has *holes*: the topology, the pieces and loops, which closes the geometric description of the cloud.
