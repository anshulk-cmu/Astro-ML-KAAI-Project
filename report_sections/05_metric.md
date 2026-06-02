## 5. The metric problem and the concentration diagnostic

Every geometric result in this report rests on one earlier choice: how do we measure the distance between two galaxies in the embedding? It sounds like a detail. It is not. Intrinsic dimension, curvature, topology, and the shape of the diffusion map are all built on a notion of "near" and "far", and that notion comes from a distance function. Pick the wrong one and you can manufacture structure that is not there, or wash out structure that is. So before we trust any downstream number, we have to ask whether the obvious choice (plain straight-line distance in 1024 dimensions) is even meaningful here, and if not, what to use instead. This section sets up that decision and reports the diagnostic we ran to make it.

### 5.1 Why straight-line distance gets unreliable in high dimensions

Start with the plain picture. Each galaxy is a point in a space with $d = 1024$ axes (the ambient dimension defined in Section 3). The most natural distance between two points $x$ and $y$ is the Euclidean distance, the length of the straight line between them:

$$d_{\mathrm{euc}}(x,y) = \sqrt{\sum_{j=1}^{d} (x_j - y_j)^2}.$$

Here $x_j$ is the value of point $x$ on axis $j$, and the sum runs over all 1024 axes. This is the ruler you used in school, extended to many axes. In two or three dimensions it behaves exactly as your intuition expects. In hundreds of dimensions it stops behaving.

The problem has a name: distance concentration. As the number of dimensions grows, the distances between points tend to bunch up around a single value. The nearest point and the farthest point end up almost the same distance away. Put differently, the contrast between "close" and "far" shrinks toward zero. In the limit of very high dimension the ratio of the farthest distance to the nearest distance approaches one, so "nearest neighbour" loses its meaning: everything is roughly equidistant from everything else.

The reason is a counting effect, not anything exotic. Euclidean distance squared is a sum of 1024 per-axis terms. Each term is one small contribution. When you add up many independent-ish small contributions, the total behaves like an average: its typical size grows, but its spread relative to that size shrinks. By a law-of-large-numbers argument, the relative spread of the squared distance falls off like roughly $1/\sqrt{d}$. With $d = 1024$ that factor is about $1/32$, so the fractional differences between pairwise distances are squeezed into a narrow band. The absolute distances are large; what collapses is the *contrast* between them. And contrast is exactly what a nearest-neighbour graph, an intrinsic-dimension estimator, or a diffusion walk feeds on. If every point is about as far from every other point, the local neighbourhood structure those methods rely on is built on numerical noise.

Two facts about our data make this worry concrete rather than theoretical. First, the ambient dimension really is 1024, deep in the regime where concentration bites. Second (from Section 4 and `_FACTS.md`), the embedding vectors have tightly clustered norms (about 48 for the multimodal set $E_{\mathrm{full}}$ and about 49.6 for the image-only set $E_{\mathrm{img}}$, roughly a 5 percent spread), so the whole cloud sits close to a thin spherical shell. Points on a thin shell are a textbook setting for distance concentration, because a large part of every pairwise distance is just the common shell radius. We standardised each of the 1024 dimensions to zero mean and unit variance (z-scoring, Section 4) before any of this, which fixes the unrelated problem of a few high-variance axes dominating the ruler, but z-scoring does not undo concentration. So we measured it directly.

### 5.2 Two numbers that quantify concentration

To turn "are distances bunched up?" into something we can plot, we computed two summary statistics on the full table of pairwise distances. Both were run on a 2,000-point random subsample of the 48,398 galaxies (the same subsample for every metric, drawn with a fixed seed), z-scored first, using a $k = 15$ nearest-neighbour graph where a graph is needed. The subsample keeps the all-pairs distance table (about two million pairs) tractable while staying large enough for stable summaries.

The first statistic is the relative distance ratio, which we write RDR:

$$\mathrm{RDR} = \frac{D_{\max} - D_{\min}}{D_{\min}},$$

where $D_{\max}$ and $D_{\min}$ are the largest and smallest distances over all pairs of distinct points. Read it as: how many times bigger is the spread of distances than the smallest distance? A small RDR means the biggest and smallest distances are nearly equal, so the cloud is strongly concentrated and contrast is low. A large RDR means there is real range between the closest pair and the farthest pair, so contrast is high and "near" means something. Higher RDR is better for everything we do downstream. (One caveat that belongs to RDR: it is built from the single most extreme pair on each end, so it is sensitive to one outlier distance. We read it alongside the second statistic, which uses a typical-pair quantity, not the extremes.)

The second statistic is the nearest-neighbour-to-mean ratio, NN/mean:

$$\mathrm{NN/mean} = \frac{\overline{d_{\mathrm{NN}}}}{\overline{d_{\mathrm{pair}}}},$$

where $\overline{d_{\mathrm{NN}}}$ is the average distance from a point to its single closest other point, and $\overline{d_{\mathrm{pair}}}$ is the average distance over all pairs. This asks how far the nearest neighbour sits compared to a typical pair. If nearest neighbours are barely closer than random pairs, the ratio climbs toward 1 and local structure is weak (strong concentration). If nearest neighbours are much closer than typical pairs, the ratio is small and local neighbourhoods are well separated from the bulk (weak concentration). Lower NN/mean is better. The two statistics come at the same question from opposite ends (RDR from the global extremes, NN/mean from the local-versus-typical comparison), so when they agree we trust the reading.

In short, the decision rule for both statistics is simple:

1. RDR up, concentration down. A larger RDR means the closest and farthest pairs differ more, so contrast is healthy.
2. NN/mean down, concentration down. A smaller NN/mean means nearest neighbours stand out from the bulk, so local structure is healthy.

When both point the same way, we have a clear reading; when they conflict, we treat the metric as borderline.

We also recorded, for each metric, how many pairwise distances came out infinite. Infinite distances appear when a graph-based metric leaves the cloud split into pieces with no path between them, which would make that metric unusable for global geometry. For every metric and both embedding sets the infinite count was zero (`inf = 0` throughout `results/metric.json`): all graphs were connected, so the distance tables are complete and comparable.

### 5.3 The honest-metric battery: five rulers, not one

Rather than argue from theory about which distance to trust, we measured five of them side by side on the same points. Call it the honest-metric battery. Each metric encodes a different assumption about what "distance along the data" should mean, and seeing all five at once tells us which assumptions buy contrast and which do not.

**Euclidean (the control).** The straight-line distance defined above, computed directly from the z-scored coordinates. This is the baseline we are worried about. It ignores the shape of the data entirely: it measures the distance through empty space, as if you could fly straight from one galaxy to another regardless of whether any galaxies lie in between. We keep it as a control precisely because it is the naive choice and the one most exposed to concentration.

**Cosine.** Distance defined as $1 - \cos\theta$, where $\theta$ is the angle between the two points' vectors (each vector normalised to unit length first). Cosine distance ignores how long the vectors are and cares only about their direction. Because the cloud already sits near a thin shell of nearly equal norms, cosine and Euclidean ought to behave similarly here, and prior expectation is that cosine would concentrate much like Euclidean. We included it to test that expectation, not because we planned to use it downstream.

**Isomap geodesic.** Instead of flying straight through empty space, walk along the data. Build a graph that connects each point to its $k = 15$ nearest neighbours, weight each edge by the local Euclidean distance, and define the distance between any two points as the length of the shortest path through that graph (computed with Dijkstra's algorithm). This is the geodesic distance, the "as the data curves" distance: it follows the manifold the points lie on rather than cutting across regions where no galaxies exist. On a curved sheet, two points on opposite folds can be close in straight-line distance yet far along the sheet, and Isomap captures that.

**Fermat (density-weighted geodesic).** Same shortest-path idea, but the edge weights are raised to a power $p = 2$ before the path search, so an edge of local length $\ell$ contributes $\ell^{2}$. Squaring the weights makes long hops disproportionately expensive, so the cheapest path prefers to take many short steps through dense regions rather than a few long jumps across sparse gaps. This is the Fermat distance (named by analogy with Fermat's principle, that a path takes the cheapest route). Two properties make it attractive for our downstream arms. It is outlier-resistant: a single stray point in a void cannot become a cheap stepping-stone, because the long edges to reach it are penalised. And it has a published convergence guarantee for recovering the topology of the underlying manifold as the sample grows, which matters for the loop-counting in Section 15 (topology). For these reasons it is our candidate primary metric.

**Diffusion distance.** Rather than committing to one shortest path, average over all paths. Turn the neighbour graph into a random walk: from any point, step to nearby points with probability set by a Gaussian falloff of their distance (we set the falloff scale to the median squared edge length of the graph). Two points are close in diffusion distance if a random walker starting at one reaches a similar distribution of places as a walker starting at the other, after the same number of steps. Concretely we form the symmetric normalised affinity matrix, take its top eigenvectors $\psi$ scaled by their eigenvalues, and measure ordinary Euclidean distance in those scaled diffusion coordinates. Averaging over many paths makes diffusion distance smooth and stable to noise in any single edge. (This is a preview of the full diffusion-map construction in Section 8, used here only as one ruler among five.)

### 5.4 What the diagnostic shows

![Figure 3. Pairwise-distance concentration across five metrics.](figures/03_metric_concentration.png)

Figure 3 plots the two concentration statistics for all five metrics, for both embedding sets, on the shared 2,000-point z-scored subsample with $k = 15$. Panel A (left) shows RDR, the relative distance ratio, where higher means more contrast and *less* concentration; its vertical axis is on a logarithmic scale (each gridline is ten times the one below), because the values span more than an order of magnitude. Panel B (right) shows NN/mean, the nearest-neighbour over mean-pair ratio, where lower means *more* concentration; its vertical axis is linear, from 0 to about 0.5. In both panels the five metrics run along the horizontal axis (Euclidean, Cosine, Isomap, Fermat with $p=2$, Diffusion), and for each metric there are two bars: blue is $E_{\mathrm{full}}$ (image plus photometry plus redshift) and orange is $E_{\mathrm{img}}$ (image only). The number printed above each bar is its value. The caption strip under the figure restates the reading. There are no reference lines; the comparison is bar-to-bar.

The full set of measured values is in the table below (read directly from `results/metric.json`), so the figure and the prose can be checked against the same numbers.

| Metric | RDR, $E_{\mathrm{full}}$ | RDR, $E_{\mathrm{img}}$ | NN/mean, $E_{\mathrm{full}}$ | NN/mean, $E_{\mathrm{img}}$ |
| --- | --- | --- | --- | --- |
| Euclidean (control) | 14.69 | 14.16 | 0.433 | 0.430 |
| Cosine | 76.80 | 79.36 | 0.193 | 0.193 |
| Isomap (geodesic) | 42.12 | 44.46 | 0.179 | 0.178 |
| Fermat ($p=2$) | 281.04 | 259.14 | 0.164 | 0.164 |
| Diffusion | 95.01 | 99.16 | 0.269 | 0.267 |

All values are measured. Higher RDR means more contrast (less concentration); lower NN/mean means more concentration. Every graph-based metric had zero infinite distances, so all five rulers describe one connected cloud.

The first thing to look for is that the blue and orange bars are nearly the same height everywhere. The two embedding sets concentrate almost identically, so the metric behaviour is a property of the geometry, not an artefact of which inputs the model saw. The numbers in the table make this concrete: for every metric the $E_{\mathrm{full}}$ and $E_{\mathrm{img}}$ entries agree to within a few percent, and the NN/mean column is nearly identical between the two sets. Whatever the metric is reacting to, both embeddings share it.

The second thing to look for is the ranking. Raw Euclidean distance is the most concentrated metric by a wide margin: it has the lowest RDR (about 14 to 15, so the largest pairwise distance is only about fifteen times the spread above the smallest) and the highest NN/mean (about 0.43, so a point's nearest neighbour sits at nearly half the average pairwise distance, barely closer than a random galaxy). That is the concentration we feared, measured. The intrinsic metrics that walk along the data give several times more contrast. Fermat is the standout: RDR around 260 to 281, roughly eighteen to twenty times the Euclidean contrast, and the lowest NN/mean at about 0.164, meaning a point's nearest neighbour is on average about six times closer than a typical pair. Isomap (RDR around 42 to 44) and diffusion (around 95 to 99) sit in between, both clearly better than Euclidean. This is the headline of the section: following the geometry instead of cutting across it restores the contrast that high dimension had erased.

Now the honest deviation from prior expectation. We expected cosine to concentrate like Euclidean, because the cloud sits near a thin shell of nearly equal norms, and on such a shell direction and position carry almost the same information. It did not. Cosine RDR is about 77 to 79, far above Euclidean's 14 to 15, and its NN/mean (0.193) is much lower than Euclidean's (0.43). So cosine pulls apart the distance distribution more than the straight-line ruler does, even on this near-spherical cloud. We do not have a clean single-cause explanation for why; the shell is not perfectly uniform, and small angular differences evidently carry more contrast than the small radial differences that Euclidean folds in. The faithful reading is narrow: cosine is a middling control here, more contrastive than Euclidean but less so than the geodesic metrics, and we treat it as one more reference point rather than as a failure of the prior or as a recommended downstream metric. We do not build any downstream arm on cosine.

One caveat keeps these readings honest. More contrast is necessary, not sufficient. A metric can spread distances apart for the wrong reason, for instance by over-weighting a handful of long edges in a sparse region, and a high RDR alone cannot tell good contrast from inflated contrast. That is exactly why we read RDR and NN/mean together (the extremes and the local-versus-typical view agree on the ranking here) and why we favour Fermat on grounds beyond its raw numbers: its outlier resistance and its topology-convergence guarantee mean its contrast comes from following dense structure, not from rewarding stray points. The diagnostic ranks the metrics; the theory tells us which high-contrast metric to trust.

### 5.5 The decision this fixes for the rest of the report

The concentration diagnostic settles a choice that every later section inherits. Plain Euclidean distance in this 1024-dimensional cloud is the most concentrated of the five rulers, with the least contrast between near and far, so taking it at face value would push the downstream methods toward the regime where neighbourhoods are unreliable. We therefore adopt Fermat (the density-weighted geodesic, $p = 2$) as the primary metric for the topology and curvature arms, because it delivers the most contrast (RDR around 260 to 281) and carries the resistance-to-outliers and topology-convergence properties those arms need. We keep Euclidean as the explicit control throughout, run side by side with Fermat, so that any geometric claim can be checked against the naive ruler and we can see when a result depends on the metric and when it does not.

The roles each metric plays from here on are fixed as follows.

- Fermat ($p=2$): primary metric for topology (Section 15) and curvature (Section 14). Most contrast, outlier-resistant, topology-convergence guarantee.
- Euclidean: the control, run beside Fermat everywhere so metric-dependence is visible.
- Diffusion: cross-check here, and the basis of the coordinate system in Section 8 (diffusion maps), where the same walk-along-the-data idea becomes the axes we read physics off.
- Isomap: a secondary geodesic cross-check; it confirms the contrast gain over Euclidean without the density weighting.
- Cosine: a measured-but-unused reference; more contrastive than Euclidean but not built on downstream.

Two limits bound what this section can claim, and they are worth stating plainly. The diagnostic ran on a single 2,000-point subsample, so the exact RDR and NN/mean values carry sampling uncertainty we did not bootstrap here; the ranking is large and stable across both embedding sets, but the specific numbers should be read as one draw, not as tight estimates. And the diagnostic measures contrast, not correctness: it tells us which metric resists concentration, not which metric reflects the true manifold geometry, a question no single number can answer. With that understood, we have a defensible distance to work in. The next two sections use it: Section 6 defines intrinsic dimension and its estimators, and Section 7 reports how many independent directions this concentrated, shell-like cloud actually uses.
