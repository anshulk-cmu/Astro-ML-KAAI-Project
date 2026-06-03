## 15. Topology: pieces and holes

The last two sections measured how the manifold is shaped.

Section 7 counted how many independent knobs it uses (intrinsic dimension around 10 to 12). Section 14 asked whether it curls or branches (mostly positive curvature, with weak local branching).

This section asks a coarser but more basic question. Forget the fine shape for a moment. How many separate pieces is the embedding made of, and does it have any holes in it?

Those two counts have names. They are the first two Betti numbers, written $\beta_0$ and $\beta_1$.

They come from topology, the branch of math that studies the properties of a shape that survive any smooth stretching or bending, as long as you do not tear or glue. A coffee mug and a doughnut are the same to topology because each has exactly one hole; you can mould one into the other without cutting. The Betti numbers are the simplest such invariants.

In plain words:

- $\beta_0$ is the number of connected pieces. A single blob has $\beta_0 = 1$. Two blobs that never touch have $\beta_0 = 2$. It counts how many parts the shape falls into.
- $\beta_1$ is the number of independent loops, or holes, in the sense of a ring or the hole of a doughnut. A solid disk has $\beta_1 = 0$ (no hole). A ring, an annulus, a circle drawn on paper has $\beta_1 = 1$. A figure-eight has $\beta_1 = 2$.

Why do we care for a galaxy embedding? Because the answer separates two very different pictures of what the model learned.

If the embedding broke into two clean pieces ($\beta_0 = 2$), that would say the model sees galaxies as belonging to two genuinely disconnected families, with a true gap between them and no objects in the middle. A natural guess would be the red sequence and the blue cloud (Section 2) sitting as two islands.

If instead $\beta_0 = 1$, the model sees one continuous body where you can walk from any galaxy to any other without ever leaving the cloud, even if the density thins out in places.

And $\beta_1$ asks a second question. Does that one body have a ring-shaped hole, some forbidden region in the middle that real galaxies avoid while the cloud wraps all the way around it?

Both questions have crisp, falsifiable answers, and we measured them.

### What persistent homology is

The tool for measuring holes from a finite set of points is persistent homology. The idea is simple to picture.

You have a scatter of points sampled from some unknown shape. Put a ball of radius $r$ around every point. Start with $r = 0$, then grow $r$ slowly.

At $r = 0$ every point is its own island, so you have as many pieces as points. As the balls grow they start to overlap and merge, and the number of pieces drops. At some larger $r$ a ring of overlapping balls can close up and surround an empty middle, which creates a loop. Grow $r$ more and the loop fills in and dies.

Persistent homology records, for every topological feature, the radius at which it is born and the radius at which it dies. The lifetime, death minus birth, is called the persistence.

A feature that is born early and dies late (long persistence) reflects real structure in the shape. A feature that is born and dies almost immediately (short persistence) is the kind of accidental loop you always get from a finite, noisy sample; it is statistical noise, not geometry.

So the rule of persistent homology is plain: long bars are real, short bars are noise. We count only features whose persistence clears a threshold.

It helps to know that short noise loops are not a sign that something went wrong. They are guaranteed. Any finite sample of points, even one drawn from a perfectly solid shape with no holes at all, produces thousands of tiny transient loops as the balls grow, simply because the points are scattered unevenly. The job of persistence is to separate those expected, short-lived loops from the rare long-lived one that would mark a true hole. A method that reported every loop as real would call every dataset a sieve.

This is the honest way to read topology from data, because it never forces a single scale on you. It watches the whole movie of growing balls and keeps the features that survive a wide range of scales.

### $\beta_0$: how many pieces (exact, on the full sample)

For the piece count we do not need subsampling or a persistence threshold. We can compute it directly on all 48,398 galaxies, and we did, two ways, one cross-checking the other.

First, the nearest-neighbour graph. We connected each galaxy to its $k = 15$ nearest neighbours (in the z-scored space, the standardised coordinates defined in Section 4) and asked how many connected components that graph has, meaning how many groups of galaxies you can reach from one another by hopping along edges.

The result is a single number: knn_components $= 1$ (measured). Every galaxy is reachable from every other. There is no second island.

Second, the minimum spanning tree, or MST. The MST is the cheapest possible set of links that still ties all $N$ points into one connected web. Take the $N$ points, and add edges (each edge weighted by the distance between its two endpoints) one at a time, cheapest first, skipping any edge that would make a loop, until everything is joined. The result is a tree (no loops) with $N - 1$ edges and the smallest possible total length.

The MST is useful for $\beta_0$ because of a clean fact. If you cut the single longest edge of the MST, the tree falls into exactly two pieces, and those two pieces are the same two clusters you would get from single-linkage clustering at that distance. Cutting the next-longest edge splits off a third piece, and so on.

So the sorted list of the longest MST edges is a direct readout of how the cloud wants to break apart. The sizes of the resulting pieces tell you whether a cut found a real division or just lopped off a stray point.

This is the test that matters. A real $\beta_0 = 2$ split (two genuine families) would cut the cloud into two large, comparable pieces. An outlier (one weird galaxy, an artefact, a star) would cut off a piece of size one. Those two outcomes look completely different.

Here is what we measured. The longest MST edges have lengths 65.0, 61.3, 60.9, 58.9, and on down. Cutting them in order gives these piece sizes:

| Cuts | Resulting component sizes |
|------|---------------------------|
| 0 | [48,398] |
| 1 | [48,397, 1] |
| 2 | [48,396, 1, 1] |
| 3 | [48,395, 1, 1, 1] |

Read the table. After one cut, the cloud is one giant component of 48,397 galaxies plus a single isolated point. After two cuts, the giant has 48,396 and there are two lone points. After three cuts, 48,395 and three lone points.

Every cut peels off exactly one outlier and leaves the bulk untouched. The giant component never splits into two comparable halves. This is the signature of one connected body with a few far-flung stragglers, not of two families.

Notice the edge lengths themselves. The longest edge is 65.0 and the next ones are 61.3, 60.9, 58.9, with no sudden jump that would mark a true seam between two clusters. The gaps between consecutive sorted edges are small and irregular, which is what you see when the longest links are just bridges to lone outliers rather than the single long span that would join two otherwise-separate populations. There is no one dominant edge whose removal halves the cloud.

Measured result: $\beta_0 = 1$.

We want to be precise about what this does and does not say.

It says there is no clean topological gap that separates galaxies into disconnected groups at the resolution of this sample. It does not say the density is uniform.

There can be (and we expect there are) dense regions and thin regions inside the one body. The red sequence and blue cloud can be two density peaks joined by a sparser bridge, the way two hills are joined by a valley without the land ever stopping. Topology counts pieces, not peaks.

The density story is the diffusion-map story (Section 8), and the two agree: a single connected body with smooth internal variation.

### $\beta_1$: how many holes (subsampled, three metrics)

The loop count is harder to compute, and here we made a deliberate trade-off that the reader should know about.

Full persistent homology of loops ($\beta_1$) on tens of thousands of points is expensive, because the computation grows steeply with the number of points. The honest fix is to work on random subsamples and require any real loop to show up again and again.

So instead of one expensive run, we did 10 independent runs, each on a fresh random subsample of 2,000 galaxies, and asked which loops recur. A genuine ring in the manifold should appear in subsample after subsample. A loop that shows up once and never again is sampling noise.

We also ran each subsample under three different distance metrics, because a real hole should be a hole no matter how you measure distance, while a noise loop is an artefact of one particular metric:

- Euclidean: ordinary straight-line distance in the standardised space (our control).
- Fermat: a density-weighted geodesic, where the distance between two points is the shortest path that prefers to travel through dense regions, which makes it outlier-resistant and is our primary metric for geometry (Section 5). Fermat has a published guarantee that it converges to the right underlying geometry as the sample grows, which is exactly the property you want for a topology claim.
- Diffusion: a distance that averages over all paths through the neighbour graph (Section 8), so it is smooth and density-aware in a different way.

All distances are diameter-normalised, meaning we divide every distance by the largest distance in that subsample so the cloud always has diameter 1. That puts the persistence on a fixed 0-to-1 scale across runs and metrics, which is what lets us set one threshold for everybody.

We call a loop significant if its persistence exceeds 0.1, that is, if it survives across at least a tenth of the full diameter of the cloud. A persistence of 0.1 is already generous; a real doughnut hole would persist over a large fraction of the diameter.

Here is what we measured, averaged over the 10 subsamples per metric:

| Metric | Mean short noise bars per run | Max loop persistence | Mean significant loops ($>0.1$) | Range over 10 runs |
|--------|------------------------------:|---------------------:|--------------------------------:|:------------------:|
| Euclidean | 2,174.8 | 0.105 | 0.1 | 0 to 1 |
| Fermat | 2,516.5 | 0.138 | 0.2 | 0 to 1 |
| Diffusion | 1,094.1 | 0.081 | 0.0 | 0 to 0 |

Read the table carefully, because it has two faces.

The middle column shows thousands of loop bars per subsample (around 2,175 under Euclidean, 2,517 under Fermat, 1,094 under diffusion). That sounds like a lot of holes. It is not. Those are all short bars, born and dead almost at the same radius, exactly the accidental loops a finite point cloud always produces.

The differences between those counts are themselves a small consistency check rather than a signal. Diffusion produces the fewest noise bars (1,094) because it is the smoothest metric: averaging over many paths blurs the fine-grained gaps between points that would otherwise spawn transient loops. Fermat produces the most (2,517) because its density-weighted paths sharpen local structure. None of that bears on whether a real hole exists; it only sets the size of the noise floor each metric carries. The point that matters is the same in all three columns: every one of those thousands of bars is short.

The honest column is the last three. The largest persistence any loop ever reached was 0.105 (Euclidean), 0.138 (Fermat), and 0.081 (diffusion). Two of those three barely clear the 0.1 line, and one (diffusion) never reaches it at all.

And the count of significant loops, averaged over 10 runs, is 0.1, 0.2, and 0.0. In plain numbers: across all 30 runs (3 metrics times 10 subsamples), a loop crossed the threshold a small handful of times at most, never more than once in any single run, and never once in the diffusion metric.

That is the definition of a non-result for holes. A real ring would recur in nearly every subsample and survive every metric. What we see instead is a rare, metric-dependent, never-repeating blip that sits right at the threshold.

We read this as noise that happens to clip the line, not as a hole. Measured result: $\beta_1 = 0$.

### Why two different method choices

The two counts are measured very differently, and the reason is a deliberate honesty choice, not a shortcut.

The piece count $\beta_0$ is cheap. Connected components and a minimum spanning tree both scale gently with the number of points, so we ran them on the full 48,398 galaxies with no approximation. That is why we can state $\beta_0 = 1$ flatly: it is an exact property of the whole sample, confirmed by two independent constructions.

The loop count $\beta_1$ is expensive. The computation of one-dimensional persistent homology grows fast enough that running it on the full sample is impractical, so some form of subsampling is forced on us. The question is how to subsample without fooling ourselves.

Our answer was to demand recurrence and metric-invariance. Ten independent draws of 2,000 points each give ten independent chances for a real loop to appear; three metrics give three independent ways for it to survive. A loop that is genuine should pass nearly all 30 of those checks. A loop that is noise should pass almost none and never repeat. By reporting the recurring-loop count rather than a single run, we turned an unavoidable approximation into a confidence statement: not "there are no loops" but "no loop survives our recurrence-and-metric test, and the strongest candidate barely touches the threshold once". That is the most we can honestly claim at this sample size, and it is what the numbers support.

### The figure

![Figure 15. Topology of the AION-1 manifold: one connected component and no robust loops.](figures/15_topology.png)

Figure 15 shows the two measurements side by side.

**Panel A (left), the piece count $\beta_0$.** The vertical axis lists four states of the cloud, from top to bottom: the uncut nearest-neighbour graph, then the cloud after 1, 2, and 3 of the longest MST edges have been cut. The horizontal axis is component size in number of galaxies, from 0 to about 50,000.

Each row is a stacked horizontal bar. The blue segment is the giant component (its size labelled inside, "giant: 48,398" down to "giant: 48,395"), and the thin red segment on the right is the outliers that each cut peels off, labelled "+1 x size-1 outlier", "+2 x size-1 outliers", "+3 x size-1 outliers". The outlier fragments are drawn at minimum width so you can see them at all; their true size is 1 galaxy each.

What to look for in Panel A: the blue giant bar barely changes length as you go down the rows, while the only thing the cuts add is one more tiny red sliver each time. The cloud is not splitting into two comparable halves; it is shedding single points. That is the visual proof of $\beta_0 = 1$ (measured), one connected continuum, not a red-versus-blue $\beta_0 = 2$ split.

**Panel B (right), the loop count $\beta_1$.** The vertical axis is the maximum loop persistence found in a subsample, diameter-normalised so it runs on the same 0-to-1 scale for every metric (the panel shows roughly 0.06 to 0.18). The horizontal axis is the three metrics: Euclidean, Fermat, diffusion.

Each coloured bar is the single longest-lived loop seen across the 10 subsamples for that metric, with its height labelled on top: 0.105 (Euclidean, blue), 0.138 (Fermat, purple), 0.081 (diffusion, green). The red dashed horizontal line at 0.1 is the significance threshold; any bar reaching above it is a loop we would have to take seriously.

The grey annotation inside each bar records two things: how many short noise bars that metric produced per subsample (around 2,175, 2,517, and 1,094) and the recurring-loop count across the 10 runs (0.1/10, 0.2/10, and 0/10, each in the range 0 to 1).

What to look for in Panel B: even the tallest bar (Fermat, 0.138) is only barely above the line, the diffusion bar sits below it, and the recurring-loop counts are essentially zero everywhere. No loop is both tall and repeatable. That is the visual proof of $\beta_1 = 0$ (measured): the thousands of noise bars never condense into a single hole that survives resampling or a change of metric.

### Seeing the gap directly: the persistence diagrams

Panel B of Figure 15 reported only the single tallest loop per metric. That is a summary, and a careful reader will want the full picture: the whole population of loops, and how cleanly the noise separates from anything that might be real. So we recomputed the loops keeping every birth and death this time, and added a proper statistical band. The result is Figure 19.

![Figure 19. H1 persistence diagrams under the three metrics, with the bootstrap noise band.](figures/19_persistence_diagrams.png)

Start with what is plotted. Each dot is one loop. The top row is the standard persistence diagram: a loop is born at radius $x$ (when its ring of points first closes) and dies at radius $y$ (when the ring fills in), so every dot sits above the 45-degree diagonal, and its distance above the diagonal is its lifetime. The bottom row plots that lifetime straight up the vertical axis against birth, which flattens the diagonal into the floor and makes the noise band horizontal and easy to read. Both rows draw the same three references over the loops: the diagonal (or the floor), the fixed 0.1 threshold from earlier as a red dashed line, and a blue band.

The blue band is the new piece, and it is what makes the call statistical rather than eyeballed. It is a bootstrap 95% confidence band built the standard way (Fasy and Chazal): we resample the 2,000-point cloud with replacement, recompute its diagram, measure how far that diagram moved from the original using the bottleneck distance, and take the 95th percentile of that movement as a half-width $c_n$. A loop has to rise past twice that, a lifetime above $2 c_n$, before we can call it distinguishable from sampling noise. We computed the band on the worst-case subsample for each metric, the one that produced the longest loop, so this is the hardest test for a "no loops" verdict.

Here are the measured bands and the longest loops, in diameter-normalised units (measured): Euclidean band $2 c_n = 0.109$, longest loop 0.105; Fermat band 0.197, longest loop 0.138; diffusion band 0.081, longest loop 0.081. Under every metric the longest loop sits inside its own band. None clears it.

The Fermat panel carries the cleanest answer, and it is the one to watch, because Fermat is our primary metric and also the metric whose strongest loop (0.138) pokes above the crude 0.1 line. That is exactly why the number looked borderline in the summary. The bootstrap band settles it. Fermat's band reaches 0.197, well above 0.138, because a 2,000-point Fermat diagram is noisier and shifts more under resampling, so it takes a longer-lived loop to stand out from its own noise. Measured against the variability of its own metric, the 0.138 loop is comfortably inside the noise. The fixed 0.1 line flagged it; the honest statistical band does not.

The bottom row makes the geometry of "no signal" visible. In each panel the loops form a dense cushion near zero lifetime that thins out smoothly as lifetime grows, and the single longest loop (red ring) sits at the upper edge of that cushion with empty space above it. That empty space is the whole point. A real hole would appear as a dot floating clear of the cushion, separated by a visible gap. There is no such dot in any panel. The longest loop is the tallest blade of grass, not a tree.

So the persistence diagrams confirm $\beta_1 = 0$ a second way, and a firmer way than the summary bars did. The candidate loop fails to recur across subsamples, fails to survive a change of metric, and now we can add that it never rises above the sampling noise of any single metric, including the Fermat metric we trust most. The raw diagrams and the band widths are saved with the other results, so the check is reproducible.

### What it means, and the honest caveats

Put the two numbers together. $\beta_0 = 1$ and $\beta_1 = 0$ describe a simply-connected continuum: one piece, no holes.

Topologically, the AION-1 galaxy embedding is a single solid body, like a filled lump of clay rather than a doughnut, a chain of beads, or two separate stones. You can travel continuously from any galaxy to any other, and you never have to go around an empty ring to do it.

This is not an isolated finding; it is the same conclusion the diffusion map reached by a completely different route (Section 8). There, the eigenvalue spectrum decayed smoothly with no dominant gap, and the cloud had a single connected component.

A spectral gap would have meant discrete clusters; its absence meant one continuous body. Here, persistent homology says one piece and no holes. Two methods, one with a random-walk operator and one with growing balls, agree.

When independent tools agree, the reading is on firmer ground. Both point at a continuum, which matches the physics laid out in Section 2: morphology is a sequence, not a set of boxes, and colour-mass structure is two density peaks joined by a sparse green valley, not two disconnected worlds.

This connection deserves a beat, because it is the single cleanest agreement between geometry and known physics in the whole report. Astronomers have long described galaxies as living on a continuum: the morphological sequence runs smoothly from smooth ellipticals through lenticulars to spirals, and the colour-mass diagram has a thinly populated green valley rather than an empty void between red and blue. If AION-1 had learned a faithful internal map of galaxies, that map should also be one continuous body with no hard breaks. It is. The model was never told this; it learned to fill in masked tokens, with no morphology labels and no instruction to make its space connected. The continuum is an emergent property of the representation, and topology is the tool that confirms it without any reference to the labels.

We are careful not to overstate it. Topology being a continuum is consistent with the physics; it does not prove the model understands the physics, and it cannot tell apart a faithful continuum from any other continuous arrangement of the same points. What it does rule out is the alternative we worried about at the start: that the model carved galaxies into disconnected discrete classes with gaps between them. It did not.

Before the caveats, one point about falsifiability. Both of these tests could have come out the other way, and we would have reported that. A single dominant MST edge whose removal split the cloud into two large halves would have given $\beta_0 = 2$ and a different story entirely. A loop that recurred across most subsamples and survived all three metrics would have given $\beta_1 = 1$ and forced us to ask what physical region galaxies were avoiding. Neither happened. The continuum reading is what the measurements returned, not an assumption we built in.

Now the caveats this section owns, stated plainly.

The $\beta_0$ result is strong. It is exact on the full sample, computed two ways, and the MST cut sizes are unambiguous: the giant stays whole and only single points peel off. We are confident there is no clean two-family split at this resolution.

The $\beta_1$ result is a confidence-set result, not an exhaustive proof. We did not compute loop homology on all 48,398 points at once; we used 10 subsamples of 2,000 and required recurrence across them and across three metrics.

This is the standard and honest way to bound $\beta_1$ at this scale, but it has a real limit. A hole that is small relative to the cloud, or a hole that only appears at a density we under-sample with 2,000 points, could in principle hide below our detection.

So the careful statement is this: we find no evidence of any persistent loop, across three metrics and ten subsamples, and the strongest candidate barely touches the 0.1 line and never repeats. That is good evidence for $\beta_1 = 0$, not a theorem. Our data cannot rule out a hole much finer than the 2,000-point resolution can see.

One more honest point about the metrics. The three metrics disagree slightly on the tallest noise bar (Fermat reaches 0.138, diffusion only 0.081), and that disagreement is itself informative.

If there were a real loop, the three metrics should agree that it is there. They do not agree, which is exactly what you expect when the tallest bar in each case is just the luckiest noise loop for that particular metric. The lack of cross-metric agreement is part of why we read $\beta_1 = 0$ rather than $\beta_1 = 1$.

So the topology is settled in the direction the rest of the report keeps finding: one smooth, simply-connected body.

The next section takes that one body and asks a sharper question within it. If the whole cloud is a single continuum, do different kinds of galaxies, passive and star-forming, still sit on sub-sheets of different intrinsic dimension inside that continuum?
