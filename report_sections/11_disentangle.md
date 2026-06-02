## 11. Disentanglement and neighbourhood purity

Section 10 (probes) asked how much of each physical property the embedding carries. A high $R^2$ tells us the information is there and readable by a straight-line rule. It does not tell us whether the model keeps two different properties on separate directions, or whether it has piled them onto the same axis and we are simply reading one through the other. This section runs two checks that are light on modelling and aimed at that second question. The first asks whether the model's internal directions for different concepts are more separated than the labels themselves would force. The second, even simpler, asks whether galaxies of one morphological type sit next to galaxies of the same type in the raw 1024-dimensional space, with no probe at all.

Why does separation matter, beyond decodability? Because a representation that smears two properties onto one shared axis has, in a real sense, only learned one of them. If the model encoded "redness" and "smoothness" on the same direction, then every time it moved a galaxy redward it would also move it smoothward, and we could never tell whether it had a concept of morphology at all or just a concept of colour that we were reading twice.

A representation that keeps the two on separate axes can vary one while holding the other fixed. That is the property people mean by "disentangled": independent factors live on independent directions, so the model can describe a red disk and a blue spheroid as easily as the common red spheroid and blue disk. The rare combinations are the test. A representation that has truly separated colour from shape can place a blue elliptical and a red spiral wherever those galaxies actually fall, instead of forcing every red galaxy to look smooth.

AION-1's authors claimed the model "organises objects along physically meaningful directions." The disentanglement test is a direct, if linear, check on the word "directions": are they actually distinct directions, or one direction wearing several names?

### 11.1 What a probe direction is, and what "disentangled" should mean

Recall from Section 10 that a linear probe is a ridge regression: it fits one weight vector $\mathbf{w} \in \mathbb{R}^{1024}$ so that the dot product $\mathbf{w} \cdot \mathbf{x}$ predicts a property of galaxy $\mathbf{x}$. That weight vector has a direction in the embedding space. We will call it the probe direction for that property. It is the single straight line along which the property changes fastest, as the model sees it.

The direction comes from the same fits already reported in Section 10, one ridge probe per label, so no new model is trained here. The weight vector $\mathbf{w}$ is what ridge regression returns; only its orientation matters for this test, so its overall length drops out when we normalise. Because the embedding was z-scored first (each of the 1024 dimensions shifted to mean zero and scaled to unit spread), the directions live in a space where no single dimension dominates by accident, which is what lets us compare angles fairly across labels.

Now take two properties, say redshift (how far away and how stretched in wavelength a galaxy's light is, a stand-in for distance and look-back time) and smooth fraction (the share of Galaxy Zoo voters who called the galaxy a smooth, featureless blob rather than a structured disk). Each gives a probe direction. We measure the angle $\theta$ between the two directions:

$$\theta = \arccos\!\left(\frac{\mathbf{w}_A \cdot \mathbf{w}_B}{\lVert \mathbf{w}_A \rVert \, \lVert \mathbf{w}_B \rVert}\right).$$

Here $\mathbf{w}_A$ and $\mathbf{w}_B$ are the two probe weight vectors, the dot in the numerator is their inner product, and the two norms in the denominator scale the result so it depends only on direction, not on length. The function $\arccos$ turns a cosine back into an angle. The quantity inside the brackets is the cosine similarity of the two directions, a number between $-1$ and $+1$ that is $+1$ when the directions coincide, $0$ when they are perpendicular, and $-1$ when they point exactly opposite. An angle of $0^\circ$ means the two directions are identical (the model encodes both properties on the very same axis). An angle of $90^\circ$ means they are orthogonal, fully independent directions, the cleanest possible separation. An angle above $90^\circ$ means the directions point partly against each other.

It helps to picture what orthogonal directions buy us. If $\mathbf{w}_A$ and $\mathbf{w}_B$ are perpendicular, then moving a galaxy's embedding along $\mathbf{w}_A$ changes property $A$ but leaves the projection onto $\mathbf{w}_B$, and therefore the read-out of property $B$, untouched. The two read-outs do not interfere.

If the directions are nearly parallel ($\theta$ small), then nudging $A$ drags $B$ along with it, and the two probes are effectively reading the same coordinate. So the angle is a clean, geometric statement about how much the model's two read-outs can be set independently.

A bare angle is not enough, because the two properties are themselves correlated in the real Universe. Redder galaxies tend to be smoother and more often passive; that is an astrophysical fact, not a modelling choice. If two labels are correlated, then any honest representation that encodes both will have probe directions that are correlated too, just from the labels. So a small angle does not automatically mean the model failed to separate the concepts. It might only be reflecting a real correlation in the data.

To handle that, we build a null: the angle we would expect from the label correlation alone, with no extra separation. If two standardised labels have Pearson correlation $r$ (Pearson correlation measures how much two quantities move together on a straight line, running from $-1$ to $+1$), then the angle implied by that correlation is

$$\theta_{\text{null}} = \arccos(r).$$

This is the baseline. It is the angle two probe directions would make if the only thing separating them were the labels' own correlation. We then define the excess:

$$\text{excess} = \theta - \theta_{\text{null}}.$$

A positive excess means the model holds the two concepts more apart than the label correlation requires (genuine separation, the directions are more orthogonal than they "have to" be). An excess near zero means the model separates them exactly as much as the labels do, no more and no less (the directions track the label correlation and nothing extra). A negative excess means the model's directions are closer together than the labels are, which would say the model has entangled the two concepts beyond what the data forces. All angles are measured on the leakage-free image-only embedding $E_{\text{img}}$ where it matters, so we are reading inference, not inputs, consistent with Section 10.

The reason the null is built from the labels and not from chance is worth stating once more, because it is the whole point of the test. Two random directions in 1024-dimensional space are almost always close to perpendicular; high-dimensional spaces are roomy, and a "default" angle of nearly $90^\circ$ is what you get from noise alone. So comparing $\theta$ to $90^\circ$ would be the wrong yardstick: it would reward the model for an orthogonality that costs it nothing.

The label-correlation null fixes that. It asks the sharper question: given that these two real properties genuinely move together in the Universe, is the model still keeping their directions further apart than that shared movement demands? Only an excess above that label baseline counts as the model doing extra work to keep the concepts apart.

A note on angles past $90^\circ$, because several pairs show them. An angle above $90^\circ$ means the cosine similarity is negative, so the two probe directions point partly opposite. That is not a problem, and it is not more entangled than orthogonal. For separation, what matters is distance from parallel; both $80^\circ$ and $100^\circ$ are well away from parallel and represent near-independent read-outs. The sign of the cosine only reflects which way each label was oriented (smooth-up versus featured-up, say), which is arbitrary. So we treat $93^\circ$ and $87^\circ$ as equally good separation, and we always judge against the same pair's null, not against the raw $90^\circ$ line.

One honest limit up front. This is a linear, correlational test. It compares the angle between two best-fit linear directions to the angle between two labels. It does not prove the model represents a concept on exactly one axis, and it cannot establish that changing one direction leaves the other untouched. That would need a causal intervention (steering or ablating a direction and watching the other), which we did not run. The excess is suggestive evidence of separation, not a guarantee of it.

### 11.2 Reading the dumbbell plot

![Figure 9. Learned probe-direction angle versus the label-correlation null, for ten label pairs.](figures/09_disentanglement_angles.png)

Figure 9 shows all ten label pairs as a dumbbell plot. The horizontal axis is the angle between directions in degrees, running from about $20^\circ$ to past $100^\circ$. Each row is one pair of labels, named on the left (for example "redshift - smooth"). Within a row there are two dots joined by a bar, which is why it is called a dumbbell.

The grey dot is the null angle $\theta_{\text{null}} = \arccos(r)$, the separation the labels' own correlation would produce. The coloured dot is the measured probe-direction angle $\theta$. The bar between them is the excess, and the number printed beside it is the excess in degrees. The bar and dot are coloured blue when the excess is positive (the model is more disentangled than the null) and red when the excess is at or below zero (no extra separation). The vertical dashed line at $90^\circ$ marks exact orthogonality, the cleanest separation a pair of directions can have. Rows are sorted from the largest positive excess at the top to the most negative at the bottom, so the figure reads as a ranking: the strongest separation is on top, the entangled or undecided pairs at the bottom.

What to look for: the horizontal position of the coloured dot relative to its grey twin. If the blue dot sits to the right of the grey dot, the learned directions are more separated than the labels force. The further right, the more separation. Read the figure together with the table below.

| Label pair | $\theta$ (deg) | $\theta_{\text{null}}$ (deg) | excess (deg) |
|---|---|---|---|
| redshift - smooth | 93.0 | 73.7 | $+19.4$ |
| redshift - r-z | 69.2 | 50.7 | $+18.5$ |
| g-r - r-z | 40.3 | 25.2 | $+15.1$ |
| redshift - g-r | 52.2 | 39.1 | $+13.1$ |
| g-r - smooth | 90.5 | 78.8 | $+11.7$ |
| r-z - smooth | 93.3 | 84.3 | $+9.0$ |
| r-z - merger | 91.9 | 86.2 | $+5.7$ |
| g-r - merger | 88.1 | 85.2 | $+2.9$ |
| redshift - merger | 82.5 | 84.5 | $-2.0$ |
| smooth - merger | 90.4 | 95.9 | $-5.6$ |

All values here are measured. The reading of them as "separation" is interpreted.

### 11.3 What the excesses say

Eight of the ten pairs have positive excess, and the largest are clear.

The redshift-smooth pair is the headline. The labels are correlated enough to imply a null angle of $73.7^\circ$, which corresponds to a label correlation of $r = \cos(73.7^\circ) \approx 0.28$: smoother galaxies do tend to sit at the redshifts where the red sequence is well populated, so the labels lean together by about a quarter. Yet the model's two probe directions sit at $93.0^\circ$, just past orthogonal, for an excess of $+19.4^\circ$ (measured). So the model holds its redshift axis and its smoothness axis nearly at right angles even though the labels themselves lean together.

We read that as the model keeping distance (a property tied to look-back time and apparent size) on a different axis from morphology (a property tied to internal structure), more cleanly than the raw label correlation would force (interpreted). That is exactly the separation you would hope a faithful representation makes: how far away a galaxy is and what shape it has are independent facts about it, and the model treats them as independent. The redshift-r-z pair tells a similar story at $+18.5^\circ$, with $\theta = 69.2^\circ$ against a null of $50.7^\circ$. Here the measured angle stays below $90^\circ$, so the two directions are not orthogonal, but they are pulled $18.5^\circ$ further apart than the labels alone would put them, which is the second-largest excess in the set.

The two colour channels, $g-r$ and $r-z$, are strongly correlated with each other (both rise as a galaxy's light reddens), so their null angle is small, $25.2^\circ$. The model's directions sit at $40.3^\circ$, an excess of $+15.1^\circ$. They stay partly distinct rather than collapsing onto one colour axis.

This is a meaningful result on its own. The symbols $g-r$ and $r-z$ are two colours, the difference in brightness between the green and red bands and between the red and near-infrared bands. They carry overlapping but not identical information: $g-r$ is most sensitive to recent star formation and the strength of the $4000$-angstrom break in a galaxy's spectrum, while $r-z$ reaches redder and picks up older stellar populations and dust. A model that kept a single "redness" axis would set their probe directions nearly on top of each other. That AION keeps them $15^\circ$ further apart than even their tight label correlation forces says it has resolved colour into more than one channel (interpreted).

The redshift-$g-r$ pair adds $+13.1^\circ$ ($\theta = 52.2^\circ$, null $39.1^\circ$). These four pairs, all involving redshift or colour, carry the strongest disentanglement signal in the set, and they are also the four properties the probes decoded best in Section 10 ($g-r$ at $R^2 = 0.958$, $r-z$ at $0.911$, redshift at $0.800$). Strong decoding and clean separation travel together here, which is the pattern we would want if the directions are real.

The morphology-colour pairs are positive but smaller: $g-r$-smooth at $+11.7^\circ$ ($\theta = 90.5^\circ$ against null $78.8^\circ$) and $r-z$-smooth at $+9.0^\circ$ ($\theta = 93.3^\circ$ against null $84.3^\circ$). Both measured angles sit right at or just past the $90^\circ$ orthogonality line, so the model is holding colour and smoothness on essentially independent axes.

The excess over the null is more modest than for the redshift and colour pairs, but it is still positive and in the right direction. Physically this is the cleanest kind of separation to ask for: colour (a property of the stellar populations) and smoothness (a property of the spatial structure) are genuinely different things about a galaxy, and the model treats them as such.

Now the honest part.

The pairs involving merger fraction (the share of voters who flagged a galaxy as a merging or disturbed system) cluster near zero, and two of them go slightly negative. The $r-z$-merger and $g-r$-merger excesses are small and positive ($+5.7^\circ$ and $+2.9^\circ$). The redshift-merger excess is $-2.0^\circ$ ($\theta = 82.5^\circ$, null $84.5^\circ$) and the smooth-merger excess is $-5.6^\circ$ ($\theta = 90.4^\circ$, null $95.9^\circ$), both negative. A negative excess means the model's two directions are marginally closer together than the labels are. So for merger we see no extra separation, and a hint of the opposite. These two rows are the ones drawn in red in Figure 9, with the coloured dot sitting to the left of its grey twin rather than to the right.

We do not over-read this. The merger labels are the weakest in the set. From Section 10, merger was the least decodable full-N property ($R^2 = 0.681$, the lowest of the morphology labels), and merger is a rare, noisy, hard-to-define class: a galaxy is "merging" only fleetingly, and the visual signs are subtle and easy to confuse with ordinary disturbed structure or projection.

A weakly decodable label gives a noisy probe direction, and a noisy direction gives a noisy angle. The small negative excesses ($-2.0^\circ$ and $-5.6^\circ$) are well inside the range we would expect from that noise alone.

The honest statement is: the model cleanly disentangles redshift, colour, and smoothness from one another, while for merger the test is inconclusive and shows no extra separation. We do not claim the model entangles merger with anything; we claim the data cannot decide it here.

So the ten pairs fall into three tiers. The top tier (excess above $+13^\circ$) is redshift and colour separated from each other and from smoothness, the model's clearest disentanglement. The middle tier (excess between $+9^\circ$ and $+12^\circ$) is colour separated from smoothness, still positive and orthogonal but with a smaller margin over the null. The bottom tier is everything touching merger, where the excess sits within a few degrees of zero on either side and the test cannot decide. The signal lives in the top two tiers, and it is consistent: every property the probes decoded well also separates well.

Two caveats belong to this whole subsection. First, $\theta_{\text{null}} = \arccos(r)$ uses the linear (Pearson) label correlation, so it is the right null only to the extent that the relationships are linear; strong nonlinearity between labels would bias the null. Second, the excess has no confidence interval attached in our results, so we lean on the size of the effect: $+19^\circ$ and $+18^\circ$ are large and consistent across the redshift and colour pairs, whereas the $\pm 2$ to $\pm 6$ degree merger numbers are small enough to be noise. Read the big positive excesses as the signal and the near-zero ones as undecided.

And the deepest caveat is the one from the start of the section, which we restate because it bounds every positive number above. This test is linear and correlational. A large excess says the two best-fit linear read-out directions are far apart; it does not prove the model stores each concept on a single axis, and it does not show that moving one concept leaves the other physically unchanged inside the network. Establishing that would take a causal intervention we did not run.

So the right summary is measured-and-modest: the model's linear directions for redshift, colour, and smoothness are more orthogonal than the labels force, which is real and consistent evidence of separation, and falls short of a proof of true disentanglement.

### 11.4 Neighbourhood purity: a check with no probe at all

The disentanglement test still fits probes. The second check throws even that away and asks a blunt question directly in the embedding: do galaxies of one morphological type sit among their own kind in the raw space?

This is model-free in the sense that no regression, no fitted weight vector, and no learned direction is involved. We only use distances between the embedding vectors themselves, on the full multimodal $E_{\text{full}}$ set. Where the disentanglement test asked whether the model's read-out axes are distinct, this asks whether the raw geometry already groups galaxies by type before any axis is drawn. The two are complementary: one looks at directions, the other at neighbourhoods.

The measure is $k$-nearest-neighbour purity, with $k = 20$. For a chosen class of galaxies, we take each galaxy in that class, find its 20 nearest neighbours in the embedding (the 20 closest other galaxies by distance), and ask what fraction of those 20 belong to the same class. Average that fraction over every galaxy in the class.

A purity of $1.0$ means every neighbour of every class member is also in the class (perfect local segregation). A purity near the class's overall share of the sample means neighbours are no more likely to be same-class than random (no local structure). High purity says "like sits near like" in the native geometry, with no probe needed to see it.

The right yardstick is again not zero but a baseline, and here the baseline is the class's own prevalence. If a class makes up a fraction $p$ of the sample and points were scattered without any structure, then a typical neighbour would be in the class with probability $p$, so the expected purity would be about $p$. Purity well above $p$ is the signal that the class clusters.

This baseline is what makes the two numbers below interpretable, and it is also what makes them not directly comparable, because the two classes have very different $p$.

We define the two classes by a vote-fraction cut. A galaxy is counted as smooth if its smooth fraction exceeds $0.7$ (more than 70% of voters called it smooth), and as featured if its featured fraction exceeds $0.7$. The cut at $0.7$ keeps only confident cases on each side.

The results (measured):

| Class | $n$ | $k$NN purity ($k=20$) |
|---|---|---|
| smooth ($> 0.7$) | 34,289 | 0.991 |
| featured ($> 0.7$) | 2,224 | 0.714 |

Smooth galaxies are almost perfectly surrounded by other smooth galaxies: 99.1% of the 20 neighbours of a confident smooth galaxy are themselves confident smooth galaxies. Featured galaxies are also surrounded by their own kind, but less tightly: 71.4% of neighbours match.

Put these against the prevalence baseline. The confident smooth set has $n = 34{,}289$ members; the confident featured set has $n = 2{,}224$. Of the galaxies that fall into one of these two confident classes, smooth is the overwhelming majority, so its prevalence is high and its purity of $0.991$ is close to the ceiling.

The featured prevalence is far lower, roughly one part in sixteen of the two-class pool. A purity of $0.714$ against a prevalence near $0.06$ is a factor of about twelve above what random scattering would give. So even though $0.714$ looks weaker than $0.991$ on the page, relative to its own much harder baseline it is a strong signal of clustering. The two numbers measure the same thing on two very different scales.

The reading is straightforward and the asymmetry is informative. Confident smooth galaxies form a dense, clean region of the embedding; you can stand on almost any smooth galaxy and your 20 nearest neighbours will all be smooth too (interpreted).

The lower featured purity has a simple structural cause we should state plainly. Smooth galaxies are by far the larger population here ($n = 34{,}289$ versus $n = 2{,}224$, about fifteen to one). A rarer class has fewer same-class points to be near, so even with real local clustering its purity is mechanically pulled down, because some of any galaxy's 20 neighbours will be drawn from the dominant smooth population at the boundary between the two regions.

So $0.714$ for featured is still well above what random mixing would give for a class that is a small minority of the sample, and we read it as real local structure, just softer and more boundary-exposed than the smooth core (interpreted). The class-imbalance caveat is the one to keep: the two purity numbers are not directly comparable as if the classes were the same size.

There is a second, gentler caveat. The class labels are themselves vote fractions predicted by a separate network and accurate only to about 5 to 10 percent (Section 4 (data and preprocessing)). A galaxy near the $0.7$ cut could fall on either side by label noise alone, which softens purity at the boundary even if the embedding geometry were perfect.

So the small shortfall of the featured number from $1.0$ is partly label noise, not only embedding structure. The smooth purity of $0.991$ is high enough that this barely matters there; it matters more for the rarer, fuzzier featured class.

One thing the purity result does NOT say, and we should be careful here. High neighbourhood purity is fully consistent with the single continuous body that the diffusion map found (Section 8 (diffusion maps)) and that the topology arm confirmed (one connected piece, no holes). Purity is a local statement: near a smooth galaxy, things are smooth.

It is not a claim that smooth and featured galaxies form two separate islands. They occupy different regions of one continuous cloud, and the smooth region happens to be large, dense, and pure. Like sits near like along a continuum, not in disconnected clusters.

Taken together, the two checks point the same way as the probes and the diffusion map (Section 8 (diffusion maps), Section 9 (reading physics off the manifold)). The model does not just carry the information that a galaxy is smooth or featured; it arranges the embedding so that smoothness, colour, and redshift live on largely separate directions, and so that galaxies of a kind cluster near their own kind. The next two sections move from probing for human-named properties to letting the embedding name its own axes, using a sparse autoencoder to ask whether the model carries concepts we never labelled.
