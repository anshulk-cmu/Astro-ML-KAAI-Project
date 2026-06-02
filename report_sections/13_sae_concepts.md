## 13. Concept discovery II: the concepts and the alien candidates

Section 12 built the tool: a TopK sparse autoencoder (SAE) that rewrites each galaxy's 1024-number embedding as a short list of active "concept" units drawn from an over-complete dictionary of $m = 4{,}096$ latents, with $k = 32$ of them switched on per galaxy. This section reads the output. We ask three questions of every latent the SAE learned. Does it line up with a property we already have a name for? Is it real, in the sense that it reappears when we retrain from a different random start? And does it carry signal that none of our named properties can account for? The third question is the one that motivated this whole arm, because a foundation model trained with no morphology labels might organise galaxies along axes that our human taxonomy never wrote down.

Throughout, keep one fact in front of you. Everything here is correlational. An SAE latent that tracks colour is a direction in embedding space whose activation rises and falls with colour; we never intervened on the model, never ablated a latent, never steered the embedding to see what changes downstream. So when we call 59 latents "alien candidates" below, read "candidate" as a literal promissory note, not a finding. We will repeat this.

### 13.1 What "alignment" means and how we score it

A latent is one number per galaxy: how strongly that concept unit fires for that galaxy. A label is also one number per galaxy: its redshift, its g-r colour, its smooth vote fraction, and so on. To ask whether a latent "is about" a label, we measure how tightly the two numbers move together across all the galaxies where the latent is active.

We use the Spearman rank correlation $\rho$. Spearman replaces each value by its rank (1st smallest, 2nd smallest, and so on) and then correlates the ranks. We chose rank correlation, not the ordinary Pearson correlation, for a physical reason: SAE activations are heavy-tailed and often near zero, and several labels (redshift, vote fractions) are bounded or skewed, so a measure that only cares about ordering is the honest choice. It does not assume a straight-line relationship; it asks the weaker, safer question of whether high goes with high.

We define the alignment of a latent as

$$\text{align} = \max_{\ell \in \{6 \text{ labels}\}} \; \big| \rho(\text{latent}, \ell) \big|.$$

In words: take the six full-sample labels (redshift, the two colours g-r and r-z, and the three Galaxy-Zoo vote fractions smooth, featured, merger), correlate the latent against each, take the absolute value (a latent that fires for blue galaxies and one that fires for red galaxies are both "colour" latents, so sign does not matter for naming), and keep the largest. That single number is how strongly the latent matches its best-matching human label. The label that wins the max is the latent's name.

The scoring ran on the $R=4$ dictionary ($m = 4{,}096$ latents) trained on the z-scored multimodal embedding E_full. Of the 4,096 dictionary slots, $n_{\text{active}} = 2{,}657$ actually fire on the data (the rest are dead or near-dead, expected at this expansion). All counts below are out of those 2,657 active latents (measured).

### 13.2 The label-shuffle null: how big an alignment is "real"

A correlation of, say, 0.04 sounds small, but with $N \approx 48{,}000$ galaxies even pure noise produces non-zero correlations, and because we take the *maximum* over six labels we get a free upward bias (the best of six noisy draws is larger than any single one). We need a baseline that bakes in both effects. We use a label-shuffle null.

The recipe is simple and assumption-free. We randomly permute each label across galaxies, breaking any true link to the embedding while keeping each label's own distribution exactly intact, then recompute every latent's alignment against the shuffled labels. Repeating the shuffle builds a distribution of alignment values that pure chance can manufacture. From that distribution we read two cutoffs (measured):

$$\text{thr95} = 0.0119, \qquad \text{thr99} = 0.0129.$$

A latent whose real alignment exceeds 0.0119 sits above the 95th percentile of chance; one above 0.0129 clears the stricter 99th percentile. So the bar for "significantly aligned" is a Spearman of about 0.012. That is a low bar in absolute terms, which is the point: at this sample size, even a weak but genuine correlation is detectable, and the null tells us where genuine starts.

Two counts follow directly (measured):

- $\text{aligned\_sig} = 717$ latents clear thr95.
- $\text{aligned\_strong} = 690$ latents clear the stricter thr99.

That the two numbers are so close (717 versus 690) tells you something quiet but useful. Almost every latent that beats the 95% bar also beats the 99% bar; there is no thick band of borderline cases sitting between the two thresholds. The aligned latents are not marginal. They are well clear of the null or not in the running at all.

The largest alignment anywhere is

$$\text{max\_align} = 0.279 \quad (\text{measured}),$$

which is about 23 times thr95. We write "about 23x" because $0.279 / 0.0119 = 23.4$; the figure rounds it to 23x. So the single most label-locked latent in the dictionary still only reaches a Spearman of 0.28. We read that as a real but modest ceiling (interpreted): the SAE does find directions that track human labels, but no single sparse latent *is* a label. The strongest match explains only a slice of the label's variation. This matters for how you picture the model's code. Physics is not stored in one neuron per concept; it is smeared across many.

### 13.3 Seed-stability: is the latent a property of the model or of the random seed?

An SAE is trained from a random initialisation, so any single run can manufacture a latent that is an accident of that particular start. The fix is to retrain. We trained five SAEs from five different random seeds and asked, for each latent, whether the same direction comes back.

The comparison is on the decoder vector. Each latent has a decoder vector $W_{\text{dec}}$, the 1024-number direction it writes back into the embedding when it fires; this is the latent's identity in embedding space. To compare a latent in one seed to a latent in another, we use cosine similarity, the cosine of the angle between the two decoder vectors. Cosine of 1 means the same direction, 0 means orthogonal (unrelated), and we set the match bar at

$$\cos \ge 0.6,$$

meaning the two directions point within about 53 degrees of each other. A latent is called seed-stable if its decoder direction reappears with $\cos \ge 0.6$ in at least half of the other four seeds. The "half of the others" rule guards against a single lucky coincidence: a stable concept should show up again and again, not just once.

Across all active latents, the median best cross-seed cosine is 0.460 (measured). Read that carefully. The *typical* latent's best match in another seed sits below the 0.6 bar, so most individual SAE latents are partly seed-specific; the dictionary as a whole is not perfectly reproducible unit-for-unit. That is the honest caveat for this method. What survives the bar is a minority, and it is exactly that minority we trust. The fraction of active latents that are seed-stable is 0.148 (measured), about one in seven.

Combining the two filters gives the headline count of dependable, human-named directions:

$$\text{aligned\_and\_stable} = 335 \quad (\text{measured}).$$

These 335 latents both beat the shuffle null (so they track a real label) and reappear across seeds (so they are a property of the model, not the seed). This is the number to quote for "physics-named concepts the model reliably carries." It is the intersection of two independent screens, which is why it is the most defensible count in this section.

### 13.4 The named concepts

Naming each aligned latent by its top-correlated label and tallying them gives the concept inventory in Figure 12.

![Figure 12. SAE features aligned to physical labels, per concept.](figures/12_sae_named_concepts.png)

Figure 12. The horizontal axis is a count: the number of active SAE latents (out of 2,657) whose best-matching label is the concept on the vertical axis. The six concepts on the vertical axis are the six full-sample labels, sorted top to bottom by total count. Each concept has two stacked bars: the light bar is the number of latents that clear the label-shuffle significance bar (thr95 = 0.012), and the darker bar is the seed-stable subset of those (the latents that also reappear across seeds). The text label at the end of each light bar, "max |rho| = ...", gives that concept's single strongest alignment, the largest absolute Spearman correlation any latent reached against that label. The two footnote lines restate the definitions and the load-bearing caveat: "aligned" is a correlational match to a shuffle null, naming is just the top-correlated label, and redshift in particular is spread over many weak latents rather than concentrated in one strong feature. What to look for: colour (g-r) and the morphology pair (smooth, featured) carry the strongest single matches (max |rho| around 0.25 to 0.28), while redshift has many aligned latents but each is individually weak (max |rho| only 0.13).

The per-concept counts (light bar $n$, dark bar $n_{\text{stable}}$, and the strongest alignment) are all measured:

| Concept | Aligned latents $n$ | Seed-stable $n_{\text{stable}}$ | Max alignment |
|---|---|---|---|
| g-r (colour) | 204 | 80 | 0.272 |
| redshift | 166 | 87 | 0.126 |
| featured (morphology) | 114 | 52 | 0.245 |
| r-z (colour) | 89 | 41 | 0.210 |
| smooth (morphology) | 81 | 37 | 0.279 |
| merger | 63 | 38 | 0.164 |

Two readings, both interpreted, follow from the table.

First, the clearest single concepts are colour and morphology. The g-r colour wins on count (204 aligned latents, the most of any label) and the smooth vote fraction wins on strength (0.279, the highest single alignment in the whole dictionary, and the same number as the global max\_align). Featured sits just behind at 0.245. So when the SAE carves out a direction that lines up tightly with a human label, that label is usually colour or a smooth/featured morphology axis. This is consistent with what every other arm of the report found: colour and the smooth-versus-featured morphology split are the model's primary organising axes (the diffusion coordinates of Section 9, the strong probes of Section 10).

Second, redshift behaves differently, and the difference is informative. Redshift has plenty of aligned latents (166, second only to g-r) and the most seed-stable ones (87), yet its single strongest alignment is only 0.126, roughly half the colour and morphology peaks. We read this as redshift being a *global, diffuse* property of the embedding rather than a sparse code (interpreted). It is written faintly into many directions at once, not loudly into a few. That fits the diffusion-map picture, where redshift loaded onto a broad smooth gradient (dc1) rather than a sharp axis, and it fits the intuition that redshift, a distance-and-time coordinate, touches almost everything about how a galaxy looks (its apparent size, brightness, and observed colours all shift with it) instead of being one isolated feature. Merger is the weakest concept on both count (63) and strength (0.164), which is unsurprising: mergers are rare, the vote fraction is noisy, and there is simply less consistent signal for the SAE to latch onto.

One caution on the table. The counts are not a partition of "how much the model knows about each property," because a latent can correlate with several labels and we assign it to only its single best one. Colour and redshift, for instance, are themselves correlated in the data (redshift is mostly photo-z, which is colour-derived, per Section 4 and Section 10), so some latents could plausibly have gone either way. The names are a useful summary, not a clean decomposition.

### 13.5 Novelty: signal the labels cannot explain

Now the question this arm was built for. Does the SAE find structure that our six labels miss?

We define novelty per latent as the fraction of the latent's activation variance that the labels *cannot* linearly account for. Concretely, we regress the latent's activations on the six labels (a linear least-squares fit, latent as the target, the six labels as predictors) and measure how much of the latent's variance the fit explains. Call that explained fraction $\text{rec}$ (for "recovered"). Then

$$\text{novelty} = 1 - \text{rec}.$$

Novelty near 0 means the labels reconstruct the latent almost perfectly, so the latent is, in effect, a relabelled combination of things we already named. Novelty near 1 means the labels are nearly useless at predicting the latent, so it carries variation orthogonal to our entire human vocabulary. It is a regression residual, plain and simple: what is left of a latent after you subtract everything the labels can linearly say about it.

We then defined an alien candidate by three conditions held at once (measured):

1. Seed-stable (decoder direction reappears in at least half the other seeds at $\cos \ge 0.6$), so it is a real property of the model, not a seed accident.
2. High novelty, $\text{novelty} > 0.7$, so the six labels explain less than 30% of its activation variance.
3. Not label-aligned (its best alignment does not clear the thr95 shuffle null), so it is not just a weak echo of a named concept.

A latent that is stable, mostly unexplained by any label, and not even weakly tied to one, is a direction the model reliably uses for *something* we have no name for. There are

$$\text{alien\_candidates} = 59 \quad (\text{measured}).$$

Figure 13 shows both the alignment distribution and where these 59 sit.

![Figure 13. SAE alignment versus the shuffle null, and alignment versus novelty with the alien candidates.](figures/13_sae_alignment_novelty.png)

Figure 13, panel A (left). The horizontal axis is alignment, each active latent's best absolute Spearman correlation to a physical label, from 0 to about 0.28. The vertical axis is a count of latents on a logarithmic scale (note: each step up is a factor of ten, so the tall leftmost bar holds well over a thousand latents). The red dashed vertical line is the shuffle null at thr95 = 0.012 and the black dotted line is thr99 = 0.013; latents to the left of these lines are statistically indistinguishable from chance. The red arrow on the right marks the single most-aligned latent at 0.279, annotated "max alignment = 0.279 (~23x the thr95 null)." What to look for: the distribution is hugely front-loaded, with most latents piled just above zero (below or near the null) and a long thin tail of genuinely aligned latents stretching out to 0.28. The shape says the same thing as the close 717-versus-690 counts above: a minority of latents carry real label signal, and they are clearly separated from the noise floor rather than blending into it.

Figure 13, panel B (right). The horizontal axis is again alignment (best |Spearman| to a label). The vertical axis is novelty, the residual activation-variance fraction with the six labels regressed out, from roughly 0.5 to 1.0. Each dot is one active latent. Colour encodes seed-stability: faint grey dots are not seed-stable (cross-seed reconstruction below 0.5), blue dots are seed-stable (at or above 0.5), and the orange-ringed markers are the 59 alien candidates. The shaded orange box in the lower region near the left is the alien selection region: low alignment (left of the thr95 line, the red dashed vertical) and high novelty (above 0.7). The inset at the top zooms into that alien region so the 59 orange points are legible against the crowd. The footnote restates the definition and confirms the count: "alien candidates: seed-stable, high-novelty, not label-aligned (align <= thr95) -- CORRELATIONAL only (no causal test). JSON-reported alien\_candidates = 59; reproduced here = 59." What to look for: most latents sit high on the novelty axis (the labels never explain a latent fully, because each latent is sparse and idiosyncratic), but the orange points are the ones that combine high novelty with near-zero alignment *and* survive the seed test. They are the model's reliable, unnamed directions.

### 13.6 What the alien candidates are, and what they are not

Here is the careful reading, because this is the most interesting and the least certain result in the whole report.

What we can say (measured). There exist 59 directions in AION-1's embedding that (a) the model reconstructs consistently across five independent random retrainings, so they are not artefacts of one optimisation run, and (b) our six physical labels cannot linearly predict, explaining less than 30% of their variance, and (c) do not even weakly track any single label above the chance floor. By construction they are stable, novel, and unnamed.

What we cannot say (the caveat, repeated because it governs every word here). These are correlational candidates only. We ran no causal test. We did not ablate any alien latent and check whether reconstruction or any downstream prediction degrades. We did not steer a latent and watch the embedding or any rendered galaxy change. So we do not know whether an alien candidate encodes a genuine, physically meaningful galaxy property that our labels happen to omit, or whether it encodes something with no clean physical reading at all: an imaging artefact, a survey or instrument systematic, a sky-position effect, a residual of the masked-modelling training objective, or a correlated nuisance we did not measure. "Alien" names the gap in our labels, not a discovery about galaxies. The honest status is: a stable, unexplained direction worth a future causal test, no more.

There is also a built-in limit on novelty's reach. We only regressed against six labels, and only linearly. A latent could be perfectly explained by some property we did not have (stellar mass, sSFR, and Sersic index exist for only a few thousand galaxies, so they were not in the six-label full-sample regression) or by a nonlinear combination of labels we did have. So novelty over 0.7 means "not linearly explained by these six," which is narrower than "genuinely beyond all known physics." A fraction of the 59 could collapse into "known" the moment a richer label set or a nonlinear probe is applied. We flag this rather than hide it.

### 13.7 What this arm establishes

Stepping back, the SAE answers the concept-discovery question (question C from Section 1) in two parts. First, the model's reliable, human-nameable code is real but modest and smeared: 335 seed-stable, label-aligned latents, dominated by colour and the smooth/featured morphology split, with redshift present everywhere but concentrated nowhere, and a strongest single match of only 0.279. No single latent is a concept; concepts live in populations of weakly aligned directions. Second, beyond the named axes the model reliably uses at least 59 stable directions our labels cannot account for. Those 59 are the candidates a full study should target first, with the causal tools this static, correlational analysis could not bring to bear: ablate them and steer them, and see what moves. Section 19 folds this concept picture into the geometric one, and Section 21 lists the causal follow-up as the first thing a complete study should do.
