## 10. Decodability: how much physics the embedding carries

Section 8 showed the embedding is one smooth body, and Section 9 showed that two of its diffusion coordinates line up with morphology, colour, and star formation. Those were correlations: we coloured a 2D picture by a property and saw a gradient.

A correlation along one or two coordinates does not tell you how much of a property the full 1024-number vector actually pins down. The diffusion picture used only the slowest two directions; the embedding has a thousand more. This section uses all of them at once.

So it answers a sharper question. Given the whole embedding, how precisely can you read off a physical property of the galaxy? And, the part that matters most, can you read off properties the model was never given as input?

The tool for this is a probe. The idea is simple and deliberately weak. We do not let a clever nonlinear model dig the answer out. We fit the simplest thing that could work, a single linear map from the 1024 numbers to the property, and we measure how well that linear map predicts the property on galaxies it never saw during fitting. If even a straight-line readout recovers a property well, then that property is laid out in the embedding in an easy, linear, accessible way. That is a strong statement about the representation, not about our cleverness.

The weakness is a feature, not a limitation we tolerate. Suppose we let a deep network probe the embedding instead. A deep network is so flexible that it could pull a property out of almost any representation, even one that buried the property in a tangled, useless form. Then a high score would tell us about the probe's power, not the embedding's organisation.

By restricting the probe to a single linear map, we tie the score directly to a property of the representation itself: is this quantity available along a direction, yes or no, and how cleanly. A linear probe answers a sharp question about geometry. That is what we want here.

One more framing point before the math. There is a difference between "the information is present" and "the information is accessible". A pile of pixels contains a galaxy's mass too, in principle, but you would need a whole physical model to extract it. When we say the embedding "carries" mass, we mean something stronger than mere presence: the embedding has arranged itself so that mass is readable by a flat, linear readout. That is the kind of structure a downstream user actually benefits from, and it is the kind of structure a faithful representation should have if the model truly organised galaxies by physics.

And one warning we will repeat. A high probe score is evidence of accessible structure, not of understanding. Decodability tells you the property lives along a direction in the embedding. It does not tell you the model "knows" the physics, nor that the direction is causal, nor that the model would use that direction the way we read it. Those are separate, harder questions. This section measures accessibility and nothing more, and we keep that boundary visible throughout.

### 10.1 What a linear probe is, in plain words

Write the embedding of a galaxy as a vector $x \in \mathbb{R}^{1024}$, meaning a list of 1024 real numbers. Write the property we want to predict (redshift, stellar mass, a vote fraction) as a single number $y$. A linear probe is a weight vector $w \in \mathbb{R}^{1024}$ and an offset $b$ (a single number) such that the prediction is

$$\hat{y} = w \cdot x + b = \sum_{j=1}^{1024} w_j x_j + b.$$

In words: multiply each of the 1024 embedding numbers by its own weight, add them all up, add a constant, and that sum is your guess for the property. The weights $w_j$ say how much each embedding dimension should count, and their signs say in which direction. Nothing here bends or curves. It is a flat, straight-line readout of the property from the embedding. That weakness is the point. A linear probe can only succeed if the property is already arranged in the embedding along some direction. It cannot manufacture structure that is not there.

There is a clean geometric picture for $w$. The weight vector points in a direction inside the 1024-dimensional embedding space, and the prediction $w \cdot x$ is the length of the galaxy's embedding projected onto that direction (up to the offset $b$). So fitting a probe is finding the direction along which the property increases most smoothly.

If such a direction exists and the property changes steadily as you move along it, the probe scores high. If the property is scattered with no preferred direction, the projection carries little information and the probe scores near zero. We will use exactly this "probe direction" idea again in Section 11, where we measure the angle between the direction for one property and the direction for another to test whether the model keeps concepts apart.

We fit $w$ and $b$ by least squares with a penalty, which is the next idea.

### 10.2 Ridge regression and the cross-validated penalty

Plain least squares picks $w$ to make the prediction errors as small as possible on the training galaxies. With 1024 weights and a target that has real noise, plain least squares will happily use tiny accidental patterns in the training set that do not generalise.

That is overfitting: the probe looks great on the galaxies it was fit on and worse on new ones. We want a measure of what the embedding genuinely carries, not of how well 1024 free numbers can memorise a few thousand training points. Overfitting would inflate every score and make a weak embedding look strong.

Ridge regression fixes this by adding a penalty on the size of the weights. It minimises

$$\sum_{i \in \text{train}} \left( y_i - (w \cdot x_i + b) \right)^2 \; + \; \alpha \, \lVert w \rVert^2,$$

where the first term is the usual sum of squared errors over training galaxies and the second term, $\alpha \lVert w \rVert^2 = \alpha \sum_j w_j^2$, punishes large weights. The number $\alpha \ge 0$ sets how hard we punish them.

A large $\alpha$ forces the weights toward zero, giving a simpler, smoother probe that ignores small accidental patterns. A small $\alpha$ lets the probe fit harder. Ridge keeps the probe honest by preferring small, spread-out weights unless the data clearly earns larger ones.

There is also a practical reason ridge is the right base method here, beyond overfitting. The 1024 embedding dimensions are not independent; many of them carry overlapping information (Section 2 noted that just a few principal components capture most of the raw variance).

When inputs are correlated, plain least squares becomes unstable: it can assign wildly large positive and negative weights to correlated dimensions that nearly cancel, and tiny changes in the data swing those weights around. The $\alpha \lVert w \rVert^2$ penalty tames this by spreading the weight smoothly across the correlated group instead of letting it blow up. So ridge is both a guard against overfitting and a guard against the instability that correlated embedding dimensions would otherwise cause.

We do not pick $\alpha$ by hand. We use RidgeCV, which means ridge regression with the penalty chosen by cross-validation.

Cross-validation works like this. Split the training galaxies into folds, hold one fold out, fit on the rest, predict the held-out fold, and measure the error. Do this for each fold and for a grid of candidate $\alpha$ values, here spanning $10^{-2}$ up to $10^{4}$ (a wide grid, six orders of magnitude, so the search is not artificially boxed in). The $\alpha$ that gives the lowest average held-out error wins. So the penalty strength is itself learned, from data the final scoring never touches.

The chosen $\alpha$ varies by target, and we report it because it is informative. Colours land on a small penalty ($\alpha = 0.01$), which says the colour signal is clean and strong enough that the probe can fit tightly without overfitting. Redshift sits a little higher ($\alpha = 0.1$). The small-sample physical targets and the morphology sub-fractions land on much larger penalties ($\alpha$ up to 100), which says the probe needs heavy smoothing to generalise from a few thousand noisy points.

A high $\alpha$ is the cross-validation telling us, in its own units, "do not trust this target's fine detail, smooth it". That is a useful side signal: the targets that demand the heaviest smoothing are exactly the ones with the smallest samples and the noisiest labels, which is what we would hope to see from an honest procedure. We did not hand-tune any of this; the pattern fell out of the cross-validation.

### 10.3 R-squared, the 80/20 split, and the bootstrap interval

We score a probe by $R^2$ (R-squared), the fraction of the property's variance that the probe explains on held-out galaxies. Define it on the test set:

$$R^2 = 1 - \frac{\sum_{i \in \text{test}} (y_i - \hat{y}_i)^2}{\sum_{i \in \text{test}} (y_i - \bar{y})^2},$$

where $y_i$ is the true property of test galaxy $i$, $\hat{y}_i$ is the probe's prediction for it, and $\bar{y}$ is the mean property over the test set. The denominator is the error you would make by always guessing the mean and ignoring the embedding entirely. The numerator is the error the probe actually makes.

So $R^2 = 1$ means perfect prediction, $R^2 = 0$ means the probe is no better than guessing the mean, and a negative $R^2$ means it is worse than the mean. Read $R^2$ as the share of the property that the embedding nails down through a straight-line readout.

A small worked feel for the scale helps. If a property has a true spread (variance) of some amount, an $R^2$ of $0.75$ means the probe's leftover errors have one quarter of that variance, so the typical error is about half the original spread ($\sqrt{0.25} = 0.5$). An $R^2$ of $0.96$ leaves four percent of the variance, so the typical error is about a fifth of the original spread.

This is why the jump from $0.72$ (mass) to $0.96$ (colour) is larger than it looks. It is the difference between recovering a property to half its spread versus to a fifth of its spread, a factor of more than two in typical error.

Two rules keep this number honest. First, the 80/20 split. We fit the probe on a random 80% of the galaxies (with a fixed random seed, seed 0, so the split is reproducible) and we compute $R^2$ only on the held-out 20% the probe never saw.

The penalty $\alpha$ was also chosen using only the training 80%, by cross-validation inside it, so the test 20% is untouched by any fitting decision. Every $R^2$ in this section is a held-out, out-of-sample number, not a training-set number. That is the difference between measuring what the embedding carries and measuring how well 1024 weights can memorise.

Second, the uncertainty. A single test set gives one $R^2$, but that one number depends on which galaxies happened to land in the 20%. To put error bars on it we use a bootstrap.

We resample the test set with replacement (draw $n_{\text{test}}$ galaxies at random, allowing repeats, to make a new pretend test set of the same size), recompute $R^2$ on that resample, and repeat 1000 times. The middle 95% of those 1000 values is the 95% confidence interval (CI). It tells you the range in which the true held-out $R^2$ plausibly sits given the finite test set.

Wherever we quote an $R^2$ below, the bracket after it is this 1000-times bootstrap 95% CI (measured). A wide bracket means a small or noisy test set; a tight bracket means the number is well pinned.

Why the bootstrap rather than a textbook formula? Because $R^2$ is a ratio of sums of squares, its sampling distribution is not a simple bell curve, especially when the test set is only a few hundred galaxies. The bootstrap sidesteps the need for any formula: it simulates "what if we had drawn a slightly different test set" by reusing the data we have, and reads the spread of $R^2$ off those simulations directly.

The price is that the bootstrap can only reflect uncertainty from the finite test sample. It does not capture uncertainty from the labels themselves being noisy (the vote fractions carry 5 to 10 percent error, Section 4) or from the single fixed train/test split. So the CIs are honest about sampling error and silent about those other sources. We flag that rather than pretend the bracket is the whole story.

### 10.4 The leakage-free design: why we probe E_img

Here is the single most important design choice in this section, and the reason the results can be trusted as real inference rather than bookkeeping.

The model has two embedding sets, defined in Section 3. $E_{\text{full}}$ is built from image plus photometry (the g, r, z fluxes) plus redshift, all fed in as input tokens. $E_{\text{img}}$ is built from the image only; the fluxes and redshift are not inputs at all.

Now think about what it means to probe redshift. If you probe redshift from $E_{\text{full}}$, redshift was handed to the model as an input. A high probe score then proves almost nothing about the model's understanding, because the model could simply copy the input forward into the embedding. The probe would be reading back something we put in.

That is leakage: information enters as an input and then gets "decoded" as if it were a discovery. The same trap applies to colour, since the fluxes that define g-r and r-z are inputs to $E_{\text{full}}$. Any probe on $E_{\text{full}}$ for redshift or colour is therefore contaminated, and we treat it as such.

So every concept claim in this section is made on $E_{\text{img}}$, the image-only embedding. On $E_{\text{img}}$, redshift and colour are not inputs. The only thing the model saw was the four-band picture of the galaxy. If a linear probe can still recover redshift or colour or stellar mass from $E_{\text{img}}$, the model genuinely inferred that property from the appearance of the galaxy. That is a real inference about physics from pixels, the kind of thing a representation can only do if it has organised images by their physical content. This is why $E_{\text{img}}$ is the honest set and why we lead with it. We will use $E_{\text{full}}$ only as a contrast in the ablation, to show the leakage gap directly.

It helps to be precise about what "leakage" does and does not spoil. Leakage does not make $E_{\text{full}}$ a bad embedding. A model is allowed to keep its inputs available, and for a downstream user who wants the best possible redshift readout, $E_{\text{full}}$ is the better tool.

Leakage spoils a specific kind of claim: the claim that a high probe score is evidence the model learned or inferred something. You cannot infer what you were told. By probing $E_{\text{img}}$, where the property was withheld, we convert the probe score from a bookkeeping number into a genuine test of inference. The distinction is the whole reason the project built two embeddings instead of one.

Note also that "image only" does not fully neutralise every shortcut. Colour is computable from a multi-band image, so even on $E_{\text{img}}$ a property that is really a colour proxy can be recovered through colour.

That loophole is exactly the photo-z issue we treat in 10.7. It does not affect the cleanest targets (mass, sSFR), which is why those carry the most weight in the section's conclusion.

### 10.5 Decodability from the image alone

With the method fixed, the result is a single ranked list: how decodable is each property from the image-only embedding. We probe eleven targets. Five are full-sample labels with about 48,000 galaxies each (the two colours, redshift, and the smooth, featured, and merger vote fractions). Three are the cross-matched physical properties on a few thousand galaxies (mass, sSFR, Sersic index). Two are the branch-conditional fine-morphology fractions (spiral arm, strong bar) on about 3,000 disks. Reading the list top to bottom is reading a ranking of which physical facts the image-only representation makes easy to access.

Figure 7 reports the headline: held-out $R^2$ for each property, decoded from the image-only embedding $E_{\text{img}}$, with bootstrap 95% CI bars.

![Figure 7. Decodability of physical labels from the image-only embedding.](figures/07_probe_decodability.png)

Figure 7. Each horizontal bar is one physical property; the bar length is the held-out $R^2$ (x-axis, dimensionless, from 0 to about 1.0) of a linear ridge probe fit on 80% of the galaxies and scored on the held-out 20%, using the image-only embedding $E_{\text{img}}$. The thin black whisker on each bar is the 1000-times bootstrap 95% confidence interval. The text at the right of each bar is the sample size $n$ used for that property.

Colour codes the sample regime. Blue bars are full-sample targets ($n \approx 48{,}000$); orange bars are small cross-matched or branch-conditional subsets ($n < 5{,}000$), where the labels come from external catalogues or from the Galaxy-Zoo decision tree and exist for only a few thousand galaxies. The vertical line at $R^2 = 0$ marks the no-skill baseline (a probe that just predicts the mean).

What to look for: the bars are sorted top to bottom from most to least decodable. Colours sit at the top near $R^2 \approx 0.91$ to $0.96$, redshift and the broad morphology fractions cluster near $0.79$ to $0.80$, the non-input physical properties stellar mass and sSFR sit at $0.72$ and $0.76$, and the small branch-conditional fine-morphology targets (strong bar, spiral arm) fall off at the bottom with wide error bars. All measured.

The numbers behind the bars (all measured, all from $E_{\text{img}}$, all held-out, brackets are the bootstrap 95% CI):

| Property | $R^2$ | 95% CI | $n$ | input to $E_{\text{img}}$? |
|---|---|---|---|---|
| g-r colour | 0.958 | [0.957, 0.960] | 48,396 | no |
| r-z colour | 0.911 | [0.903, 0.918] | 48,397 | no |
| redshift (photo-z) | 0.800 | [0.792, 0.809] | 48,398 | no |
| featured fraction | 0.794 | [0.783, 0.804] | 48,398 | no |
| smooth fraction | 0.792 | [0.782, 0.802] | 48,398 | no |
| sSFR | 0.760 | [0.692, 0.816] | 4,760 | no |
| stellar mass $\log M_*$ | 0.721 | [0.659, 0.772] | 3,728 | no |
| merger fraction | 0.681 | [0.658, 0.702] | 48,398 | no |
| Sersic index $n$ | 0.664 | [0.608, 0.713] | 3,730 | no |
| strong-bar fraction | 0.554 | [0.476, 0.625] | 3,034 | no |
| spiral-arm fraction | 0.252 | [0.153, 0.334] | 3,034 | no |

Read the tiers from the top. Colour is decoded almost perfectly: g-r at $R^2 = 0.958$ [0.957, 0.960] and r-z at $0.911$ [0.903, 0.918], both with very tight intervals.

The colour g-r is the difference in brightness between the green (g) and red (r) bands, and r-z between the red and near-infrared (z) bands; together they say whether a galaxy's light is bluer (young stars, recent star formation) or redder (old stars, passive). These fluxes were not inputs to $E_{\text{img}}$, so the model is reading the colour of a galaxy straight off its image, which is a thing you can in fact see in a multi-band picture.

That this works at $0.95$-plus tells us colour is encoded along a clean, near-linear direction in the embedding, which is consistent with the dc2 colour axis we found in Section 9. The tightness of the CI (a span of $0.003$ for g-r) is what you expect when 48,000 galaxies vote on a strong, clean signal.

It also sets a rough ceiling for everything below. If the easiest visible property tops out near $0.96$ rather than at exactly $1.0$, some of the missing few percent is label and measurement noise that no probe could recover. So a target scoring $0.72$ is not $0.72$ out of a possible $1.0$ of clean signal; it is $0.72$ against a practical ceiling somewhat below $1.0$. That makes the mass and sSFR numbers look stronger, not weaker.

Redshift comes next at $R^2 = 0.800$ [0.792, 0.809]. The model never saw redshift for $E_{\text{img}}$, yet a straight readout recovers four-fifths of its variance from the image. Hold that number; the photo-z caveat in 10.7 qualifies what it means.

There is a physical reading of the whole ranking, and it is coherent. The properties that decode best are the ones most directly visible in a multi-band image (colour, then coarse shape). The middle tier is properties that are inferred from the image but still strongly tied to visible cues (mass, sSFR, concentration). The bottom tier is fine structural detail that needs resolution and clean conditions to see (bars, arms).

The model's accessibility ranking tracks how visible each property is, which is what an honest visual representation should produce. We read this as a consistency check passing, not as a separate discovery.

Morphology, as the broad smooth and featured vote fractions, sits at $R^2 \approx 0.79$ (smooth $0.792$ [0.782, 0.802], featured $0.794$ [0.783, 0.804]). A vote fraction is the share of human (or human-trained CNN) votes that called a galaxy "smooth" or "featured" (Section 4 defines these). These are the broad "is it a featureless blob or does it have visible structure" judgements, and the image-only embedding tracks them well. That smooth and featured land at almost the same $R^2$ is expected, since they are close to two sides of one coin.

The merger fraction is lower at $0.681$ [0.658, 0.702], which fits intuition: mergers are rarer and messier, the visual signature (tidal tails, double nuclei, disturbed shapes) is more varied, and a single linear direction captures less of it. This lower score is also consistent with the disentanglement result in Section 11, where the merger direction turns out to behave differently from the colour and redshift directions.

The non-input physical properties are the most interesting, and they are in orange because they come from a few thousand cross-matched galaxies. Stellar mass decodes at $R^2 = 0.721$ [0.659, 0.772] and specific star-formation rate (sSFR) at $0.760$ [0.692, 0.816], both from the image alone.

We define these terms in Section 2. Stellar mass is the total mass locked in stars (in solar masses), and sSFR is the star-formation rate divided by the stellar mass, a measure of how fast a galaxy is building new stars relative to its size. Neither was ever an input to $E_{\text{img}}$, and neither is directly "visible" the way colour is. A galaxy's mass and star-forming state are inferred quantities even for a human expert, recovered from a physical model of how light maps to physical properties.

Recovering roughly three-quarters of their variance from pixels means the visual representation encodes the galaxy's mass and current star-forming state, not just its surface appearance.

There is a physical reason this is plausible rather than magical. Mass correlates with size and brightness and concentration, and star formation correlates with colour, clumpy bright regions, and disk structure, all of which are visible in the image. The model appears to have folded those visible cues into directions that line up with the underlying physical quantities.

This connects to the colour-mass bimodality from Section 2 (the red sequence of passive galaxies and the blue cloud of star-forming ones). If the embedding lays out colour, mass, and sSFR along accessible directions, it has the raw material to reproduce that bimodality on its own, which is what the diffusion picture in Section 9 hinted at and what the SAE in Section 13 will probe directly. We return to why mass and sSFR are the cleaner evidence in 10.7.

The Sersic index $n$ (a number describing how steeply a galaxy's light falls off from centre to edge; high $n$ is a concentrated bulge-like profile, low $n$ is a flatter disk) decodes at $R^2 = 0.664$ [0.608, 0.713]. The structure is partly there but less cleanly linear. Concentration is a real visible feature, so it makes sense the embedding tracks it, but the single-number Sersic fit is itself a lossy summary of a galaxy's light profile, which caps how high any probe can score against it.

The bottom two bars are the branch-conditional fine-morphology targets, defined only for featured non-edge-on disks (the Galaxy-Zoo decision tree, hence the small $n = 3{,}034$, explained in Section 4). Strong-bar fraction decodes weakly at $R^2 = 0.554$ [0.476, 0.625], and spiral-arm fraction barely at all, $R^2 = 0.252$ [0.153, 0.334].

The wide intervals are honest. With 3,034 galaxies and a fine, conditional label, the probe has little to work with, and the spiral CI runs from $0.15$ to $0.33$, so we cannot even pin its value to within a factor of two.

We read this plainly. The image-only embedding carries the coarse morphology (smooth vs featured) strongly and the fine sub-structure (bars, arms) weakly. We do not claim the model resolves spiral arms; the data says it mostly does not, at least not in a linearly accessible way.

Two cautions before reading too much into the weak fine-morphology scores. First, these labels are conditional, computed only for disks that already passed the "featured and not edge-on" branch, so the probe is asked to do a harder, narrower job on a pre-filtered population.

Second, spiral-arm and bar fractions are exactly the labels where the underlying CNN vote fractions are noisiest, so part of the low $R^2$ is label noise rather than missing structure in the embedding. A low score here is a soft "not clearly accessible", not a hard "absent". We flag this rather than read the weak scores as proof the model is blind to spiral structure.

### 10.6 The modality ablation: seeing the leakage directly

To make the leakage argument concrete rather than asserted, we run the same probe on both embeddings for the three properties that are inputs to $E_{\text{full}}$ but not to $E_{\text{img}}$: redshift and the two colours. The gap between the two scores is exactly the leakage we warned about.

This doubles as a check on the method itself. If our probe were broken or our setup leaky in some hidden way, the $E_{\text{img}}$ and $E_{\text{full}}$ scores might come out the same, or in the wrong order. Seeing $E_{\text{full}}$ score higher on exactly the quantities it was fed, and by an amount that shrinks as the image already carries the signal, is the behaviour a correct leakage-aware setup should show. So the ablation confirms both the leakage story and that the pipeline is doing what we think.

![Figure 8. Decodability: image-only versus multimodal embedding.](figures/08_modality_ablation.png)

Figure 8. Grouped bars comparing held-out $R^2$ (y-axis, dimensionless, 0 to just above 1.0) for three properties (x-axis: redshift, g-r colour, r-z colour) decoded two ways. The blue bar in each pair is $E_{\text{img}}$, the image-only embedding (the honest set, where these three were never inputs). The grey bar is $E_{\text{full}}$, the multimodal embedding, which ingested the g, r, z fluxes and the redshift as inputs. Each bar is labelled with its $R^2$ value on top. What to look for: the grey bars are always higher than the blue bars, and the gap is largest for redshift. That gap is the leakage. Measured.

The numbers (measured): redshift goes from $E_{\text{img}} = 0.800$ to $E_{\text{full}} = 0.976$; g-r from $0.958$ to $0.989$; r-z from $0.911$ to $0.968$. In every case the multimodal embedding scores higher on exactly the quantities it was fed.

The redshift jump is the biggest, from $0.80$ to $0.98$, because redshift was handed to $E_{\text{full}}$ as a scalar input that the probe can largely copy back. The colour jumps are smaller ($0.958 \to 0.989$, $0.911 \to 0.968$) because the image already carries most of the colour signal, so the extra flux inputs only top it up.

The size of each gap is itself a readout of how much new information the input added on top of the image: large for redshift (the image alone gets it only to $0.80$), small for colour (the image alone already nearly saturates). Reading the gaps this way, the ablation is not just a warning; it quantifies how much each non-image input mattered.

The logic to take away: if we had reported the $E_{\text{full}}$ numbers as evidence that "the model understands redshift to $R^2 = 0.98$", we would be inflating a real but partly circular result. The model can score that high partly because it was told the answer. The $E_{\text{img}}$ numbers, where the answer was withheld, are the honest measure of what the model inferred. This is why the whole concept analysis (this section and the disentanglement and SAE sections that follow) is anchored on $E_{\text{img}}$ unless a property could not leak.

One nuance worth stating. For the small-sample physical properties (mass, sSFR), $E_{\text{full}}$ also scores higher: mass $0.870$ versus $0.721$ on $E_{\text{img}}$, and sSFR $0.848$ versus $0.760$. But here the gap is not pure leakage, because mass and sSFR were not direct inputs to $E_{\text{full}}$ either.

$E_{\text{full}}$ does better because it additionally saw the fluxes and redshift, which are physically informative about mass and sSFR. Brighter and redder galaxies tend to be more massive, and redshift sets the distance and so the conversion from observed brightness to physical luminosity.

So the $E_{\text{full}}$ advantage there is partly legitimate extra information, not copying. We still report the $E_{\text{img}}$ numbers as the headline because they isolate what the image alone supports.

The Sersic index is the exception that proves the rule. $E_{\text{full}} = 0.668$ and $E_{\text{img}} = 0.664$ are essentially identical, which says structure (light concentration) is read from the image and adding fluxes and redshift buys nothing for it. That makes sense: concentration is a purely geometric, distance-independent feature of the picture, so the extra inputs have nothing to add.

### 10.7 Predicted versus true, and the photo-z caveat

$R^2$ is one summary number. To see what the probe is actually doing, Figure 10 plots predicted against true for the three properties where the stakes are highest: redshift (the famous one), and the two non-input physical properties, mass and sSFR.

![Figure 10. Image-only probe: predicted versus true on the held-out test set.](figures/10_probe_pred_vs_true.png)

Figure 10. Three panels, one per property, all from the image-only embedding $E_{\text{img}}$ and all on the held-out 20% test set. Left: redshift (mostly photo-z). Middle: stellar mass $\log_{10}(M_*/M_\odot)$ (mass in solar masses, log scale). Right: $\log_{10}$ sSFR in units of $\text{yr}^{-1}$ (inverse years, the natural unit for a rate per unit mass).

In each panel the x-axis is the true catalogue value and the y-axis is the probe's prediction, in the same units. The points are galaxies, drawn as a density: the colour scale (a perceptual colour map, dark to bright) counts galaxies per bin, so bright regions are where many galaxies pile up and dark scattered points are sparse. The red dashed diagonal is the line $y = x$, perfect prediction; points on it are exactly right. Each panel is annotated with its held-out $R^2$ and its test-set size $n_{\text{test}}$.

What to look for: in all three panels the bright ridge of galaxies lies along the $y = x$ diagonal, which means the probe tracks the true value across the whole range, not just on average. The redshift panel ($R^2 = 0.800$, $n_{\text{test}} = 9{,}680$) is the tightest, with a dense narrow ridge. Mass ($R^2 = 0.721$, $n_{\text{test}} = 746$) and sSFR ($R^2 = 0.761$, $n_{\text{test}} = 895$) are looser, with more scatter around the diagonal, consistent with their lower $R^2$ and smaller test sets.

A useful detail to check in the plot is that the ridge stays on the diagonal at both ends, not just in the crowded middle. A probe that only got the average right would bend away from $y = x$ at the extremes (predicting too high for low-mass galaxies and too low for high-mass ones, the regression-to-the-mean tilt). The mass and sSFR ridges stay reasonably straight across their range, which says the probe captures the property itself and not merely its average. Measured.

The slight discrepancy in the panel labels is just rounding from the same underlying numbers: the sSFR panel prints $R^2 = 0.761$ while the table value is $0.760$ (the table rounds $0.7603$, the figure rounds the bootstrap-resampled point estimate). They are the same measurement.

Now the caveat this section owns, and it is the load-bearing one. The redshift labels are mostly photometric redshifts, "photo-z" for short.

A spectroscopic redshift comes from measuring how far a galaxy's spectral lines are shifted, which is a direct distance-and-recession measurement. A photometric redshift is an estimate of redshift derived from the galaxy's broad-band colours, because galaxies of different redshift have systematically different colours.

In our sample only 6,883 galaxies have a true spectroscopic redshift; the rest carry photo-z (Section 4). So the redshift label we probe against is, for most galaxies, itself a colour-based estimate rather than a direct measurement.

The catch is that photo-z is itself a function of colour. So when the image-only probe "predicts redshift" at $R^2 = 0.80$, part of what it is doing is this: read the colour off the image (which it does at $R^2 \approx 0.96$), then map colour to photo-z (which is roughly what the photo-z pipeline did in the first place).

In short, image $\to$ redshift here is partly image $\to$ colour $\to$ photo-z. The $0.80$ is real and the model genuinely recovers a redshift-correlated quantity from pixels, but we must not over-read it as the model independently understanding cosmological distance. The chain may run through colour.

We cannot fully separate the two routes with the data in hand. Splitting the spectroscopic-only subset would let us probe true redshift directly, but with only 6,883 spectroscopic galaxies the test would be far noisier and we would be comparing different samples.

So we state the limit honestly. The redshift number is a decodability result with a known shortcut, not clean proof of independent redshift inference. Section 20 carries this as a named limitation, and the synthesis in Section 19 leans on mass and sSFR, not redshift, when it argues that the embedding encodes physics.

This is exactly why mass and sSFR are the cleaner evidence, and we say so plainly.

Stellar mass and specific star-formation rate come from external SED-fitting catalogues (fits to the galaxy's spectral energy distribution across many bands), not from a colour-based shortcut, and they were never inputs to $E_{\text{img}}$. They are not trivially "visible" in the image the way colour is. There is no analogue of the photo-z chain for them: the catalogue values do not pass through a single colour bottleneck, so a probe cannot recover them by the colour shortcut alone.

Recovering mass at $R^2 = 0.721$ and sSFR at $0.760$ from the image alone is therefore the strongest single piece of evidence in this section that the visual representation encodes galaxy physics it was not given, with no obvious leakage path and no colour-shortcut to explain it away.

The intervals are wider than for the full-sample colours ([0.659, 0.772] for mass and [0.692, 0.816] for sSFR) because the cross-matched samples are small (a few thousand galaxies, with held-out test sizes of 746 and 895). We report that width honestly rather than rounding it away.

But both intervals sit well above zero, so the signal is not a fluctuation. Even at the low end of its CI, mass is recovered at $R^2 \approx 0.66$ and sSFR at $\approx 0.69$, which is still most of the variance. The result does not depend on reading the point estimate optimistically; the whole interval tells the same story.

### 10.8 Connecting back to the original claim

The authors of AION-1 said the model "organises objects along physically meaningful directions" (Section 1). This section is the most direct test of that claim that the report contains, and it largely supports it, with one sharpened qualification.

The support: physically meaningful properties really are accessible along directions in the embedding, strongly enough that a flat linear readout recovers most of their variance, and this holds even for properties (mass, sSFR) that the image-only model was never given.

A representation that scattered physics randomly across its 1024 dimensions could not produce these scores. So "physically meaningful directions" is a fair description of what we measure, at the level of linear accessibility.

The qualification: "directions" in the plural is doing real work, and decodability alone cannot say the directions are clean or separate. That is the next question.

### 10.9 What this section establishes and what it does not

What it establishes, measured. From the image alone, a linear probe recovers colour almost perfectly ($0.91$ to $0.96$), redshift well ($0.80$), broad morphology well ($0.68$ to $0.79$), and, the key result, two non-input physical properties, stellar mass ($0.72$) and sSFR ($0.76$), at roughly three-quarters of their variance. Decodability falls off only for fine conditional morphology (bars at $0.55$, arms at $0.25$), where the labels are sparse and the intervals are wide.

The modality ablation shows directly why we trust the image-only numbers. The multimodal embedding scores higher on exactly the quantities it was fed (redshift $0.98$, colours $0.97$ to $0.99$), which is leakage, so $E_{\text{img}}$ is the fair measure.

What it does not establish. A high $R^2$ says a property is linearly accessible in the embedding; it does not say the model has an isolated, clean concept for it. A property could be decodable while being smeared across many entangled directions. Whether the model keeps these concepts separate is a different question, the disentanglement question, and Section 11 takes it up by asking whether the probe directions for different properties point in genuinely different directions or just inherit the correlations among the labels themselves.

And the photo-z caveat above bounds the redshift claim. Read mass and sSFR, not redshift, as the evidence that pixels encode physics the model was never told. That single sentence is the honest summary of this section's strongest result.
