# The Native Representational Geometry of an Astronomical Foundation Model: A Geometry-and-Concept Study of AION-1's Galaxy Embedding

*Technical report. N = 48,398 galaxies; AION-1-Large, frozen. 18 figures. Every number is
measured from the committed analysis; interpretations are tagged "interpreted". Figure
paths are relative to this file's folder, so preview from the repository root.*

## Table of contents

- [0. Front matter](#0-front-matter)
- [1. The question, and why it is still open](#1-the-question-and-why-it-is-still-open)
- [2. Background: galaxies as a measurable shape](#2-background-galaxies-as-a-measurable-shape)
- [3. The model and the representation we study](#3-the-model-and-the-representation-we-study)
- [4. Data and preprocessing](#4-data-and-preprocessing)
- [5. The metric problem and the concentration diagnostic](#5-the-metric-problem-and-the-concentration-diagnostic)
- [6. Intrinsic dimension I: definition and estimators](#6-intrinsic-dimension-i-definition-and-estimators)
- [7. Intrinsic dimension II: validation and the AION result](#7-intrinsic-dimension-ii-validation-and-the-aion-result)
- [8. The shape of the manifold: diffusion maps](#8-the-shape-of-the-manifold-diffusion-maps)
- [9. Reading physics off the manifold](#9-reading-physics-off-the-manifold)
- [10. Decodability: how much physics the embedding carries](#10-decodability-how-much-physics-the-embedding-carries)
- [11. Disentanglement and neighbourhood purity](#11-disentanglement-and-neighbourhood-purity)
- [12. Concept discovery I: the sparse autoencoder](#12-concept-discovery-i-the-sparse-autoencoder)
- [13. Concept discovery II: the concepts and the alien candidates](#13-concept-discovery-ii-the-concepts-and-the-alien-candidates)
- [14. Curvature and tree-likeness](#14-curvature-and-tree-likeness)
- [15. Topology: pieces and holes](#15-topology-pieces-and-holes)
- [16. Heterogeneity: different types, different dimension?](#16-heterogeneity-different-types-different-dimension)
- [17. The geometry of concepts: an honest null](#17-the-geometry-of-concepts-an-honest-null)
- [18. The manifold made visible](#18-the-manifold-made-visible)
- [19. Synthesis: the coherent picture](#19-synthesis-the-coherent-picture)
- [20. Limitations and threats to validity](#20-limitations-and-threats-to-validity)
- [21. Conclusions and what a full study should do next](#21-conclusions-and-what-a-full-study-should-do-next)
- [22. Glossary of terms and symbols](#22-glossary-of-terms-and-symbols)
- [23. Appendix: thresholds, statistical claims, reproducibility](#23-appendix-thresholds-statistical-claims-reproducibility)

## 0. Front matter

### Thesis in one paragraph

A foundation model is a large neural network trained once on a flood of unlabeled data,
then reused for many tasks without retraining.

AION-1 is one such model for astronomy. It reads galaxy images and catalogue measurements
and turns each galaxy into a list of 1024 numbers, a point in a 1024-dimensional space we
call the embedding. Its authors said the model "organises objects along physically
meaningful directions," but they never measured the shape of that space.

This report measures it. We take the model frozen, with no extra training, and ask three
plain questions about the cloud of 48,398 galaxy points it produces. How many independent
directions does the cloud really use (its intrinsic dimension)? What is its shape: one
connected body or separate clusters, flat or curved, holed or simply connected? And does
the model carry concepts of its own, beyond the human labels we already have?

The short answer is that AION-1's galaxy space is a single, smooth, low-dimensional
continuum, about 10 to 12 real directions out of 1024, gently curved and not tree-like,
that encodes real galaxy physics, including properties it was never told, strongly enough
to read back off the image alone. That is the whole report in one sentence. The rest earns
it.

### Executive summary: the four headline findings

Before the findings, the scope in one breath. We analyse $N = 48{,}398$ galaxies (drawn
from Galaxy Zoo DESI, redshift $0 < z < 1$, fixed random seed; 96.8% of a 50,000 draw had
usable four-band images). The model is AION-1-Large, 800 million parameters, frozen and
never fine-tuned, with no morphology labels in its training. It emits one vector per image
token, and we mean-pool those into a single 1024-number vector per galaxy. We study two
versions of that vector: $E_{\text{full}}$, which fuses image plus photometry plus
redshift, used for the geometry; and $E_{\text{img}}$, image only, the leakage-free set
used for concept probes, because anything fed in as an input is trivially decodable and
would make a "concept axis" circular. Every distance below is computed after z-scoring
each of the 1024 dimensions (subtracting its mean, dividing by its standard deviation) so
that no single high-variance dimension dominates.

We organise the work around four results. Each carries a number, a confidence interval or
null where we have one, and an explicit tag: "measured" means a direct readout from the
data, "interpreted" means an inference we draw from the measurement and would defend but
cannot prove.

One discipline runs under all four. Every estimator was first run on synthetic data of
known answer at the same sample size, so a number is only trusted on AION-1 after it
recovered the truth on a control. The dimension estimators were checked on a sphere, a
swiss roll, and a plane; the curvature and topology tools were checked against a matched
random cloud and a synthetic tree; the concept scores were checked against label-shuffle
and permutation nulls. When a tool failed its control (the small-scale TwoNN inflating on
tight clusters, Forman-Ricci going structurally negative on a neighbour graph), we say so
and do not lean on it. That is why some readings below are demoted to controls rather than
estimates.

**Finding 1: the embedding uses only about 10 to 12 real directions, not 1024.** The
intrinsic dimension is the count of independent knobs you would need to describe a
galaxy's position in the cloud, which can be far below the ambient 1024. We estimated it
four ways, each resting on a different piece of math, after first checking every estimator
on synthetic shapes of known dimension at the same sample size.

On AION-1, three estimators that work at manifold scale agree: the Gride scale curve
flattens near 10, the local Levina-Bickel maximum-likelihood estimator plateaus near 11.4,
and the linear PCA participation ratio is about 11 (all measured). The smallest-scale
TwoNN reading of about 16.5 is inflated by sampling noise and is not the estimate; the
plateau at larger scale is (interpreted). All of these sit below the resolution ceiling of
$\log_2 N = 15.56$ that a sample of 48,398 points can support.

Compressing 1024 nominal dimensions down to about 11 is a large reduction, and the
closeness of the nonlinear estimate (about 10) to the linear one (about 11) tells us the
manifold is only weakly curved at large scale (interpreted). Honest caveat: this dimension
sits at or just above an optimistic astrophysical prior band of 4 to 10, so
"low-dimensional" is true but not as low as the best case.

**Finding 2: it is one smooth, simply-connected continuum, not a set of discrete
clusters.** Three independent geometric tests agree. The diffusion-map eigenvalue spectrum
decays smoothly with no dominant gap (a gap would mean the cloud breaks into separate
groups), and there is a single connected component (measured).

The number of separate pieces, the topological invariant $\beta_0$, is 1: cutting the
longest links of the minimum spanning tree peels off only lone outliers while the roughly
48,000-point body stays whole, so this is not a two-piece red/blue split (measured). The
number of independent loops or holes, $\beta_1$, is 0: significant loops essentially never
recur across ten independent subsamples or survive a change of distance metric (measured,
from 2,000-point subsamples).

We read the continuum as the model's encoding of the known morphological and colour
sequences of galaxies, which themselves vary continuously (interpreted).

**Finding 3: physics is strongly decodable from the image alone, and the concept
directions are separated.** A linear probe is a single straight-line map fit from the 1024
numbers to a physical property; its $R^2$ is the fraction of held-out variance it
explains, where 1 is perfect and 0 is no better than guessing the mean.

Probing the image-only embedding, where redshift and colour are never given as inputs so
decoding them is a true inference, we measure stellar mass $R^2 = 0.721$ (95% interval
0.659 to 0.772, $n = 3{,}728$) and specific star-formation rate $R^2 = 0.760$ (0.692 to
0.816, $n = 4{,}760$), both from pixels alone (measured). Colours read back at 0.91 to
0.96 and morphology vote fractions at 0.68 to 0.79 (measured). The model also separates
concepts more than the labels' own correlations would force: the angle between probe
directions exceeds the label-correlation null by up to about 19 degrees (measured).

We read this as a visual representation that has internalised galaxy physics it was not
handed (interpreted). Honest caveat: the redshift label is mostly photometric
(colour-derived), so image-to-redshift is partly image-to-colour-to-photo-z; the mass and
star-formation results are the cleaner evidence because those quantities were never
inputs.

**Finding 4: sparse autoencoders surface 335 seed-stable, physics-named concept directions
plus 59 candidate "alien" directions.** A sparse autoencoder learns an over-complete
dictionary so that each galaxy vector is rebuilt from only a few active concept units
(here 32 of them), which lets the embedding reveal its own axes instead of us probing for
human-named ones.

After scoring each unit against six labels with a label-shuffle null (the 95th-percentile
threshold is 0.0119), we find 717 units aligned to a known property and 335 of those also
stable across five random training seeds (measured). The strongest single alignment is
0.279, about 23 times the null (measured); colour and morphology are the clearest single
concepts, while redshift spreads over many weak units.

Separately, 59 units are seed-stable and high-novelty (most of their activation variance
is not linearly explained by any label we have) yet not aligned to any label, the "alien"
candidates (measured). We flag these hard: they are correlational only, no causal or
ablation test was run, so they are the most interesting and the least certain result in
the report (interpreted).

### The honest one-line verdict

AION-1's galaxy embedding is a low-dimensional (about 10 to 12), single, simply-connected,
mostly positively curved continuum that encodes real galaxy physics strongly, with weak
localised branching and a small set of candidate concepts beyond the human labels; the
dimension sits at or just above an optimistic prior, the redshift signal is partly photo-z
routed through colour, and the alien concepts are correlational and untested for cause.

### How to read this report

The audience is one technical reader who may know astronomy or machine learning but
probably not both, so every term is defined in plain words the first time it appears, and
we explain both the math and the physics.

We separate two kinds of statement throughout. A "measured" claim is a direct readout from
data and carries its uncertainty (a confidence interval, a bootstrap range, a null
threshold, or an anchor to compare against). An "interpreted" claim is an inference,
flagged with words like "we read this as," that we will defend but cannot prove from this
data alone. Where the data cannot decide, we say so.

The sections build in order. Sections 1 to 4 set up the question, the galaxy physics a
faithful embedding should reflect, the model itself, and the data and preprocessing.
Section 5 explains why the choice of distance matters in high dimensions and picks the
metric we use downstream. Sections 6 and 7 define and validate the intrinsic-dimension
estimators and report Finding 1. Sections 8 and 9 give the diffusion-map shape and read
physics off it.

Sections 10 and 11 cover decodability and concept separation (Finding 3). Sections 12 and
13 cover the sparse autoencoder and its concepts (Finding 4). Sections 14 through 17 cover
curvature, topology (Finding 2), population-dependent dimension, and one honest null
result. Section 18 shows real galaxy cutouts along the recovered axes. Sections 19 to 21
synthesise, list limitations, and state conclusions. Section 22 is a glossary of every
term and symbol, and Section 23 is an appendix of every threshold and statistical decision
rule with a reproducibility note.

If you want only the bottom line, read this front matter and Section 19. If you doubt a
number, Section 23 and the glossary point you to the exact result file.

A note on what this study is and is not. The embedding is a static snapshot of present-day
galaxies, so it shows the density of populations that exist now, not a movie of galaxies
evolving over time. We never claim the model "learned evolution." The question of whether
the model rediscovered the classic Hubble tuning-fork sequence of galaxy shapes is a
control, not the thesis; the thesis is the native geometry and the concepts.

### Conventions and a few symbols you meet early

So you do not have to hunt later, here are the conventions used everywhere and the handful
of symbols that appear before their home section defines them in full. Each is restated in
plain words where it is first used in earnest, and all of them live in the Section 22
glossary.

| Symbol or term | Plain meaning |
|---|---|
| $N = 48{,}398$ | Number of galaxies analysed. |
| $d = 1024$ | Ambient embedding dimension, the length of each galaxy's vector. |
| $E_{\text{full}}$ | Embedding from image plus photometry plus redshift; used for geometry. |
| $E_{\text{img}}$ | Image-only embedding; the leakage-free set used for concept probes. |
| Intrinsic dimension | Count of independent directions the cloud actually uses, far below $d$. |
| $R^2$ | Fraction of held-out variance a linear probe explains (1 perfect, 0 mean-guess). |
| $\beta_0,\ \beta_1$ | Number of separate pieces, and number of independent loops or holes. |
| "measured" | A direct readout from the data, reported with its uncertainty. |
| "interpreted" | An inference we draw from a measurement and would defend but not prove. |

Two reading rules follow from the table. First, treat any number without an interval,
anchor, or null as still measured but read its uncertainty in the owning section, because
the headline here is a summary. Second, when a sentence says "we read this as" or carries
an "(interpreted)" tag, that is the boundary between what the data shows and what we think
it means; the report keeps that boundary visible on purpose.

### Figure index

Eighteen figures carry the quantitative load. Each is embedded in the section that owns
it, with a full caption naming every axis, colour scale, legend, and reference line.
Figures are numbered by the two-digit number in their filename. This index is the map.

| Figure | File | One-line description |
|---|---|---|
| 1 | figures/01_id_synthetic_validation.png | Intrinsic-dimension estimators recover known dimensions on synthetic manifolds (the validation check). |
| 2 | figures/02_id_aion_scale_curves.png | AION intrinsic dimension versus neighbour scale (Gride and local-MLE), showing the 10 to 12 plateau. |
| 3 | figures/03_metric_concentration.png | Distance contrast (RDR) and nearest-neighbour ratio across five candidate metrics. |
| 4 | figures/04_diffusion_spectrum.png | Diffusion-map eigenvalue spectrum and consecutive gaps (smooth decay, no dominant gap). |
| 5 | figures/05_diffusion_full_labels.png | Diffusion embedding (coords 1 by 2) coloured by six full-sample labels. |
| 6 | figures/06_diffusion_sparse_physics.png | Same embedding coloured by stellar mass, sSFR, and Sersic index (cross-matched subset). |
| 7 | figures/07_probe_decodability.png | Linear-probe $R^2$ with 95% intervals, image-only, full-sample and subset labels. |
| 8 | figures/08_modality_ablation.png | Image-only versus multimodal decodability, exposing input leakage. |
| 9 | figures/09_disentanglement_angles.png | Probe-direction angle versus the label-correlation null (the orthogonality excess). |
| 10 | figures/10_probe_pred_vs_true.png | Predicted versus true on held-out data for redshift, mass, and sSFR. |
| 11 | figures/11_sae_frontier.png | Sparse-autoencoder reconstruction error (FVU) versus sparsity level $k$. |
| 12 | figures/12_sae_named_concepts.png | Counts of aligned and seed-stable autoencoder features per named concept. |
| 13 | figures/13_sae_alignment_novelty.png | Feature alignment versus the shuffle null, and alignment versus novelty, marking the alien candidates. |
| 14 | figures/14_curvature.png | Delta-hyperbolicity against anchors plus the Ollivier-Ricci curvature summary. |
| 15 | figures/15_topology.png | $\beta_0$ via minimum-spanning-tree cuts and $\beta_1$ via loop persistence versus threshold. |
| 16 | figures/16_stratified_id.png | The g-r colour split plus passive versus star-forming intrinsic dimension and the delta-ID. |
| 17 | figures/17_pca_variance.png | Linear PCA cumulative variance, the linear baseline for the dimension result. |
| 18 | figures/18_manifold_galaxy_montage.png | Real galaxy cutouts sampled along the colour and morphology axes. |

With the map in hand, Section 1 states the question and why, despite the model's own
claims, it is still open.


## 1. The question, and why it is still open

Start with the object at the center of this report: an embedding. When a large neural
network looks at a galaxy, it does not store the picture. It produces a list of numbers.
In our case that list has 1024 entries, so each galaxy becomes a single point in a
1024-dimensional space.

That point is the model's compressed opinion about everything it can tell from the inputs
it was given. Two galaxies the model judges to be similar land close together. Two it
judges different land far apart. The collection of all these points, one per galaxy, is
the embedding, and it is the only thing we study here.

Why 1024 and not some other number? Because that is the width the model's designers chose
for its internal vectors. The width is a capacity, not a count of meaningful features. A
model can have a wide internal vector and still use only a small part of it, the way a
spreadsheet with a thousand columns might have real data in only ten. Distinguishing the
capacity (1024) from the part actually used (the intrinsic dimension) is one of the first
things this report does.

We never ask the model to label anything. We ask where it puts things and what the
arrangement looks like.

This is a different exercise from the usual way machine-learning models are judged. The
common question is "how well does it perform on a task," scored by accuracy on some
benchmark. Our question is "what shape did it build," which is prior to any task. A model
can score well and still have an arrangement that is opaque, tangled, or
higher-dimensional than the data warrants. By measuring the shape directly, we get at
something the benchmark numbers hide: not whether the representation works, but what it
is.

### 1.1 What an embedding is, in plain words

A foundation model is a single large model trained once on a large pile of unlabeled data,
then reused for many tasks without retraining. The phrase comes from the idea that the
model is a shared foundation you build on top of. The model we study, AION-1, is one of
these.

It was trained on astronomical survey data by a self-supervised objective, meaning it
learned from the data alone with no human-supplied answer key. Nobody told it which
galaxies are spirals and which are ellipticals. Nobody handed it a star-formation rate. It
learned by filling in deliberately hidden pieces of its inputs, and to make those guesses
it had to build an internal summary of each object. That summary, read out as a
1024-number vector, is the embedding we measure.

So the embedding is a learned 1024-number summary of a galaxy. Hold onto that plain
definition.

Everything in this report is an attempt to describe the shape of the cloud those summaries
make, and to ask what the shape means. The number 1024 is the ambient dimension, the count
of coordinate axes the vectors literally live along. It is not the number of axes the
cloud actually uses, which is almost always far smaller, and which Sections 6 and 7
measure.

One more word on why a cloud of points is the right way to think about this. We are not
studying the model's weights, the millions of internal numbers fixed during training. We
are studying its outputs, one vector per galaxy. The weights would tell us how the machine
works; the outputs tell us how it sees the data. Those are different questions, and this
report is firmly about the second one. The shape of the output cloud is a fingerprint of
how the model has chosen to arrange galaxies, and that arrangement is what we can compare
against known physics.

The size of the sample matters too. We analyze $N = 48{,}398$ galaxies (Section 4 gives
the selection). That count is not just a detail of scale. It sets a hard ceiling on how
fine a geometry we can even resolve, because you cannot measure structure with more
independent directions than your data can populate. We make that ceiling explicit later as
$\log_2(N) \approx 15.56$, and every dimension estimate is read against it.

### 1.2 The claim the authors made, and the gap they left

The team that built AION-1 reported that the model organizes objects along directions that
line up with physically meaningful properties. In plain words: move along one direction in
the embedding and a real galaxy property changes smoothly. That is a strong and
interesting claim, and our own probes in later sections give it real support.

But it leaves a gap. Saying that some directions are meaningful is not the same as
describing the whole shape.

It does not tell you how many independent directions the cloud actually uses. It does not
tell you whether the cloud is one connected body or several separate islands. It does not
tell you whether the body is flat, curved, or branched like a tree. It does not tell you
whether the model found structure that no human label names.

The authors measured that the embedding is useful. They did not measure its geometry. That
gap is the opening for this work.

To be fair to them, usefulness is the harder thing to demonstrate and the more immediately
valuable one. A model that lets you read off galaxy properties cheaply is worth a great
deal regardless of its shape. Our point is not that the shape question is more important.
It is that the shape question is separate and unanswered, and that answering it tells you
something the usefulness numbers cannot: whether the model's view of galaxies is simple or
tangled, continuous or broken, and whether it contains structure no one has named.

### 1.3 Geometry, in a concrete and measurable sense

We treat "geometry" here as something measurable, not as a metaphor. A cloud of points in
high-dimensional space has a shape, and that shape has properties you can estimate with
numbers and report with error bars.

Five properties cover most of what we mean by shape.

How many directions does the cloud really fill, its intrinsic dimension? Does distance
behave sensibly inside it, or does every point look equally far from every other, the
concentration problem? Is it one piece or many, its connectedness? Does it have loops or
holes, its topology? Does it curve like a ball, bend like a saddle, or split like a
branch, its curvature?

Each maps to a section. Dimension is Sections 6 and 7, concentration is Section 5,
connectedness and topology are Sections 8 and 15, and curvature is Section 14. Reading
them together is how we assemble a single coherent description of the shape rather than
five disconnected numbers.

These are not vague questions. Each has a definition, an estimator, a validation on data
with a known answer, and a confidence interval. The body of this report is the set of
those answers.

A word on the word "manifold," which recurs throughout. A manifold is a shape that looks
flat if you zoom in close enough, even when it curves on a large scale, the way the ground
under your feet looks flat although the Earth is a ball. When we say the embedding may
live on a low-dimensional manifold inside the 1024-dimensional space, we mean the points
may sit on a thin curved sheet with only a handful of independent directions, not
scattered through the full volume. Whether that picture holds, and how thin the sheet is,
is the first thing we measure. We do not assume it. A self-supervised model could in
principle produce a messy blob with no clean low-dimensional structure, and part of the
work in Sections 6 to 8 is checking that it did not.

### 1.4 The two open questions we ask

We organize the work around two questions, and we name them so later sections can point
back.

Question B is the native-geometry question. What is the intrinsic shape of the embedding?
Concretely: its intrinsic dimension, its connectedness and topology, its curvature, and
whether distance is even reliable inside it.

We call this the native geometry because we want the shape the model itself imposes, read
with as few of our own assumptions as we can manage. Where a choice has to be made, which
distance to use, which estimator to trust, we make it explicitly, test it on synthetic
data with a known answer, and report what the choice does.

The word "native" is doing real work in that sentence. A high-dimensional cloud does not
hand you a distance for free. You have to choose how to measure how far apart two points
are, and that choice can change the answer to nearly every geometric question.
Straight-line distance is the obvious option and often the wrong one in high dimensions.
Section 5 makes this concrete and explains why we adopt a distance that follows the
cloud's own density rather than cutting across empty space. "Native" means we work hard to
read the geometry the model built, not an artifact of a careless distance choice.

We label them B and C, rather than starting at A, on purpose. There is an implicit
Question A behind both, the confirmatory control discussed just below, and we keep the
numbering to remind ourselves that the control is prior to and separate from the two
questions that carry the report.

Question C is the concept question. Does the model organize galaxies by structure that
goes beyond the human taxonomy? A taxonomy is just a naming scheme, the set of categories
people use to sort galaxies: spiral, elliptical, merger, and so on.

A model trained without those names is free to carve the data along axes we never wrote
down. Question C asks whether it did, and if so, whether those self-found axes correspond
to anything we can recognize.

We answer it with a sparse autoencoder, a second small model that forces the embedding to
explain each galaxy using only a few active units at a time, so we can read those units
one by one (Sections 12 and 13). The appeal of this method is that it lets the embedding
reveal its own axes, rather than us probing for human-named ones. When we fit a probe for
"redshift," we have already decided redshift is the thing to look for. A sparse
autoencoder does not start from our list of names. It carves the cloud along whatever
directions are most economical, and only afterward do we ask which of those directions
match something we recognize and which do not.

We flag now, and will repeat, that any "new" axis we find this way is correlational. We
can show a unit tracks something, but we ran no causal test that would prove the model
uses it. The honest version of the finding is "here are stable directions the model uses
that no label we hold explains," not "here is a new physical concept the model
discovered." That distinction is part of the answer.

### 1.5 The confirmatory question is a control, not the thesis

There is a third question floating in the background, and we keep it firmly in the
background on purpose.

Call it the confirmatory question: did the model rediscover the things astronomers already
know, like the smooth sequence from spirals to ellipticals (the Hubble tuning fork, named
for the fork-shaped diagram Edwin Hubble drew to arrange galaxy shapes)? That is a
control, not a thesis.

If the model had failed to recover well-established physics, we would distrust everything
else it told us, so we check. But recovering known physics is the floor, not the finding.
The finding we are after is the full shape and the candidate structure beyond the named
categories.

We will be careful throughout not to dress up a passed control as a discovery.

There is one fact about the model that makes even the control interesting, and it is worth
stating early because it changes how to read every result. The model never saw a
morphology label. It was never told which galaxies are spirals, never given a
star-formation rate, never shown the human categories. So when its embedding turns out to
encode those properties, that is not the model repeating a lesson. It is the model having
inferred the structure on its own from the raw light. A passed control here still says
something: that the relevant physics is recoverable from the data without supervision. We
hold that as the floor and look above it for the rest.

### 1.6 Where this sits in the machine-learning lineage

None of our tools are new to machine learning, and it helps to say plainly which lines of
earlier work we stand on. We are not claiming to invent methods. We are applying
established representation-geometry tools to an astronomical model that had not been
measured this way, validating each tool on data with a known answer before we trust it on
the real embedding.

There are three such lines, and we touch each one only as far as our results require.

The first line is the study of the intrinsic dimension of learned representations. A
network may emit a 1024-number vector, but those numbers are usually far from independent.
The data may really live on a much thinner sheet inside the big space, the way a sheet of
paper is a two-dimensional thing even when it sits in a three-dimensional room.

People have repeatedly found that deep-network representations have an intrinsic dimension
far below their nominal size, and that the small number is informative: it bounds how much
independent structure the representation actually carries.

There is a tension in this literature that we have to respect. A low intrinsic dimension
can mean two very different things. It can mean the model found the real, compact
structure of the data, which is the good story. Or it can mean the model collapsed,
throwing away usable directions and crowding everything onto a thin sheet, which is a
known failure mode of self-supervised training called dimensional collapse. A small number
alone does not tell you which. So we do not treat a low dimension as automatic proof of a
clean manifold. We check the shape itself (Sections 8, 14, 15) and ask whether the
low-dimensional sheet is smooth and physically organized, which is what separates genuine
compression from collapse.

We measure the dimension directly for AION-1 with four independent estimators (Sections 6
and 7). Our headline, an intrinsic dimension near 10 to 12 out of 1024, is read against
the $\log_2(N)$ sampling ceiling and against several synthetic controls, not asserted on
faith.

The second line is the geometry of concepts: the idea that a human-understandable property
can correspond to a direction in representation space, so that adding a fixed vector moves
you from "without the property" to "with it."

This is the picture behind word-vector analogies in language models and behind
linear-probe studies more broadly. The classic example from language is that the direction
from "man" to "woman" is roughly the same as the direction from "king" to "queen," so a
property like gender behaves like a fixed step in the space. We borrow the same idea and
ask whether galaxy properties like color or redshift behave like fixed steps in AION-1's
space.

We use it in two ways. We fit linear probes to test how much of a property a single
direction can recover (Sections 10 and 11). And we test one specific structural prediction
from this literature, that a child concept should equal its parent direction plus an
orthogonal part, in Section 17. We report the second test as an honest null: our data does
not establish that clean hierarchy here.

A related idea from this line of work is disentanglement: the notion that a good
representation keeps distinct concepts on distinct, ideally separate, directions, so that
changing one property does not drag others along. Two galaxy properties are correlated in
nature (redder galaxies tend to be more smooth, for instance), so we cannot expect their
directions to be exactly perpendicular. The sharper question is whether the model
separates them more than their natural correlation forces. Section 11 measures exactly
that, comparing the angle between two learned directions to the angle the labels' own
correlation would imply. We mention it here only to mark it as a third use of the
concept-geometry idea.

The third line is broader and more speculative.

It is the idea that large models trained on enough data may converge toward a shared
internal structure, so that the geometry of one model's representation reflects the
structure of the world rather than the quirks of one training run. The appeal is obvious:
if true, the geometry of a representation becomes a window onto the data's own structure.
The risk is equally obvious: the appeal can lead you to over-read a single model's shape
as a fact about nature.

We do not test convergence across models here; we have one model. We mention the idea only
to place our question.

If a representation's geometry reflects real structure in galaxies, then measuring that
geometry is a way of asking what the data itself is shaped like, seen through one careful
instrument. That is a motivation, not a result, and we keep it labeled as such.

The caution cuts both ways. If two models agreed on a geometry, that might reflect shared
structure in galaxies, or it might reflect shared training recipes and shared biases. With
one model we cannot separate those, so we make no convergence claim at all. What we can do
is measure one instrument carefully and report what it shows, which is a clean and modest
goal. The cross-model comparison is something we flag in Section 21 as a clear next step,
not something we attempt here.

### 1.7 What counts as an answer, and what does not

Two ground rules carry through every later section, and we set them here so they are not a
surprise.

First, we separate what we measured from what we read into it. A measured statement is a
direct readout with an error bar: an intrinsic-dimension estimate with a bootstrap
interval, a probe score with a confidence interval, a count of connected pieces. An
interpreted statement is the physical story we attach: that a smoothly connected cloud
matches the known smoothness of galaxy populations, or that a thin region of negative
curvature is consistent with a branch point.

The measurements are the spine of the report. The interpretations are flagged as
interpretations, and where two readings are both consistent with the data, we say the data
cannot decide.

This is not pedantry. The whole value of a geometry study is that the geometry is
measured, not asserted. The moment we let an attractive physical story stand in for a
measurement, we have given up the one advantage this approach has over hand-waving.

We want this line to be visible in the prose, not just promised here. So when you read a
sentence like "the cloud is one connected body," that is measured: a counted number of
pieces. When you read "this matches the smooth physical sequence from spirals to
ellipticals," that is interpreted: a physical reading laid over the measurement. We try
never to let the second kind of sentence stand in for the first.

Second, we always show the uncertainty and the baseline. A number with no error bar and no
point of comparison is not evidence.

So an intrinsic dimension is reported against the resolution ceiling set by the sample
size and against synthetic manifolds whose dimension we already know. A curvature is
reported against a matched random cloud and against a synthetic tree, because the absolute
number means little until you see what "more curved" and "less curved" look like for known
shapes. A concept-alignment score is reported against a shuffled-label null, the value you
would get by chance. Throughout, the comparison is the point.

### 1.8 Validate the ruler before measuring the thing

There is a third habit that runs through the report, and it follows from the second.
Before we trust any estimator on the real embedding, we run it on synthetic data whose
answer we already know, at the same sample size, with the same settings.

The reason is simple. Every estimator has biases. A dimension estimator might read high on
a curved sheet, or inflate on a tight cluster, or sag at small scales where sampling noise
dominates. You cannot tell a true reading from a biased one by looking at the real data
alone, because you do not know the truth for the real data. That is the whole problem.

So we build a sphere whose dimension is exactly 5, a rolled-up sheet whose dimension is
exactly 2, a flat plane of dimension 5, and a couple of deliberate trap shapes, and we
check that each estimator returns the known answer on these before we point it at AION-1.
When an estimator passes the knowns and the estimators agree with each other on the real
data, the result is trustworthy. When an estimator reads oddly even on a known shape, we
say so and down-weight it. Sections 6 and 7 carry this out in full, and it is why our
intrinsic-dimension headline rests on agreement among independent methods that each passed
their controls, not on any single number.

This is also why a negative result is a real result in this report. When the validated
tool says "no significant effect," as it does for the concept-hierarchy test in Section
17, that null is as informative as a positive finding, because we have already shown the
tool can detect the effect when it is present. We report the nulls plainly and do not bury
them.

### 1.9 The arc of the report

It helps to know where this is going. The report moves from the instrument outward to the
meaning.

After this question and the physics background, Section 3 describes the model and Section
4 the data. Section 5 settles the distance question. Sections 6 and 7 measure intrinsic
dimension. Section 8 maps the shape and shows it is one continuous body. Sections 9
through 11 read physics off the manifold and test how much of it the embedding carries and
how cleanly the concepts separate. Sections 12 and 13 hunt for structure beyond the human
labels. Sections 14 through 16 measure curvature, topology, and whether different galaxy
populations have different dimension. Section 17 reports the one test that returned a
clean null, and Section 18 puts real galaxy pictures on the axes. The final sections
synthesize, list the limits honestly, and say what a fuller study should do next.

Every one of those measurements is validated on known-answer data first, reported with its
uncertainty, and compared to a baseline. That is the contract.

With those rules fixed, the next thing we need is the physics. Before we can say whether a
geometry is faithful, we have to know what a faithful embedding of galaxies should
reflect: the smooth sequence of shapes, the split between red and blue populations, and
the rough number of independent knobs the data really has. Section 2 lays that out, and it
ends with the load-bearing caveat that shapes the whole reading: a present-day embedding
shows a density of populations, not a movie of galaxies evolving.

## 2. Background: galaxies as a measurable shape

To judge whether the embedding is faithful, we first have to know what a faithful summary
of galaxies should reflect. This section lays out the physics in plain words, defines
every term we use later, and ends with a quantitative prior on how many independent knobs
the data really has.

It also states the one caveat that constrains every interpretation in this report: a
present-day embedding shows a density of populations, not a movie of evolution.

The structure of the section is simple. First the vocabulary, every term defined once.
Then the two big facts a faithful embedding should respect, the morphological continuum
and the color-mass bimodality. Then quenching and its multiple channels, which is where
the geometry gets interesting. Then the quantitative prior on dimension. Then the caveat.

A galaxy is a gravitationally bound system of stars, gas, dust, and dark matter. We do not
observe any of that directly. We observe the light that reaches us, spread across the sky
as an image and across wavelength as a spectrum or a set of broadband colors.

Almost everything astronomers say about a galaxy's physical state is inferred from that
light. So when we ask whether a model's embedding reflects galaxy physics, we are really
asking whether the arrangement of points lines up with the physical quantities astronomers
themselves infer from the same light. Those quantities are the vocabulary of this section.

The light comes in two forms, and the difference between them runs through the whole
report. An image is the light laid out on the sky: a picture, in a few broad filters, that
shows shape and structure. A spectrum is the same light laid out by wavelength: a
fine-grained curve that shows which atoms are present, how fast the gas moves, and exactly
how far the light has been stretched. Images are cheap and exist for huge numbers of
galaxies. Spectra are expensive and exist for far fewer. Most of the physical quantities
below can be estimated from either form, with spectra giving the sharper answer. The model
we study can in principle use both, though our run uses images plus a few catalogue
numbers and no spectra (Section 4 is explicit about this).

### 2.1 The terms, defined once

**Redshift** ($z$) is how much the light from a galaxy has been stretched to longer
(redder) wavelengths by the expansion of the Universe. Because the Universe expands, more
distant galaxies recede faster and their light is stretched more, so a larger $z$ means a
more distant galaxy seen at an earlier cosmic time. In our sample $0 < z < 1$, which
reaches back several billion years.

The range $0 < z < 1$ in our sample is worth a sentence. A redshift of zero is
here-and-now; a redshift of one corresponds to light that left its galaxy when the
Universe was roughly half its present age. So our sample spans the recent half of cosmic
history, a wide enough reach that redshift is a real source of variation the model has to
handle, not a fixed background.

Redshift can be measured two ways. **Spectroscopy** disperses the light into a full
wavelength spectrum and reads $z$ from the shift of known spectral lines; this is precise
but expensive, so it exists for only a fraction of galaxies. **Photometry** measures the
total brightness through a few broad filters and estimates $z$ from how the brightness is
distributed across those filters; this is cheap and exists for almost everything, but the
resulting **photometric redshift** (photo-z) is an estimate with real scatter, not a
direct line measurement.

This distinction matters later. Most of our redshift labels are photo-z, so they are
partly derived from color, and we treat color-related results accordingly.

The reason photo-z works at all is the same physics that makes color a useful number. The
expansion of the Universe shifts a galaxy's whole spectrum toward longer wavelengths, so a
more distant galaxy looks redder in the same set of filters. That shift is correlated with
the galaxy's intrinsic color, so an estimate of redshift built from broadband filters is
partly an estimate of color in disguise. When we later recover redshift from an image
alone, we keep in mind that part of that skill could be the model reading color and
inferring a color-based photo-z, rather than measuring distance from something
independent.

**Photometry** more generally means brightness measured through filters. Our images use
four filters named $g$, $r$, $i$, $z$. (Note: this filter $z$ is a band name and is
unrelated to redshift $z$; we keep them apart in the text.) They run from bluer ($g$) to
redder ($i$, and the $z$ band). A **magnitude** is an astronomer's brightness unit,
defined so that a smaller number means brighter and a difference of 5 magnitudes is a
factor of 100 in brightness. We will mostly use magnitudes through differences.

A **color** is the difference between two magnitudes, for example $g - r$ or $r - z$.
Because magnitude is a logarithmic brightness, a color is essentially the ratio of
brightness in two bands, and it tells you the shape of the light. A blue galaxy emits
relatively more short-wavelength light and has a small $g - r$. A red galaxy emits
relatively more long-wavelength light and has a large $g - r$.

Color is one of the most informative single numbers about a galaxy because it responds to
the ages of its stars. Young, massive, hot stars are blue and short-lived; old, low-mass
stars are red.

A galaxy still forming stars keeps making blue stars and stays blue. A galaxy that has
stopped reddens as its blue stars die off and only the long-lived red stars remain. So
color is a clock and a thermometer for star formation at once.

Two colors are better than one because they separate causes that a single color confuses.
The $g - r$ color is sensitive mainly to stellar age and recent star formation. The
$r - z$ color reaches farther into the red and adds sensitivity to dust and to redshift.
Using both, as we do, gives the model and the probes two partly independent handles on
what the light is doing, which is why we report decodability for $g - r$ and $r - z$
separately in later sections rather than collapsing color into one number.

**Stellar mass** ($M_\star$, in units of the Sun's mass $M_\odot$) is the total mass
locked up in stars. We report it as $\log_{10}(M_\star/M_\odot)$ because it ranges over
many factors of ten. The galaxies in samples like ours typically span roughly
$\log_{10}(M_\star/M_\odot)$ from about 9 to 11.5, that is, from about a billion to a few
hundred billion solar masses, so the log scale is not a convenience but a necessity.

Stellar mass is inferred by fitting models of stellar populations to the observed light, a
procedure called **SED fitting** (spectral energy distribution fitting, where the SED is
just the brightness as a function of wavelength). It is the single best proxy for "how big
is this galaxy" in a physical sense.

Mass matters for geometry because it is one of the strongest organizing variables in the
galaxy population. More massive galaxies are more likely to be passive and red; less
massive ones are more likely to be star-forming and blue. So mass, color, and morphology
are all braided together, and a faithful embedding should reflect that braiding rather
than treating them as independent. The disentanglement question in Section 11, whether the
model separates these correlated properties more than their natural correlation forces, is
sharpened by exactly this braiding.

A galaxy is called **star-forming** (or active) when it is currently making new stars at a
healthy rate, and **passive** (or quiescent) when it has largely stopped. These are the
two physical states that the red and blue populations below correspond to. The cleanest
single number to tell them apart is the specific star-formation rate, defined next; the
color $g - r$ is a cheaper proxy for the same split. We use "passive" and "star-forming"
throughout as the names for these two states.

**Star-formation rate** is how fast a galaxy is currently turning gas into new stars, in
solar masses per year. Dividing it by stellar mass gives the **specific star-formation
rate** (**sSFR**), in units of inverse time.

The sSFR answers a sharper question than the raw rate: not "how many stars per year" but
"how fast is this galaxy growing relative to the stars it already has." A high sSFR means
a galaxy is actively building itself. A low sSFR means it has largely stopped. The sSFR is
the cleanest single number for the active-versus-passive distinction below, and like
stellar mass it comes from SED fitting, so it exists for only a subset of our galaxies.

**Morphology** is the visual shape and structure of a galaxy: smooth and round, or
featured with spiral arms, bars, and clumps. Astronomers compress shape into a sequence.

At one end are **early-type** galaxies, smooth featureless ellipsoids with little gas and
few young stars. At the other are **late-type** galaxies, disks with spiral arms, gas, and
ongoing star formation. The words "early" and "late" are historical labels for position
along this sequence and carry no claim about which formed first; we use them only as names
for the two ends.

Hubble arranged this sequence in his fork-shaped diagram, the tuning fork mentioned in
Section 1, with smooth ellipticals on the handle and the spiral types fanning out along
the tines.

A compact numerical handle on morphology is the **Sersic index** ($n$). Fit a galaxy's
light profile, meaning how surface brightness falls off from the center outward, with a
standard family of curves controlled by one number $n$. A small $n$ (around 1) describes
an exponential disk, the light profile of a spiral. A large $n$ (around 4) describes a
steep, centrally concentrated profile, the signature of an elliptical or a bulge-dominated
galaxy.

So $n$ is a one-number summary that slides from disky to spheroidal, a structural cousin
of the visual early-to-late sequence.

Like stellar mass and sSFR, the Sersic index comes from fitting a model to the image, so
it exists for only a few thousand of our galaxies. We use it as one of the cleaner
physical labels precisely because it is structural: it describes the shape of the light,
not the colors that were partly fed to the model. Recovering a structural quantity like
$n$ from the image is some of the better evidence that the embedding encodes real galaxy
structure rather than echoing its inputs.

The Sersic index is worth keeping separate from color in your mind, because the two can
disagree, and the disagreement is informative. Color reports on the stars' ages and the
gas. Sersic index reports on the arrangement of light in space. A galaxy can be red yet
still disky, or blue yet centrally concentrated. When color and structure agree, the
galaxy sits cleanly at one end of both sequences. When they disagree, the galaxy is doing
something interesting, and the quenching channels below are partly a story about which of
the two changes first.

We also use **vote fractions** from a citizen-science classification (Galaxy Zoo), where
many people answer questions like "is this galaxy smooth or featured?" and the fraction
voting each way becomes a soft label. In our sample these fractions are predicted by a
separate trained network from the images, accurate to roughly five to ten percent.

We use three at full sample size, **smooth**, **featured**, and **merger** (a galaxy in
the act of colliding or merging with another), and two conditional ones (spiral-arm and
bar fractions) defined only for galaxies already judged to be featured, non-edge-on disks.
Section 4 covers these labels and their coverage in detail; here we only need them as
named physical properties the embedding might track.

The conditional ones deserve a flag now because they shape what we can say later. A
spiral-arm fraction only makes sense for a galaxy that is already a disk seen face-on;
asking whether an elliptical has spiral arms is meaningless. So these labels exist for
only a few thousand galaxies by design, not by accident, and any result that leans on them
carries a wider error bar. When we test the "featured to spiral" concept hierarchy in
Section 17 and report a null, the small spiral sample is part of why that test has limited
power, and we say so there.

A last word on what these labels are. A vote fraction is a soft, human-flavored quantity:
the share of people (here, the share predicted by a network trained on people) who would
call a galaxy smooth. It is not a clean physical measurement like a flux. It carries the
roughly five-to-ten-percent error of the predicting network and the fuzziness of human
judgment. We use vote fractions because they are the best morphology labels available at
full sample size, and we keep their softness in mind whenever a morphology result looks
marginal.

### 2.2 Morphology is a continuum, not a set of boxes

The first physical fact a faithful embedding should reflect is that galaxy shapes form a
continuum. Ellipticals and spirals are not two disconnected categories with a gap between
them. They blend.

Between a pure smooth elliptical and a grand spiral sit lenticulars (disk galaxies with
little gas and no arms), weak spirals, barred spirals, and every gradation in between. The
Sersic index slides continuously from about 1 to about 4 with no forbidden values. The
smooth and featured vote fractions trade off smoothly rather than splitting the sample
into two islands.

This matters for what we expect to measure. If morphology were a small set of discrete
boxes, a faithful embedding might show separate clusters with empty space between them,
and a geometry measurement would find several disconnected pieces. If morphology is a
continuum, a faithful embedding should be one connected body with smooth gradients across
it.

The known physics predicts the second picture. Section 8 (diffusion maps) and Section 15
(topology) test directly whether the embedding is one continuous body or many islands. The
morphological continuum is the physical reason we expect, and the standard against which
we read, a single connected cloud.

There is a deeper reason the continuum picture is the right default, and it is worth
stating because it guards against a common mistake. The human categories (elliptical,
lenticular, spiral, irregular) are names we impose for convenience, not seams in the data.
Nature did not draw boundaries between them. A model trained without those names has no
reason to insert gaps where we drew lines. So if the embedding turned out to be cleanly
clustered along the human categories, that would actually be surprising and would suggest
the model had somehow absorbed the human scheme. A smooth continuum is both what the
physics predicts and what an unsupervised model should produce, and finding one is
therefore consistent rather than remarkable. We treat it as a passed expectation, not a
discovery.

### 2.3 Color and mass come in two populations: the bimodality

The second fact is sharper and, at first sight, in tension with the first. When you plot
galaxies by color and stellar mass, they do not spread out evenly. They pile up into two
populations with a relatively empty zone between them. This is the **color-mass
bimodality** ("bi-modality" meaning two peaks).

The **red sequence** is a tight band of red galaxies. These are mostly the early-type
spheroidals: massive, gas-poor, with old stellar populations and little ongoing star
formation. They are red because their blue stars have died and not been replaced. They are
called passive (or quenched, defined below) because they have largely stopped forming
stars. The red sequence is narrow, which tells you these galaxies share a fairly uniform
old-stellar makeup.

The **blue cloud** is a looser group of blue galaxies. These are mostly late-type disks:
gas-rich, actively forming stars, with young blue stars keeping them blue. It is called a
cloud rather than a sequence because it is broader and more scattered, reflecting a wider
range of ongoing star-formation activity.

Between them lies the **green valley**, a sparsely populated zone of intermediate color.
Few galaxies sit here at any given moment. The standard reading is that galaxies cross the
green valley relatively quickly as they shut down star formation, so you catch few of them
mid-transition, the way few cars are parked in the middle of a one-way bridge.

The valley is not perfectly empty, and it is the region where the active-to-passive
transition is happening.

Why two modes and not a smooth spread? Because the physics of star formation has a
near-switch quality. While a galaxy has cold gas it keeps making stars and stays blue;
once the gas is gone or stabilized, star formation drops sharply and the galaxy reddens
over a comparatively short time. A process with a fast transition produces few objects
caught mid-way, which is exactly the sparse green valley. The bimodality is therefore a
fingerprint of how quenching works, not an arbitrary split, and that is why the next
subsection is about quenching.

So there is no contradiction with the continuum picture, but there is a subtlety we have
to hold. Morphology runs smoothly, yet color and star formation pile into two modes with a
thin bridge.

A faithful embedding could plausibly show both: one connected body (because the
populations are joined by the green-valley bridge and by the morphological continuum) that
nonetheless has a denser red region and a denser blue region with a thinner zone between.
Whether the model shows a single continuum with internal density structure, or genuinely
separate clusters, is something we measure rather than assume.

The bimodality is the physical reason we look closely for any sign of a split. Section 16
(stratified intrinsic dimension) splits the sample at the color boundary on purpose to ask
whether the two populations even have the same geometric dimension.

It is worth being precise about what bimodality is and is not, because the two-peaks
picture can be over-read. Bimodality is a statement about density: two crowded regions and
a thinner one between them. It is not a statement that the two regions are disconnected. A
mountain range with two peaks and a low pass between them is bimodal in height yet fully
connected as a surface. The red and blue populations are like that. The green valley is a
low-density pass, not a chasm, and the morphological continuum and the transitioning
galaxies in the valley keep the whole surface joined. Holding density and connectedness
apart is what lets us reconcile the smooth-continuum fact of Section 2.2 with the
two-population fact here: the embedding can be one connected body that is denser in two
places. Which of those it actually is, we measure, and the answer (Sections 8 and 15) is
the connected-but-uneven one.

### 2.4 Quenching, and the at-least-two channels

The process by which a galaxy stops forming stars is called **quenching**. A star-forming
galaxy is quenched when its gas supply is removed, heated, or stabilized so that it can no
longer collapse into new stars. After quenching, the galaxy reddens and joins the red
sequence.

Quenching is the physical event that moves a galaxy from the blue cloud, through the green
valley, to the red sequence.

What removes or stabilizes the gas is itself a list of distinct physical processes. Gas
can be heated by energy released near a galaxy's central black hole. It can be stripped
away as a galaxy falls through the hot gas of a larger group or cluster. It can be used up
faster than it is replenished. It can be locked into a state stabilized by the galaxy's
own deep gravity. We do not need the details for this report. We need only the
consequence: because the causes differ, the routes to the red sequence differ, and that is
the multi-channel picture.

The important point for geometry is that quenching is not believed to happen by a single
mechanism. There appear to be at least two distinct channels, and they differ in both
speed and accompanying structural change.

One channel is fast and is accompanied by a transformation of shape, a disk being
disrupted into a spheroid. A violent event such as a major merger between two galaxies, or
a strong burst that drives gas out, can shut down star formation quickly and rebuild a
disky galaxy into a centrally concentrated spheroidal one. In this channel morphology and
color change together and fast: the galaxy crosses the green valley quickly and arrives on
the red sequence with a high Sersic index.

The other channel is slow and need not destroy the disk. A galaxy can run low on fresh
gas, or have its gas held in a state that resists collapse, and fade in star formation
gradually while keeping much of its disk structure. Here color reddens over a long time
while morphology changes little, so the galaxy can arrive in the red population while
still looking disky.

These are physical hypotheses from the galaxy-evolution literature, stated here as the
prior context for our interpretations, not as something this report proves.

The reason they matter geometrically is specific. If there really are distinct routes from
blue to red, an embedding that captures galaxy physics might show weak branching: a mostly
connected body that splits, near the transition region, into separable strands
corresponding to different quenching paths. That is a precise geometric prediction, and it
connects to curvature.

A branch point in a manifold is a place of locally negative curvature, a saddle where the
body divides. Section 14 (curvature) measures exactly this and asks whether the embedding
shows the thin negative-curvature bridges that branching would produce. We will find weak,
localized branching rather than a clean tree, and the two-channel picture is the physical
hypothesis we hold that result up against.

We are careful, there and here, not to claim the geometry proves the channels exist.
Consistency is not proof.

There is a second reason the channels matter for what we measure, and it concerns
dimension rather than branching. If passive galaxies are assembled by more than one route,
and from a wider range of starting structures, the passive population could occupy a
sub-region with more independent degrees of freedom than the star-forming one, which is a
younger and more uniform set still on its first build-up. That is a prediction about the
relative intrinsic dimension of the two populations, and Section 16 tests it directly by
measuring the dimension of the red and blue halves separately. We will find a small but
real difference, with the passive half sitting slightly higher, and we hold the
multi-channel picture up against that result as one possible reading among several.

### 2.5 How many knobs? The intrinsic-dimension prior

The last piece of background is a number, or rather a range, that sets expectations for
the central geometry result. The question is how many independent quantities you need to
describe a galaxy's light. This is the physical version of intrinsic dimension, the count
of independent knobs, which Sections 6 and 7 measure directly for the embedding.

Start from photometry. Clean broadband photometry of galaxies, the handful of magnitudes
and colors, is known to be highly redundant. The colors of normal galaxies trace out a
thin, low-dimensional locus rather than filling the available color space, because the
underlying drivers (stellar age, the amount of dust, the metal content, the redshift) are
few and tightly correlated.

Studies that estimate the effective dimension of galaxy photometry land low, in roughly
the **2 to 5** range. In words: once you know a few numbers about a galaxy's light, the
rest is largely predictable.

Now add spectra. A galaxy spectrum has thousands of wavelength channels, but those
channels are not independent either. The spectrum is generated by a modest number of
physical drivers (the mix of stellar ages, the metal content, the dust, the gas emission,
the velocity broadening), so its effective dimension is far below its channel count.

Analyses of large spectroscopic samples find that a few principal components, on the order
of **3 to 10**, reconstruct most galaxy spectra. The spectrum carries more independent
structure than broadband photometry, but still only a handful of effective axes.

A principal component, used loosely here, is a single combined direction that captures as
much of the variation as possible; saying "a few principal components reconstruct most
spectra" means a few combined directions explain most of how spectra differ. Section 6
makes this precise and turns it into one of our four dimension estimators. For now the
takeaway is that both the cheap measurement (photometry) and the rich one (spectra) point
to a small number of real degrees of freedom in a galaxy's light.

Put these together for a model that, in principle, fuses images, photometry, and spectra.
An honest multimodal embedding of galaxies plausibly needs on the order of **5 to 10**
independent axes to capture the real variation, with maybe a few more for structural and
morphological detail that pure photometry misses.

We stress what this number is and is not. It is an optimistic ceiling assembled from prior
work, a plausibility band, not a point prediction and not a target. The true value could
sit above it if the model encodes structure the photometric and spectroscopic studies did
not, or if some axes carry observational nuisances rather than physics.

We carry the **4 to 10** band into Sections 6 and 7 as a reference line on the plots. And
we report honestly that our measured intrinsic dimension, near 10 to 12, sits at or just
above the top of that optimistic band rather than comfortably inside it. The prior frames
the result; it does not decide it.

Why might the measured value land above the optimistic band? A few honest reasons. The
model encodes morphology, the detailed arrangement of light, which pure photometry throws
away and which plausibly adds a handful of real axes. The image carries observational
nuisances (orientation, the point at which the galaxy sits in the survey, instrumental
effects) that a faithful representation might keep, inflating the count above the purely
physical degrees of freedom. And the prior band itself was assembled for cleaner, more
curated measurements than a raw multimodal embedding. So a value modestly above the band
is not a contradiction of the physics; it is the kind of result the physics leaves room
for. What would be a contradiction is a value near the ambient 1024, which would say the
model failed to compress at all. We are far from that.

### 2.6 The load-bearing caveat: a density, not a movie

One caution governs every physical reading in this report, and it is easy to forget, so we
state it as plainly as possible.

Our embedding is a snapshot. Each galaxy is observed once, at one moment in cosmic time,
and contributes one point. The cloud of points is therefore a **density of populations**:
it shows where galaxies of various kinds sit and how common each kind is right now.

It is not a **movie**. No single galaxy is tracked as it ages, reddens, and quenches. We
never see a point move.

A density map and a movie can look deceptively alike. Both can show a smooth path from
blue to red. But one is a count of where galaxies sit today, and the other would be a
record of individual histories, and only the first is in our data.

This sounds obvious but it has teeth, because the language of evolution is tempting and
wrong here. When we find a smooth bridge between the blue and red regions, it is tempting
to say the model "learned" that galaxies travel across the green valley. It did not, and
we cannot show that.

What we can say is that the present-day population has galaxies of intermediate color in
between, and the embedding places them in between. That is a statement about where
populations sit, not about any galaxy's history.

Likewise, when we find weak branching near the transition region and connect it to the two
quenching channels, we are matching a static geometric feature to a dynamical hypothesis.
The match is suggestive and worth reporting. It is not evidence that the model represents
time.

So throughout, "the manifold connects the blue and red populations" is a measured
statement about a static density. "Galaxies move along the manifold as they quench" is a
story we are forbidden from claiming, because nothing in a single-epoch embedding can
establish motion. We keep that line bright. Where we slip toward evolutionary language for
readability, read it as shorthand for population structure, never as a claim that the
model captured evolution.

The same caution applies to the multi-channel quenching picture. When we find weak
branching and connect it to distinct quenching routes, the branching is a measured static
feature and the routes are a dynamical story. The embedding could show that geometry for
reasons that have nothing to do with quenching dynamics. We report the match because it is
genuine and interesting, and we refuse to upgrade it to proof.

With the physics fixed and the prior stated, the next thing to pin down is the instrument:
what AION-1 is, how it turns a galaxy into 1024 numbers, and which exact version of the
embedding we measure. Section 3 sets out the model and the representation, including why
we study two embedding sets rather than one and why the image-only set is the honest place
to test for genuine inference.

## 3. The model and the representation we study

Everything in this report is a measurement of one specific object: the set of vectors that
AION-1 produces when it looks at our galaxies. Before we can say anything about the shape
of that set, we have to be exact about what the model is, what a "vector" means here, and
which version of the vectors we use for which question. This section does that. It does
not cover how the model was trained at scale or where the computation ran. It covers only
what you need to read the geometry honestly.

### 3.1 What AION-1 is, in plain words

AION-1 is a foundation model for astronomical objects. "Foundation model" means a single
large neural network trained once on a broad pile of data with no task-specific labels,
then reused as a fixed feature extractor for many downstream questions. The variant we
study is AION-1-Large: about 800 million tunable numbers (parameters), trained by
self-supervision. Self-supervised means the training signal comes from the data itself,
not from human annotations. Nobody told the model which galaxies are spirals and which are
ellipticals. It never saw a morphology label. That fact is what makes the rest of this
report interesting: any morphology structure we find in its vectors was not handed to it.

The model is a transformer. A transformer is a neural network built around attention, an
operation that lets every part of the input look at and mix information from every other
part. The input to a transformer is not a raw image or a raw number. It is a sequence of
tokens. A token is a small chunk of the input turned into a vector, the network's basic
unit of "a thing to attend to." AION-1 tokenises several kinds of astronomical data into
one shared sequence: image patches, spectra, and individual catalogue scalars (single
measured numbers like a flux or a redshift) each become one or more tokens. The point of
putting different data types into one token stream is that attention can then relate, say,
an image patch to a brightness measurement directly.

The training objective is masked modelling. During training the model hides (masks) a
random fraction of the tokens and is asked to fill them back in from the tokens it can
still see. To predict a masked image patch from the surviving patches and the surviving
catalogue numbers, the network has to learn the real correlations among brightness, shape,
colour, and distance, because those correlations are the only thing that makes the missing
token predictable. This is the same trick that lets a language model learn grammar and
facts by predicting hidden words. Here the "language" is the joint statistics of how
galaxies actually look and measure.

Architecturally AION-1 is an encoder paired with a decoder. The encoder reads the visible
tokens and produces a rich internal representation of each one. The decoder uses that
representation to reconstruct the masked tokens. For our purposes the decoder is
scaffolding. It exists to create a training signal, and once training is done we have no
use for it. We keep the encoder.

### 3.2 From a galaxy to a single 1024-number vector

We use the model frozen. Frozen means we never update its parameters: no fine-tuning, no
task-specific retraining, no gradient ever flows back into the weights. We feed a galaxy
in, read numbers out, and that is all. Freezing matters scientifically. We want to
characterise the representation the model already built on its own, not a representation
we bent toward our labels. If we had fine-tuned, any concept axis we later "discovered"
could just be an axis we trained in. The frozen model removes that loop.

When a galaxy passes through the frozen encoder, the encoder emits one vector per token. A
galaxy is described by many tokens (many image patches, plus its catalogue tokens), so the
raw output is a whole sequence of vectors, not one. We want a single fixed-length summary
per galaxy so that every galaxy lives in the same space and we can compare them. We get it
by mean pooling: take the per-token output vectors and average them, element by element,
into one vector. Mean pooling is the simplest order-independent way to collapse a
variable-length sequence into a fixed-length summary. It treats the galaxy's
representation as the average of its parts. Other choices exist (for example using a
special summary token, or attention-weighted pooling), and they could change fine detail;
we note this as a method choice, not a tested variable. The pooled vector has length
$d = 1024$. We call $d$ the ambient dimension: the number of coordinate axes of the space
the vectors literally live in.

So the object we study is a cloud of points. Each galaxy is one point in a
1024-dimensional space. We have $N = 48{,}398$ such points (Section 4 covers the sample).
A central question of this whole report is how many directions in that 1024-dimensional
space the cloud actually uses, which is almost always far fewer than 1024. That is the
intrinsic dimension, and Sections 6 and 7 measure it. For now the thing to hold onto is
the data type: a fixed table of 48,398 rows by 1024 columns, one row per galaxy, produced
by a frozen self-supervised model that never saw a morphology label.

To keep the vocabulary straight, the terms just defined:

| Term | Plain meaning |
|---|---|
| token | one chunk of input (image patch, spectrum piece, or one catalogue number) turned into a vector |
| encoder | the part of the transformer that reads visible tokens into an internal representation |
| frozen | parameters never updated; the model is used as a fixed feature extractor |
| mean pooling | averaging the per-token output vectors into one vector per galaxy |
| embedding | that final per-galaxy vector (a learned 1024-number summary) |
| ambient dimension $d$ | the literal number of coordinates, here 1024 |

### 3.3 Two embedding sets, and why we need both

Here is the design choice that the rest of the report leans on. We do not study one
embedding per galaxy. We study two, built from the same frozen model but fed different
inputs.

The first set is $E_{\text{full}}$. To build it we give the model the full multimodal
input it expects: the four-band image (the $g$, $r$, $i$, $z$ filters, explained in
Section 4), the photometric fluxes ($g$, $r$, $z$ brightnesses), and the galaxy's redshift
(a distance proxy, also defined in Section 4). Image, photometry, and redshift are fused
as input tokens, the encoder runs, and we mean-pool. $E_{\text{full}}$ is the richest
representation the model can form for each galaxy, because it gets to use every
measurement we have. We use $E_{\text{full}}$ for the pure-geometry arms of the report:
intrinsic dimension, the diffusion map, curvature, topology. Those arms ask "what shape is
the cloud," and for that question we want the model's best, most complete picture of each
galaxy.

The second set is $E_{\text{img}}$. Here we feed the model only the image. No flux tokens,
no redshift token. Same frozen encoder, same mean pooling, just a thinner input.
$E_{\text{img}}$ is the image-only representation: whatever the model can say about a
galaxy from its picture alone.

Why keep both? Because of leakage. Leakage is when the thing you are trying to predict
was, in effect, an input, so predicting it proves nothing. Suppose you take
$E_{\text{full}}$ and ask "can a simple linear readout recover the galaxy's redshift from
this vector?" The redshift was fed in as an input token. A decodable redshift then tells
you only that the model did not throw its own input away. That is a trivial, circular
result. The same circularity hits colour, because the photometric fluxes that define
colour were also fed in. Any "concept direction" you draw for redshift or colour in
$E_{\text{full}}$ might be nothing more than the model echoing its input.

$E_{\text{img}}$ closes that loophole. Redshift and colour are not inputs to
$E_{\text{img}}$; only the image is. So if a linear probe recovers redshift or colour from
$E_{\text{img}}$, that is a genuine inference the model drew from pixels, not a readback.
This is why we call $E_{\text{img}}$ the leakage-free set, and why every concept-probing
arm (decodability in Section 10, disentanglement in Section 11) runs on $E_{\text{img}}$
when the property in question could have been an input. The cleanest evidence of all comes
from properties that were never inputs to either set, like stellar mass and star-formation
rate (Section 4): recovering those from $E_{\text{img}}$ cannot be leakage under any
reading, because the model never received them in any form.

We can also turn the gap between the two sets into a measurement. By probing the same
property in both $E_{\text{full}}$ and $E_{\text{img}}$ and comparing, we read off how
much of the model's apparent skill is honest image inference versus input echo. Section 10
reports that ablation directly: for redshift the image-only probe reaches $R^2 = 0.80$
(measured) while the full multimodal probe reaches $0.98$ (measured), and the difference
is the leakage we deliberately excluded by working in $E_{\text{img}}$. We flag the
epistemic split now and quantify it there.

One reassurance about treating the two sets as the same kind of object. They sit at almost
the same scale. The pooled vectors of $E_{\text{full}}$ have an average length (Euclidean
norm) of about 48.4, and those of $E_{\text{img}}$ about 49.6 (measured, Section 2 facts).
Both spreads are tight, only a few percent, so neither set is dominated by a handful of
giant outlier vectors, and switching inputs did not blow the representation up or collapse
it. The two clouds are comparable in size and live in the same 1024-dimensional space,
which is what lets us read the geometry of one against the other.

The short version, carried forward into every later section: $E_{\text{full}}$ for the
shape of the manifold, $E_{\text{img}}$ for honest concept claims, and the difference
between them as a leakage meter.

## 4. Data and preprocessing

The geometry we measure is only as trustworthy as the sample, the images, and the labels
behind it. This section sets all three out plainly, says how much we trust each label,
defines the terms a non-astronomer needs, and then describes the one transform we apply to
the vectors before any geometry is computed (z-scoring) and why it is not optional. It
ends with an honest list of what we deliberately did not use. No logistics.

### 4.1 The galaxy sample

The galaxies come from Galaxy Zoo DESI, a catalogue of galaxies imaged by the DESI Legacy
Imaging Surveys and given morphology labels through the Galaxy Zoo project. We drew 50,000
of them with a fixed random seed, restricted to redshift $0 < z < 1$. Redshift, written
$z$, is the fractional stretching of a galaxy's light toward longer (redder) wavelengths
caused mostly by cosmic expansion: the further away and further back in time a galaxy is,
the larger its $z$. The cut $0 < z < 1$ keeps the sample to the nearby-to-intermediate
universe, where the imaging resolves galaxy shapes well enough for morphology to mean
something. Of the 50,000 drawn, 96.8% had usable images in all four bands, which leaves
the working sample of $N = 48{,}398$ galaxies analysed throughout this report (measured).
Fixing the seed means the draw is reproducible: rerun the selection and you get the same
48,398 galaxies.

That number, 48,398, is not just a sample size. It also sets a hard ceiling on what we can
claim about dimension. A finite point cloud can only resolve so many independent
directions before it runs out of points to fill them, and a standard rule of thumb is that
$N$ points can reliably support an intrinsic dimension up to about $\log_2 N$. Here
$\log_2(48{,}398) = 15.56$ (measured). So any dimension estimate we report near or above
$\sim$ 15.6 is at the edge of what this sample can even see, a caveat we carry into
Sections 6 and 7. We mention it here because it is a property of the data, not of the
method.

### 4.2 The images: four bands

Each galaxy enters the model as a four-band image. A band is the picture taken through one
colour filter, recording brightness in a fixed slice of the spectrum. The four bands are
$g$ (green), $r$ (red), $i$ (near-infrared), and $z$ (further near-infrared). The single
letter $z$ for a band is an unlucky clash with $z$ for redshift; we always say "band" or
"redshift" to keep them apart. Having several bands is what lets a model perceive colour:
the same galaxy is brighter in some filters than others, and the pattern of those ratios
carries physical information (Section 2). The image is the only input shared by both
embedding sets. For the image-only set $E_{\text{img}}$ (Section 3), these four bands are
the entire input.

### 4.3 The labels, with coverage and trust

We never feed labels into the model. We hold them out and use them only to interpret the
geometry the model built on its own: to ask "does this measured direction in the cloud
line up with a known physical property?" A label is only as useful as it is trustworthy
and as widely available, so for each one we state both its coverage (how many of the
48,398 galaxies have it) and its trust level (how it was produced and how accurate it is).
The labels fall into three tiers.

Tier one, available for essentially the full sample (~48,398). This tier has the redshift,
the colours, and the morphology vote fractions. The colours are differences of magnitudes.
A magnitude is an inverted logarithmic brightness scale (fainter objects have larger
magnitudes), and we use $\text{mag}_g$, $\text{mag}_r$, $\text{mag}_z$ to form two
colours: $g-r$ and $r-z$. A colour is just how much brighter a galaxy is in one band than
another, and it tracks the mix of stellar populations: red, old, passive galaxies have
large $g-r$, while blue, young, star-forming ones have small $g-r$ (Section 2 explains the
physics). The morphology vote fractions are smooth, featured, and merger, defined next.

A vote fraction is the share of classifiers who said a galaxy shows a given feature.
Galaxy Zoo originally asked human volunteers questions like "is this galaxy smooth and
rounded, or does it have features such as a disk or spiral arms?" The fraction of people
who answered "smooth" is the smooth vote fraction, a number between 0 and 1; likewise
featured and merger (signs of a collision or interaction). A vote fraction is a soft
label, not a hard class: a galaxy can be 0.7 smooth and 0.3 featured, which honestly
captures the fact that morphology is a continuum, not a set of boxes (Section 2). In this
catalogue the vote fractions are not from live humans on each galaxy. They are predicted
by a separate convolutional neural network (a CNN: a neural network specialised for
images, here Zoobot built on an EfficientNet backbone) that was trained on the real human
votes and then applied to every galaxy. That CNN reproduces human vote shares to within
roughly 5 to 10 percent (interpreted as a trust level, not a number we measured here). So
tier-one morphology is good but not exact, and a probe that recovers it can never beat the
label's own 5-to-10-percent noise floor. We keep that ceiling in mind when reading
morphology decodability in Section 10.

The redshift in this tier needs its own caveat, the most important trust caveat in the
report. Redshift can be measured two ways. A spectroscopic redshift comes from a full
spectrum, where sharp emission and absorption lines pin the wavelength shift precisely; it
is the gold standard. A photometric redshift (photo-z) is estimated from just the few
broadband brightnesses, by noticing that more distant galaxies look redder in a
characteristic way; it is cheaper and available for far more galaxies, but much less
accurate, and it is derived largely from colour. In our sample only 6,883 galaxies have a
spectroscopic redshift (measured); the great majority carry a photo-z. This matters
because a photo-z is essentially a function of colour. So when a probe recovers redshift
from an image, part of what it is really doing is image $\to$ colour $\to$ photo-z, a
chain that loops back through the very colour the photo-z was built from. We will not
present image-to-redshift as clean evidence that the model "understands distance." The
clean evidence lives in tier three.

Tier two, branch-conditional ($n = 3{,}034$). This tier has the spiral-arm fraction and
the strong-bar fraction. They are small not by accident but by design, because Galaxy Zoo
is a decision tree. A decision tree means later questions are only asked when earlier
answers warrant them. You are only asked "does this galaxy have spiral arms?" if you
already said it is featured and not edge-on, because spiral arms are undefined for a
smooth blob or a disk seen exactly side-on. So the spiral and bar labels exist only for
featured, non-edge-on disks, which is a small slice of the sample. The conditionality is a
feature, not a flaw: it keeps the labels meaningful. But it means any spiral or bar result
rests on about 3,034 galaxies, which is why those probe scores in Section 10 have wide
confidence intervals (spiral $R^2 = 0.25$ with a 95% interval of $[0.15, 0.33]$, measured)
and we read them cautiously.

Tier three, the cross-matched physical subset (a few thousand each). This tier has the
properties an astronomer would call genuinely physical rather than descriptive, taken from
external catalogues by matching our galaxies to them on sky position. Stellar mass,
written $\log M_*/M_\odot$ (the logarithm of the galaxy's total mass in stars, in units of
the Sun's mass), is available for $n = 3{,}728$ galaxies (measured). Specific
star-formation rate (sSFR: how fast a galaxy is forming new stars per unit of existing
stellar mass, a measure of how actively it is still growing) is available for
$n = 4{,}760$ (measured). Sersic index $n$ (a single number describing how steeply a
galaxy's light falls off from its centre; low values $\sim$ 1 mean a flat disk-like
profile, high values $\sim$ 4 mean a concentrated bulge-like profile) is available for
$n = 3{,}730$ (measured). These come from SED-fitting and structural-fitting catalogues,
and they were never fed to the model in either embedding set. That makes them the cleanest
possible test: if $E_{\text{img}}$ can predict stellar mass or sSFR from the image alone,
there is no leakage path, because the model never saw those numbers in any form. Section
10 reports that it can (mass $R^2 = 0.72$, sSFR $R^2 = 0.76$, both measured on
$E_{\text{img}}$), and those two numbers are the strongest honest evidence in the whole
study that the visual representation encodes real physics.

The label tiers at a glance:

| Tier | Labels | Coverage $n$ | How produced / trust |
|---|---|---|---|
| 1 | redshift; $g-r$, $r-z$; smooth, featured, merger | ~48,398 | redshift mostly photo-z (only 6,883 spectroscopic); colours from photometry; vote fractions from a CNN, ~5-10% error |
| 2 | spiral-arm fraction, strong-bar fraction | 3,034 | Galaxy-Zoo decision tree: only for featured, non-edge-on disks |
| 3 | $\log M_*$; sSFR; Sersic index $n$ | 3,728 / 4,760 / 3,730 | external SED-fit and structural catalogues, cross-matched; never model inputs |

### 4.4 Preprocessing: z-scoring, and why it is mandatory

Before any geometry is computed we apply one transform to the embeddings: we z-score each
of the 1024 dimensions. Z-scoring a dimension means subtracting that dimension's mean
across all galaxies and dividing by its standard deviation (std, the typical spread of
values), so every dimension ends up centred at zero with a spread of one. We do this to
every column of the 48,398-by-1024 table independently.

The reason is concrete and is forced by the data. Across the 1024 dimensions, the
per-dimension std runs from about 0.129 to about 2.93 in $E_{\text{full}}$ (measured), a
spread of roughly 23 times between the quietest and loudest dimension. Geometry is built
on distances, and almost every distance we use is Euclidean at heart, meaning it sums
squared differences across dimensions. A dimension with 23 times the spread contributes on
the order of its spread squared to that sum, so without z-scoring a mere handful of
high-variance dimensions would dominate every pairwise distance and the other thousand-odd
dimensions would be invisible. Intrinsic dimension, nearest-neighbour structure,
curvature, all of it would then be reporting the geometry of a few loud axes rather than
the geometry of the representation. Z-scoring puts every dimension on equal footing so
distances reflect the full vector. This is a deliberate analysis choice, and it has one
honest side effect worth flagging: spreading variance evenly across dimensions slightly
raises linear spread measures. The PCA participation ratio (a count of effective linear
directions, defined in Section 6) is about 7.4 on the raw vectors but about 11 after
z-scoring (measured). We report both and use the z-scored value consistently, because the
z-scored space is the one all our geometry lives in.

### 4.5 The thin-shell observation

One structural fact about the cloud is visible before any heavy machinery. The embedding
vectors have nearly equal length. In $E_{\text{full}}$ the norm averages about 48.4 with a
spread of only about 5 percent (min 34.4, max 56.5; measured), and $E_{\text{img}}$ is
similar at about 49.6 (measured). If every vector has almost the same length, then all the
points sit at almost the same distance from the origin, which means they lie close to the
surface of a high-dimensional sphere, a thin shell, rather than filling the interior of a
ball. This is a common situation for deep-network embeddings and it is worth naming
because it shapes intuition: the interesting variation among galaxies is in direction on
that shell, not in how far out they sit. It is an observation about the raw geometry; the
z-scoring above changes the per-dimension scaling but the qualitative picture of a thin
populated shell is what the norms report (measured). We do not over-read it. It simply
tells us the cloud is hollow-ish and roughly spherical at the coarsest level, with all the
physics in the angular structure that the later sections dissect.

### 4.6 What we did not use

Honesty about scope. AION-1 is a multimodal model that can also ingest spectra, the
wavelength-by-wavelength fingerprint of a galaxy's light that carries the richest physical
detail. In this run we did not feed it spectra. Both embedding sets are built from images,
photometry, and redshift only; $E_{\text{img}}$ is image alone. This is a deliberate
deviation from the fullest possible multimodal embedding, and it bounds our claims: some
physical axes that only spectra would reveal may simply be absent from the representations
we study, so a "missing" concept here is not proof the model cannot form it with richer
input. We also did not use the gold-standard spectroscopic redshift as our redshift label
for the full sample; with only 6,883 spectroscopic redshifts available we work with the
photo-z-dominated label and carry its caveat (Section 4.3) rather than shrink the sample.
And we did not use the absolute extremes of any axis for the visual montage in Section 18,
because at the very tails the cutouts include stars and imaging artefacts rather than
clean galaxies; that sampling choice is owned by that section. Naming these omissions up
front keeps the later results readable as what they are: a faithful account of one
image-plus-photometry representation of 48,398 galaxies, not a claim about the limit of
what AION-1 could encode.

## 5. The metric problem and the concentration diagnostic

Every geometric result in this report rests on one earlier choice: how do we measure the
distance between two galaxies in the embedding? It sounds like a detail. It is not.
Intrinsic dimension, curvature, topology, and the shape of the diffusion map are all built
on a notion of "near" and "far", and that notion comes from a distance function. Pick the
wrong one and you can manufacture structure that is not there, or wash out structure that
is. So before we trust any downstream number, we have to ask whether the obvious choice
(plain straight-line distance in 1024 dimensions) is even meaningful here, and if not,
what to use instead. This section sets up that decision and reports the diagnostic we ran
to make it.

### 5.1 Why straight-line distance gets unreliable in high dimensions

Start with the plain picture. Each galaxy is a point in a space with $d = 1024$ axes (the
ambient dimension defined in Section 3). The most natural distance between two points $x$
and $y$ is the Euclidean distance, the length of the straight line between them:

$$d_{\mathrm{euc}}(x,y) = \sqrt{\sum_{j=1}^{d} (x_j - y_j)^2}.$$

Here $x_j$ is the value of point $x$ on axis $j$, and the sum runs over all 1024 axes.
This is the ruler you used in school, extended to many axes. In two or three dimensions it
behaves exactly as your intuition expects. In hundreds of dimensions it stops behaving.

The problem has a name: distance concentration. As the number of dimensions grows, the
distances between points tend to bunch up around a single value. The nearest point and the
farthest point end up almost the same distance away. Put differently, the contrast between
"close" and "far" shrinks toward zero. In the limit of very high dimension the ratio of
the farthest distance to the nearest distance approaches one, so "nearest neighbour" loses
its meaning: everything is roughly equidistant from everything else.

The reason is a counting effect, not anything exotic. Euclidean distance squared is a sum
of 1024 per-axis terms. Each term is one small contribution. When you add up many
independent-ish small contributions, the total behaves like an average: its typical size
grows, but its spread relative to that size shrinks. By a law-of-large-numbers argument,
the relative spread of the squared distance falls off like roughly $1/\sqrt{d}$. With
$d = 1024$ that factor is about $1/32$, so the fractional differences between pairwise
distances are squeezed into a narrow band. The absolute distances are large; what
collapses is the *contrast* between them. And contrast is exactly what a nearest-neighbour
graph, an intrinsic-dimension estimator, or a diffusion walk feeds on. If every point is
about as far from every other point, the local neighbourhood structure those methods rely
on is built on numerical noise.

Two facts about our data make this worry concrete rather than theoretical. First, the
ambient dimension really is 1024, deep in the regime where concentration bites. Second
(from Section 4 and `_FACTS.md`), the embedding vectors have tightly clustered norms
(about 48 for the multimodal set $E_{\mathrm{full}}$ and about 49.6 for the image-only set
$E_{\mathrm{img}}$, roughly a 5 percent spread), so the whole cloud sits close to a thin
spherical shell. Points on a thin shell are a textbook setting for distance concentration,
because a large part of every pairwise distance is just the common shell radius. We
standardised each of the 1024 dimensions to zero mean and unit variance (z-scoring,
Section 4) before any of this, which fixes the unrelated problem of a few high-variance
axes dominating the ruler, but z-scoring does not undo concentration. So we measured it
directly.

### 5.2 Two numbers that quantify concentration

To turn "are distances bunched up?" into something we can plot, we computed two summary
statistics on the full table of pairwise distances. Both were run on a 2,000-point random
subsample of the 48,398 galaxies (the same subsample for every metric, drawn with a fixed
seed), z-scored first, using a $k = 15$ nearest-neighbour graph where a graph is needed.
The subsample keeps the all-pairs distance table (about two million pairs) tractable while
staying large enough for stable summaries.

The first statistic is the relative distance ratio, which we write RDR:

$$\mathrm{RDR} = \frac{D_{\max} - D_{\min}}{D_{\min}},$$

where $D_{\max}$ and $D_{\min}$ are the largest and smallest distances over all pairs of
distinct points. Read it as: how many times bigger is the spread of distances than the
smallest distance? A small RDR means the biggest and smallest distances are nearly equal,
so the cloud is strongly concentrated and contrast is low. A large RDR means there is real
range between the closest pair and the farthest pair, so contrast is high and "near" means
something. Higher RDR is better for everything we do downstream. (One caveat that belongs
to RDR: it is built from the single most extreme pair on each end, so it is sensitive to
one outlier distance. We read it alongside the second statistic, which uses a typical-pair
quantity, not the extremes.)

The second statistic is the nearest-neighbour-to-mean ratio, NN/mean:

$$\mathrm{NN/mean} = \frac{\overline{d_{\mathrm{NN}}}}{\overline{d_{\mathrm{pair}}}},$$

where $\overline{d_{\mathrm{NN}}}$ is the average distance from a point to its single
closest other point, and $\overline{d_{\mathrm{pair}}}$ is the average distance over all
pairs. This asks how far the nearest neighbour sits compared to a typical pair. If nearest
neighbours are barely closer than random pairs, the ratio climbs toward 1 and local
structure is weak (strong concentration). If nearest neighbours are much closer than
typical pairs, the ratio is small and local neighbourhoods are well separated from the
bulk (weak concentration). Lower NN/mean is better. The two statistics come at the same
question from opposite ends (RDR from the global extremes, NN/mean from the
local-versus-typical comparison), so when they agree we trust the reading.

In short, the decision rule for both statistics is simple:

1. RDR up, concentration down. A larger RDR means the closest and farthest pairs differ more, so contrast is healthy.
2. NN/mean down, concentration down. A smaller NN/mean means nearest neighbours stand out from the bulk, so local structure is healthy.

When both point the same way, we have a clear reading; when they conflict, we treat the
metric as borderline.

We also recorded, for each metric, how many pairwise distances came out infinite. Infinite
distances appear when a graph-based metric leaves the cloud split into pieces with no path
between them, which would make that metric unusable for global geometry. For every metric
and both embedding sets the infinite count was zero (`inf = 0` throughout
`results/metric.json`): all graphs were connected, so the distance tables are complete and
comparable.

### 5.3 The honest-metric battery: five rulers, not one

Rather than argue from theory about which distance to trust, we measured five of them side
by side on the same points. Call it the honest-metric battery. Each metric encodes a
different assumption about what "distance along the data" should mean, and seeing all five
at once tells us which assumptions buy contrast and which do not.

**Euclidean (the control).** The straight-line distance defined above, computed directly
from the z-scored coordinates. This is the baseline we are worried about. It ignores the
shape of the data entirely: it measures the distance through empty space, as if you could
fly straight from one galaxy to another regardless of whether any galaxies lie in between.
We keep it as a control precisely because it is the naive choice and the one most exposed
to concentration.

**Cosine.** Distance defined as $1 - \cos\theta$, where $\theta$ is the angle between the
two points' vectors (each vector normalised to unit length first). Cosine distance ignores
how long the vectors are and cares only about their direction. Because the cloud already
sits near a thin shell of nearly equal norms, cosine and Euclidean ought to behave
similarly here, and prior expectation is that cosine would concentrate much like
Euclidean. We included it to test that expectation, not because we planned to use it
downstream.

**Isomap geodesic.** Instead of flying straight through empty space, walk along the data.
Build a graph that connects each point to its $k = 15$ nearest neighbours, weight each
edge by the local Euclidean distance, and define the distance between any two points as
the length of the shortest path through that graph (computed with Dijkstra's algorithm).
This is the geodesic distance, the "as the data curves" distance: it follows the manifold
the points lie on rather than cutting across regions where no galaxies exist. On a curved
sheet, two points on opposite folds can be close in straight-line distance yet far along
the sheet, and Isomap captures that.

**Fermat (density-weighted geodesic).** Same shortest-path idea, but the edge weights are
raised to a power $p = 2$ before the path search, so an edge of local length $\ell$
contributes $\ell^{2}$. Squaring the weights makes long hops disproportionately expensive,
so the cheapest path prefers to take many short steps through dense regions rather than a
few long jumps across sparse gaps. This is the Fermat distance (named by analogy with
Fermat's principle, that a path takes the cheapest route). Two properties make it
attractive for our downstream arms. It is outlier-resistant: a single stray point in a
void cannot become a cheap stepping-stone, because the long edges to reach it are
penalised. And it has a published convergence guarantee for recovering the topology of the
underlying manifold as the sample grows, which matters for the loop-counting in Section 15
(topology). For these reasons it is our candidate primary metric.

**Diffusion distance.** Rather than committing to one shortest path, average over all
paths. Turn the neighbour graph into a random walk: from any point, step to nearby points
with probability set by a Gaussian falloff of their distance (we set the falloff scale to
the median squared edge length of the graph). Two points are close in diffusion distance
if a random walker starting at one reaches a similar distribution of places as a walker
starting at the other, after the same number of steps. Concretely we form the symmetric
normalised affinity matrix, take its top eigenvectors $\psi$ scaled by their eigenvalues,
and measure ordinary Euclidean distance in those scaled diffusion coordinates. Averaging
over many paths makes diffusion distance smooth and stable to noise in any single edge.
(This is a preview of the full diffusion-map construction in Section 8, used here only as
one ruler among five.)

### 5.4 What the diagnostic shows

![Figure 3. Pairwise-distance concentration across five metrics.](figures/03_metric_concentration.png)

Figure 3 plots the two concentration statistics for all five metrics, for both embedding
sets, on the shared 2,000-point z-scored subsample with $k = 15$. Panel A (left) shows
RDR, the relative distance ratio, where higher means more contrast and *less*
concentration; its vertical axis is on a logarithmic scale (each gridline is ten times the
one below), because the values span more than an order of magnitude. Panel B (right) shows
NN/mean, the nearest-neighbour over mean-pair ratio, where lower means *more*
concentration; its vertical axis is linear, from 0 to about 0.5. In both panels the five
metrics run along the horizontal axis (Euclidean, Cosine, Isomap, Fermat with $p=2$,
Diffusion), and for each metric there are two bars: blue is $E_{\mathrm{full}}$ (image
plus photometry plus redshift) and orange is $E_{\mathrm{img}}$ (image only). The number
printed above each bar is its value. The caption strip under the figure restates the
reading. There are no reference lines; the comparison is bar-to-bar.

The full set of measured values is in the table below (read directly from
`results/metric.json`), so the figure and the prose can be checked against the same
numbers.

| Metric | RDR, $E_{\mathrm{full}}$ | RDR, $E_{\mathrm{img}}$ | NN/mean, $E_{\mathrm{full}}$ | NN/mean, $E_{\mathrm{img}}$ |
| --- | --- | --- | --- | --- |
| Euclidean (control) | 14.69 | 14.16 | 0.433 | 0.430 |
| Cosine | 76.80 | 79.36 | 0.193 | 0.193 |
| Isomap (geodesic) | 42.12 | 44.46 | 0.179 | 0.178 |
| Fermat ($p=2$) | 281.04 | 259.14 | 0.164 | 0.164 |
| Diffusion | 95.01 | 99.16 | 0.269 | 0.267 |

All values are measured. Higher RDR means more contrast (less concentration); lower
NN/mean means more concentration. Every graph-based metric had zero infinite distances, so
all five rulers describe one connected cloud.

The first thing to look for is that the blue and orange bars are nearly the same height
everywhere. The two embedding sets concentrate almost identically, so the metric behaviour
is a property of the geometry, not an artefact of which inputs the model saw. The numbers
in the table make this concrete: for every metric the $E_{\mathrm{full}}$ and
$E_{\mathrm{img}}$ entries agree to within a few percent, and the NN/mean column is nearly
identical between the two sets. Whatever the metric is reacting to, both embeddings share
it.

The second thing to look for is the ranking. Raw Euclidean distance is the most
concentrated metric by a wide margin: it has the lowest RDR (about 14 to 15, so the
largest pairwise distance is only about fifteen times the spread above the smallest) and
the highest NN/mean (about 0.43, so a point's nearest neighbour sits at nearly half the
average pairwise distance, barely closer than a random galaxy). That is the concentration
we feared, measured. The intrinsic metrics that walk along the data give several times
more contrast. Fermat is the standout: RDR around 260 to 281, roughly eighteen to twenty
times the Euclidean contrast, and the lowest NN/mean at about 0.164, meaning a point's
nearest neighbour is on average about six times closer than a typical pair. Isomap (RDR
around 42 to 44) and diffusion (around 95 to 99) sit in between, both clearly better than
Euclidean. This is the headline of the section: following the geometry instead of cutting
across it restores the contrast that high dimension had erased.

Now the honest deviation from prior expectation. We expected cosine to concentrate like
Euclidean, because the cloud sits near a thin shell of nearly equal norms, and on such a
shell direction and position carry almost the same information. It did not. Cosine RDR is
about 77 to 79, far above Euclidean's 14 to 15, and its NN/mean (0.193) is much lower than
Euclidean's (0.43). So cosine pulls apart the distance distribution more than the
straight-line ruler does, even on this near-spherical cloud. We do not have a clean
single-cause explanation for why; the shell is not perfectly uniform, and small angular
differences evidently carry more contrast than the small radial differences that Euclidean
folds in. The faithful reading is narrow: cosine is a middling control here, more
contrastive than Euclidean but less so than the geodesic metrics, and we treat it as one
more reference point rather than as a failure of the prior or as a recommended downstream
metric. We do not build any downstream arm on cosine.

One caveat keeps these readings honest. More contrast is necessary, not sufficient. A
metric can spread distances apart for the wrong reason, for instance by over-weighting a
handful of long edges in a sparse region, and a high RDR alone cannot tell good contrast
from inflated contrast. That is exactly why we read RDR and NN/mean together (the extremes
and the local-versus-typical view agree on the ranking here) and why we favour Fermat on
grounds beyond its raw numbers: its outlier resistance and its topology-convergence
guarantee mean its contrast comes from following dense structure, not from rewarding stray
points. The diagnostic ranks the metrics; the theory tells us which high-contrast metric
to trust.

### 5.5 The decision this fixes for the rest of the report

The concentration diagnostic settles a choice that every later section inherits. Plain
Euclidean distance in this 1024-dimensional cloud is the most concentrated of the five
rulers, with the least contrast between near and far, so taking it at face value would
push the downstream methods toward the regime where neighbourhoods are unreliable. We
therefore adopt Fermat (the density-weighted geodesic, $p = 2$) as the primary metric for
the topology and curvature arms, because it delivers the most contrast (RDR around 260 to
281) and carries the resistance-to-outliers and topology-convergence properties those arms
need. We keep Euclidean as the explicit control throughout, run side by side with Fermat,
so that any geometric claim can be checked against the naive ruler and we can see when a
result depends on the metric and when it does not.

The roles each metric plays from here on are fixed as follows.

- Fermat ($p=2$): primary metric for topology (Section 15) and curvature (Section 14). Most contrast, outlier-resistant, topology-convergence guarantee.
- Euclidean: the control, run beside Fermat everywhere so metric-dependence is visible.
- Diffusion: cross-check here, and the basis of the coordinate system in Section 8 (diffusion maps), where the same walk-along-the-data idea becomes the axes we read physics off.
- Isomap: a secondary geodesic cross-check; it confirms the contrast gain over Euclidean without the density weighting.
- Cosine: a measured-but-unused reference; more contrastive than Euclidean but not built on downstream.

Two limits bound what this section can claim, and they are worth stating plainly. The
diagnostic ran on a single 2,000-point subsample, so the exact RDR and NN/mean values
carry sampling uncertainty we did not bootstrap here; the ranking is large and stable
across both embedding sets, but the specific numbers should be read as one draw, not as
tight estimates. And the diagnostic measures contrast, not correctness: it tells us which
metric resists concentration, not which metric reflects the true manifold geometry, a
question no single number can answer. With that understood, we have a defensible distance
to work in. The next two sections use it: Section 6 defines intrinsic dimension and its
estimators, and Section 7 reports how many independent directions this concentrated,
shell-like cloud actually uses.

## 6. Intrinsic dimension I: definition and estimators

Each galaxy leaves the model as a list of 1024 numbers. That is the ambient dimension: the
width of the box the data lives in. It is not the same thing as the number of independent
quantities you actually need to pin a galaxy down. A point that can only ever sit on the
surface of a ball in three-dimensional space has two real degrees of freedom, not three,
because once you fix its latitude and longitude you have fixed the point. The two free
quantities are the intrinsic dimension. The third coordinate is along for the ride, locked
to the other two by the constraint that says "stay on the sphere."

Intrinsic dimension (ID) is the number of independent knobs you would have to turn to move
to any nearby point on the data cloud. Said another way, it is the dimension of the smooth
surface (the manifold) that the points hug, measured from the inside, ignoring how that
surface happens to be folded into the wider 1024-dimensional space it is drawn in. If the
galaxies really vary along only a handful of physical axes (how red, how massive, how
feathered with spiral arms, how far away), then even though every galaxy is written with
1024 numbers, those 1024 numbers should move together in tight lockstep, and the true
count of free directions, the ID, should be small.

This number matters for the whole report. A small ID would say the model has compressed a
very wide representation down onto a thin sheet, and it would tell us roughly how many
physical axes that sheet can hold. A large ID, close to 1024, would say the representation
is spread out and the 1024 numbers are mostly independent, which would make every later
geometric claim (curvature, topology, concept axes) much harder to trust, because there
would be no low-dimensional shape to talk about. So before we describe the shape of the
manifold, we have to measure how many dimensions that shape even has.

### 6.1 Why one number, and one method, is not enough

ID is not a single quantity you read off a dial. It depends on the scale you look at, and
every way of measuring it has its own blind spot. Two facts force us to be careful.

First, ID changes with scale. Zoom in close enough to any smooth curved surface and it
looks flat and full-dimensional, the way the ground under your feet looks flat even though
Earth is round. Zoom in far enough on real data and you stop seeing the manifold at all;
you see the measurement noise, which is full-dimensional fuzz sprayed in every direction.
So at the very smallest distances, the distances between a point and its one or two
nearest neighbours, an ID estimate is contaminated by noise and reads too high. At larger
distances the estimate settles onto the manifold's true dimension, then eventually starts
to feel the finite size and curvature of the whole cloud. The honest reading is the value
on the plateau in between, not the value at the smallest scale. Any method that only
reports a single number at a single scale can be fooled.

Second, every estimator makes an assumption that can break. A method that assumes the data
is locally uniform will misread a region where density changes fast. A method that assumes
the surface is flat will read a curved surface as slightly higher-dimensional than it is.
A method built on counting nearest neighbours will inflate wildly if the data is actually
two tight blobs with almost no spread, because then the "nearest neighbour" is essentially
on top of the point and the ratio the method depends on blows up. No single estimator is
trustworthy on its own.

Our response is to use four estimators that fail in different, known ways, and to believe
the answer only where independent methods agree. Two of them, TwoNN and the local
Levina-Bickel MLE, work from the statistics of nearest-neighbour distances. One, Gride,
deliberately sweeps across scale so we can see the noise inflation separate from the true
dimension. The fourth, the PCA participation ratio, is a purely linear measure that does
not know about manifolds at all; it counts effective straight-line directions, and we keep
it as a baseline precisely because it answers a slightly different question. When a
neighbour-based nonlinear method and a linear-spread method both land near the same
number, that agreement is informative: it says the manifold is not curved much at the
scale we are measuring, because strong curvature would push the nonlinear estimate well
below the linear one. We will lean on exactly that comparison in Section 7.

Before any of this touches AION, we ran all four estimators on synthetic data of known
dimension at the same sample size (N = 48,398), so that any bias from finite samples or
curvature is one we have already seen and calibrated. That validation is the subject of
Section 7; here we define the four tools and the two ceilings that bound what they can
possibly tell us.

A note on preprocessing that applies to every estimator below. We z-score each of the 1024
embedding dimensions first, meaning we subtract that dimension's mean and divide by its
standard deviation, so every dimension has mean 0 and spread 1. The per-dimension spread
in the raw embedding ranges over a factor of about 23, so without this step a handful of
high-variance dimensions would dominate every distance and the ID estimate would mostly
reflect those few axes. z-scoring puts the dimensions on an equal footing before we ask
how many of them are really free. (Section 4 covers the preprocessing and the thin-shell
norm structure in full; this is the one-line reminder.)

### 6.2 TwoNN: dimension from the second-to-first neighbour ratio

TwoNN is the simplest of the four and the one we report first for each cloud. The idea
rests on a clean piece of geometry. Take a point. Find its nearest neighbour at distance
$r_1$, and its second-nearest neighbour at distance $r_2$. Form the ratio

$$\mu = \frac{r_2}{r_1}.$$

Here $r_1$ is the distance to the closest other point, $r_2$ is the distance to the next
closest, and $\mu \ge 1$ always, because the second neighbour cannot be closer than the
first. The symbol $\mu$ (mu) is just a name for that ratio.

Why does this ratio carry the dimension? In a space of dimension $d$, the amount of room
inside a ball of radius $r$ grows like $r^d$ (area grows like $r^2$ in a plane, volume
like $r^3$ in 3-space, and so on). If points are scattered roughly uniformly near our
chosen point, then the chance of finding a neighbour within radius $r$ is set by how much
volume that radius encloses, and volume scales with the power $d$. Working through that
probability gives a strikingly simple law for the ratio $\mu$: its cumulative distribution
is

$$P(\mu > x) = x^{-d}, \qquad x \ge 1.$$

In words: the probability that the second neighbour is more than $x$ times farther than
the first falls off as $x$ to the power minus $d$. The steeper that fall-off, the higher
the dimension, because in higher dimensions there is so much room just outside $r_1$ that
the second neighbour is almost always close behind the first, so large values of $\mu$ are
rare. This is a Pareto distribution, and the key point is that it depends only on $d$. The
unknown local density of points cancels out of the ratio, which is what makes TwoNN
attractive: it does not need to know how crowded the neighbourhood is.

From that law we estimate $d$. Taking logarithms turns the Pareto law into a statement
that $\log \mu$ follows an exponential distribution with rate $d$. The maximum-likelihood
estimate, the value of $d$ that makes the observed ratios most probable, has a closed
form. With one $\mu_i$ per point and $N$ points,

$$\hat{d}_{\text{MLE}} = \frac{N-1}{\sum_{i=1}^{N} \log \mu_i}.$$

Each $\mu_i$ is the second-to-first neighbour ratio at point $i$, $\log \mu_i$ is its
natural logarithm, and the sum runs over all points. The numerator is $N-1$ rather than
$N$ because using $N-1$ removes a small bias that the plain $N$ version carries; this is
the unbiased form of the estimator, and it is the number we quote. The intuition behind
the formula is direct: if the typical $\log \mu$ is small (second neighbours crowd close
behind first neighbours), the denominator is small and $\hat d$ is large, signalling high
dimension; if $\log \mu$ values are large, $\hat d$ is small.

A point estimate without an error bar is not enough, so we attach an interval. Because
$\sum \log \mu_i$ behaves like a sum of exponential variables, the quantity
$d \sum \log \mu_i$ follows a Gamma distribution, and that gives an exact posterior for
$d$ given the data. From it we read a 95% credible interval, the range that contains the
true $d$ with 95% probability under the model. We call this the Gamma-posterior interval.
As a second, assumption-light check we also run a 200-times bootstrap: we resample the
points with replacement 200 times, recompute $\hat d$ each time, and take the spread of
those values as an empirical interval. When the Gamma interval and the bootstrap interval
agree (they do, throughout Section 7), we trust the error bar.

We also keep the original TwoNN estimator as a cross-check on the MLE. In its first
published form, TwoNN did not use the MLE formula; instead it fit a straight line. Sort
the points by their $\mu$ value, build the empirical cumulative distribution $F(\mu)$ (the
fraction of points with ratio below each $\mu$), and note that the Pareto law
$P(\mu > x) = x^{-d}$ implies a linear relation:

$$\log\!\big(1 - F(\mu)\big) = -\,d \,\log \mu.$$

So a plot of $-\log(1-F)$ against $\log \mu$ should be a straight line through the origin
whose slope is exactly $d$. We fit that slope by least squares and report it as the
"linear" TwoNN value. It uses the same data as the MLE but weighs the tail of the
distribution differently, so a close match between the linear fit and the MLE (they agree
to within a few hundredths on every cloud we tested) tells us no single anomalous
neighbour is driving the result.

TwoNN's blind spot is exactly the scale problem from Section 6.1. It is built on the first
and second neighbours, the very smallest distances in the data, so it sees the
noise-inflated regime. On a clean synthetic manifold that does not matter, because there
is no noise and the small-scale and large-scale dimensions are the same. On real, noisy
embeddings it matters a lot: the TwoNN number will read high, and we treat it as an upper,
small-scale reading, not as the headline ID. The honest estimate comes from the methods
that look at larger scales.

### 6.3 Gride: an ID-versus-scale curve

Gride (the name is short for generalised ratios ID estimator) takes the same
neighbour-ratio idea as TwoNN and stretches it across scale on purpose, so that we can
watch the dimension change as we zoom out and pick out the plateau.

The generalisation is small but powerful. Instead of using the first and second
neighbours, Gride uses the neighbours at ranks $n_1$ and $2 n_1$. That is, it forms the
ratio of the distance to the $(2n_1)$ -th nearest neighbour over the distance to the $n_1$
-th nearest neighbour:

$$\mu^{(n_1)} = \frac{r_{2 n_1}}{r_{n_1}}.$$

When $n_1 = 1$ this is exactly TwoNN ($r_2/r_1$). When $n_1 = 256$ it compares the 512th
neighbour to the 256th, a ratio measured over a much larger neighbourhood and therefore at
a much larger physical scale. The probability law for this generalised ratio is again a
clean function of dimension $d$ alone (a Beta-prime form that reduces to the Pareto law
when $n_1 = 1$), and Gride solves for the $d$ that best fits the observed ratios at each
chosen $n_1$, with no need to throw any data away or subsample.

The output is therefore not one number but a curve: estimated ID as a function of $n_1$,
which is the same as ID as a function of scale. We compute it at
$n_1 \in \{1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256\}$. Reading this curve is how
we separate the two effects from Section 6.1. At the smallest $n_1$ the curve sits high,
because it is contaminated by full-dimensional noise. As $n_1$ grows the curve falls and
then flattens onto the manifold's true dimension; that flat stretch is the plateau, and
its height is the estimate we believe. If the curve never flattens but keeps sliding, that
itself is information (it says the cloud has no single clean dimension), and we report the
value at large $n_1$ as the large-scale reading. Gride needs no subsampling and uses every
point at every scale, which makes its curve smooth and its plateau easy to read. It is the
workhorse of our ID measurement.

The blind spot to keep in mind: at very large $n_1$ the neighbourhood starts to wrap
around the whole finite cloud, so the curve can bend down further not because the
dimension is truly lower but because we are running out of points and feeling the global
curvature. We read the plateau, not the extreme tail, and we cross-check its height
against the other two estimators.

### 6.4 Local Levina-Bickel maximum likelihood: where is the dimension higher?

The third estimator gives a dimension for every single point, which lets us ask not just
"what is the ID" but "is the ID higher in some regions than others." It is the
maximum-likelihood estimator of Levina and Bickel, applied locally.

Fix a point and look outward at its $K$ nearest neighbours, at distances
$r_1 \le r_2 \le \dots \le r_K$. Treat the arrival of neighbours as you move out from the
centre like events in a Poisson process whose rate grows with radius as $r^{d-1}$, the
natural rate for a $d$ -dimensional ball (its surface area, where new neighbours appear,
scales that way). Writing down the likelihood of seeing exactly this sequence of neighbour
distances and maximising it over $d$ gives a closed-form local estimate at that point:

$$\hat{d}_K = \left[ \frac{1}{K-1} \sum_{j=1}^{K-1} \log \frac{r_K}{r_j} \right]^{-1}.$$

Here $r_K$ is the distance to the outermost ($K$ -th) neighbour, $r_j$ is the distance to
the $j$ -th neighbour, and the sum runs over the inner $K-1$ neighbours. The bracket is
the average log-ratio of the boundary distance to each inner distance; its reciprocal is
the local dimension. The intuition matches TwoNN: if the inner neighbours are all bunched
far inside the boundary (large log-ratios, large bracket), the local dimension is small;
if they spread out toward the boundary, the dimension is large. To turn these per-point
values into one number for the cloud we average $\hat d_K$ over all points (the standard
pooled form), and we report that average.

The lever here is $K$, the neighbourhood size, and it plays the same role that $n_1$ plays
in Gride: it sets the scale. A small $K$ looks only at the closest neighbours and sees the
noise-inflated small-scale regime; a large $K$ averages over a wider neighbourhood and
settles toward the manifold dimension. So we sweep
$K \in \{10, 14, 20, 28, 40, 56, 80, 113, 160, 226\}$ and watch the pooled estimate as $K$
grows, reading the large- $K$ value as the manifold-scale dimension. Because the estimator
also produces a dimension per point, it can in principle map out regions of higher and
lower local dimension, which is the same machinery we reuse in Section 16 to compare
passive and star-forming populations.

The known biases: the local MLE assumes the neighbourhood is locally uniform and flat, so
it reads slightly low when $K$ is small (curvature and edge effects bite hardest in tiny
neighbourhoods) and can read slightly high on a strongly curved patch. Sweeping $K$
exposes the trend rather than hiding it, which is exactly why we sweep rather than fix one
$K$.

### 6.5 PCA participation ratio: the linear baseline

The fourth estimator is different in kind. The three above are nonlinear: they follow the
data along whatever curved surface it sits on, using only local neighbour distances. The
participation ratio is linear: it asks how many straight-line directions the cloud spreads
along, with no notion of a curved manifold at all. We include it on purpose, as a baseline
that answers a cleaner but cruder question, because the gap between it and the nonlinear
estimates is itself a measurement of curvature.

Start with principal component analysis (PCA). PCA finds the orthogonal directions in the
1024-dimensional space along which the cloud spreads the most. Each direction (a principal
component) comes with an eigenvalue $\lambda_i$, the variance of the data along that
direction: a big $\lambda_i$ means the cloud is stretched far along that axis, a tiny
$\lambda_i$ means it is nearly flat there. Order them $\lambda_1 \ge \lambda_2 \ge \dots$.
If the cloud were a thin pancake lying in a 3-dimensional slab inside the 1024-dimensional
space, only three eigenvalues would be large and the rest would be near zero.

The participation ratio turns the full set of eigenvalues into one effective count of
directions:

$$\text{PR} = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}.$$

Read it like this. If $D$ eigenvalues are all equal and the rest are zero, the formula
returns exactly $D$: the cloud genuinely fills $D$ directions evenly. If one eigenvalue
dwarfs all the others, the formula returns nearly 1: almost all the spread is along a
single line. In between, PR is a soft, continuous count of how many directions carry
meaningful variance, with directions weighted by how much they contribute. It is not an
integer and it is not meant to be; it is an effective dimension. The squaring in numerator
and denominator is what makes large eigenvalues dominate and tiny ones contribute almost
nothing, so PR ignores the long tail of near-zero noise directions automatically.

Why keep a linear measure when the manifold is presumably curved? Because PR sets a
ceiling that the nonlinear methods should sit at or below. A curved sheet uses up more
straight-line directions than its intrinsic dimension: a circle is a one-dimensional
curve, but it needs two straight axes to draw, so its PR would read near 2 while its true
ID is 1. So PR is biased upward by curvature, and the size of that upward bias, PR minus
the nonlinear ID, is a direct readout of how curved the manifold is at full scale. If PR
and the large-scale nonlinear ID come out close, the manifold is barely curved at that
scale; if PR sits well above, the manifold is strongly folded. We compute PR on the same
z-scored embedding as the other estimators. (We also report, in the sanity checks, a PR on
the raw, un-z-scored embedding, which is lower because the raw variance is concentrated in
fewer dimensions; the two are answering questions about two different preprocessings and
we keep them clearly labelled.)

PR's blind spot is the flip side of its design: it cannot tell a genuinely
high-dimensional flat cloud from a low-dimensional curved one, because both spread along
many straight axes. That is precisely why it cannot stand alone, and why we read it
together with the neighbour-based methods rather than instead of them.

### 6.6 The four estimators side by side, and how we read their agreement

It helps to see the four tools laid out together before we use them, because each row of
the table below is a different lever on the same question, and the whole strategy is to
cross the levers.

| Estimator | What it uses | Scale it sees | Manifold-aware? | Main bias | Role here |
|---|---|---|---|---|---|
| TwoNN | ratio $\mu = r_2/r_1$ of 2nd-to-1st neighbour distance, per point | smallest (1st-2nd neighbour) | yes (nonlinear) | reads high on noisy data (small-scale only) | small-scale / upper reading with a Gamma 95% interval |
| Gride | ratio $r_{2n_1}/r_{n_1}$ swept over rank $n_1$ | small to large (set by $n_1$) | yes (nonlinear) | tail bends at very large $n_1$ (global curvature) | the ID-versus-scale curve; its plateau is the headline |
| Local MLE | log-ratios of $K$ neighbour distances, per point, swept over $K$ | small to large (set by $K$) | yes (nonlinear) | low at small $K$, high on strong curvature | plateau cross-check; gives a per-point dimension |
| PCA PR | eigenvalues $\lambda_i$ of the full covariance | global, linear | no (linear) | reads high when the manifold is curved | linear baseline; PR minus nonlinear ID measures curvature |

Read the table this way. The three nonlinear estimators (TwoNN, Gride, local MLE) all
follow the data along its curved surface, but they look at different scales, so
disagreement between them is a scale effect we can interpret rather than a contradiction.
TwoNN sits at the smallest scale and is expected to read highest. Gride and the local MLE,
swept out to large neighbourhoods, should converge onto the same plateau if there is one.
The PCA participation ratio sits apart, linear and global, and its job is to bracket the
nonlinear answer from above.

Our decision rule for the headline ID is explicit. We do not average the four numbers,
because they answer slightly different questions and a blind average would mix the
noise-inflated small-scale value into the estimate. Instead we look for a plateau: a
stretch of the Gride curve, and a large- $K$ stretch of the local-MLE sweep, where the
estimate stops changing fast. The height of that shared plateau, cross-checked against the
PCA participation ratio, is what we report, with the small-scale TwoNN value quoted
separately as an upper bound. We trust the result to the extent that three mechanistically
independent methods (a large-scale neighbour ratio, a large- $K$ likelihood estimate, and
a linear spread count) land near each other. Where they split, we say so and explain which
scale each is reporting. This is the agreement-only standard promised in Section 6.1, made
concrete.

One more reading follows directly from the table's last column. If the large-scale
nonlinear ID and the linear PR come out close, the manifold is barely curved at the scale
we measure, because a strongly curved sheet would force the nonlinear estimate well below
the linear one (the circle-needs-two-axes effect from Section 6.5). That
global-minus-local gap is a number we extract in Section 7 and reuse in the curvature
discussion of Section 14. So the participation ratio is doing double duty: a sanity
ceiling on the ID, and a coarse curvature gauge.

A small illustrative arithmetic makes the TwoNN and local-MLE formulas concrete (these
numbers are made up for teaching, not measured). Suppose at some point the second
neighbour is 1.3 times as far as the first, so $\mu = 1.3$ and $\log \mu = 0.262$. A
single point gives $1/\log\mu = 1/0.262 \approx 3.8$, but that is a wildly noisy one-point
read: the true dimension only emerges once you average $\log \mu$ over tens of thousands
of points, which is exactly why TwoNN pools the whole sample in its
$\hat d_{\text{MLE}} = (N-1)/\sum \log \mu_i$ form. Single neighbour ratios are erratic;
their pooled statistic is stable. The same logic explains why the local MLE needs a
reasonably large $K$ before its per-point estimate means anything: one or two neighbours
carry almost no information about dimension, and only the accumulated log-ratios over many
neighbours pin it down.

### 6.7 The two ceilings: sample-size resolution and dimensional collapse

Two limits bound what any of these estimators can say, and stating them up front keeps the
Section 7 results honest.

The first is a hard resolution ceiling set by the sample size. A finite set of $N$ points
can only support so many independent directions before there simply are not enough points
to populate them; nearest-neighbour statistics degrade once the dimension climbs past
roughly the base-2 logarithm of the sample size. With $N = 48{,}398$,

$$\log_2 N = \log_2(48{,}398) = 15.56.$$

This is the largest ID our sample could reliably resolve. The exact constant is a rule of
thumb, not a theorem, but the message is firm: any ID estimate that pushes up toward 15 or
16 is bumping against the ceiling of what 48,398 points can resolve, and a value reported
above it would be untrustworthy regardless of method. We will see that the small-scale
TwoNN readings sit right under this line, which is one more reason to treat them as
inflated rather than real, and that the plateau estimates sit comfortably below it, in the
range where the sample size is not the limiting factor. We carry the 15.56 line into the
Section 7 figures as an explicit reference.

The second ceiling is about the model, not the data, and it cuts the other way.
Self-supervised encoders like AION are known to under-use their nominal width: through
training, the representation often collapses onto a subspace much narrower than its 1024
dimensions, a phenomenon called dimensional collapse. This means a small measured ID is
the expected outcome, not a surprise, and here is the caveat that goes with it: a small ID
by itself is not proof that the model has found a clean, physically meaningful
low-dimensional manifold. Collapse can produce a low ID for boring reasons (the
optimisation simply ignored most directions) as easily as for interesting ones (the data
genuinely lives on a thin physical sheet). The ID number tells us how thin the
representation is; it does not, on its own, tell us whether the thinness is meaningful.
That question is what the rest of the report is for. The diffusion-map shape (Section 8),
the strong decodability of physics from the embedding (Section 10), and the curvature and
topology measurements (Sections 14 and 15) are the evidence that the low-dimensional sheet
we find is organised by real galaxy properties and not just a collapsed leftover. So we
read the ID as a necessary first measurement and an upper bound on complexity, while
withholding the physical interpretation until the geometry backs it up.

With the four estimators defined, their failure modes named, and both ceilings fixed, we
can now run them. Section 7 first checks all four against synthetic manifolds of known
dimension at the same sample size, then turns them on AION and reads off the result.

## 7. Intrinsic dimension II: validation and the AION result

Section 6 defined intrinsic dimension and laid out the four estimators. Now we use them.
This section does two things. First it shows that the estimators work, by running them on
synthetic shapes whose dimension we already know. Then it reports the dimension of the
real AION-1 galaxy cloud and gives the headline number with its uncertainty and its
caveats.

A reminder of the terms, in one line each, so this section reads on its own. Intrinsic
dimension (ID) is the number of independent knobs you need to specify a point, which can
be far below the ambient count of 1024 numbers. TwoNN is the estimator that uses the ratio
of each point's second-nearest to first-nearest neighbour distance. Gride generalises that
ratio to larger neighbour ranks, so it returns a dimension at each scale rather than one
number. The local Levina-Bickel MLE (we will call it local-MLE) estimates a dimension per
point from its K nearest neighbours, then we average. PCA participation ratio (PCA-PR) is
the linear baseline: it counts how many principal directions carry real variance. All four
were run on the z-scored embeddings (each of the 1024 coordinates shifted to mean zero and
scaled to unit standard deviation), with GPU-accelerated nearest-neighbour search out to k
= 512 neighbours.

### 7.1 Why validate first

An ID estimator is a piece of math with assumptions baked in. TwoNN assumes the points
sample a smooth manifold with roughly uniform density at the smallest scale. Gride assumes
the same law holds as you step out to larger neighbour ranks. The local-MLE assumes the
neighbourhood of each point looks like a flat patch of a d-dimensional space. PCA-PR
assumes the spread you care about is linear. None of these assumptions is guaranteed to
hold on a real learned embedding, and a wrong answer from a confidently-stated estimator
is worse than no answer.

So before trusting any number on AION, we ran the full battery on synthetic data of known
dimension, at the same sample size as the real run (N = 48,398 points), with the same code
path. If an estimator cannot recover a dimension we already know, we have no reason to
believe it on a cloud we don't. This is the cheapest insurance in the whole study, and it
is the part most papers skip.

We used three synthetic manifolds plus one deliberate trap.

- A **hypersphere** of intrinsic dimension 5: points sampled on the surface of a sphere living in a higher-dimensional space. The surface itself is 5-dimensional and curved. Truth d = 5.
- A **Swiss roll** of intrinsic dimension 2: a flat 2D sheet rolled up into a spiral in 3D, the textbook curved-but-flat-intrinsically test. Truth d = 2.
- A **linear 5-plane**: a flat 5-dimensional slab with no curvature at all, embedded linearly in the ambient space. Truth d = 5.
- A **two-blob control** (`twoblobs`): two tight Gaussian clusters with no manifold structure between them. There is no honest intrinsic dimension here, so the right behaviour is for the estimator to misbehave in a recognisable way. This is a trap, not a target.

### 7.2 What the validation found

![Figure 1. Intrinsic-dimension estimators recover the known dimension of synthetic manifolds.](figures/01_id_synthetic_validation.png)

Figure 1 shows the validation. The horizontal axis groups the three synthetic manifolds:
hypersphere (truth d = 5), Swiss roll (truth d = 2), and linear 5-plane (truth d = 5). The
vertical axis is the estimated intrinsic dimension, a pure number with no units. Within
each group there are four coloured bars, one per estimator: TwoNN-MLE (blue), Gride at
neighbour rank n1 = 256 (orange), the local-MLE plateau at K = 226 (green), and PCA-PR
(purple). The black dashed lines mark the known truth (5, 2, 5) and are labelled at the
right. The thin black caps on the blue TwoNN bars are its 95% interval, derived from the
Gamma posterior of the maximum-likelihood fit. The trap case `twoblobs` is not drawn, on
purpose, and the note at the foot of the figure says why.

The three manifold-aware estimators land essentially on truth. On the hypersphere the
measured values are TwoNN 4.96 (95% interval [4.91, 5.00]), Gride at n1 = 256 equal to
4.83, and local-MLE at K = 226 equal to 4.88 (all measured). On the Swiss roll: TwoNN 1.95
[1.94, 1.97], Gride 1.93, local-MLE 1.96. On the linear 5-plane: TwoNN 5.08 [5.03, 5.12],
Gride 4.97, local-MLE 5.02. These are tight, and the TwoNN intervals are narrow because N
is large. The estimators do not just get the right ballpark; they get the right number to
within a few percent.

PCA-PR (purple) behaves exactly as its math predicts, and that prediction includes a known
wrinkle. On the flat linear 5-plane it reads 4.99, spot on, because there the spread
really is linear and PCA-PR is built to count linear directions. On the curved hypersphere
it reads 6.00, slightly **above** the true 5. That is not a bug. A 5-sphere lives on a
curved surface, and to wrap a curved 5D surface into a linear coordinate system you need a
sixth linear direction to hold the curvature. PCA-PR counts that extra linear direction
because it cannot see that the surface is intrinsically only 5D. On the Swiss roll the
same effect is larger: PCA-PR reads 2.98 for a truly 2D sheet, because the roll's
curvature spends a third linear direction. So PCA-PR is a faithful **linear** measure, and
on a curved manifold it reads a little high by construction. We keep it, but we read it as
the linear baseline, not as the manifold dimension. This distinction matters for the AION
result below.

Now the trap. The two-blob control returns a TwoNN value of about 212 (measured; the JSON
gives 212.4 with interval [210.5, 214.3]). That is not a dimension. It is the estimator
breaking on a structure that violates its assumption. TwoNN reads the ratio of
second-to-first neighbour distances; inside a tight cluster of near-zero diameter, those
two distances are both tiny and nearly equal, their ratio sits just above 1, and the
formula $d = (N-1)/\sum \log \mu$ (where $\mu$ is that ratio) divides by a sum of
logarithms that is almost zero, which blows the estimate up. The PCA-PR for the same two
blobs reads about 1.02, because two point clouds separated along one direction have
essentially one linear axis of spread. The two estimators disagree wildly, and that
disagreement is itself the signal: there is no manifold here. We exclude `twoblobs` from
any dimension claim. We keep it in the report because it teaches the failure mode: a
single huge ID with no agreement across estimators means the manifold assumption is
broken, not that the dimension is huge. Hold that lesson; it is exactly how we read the
small-scale AION number next.

The takeaway from Figure 1 is simple and it is the licence for everything after: on data
of known dimension, our estimators recover the truth, the manifold-aware three agree with
each other, and PCA-PR reads a touch high on curvature in a way we understand. We can
trust them on AION.

### 7.3 The AION result: dimension versus scale

Now the real cloud. The honest way to report ID on a learned embedding is not a single
number but a curve: dimension as a function of the scale at which you look. The reason is
that real data has noise on top of structure. At the very smallest scale, the distance to
your first and second neighbours is dominated by sampling noise and by the finite spacing
of points, which inflates the dimension. As you step out to larger neighbourhoods, the
noise averages down and the true manifold dimension shows through as a flat stretch, a
plateau. The plateau is the estimate. The small-scale peak is not.

![Figure 2. AION galaxy-embedding intrinsic dimension across neighbour scales; plateau ID approx 10-12.](figures/02_id_aion_scale_curves.png)

Figure 2 shows this for AION in two panels. Both panels share the same vertical axis:
intrinsic dimension, a pure number. The left panel uses Gride; its horizontal axis is the
Gride neighbour rank $n_1$ on a base-2 log scale, running from 1 (smallest scale, closest
neighbours) to 256 (largest scale shown). The right panel uses the local-MLE; its
horizontal axis is the neighbourhood size $K$, also base-2 log, from 10 to 226. In both
panels there are two solid lines: blue is E_full (the embedding built from image,
photometry, and redshift) and red is E_img (the image-only embedding, our leakage-free
set). Three horizontal reference lines run across both panels. The red dashed line is the
PCA-PR linear anchor (E_img 11.93, E_full 11.18, both measured). The green shaded band
from 4 to 10 is the astrophysical prior, the optimistic range we expected from the physics
(Section 2). The grey dotted line at the top is the resolution ceiling
$\log_2(N) = 15.56$: a sample of N points can only resolve a dimension up to about
$\log_2 N$, so any reading above that line is at or beyond the limit of what 48,398 points
can support. The "plateau = the estimate" annotation in the left panel points at the flat
tail of the Gride curve.

Read the left panel first. The Gride curve starts high at the smallest scale and falls
steadily as the scale grows. For E_full the measured values are 16.56 at $n_1 = 1$, then
12.09 at $n_1 = 32$, 11.44 at $n_1 = 64$, and 10.10 at $n_1 = 256$ (all measured). For
E_img the same descent runs 16.05 down to 9.89 at $n_1 = 256$. The curve is not flat at
small scale and flat at large scale; it descends throughout and is flattening into a
plateau near 10 by the largest scale we trust. That shape is the textbook signature of
noise-inflated small scales settling onto the manifold dimension. The starting value near
16.5 is **not** the dimension. It is the same noise-inflation that, in an extreme form,
gave the two-blob trap its absurd 212: first- and second-neighbour distances at the
smallest scale carry sampling noise that pushes the estimate up. We read the value where
the curve settles, near 10, as the structural estimate.

The right panel tells the same story from a different estimator with a different
mechanism, which is the point of running both. The local-MLE rises slightly to a gentle
peak and then declines to a plateau. For E_full the measured values are 12.97 at K = 10, a
peak near 13.1 at K = 14, and 11.40 at K = 226. For E_img: 12.71 at K = 10 down to 11.16
at K = 226. The local-MLE plateau sits near 11.4. It does not start as high as Gride at
the smallest scale because it averages a per-point dimension over a whole neighbourhood
from the start, which damps the noise spike, but it agrees on where things settle.

Here is why the agreement matters. Gride and the local-MLE are mechanistically
independent. Gride uses one global ratio law evaluated at a chosen neighbour rank; the
local-MLE fits a separate maximum-likelihood dimension at every point from its K neighbour
distances and then averages. They share the nearest-neighbour graph but not the statistic.
When two methods built on different assumptions both land near the same place, that place
is more believable than either alone. They land near 10 to 11.4. And the PCA-PR linear
anchor, a third and completely different construction that counts linear directions rather
than fitting neighbour ratios, sits at 11.18 (E_full) and 11.93 (E_img), right in the same
window.

So three mechanistically-independent estimators agree: Gride at large scale near 10,
local-MLE plateau near 11.4, PCA-PR near 11. All three sit below the $\log_2 N = 15.56$
ceiling, so the sample is large enough to resolve them; we are not bumping the limit of
the data. The TwoNN small-scale reading (16.56 for E_full, 16.05 for E_img, both measured
with tight intervals) sits right at the ceiling and is the noise-inflated small-scale
value, not the estimate. We report it for completeness and we do not use it as the
headline.

The two embeddings deserve a direct comparison here. E_full fuses image, photometry, and
redshift; E_img sees the image alone. If the extra inputs added genuinely new independent
axes of variation, E_full would carry a clearly higher dimension. It does not. The two
curves in Figure 2 track each other closely at every scale, and the plateau values differ
by under one dimension (Gride at $n_1 = 256$: 10.10 for E_full versus 9.89 for E_img;
local-MLE at K = 226: 11.40 versus 11.16; PCA-PR: 11.18 versus 11.93, where E_img is even
slightly higher). The reading (interpreted): adding photometry and redshift does not
expand the dimension of the cloud. The image already carries almost all the degrees of
freedom, and the extra modalities mostly reinforce directions the image had rather than
opening new ones. This is consistent with the probe results in Section 10, where colour
and redshift turn out to be strongly decodable from the image alone.

The measured numbers that anchor the headline are collected below, so the reader can see
the agreement and the spread at one glance.

| Estimator | What it reads | E_full | E_img | Role |
|---|---|---|---|---|
| TwoNN MLE (smallest scale) | 2nd/1st neighbour ratio | 16.56 [16.42, 16.71] | 16.05 [15.91, 16.19] | noise-inflated, not the estimate |
| Gride at $n_1 = 256$ | global ratio, large scale | 10.10 | 9.89 | plateau, the estimate |
| local-MLE at K = 226 | per-point MLE, averaged | 11.40 | 11.16 | plateau, the estimate |
| PCA participation ratio | linear effective directions | 11.18 | 11.93 | linear anchor |
| $\log_2 N$ ceiling | resolution limit | 15.56 | 15.56 | upper bound on resolvable ID |

All entries are measured (the bracketed pairs are 95% intervals from the Gamma posterior).
The three estimate-bearing rows (Gride plateau, local-MLE plateau, PCA-PR) cluster between
about 10 and 12; the TwoNN row is the small-scale artefact; the ceiling row shows the
estimates sit comfortably under the resolution limit.

**Headline (interpreted from the measured curves): the intrinsic dimension of the AION
galaxy embedding is about 10 to 12.** The model takes 1024 input numbers per galaxy and
uses only about 10 to 12 independent degrees of freedom to place a galaxy in its space.
That is a large compression, roughly a hundredfold drop from the ambient count.

One honest caveat on the size of that number, owed from Section 6. Self-supervised
encoders are known to under-use their nominal dimension; a representation can collapse
onto a low-dimensional subspace for reasons that have nothing to do with the data lying on
a clean manifold. A small ID is therefore expected and is not, by itself, proof of a
meaningful low-dimensional geometry. What makes the ID here more than a collapse artefact
is that the rest of the report (the smooth diffusion spectrum in Section 8, the strong
physics decodability in Section 10, the named concepts in Section 13) shows that the
low-dimensional space is organised by real galaxy properties. The ID number alone cannot
carry that claim; it is the agreement with the physics that does.

### 7.4 The linear baseline, and what the gap to it means

The ID curves above are nonlinear: they ask how many knobs you need if you are allowed to
bend the coordinate system to follow a curved manifold. A fair question is how the answer
compares to the simplest possible measure, plain linear PCA, which is not allowed to bend
anything.

![Figure 17. Linear PCA cumulative explained variance, the linear null for intrinsic dimension.](figures/17_pca_variance.png)

Figure 17 is that linear baseline. The horizontal axis is the number of principal
components (PCs) included, from 0 to 200. The vertical axis is the cumulative explained
variance as a percentage: how much of the total z-scored spread the first so-many PCs
capture between them. The red curve is E_img (image only), the blue curve is E_full (image
plus spectra/photometry); they nearly overlap. The faint horizontal dotted lines mark the
50%, 90%, 95%, and 99% levels (labelled on the right). The coloured dots and the in-panel
text mark where each curve crosses 95% and 99%. The note at the foot states plainly that
this is a baseline, not the headline.

The numbers on this z-scored data (measured, recomputed on the z-scored embeddings to
match the ID run, and printed on the figure): linear PCA needs about 58 components to
reach 95% of the variance for E_img and about 60 for E_full, and about 163 (E_img) / 169
(E_full) components to reach 99%. A small clarification on these counts, because they
differ from a number elsewhere in this report. The sanity check reported PCA reaching 95%
in about 44 to 45 components, but that was on the **raw** (not z-scored) embeddings.
Z-scoring spreads the variance across more dimensions on purpose, so the same 95% target
now takes more components (about 58 to 60). Both are correct; they describe different
preprocessing. The figure uses z-scored data so that it sits on the same footing as the ID
estimators, which all ran z-scored. We state both honestly.

Now the contrast that is the whole reason for the figure. A purely linear summary of this
embedding needs roughly 58 to 60 directions to recapture 95% of the spread. The nonlinear
intrinsic dimension is only about 10 to 12. Those two numbers are not in conflict; they
answer different questions. Linear PCA must spend a separate axis on every direction the
data wanders, including directions that only exist because a low-dimensional manifold is
folded inside the 1024-dimensional box. The nonlinear estimators are allowed to follow the
fold, so they need far fewer knobs. The factor of roughly five between the linear count
(~58) and the nonlinear ID (~11) is a direct, measured statement that the geometry is
curved, not flat: if the manifold were a flat linear slab, the two numbers would match.

But there is a second, subtler comparison that points the other way, and we have to report
it. PCA-PR is also a linear measure, and it reads about 11, essentially the same as the
large-scale nonlinear ID of about 10. How can the 95%-variance count say "very nonlinear"
(58 vs 11) while PCA-PR says "barely nonlinear" (11 vs 10)? Because they measure different
things. The 95% count is a tail measure: it keeps adding small-variance directions until
it has swept up 95% of the spread, so it is sensitive to a long tail of low-variance axes.
PCA-PR is a concentration measure: $(\sum \lambda)^2 / \sum \lambda^2$, dominated by the
handful of large-variance directions and largely blind to the tail. So PCA-PR and the
large-scale nonlinear ID both describe the **core** of the cloud, the directions that
carry most of the spread, and there they agree at about 10 to 11. The 95%-variance count
describes the **full** cloud including its faint tail, and there the linear cost is much
higher.

The reading we take (interpreted): at the scale of the core manifold, the nonlinear ID
(~10) is close to the linear PCA-PR (~11), which means the manifold is only **weakly
curved at that scale**. The gap between a nonlinear ID and its linear PCA-PR anchor is a
curvature gauge: a large gap means the manifold bends sharply at manifold scale; a small
gap means it is nearly flat there. Here the gap is small. So the picture is a
low-dimensional core that is gently curved, sitting inside a higher-dimensional linear box
whose extra ~50 linear directions are the folding of that core, not new independent
structure. This small global-minus-local gap is the same conclusion the curvature arm
reaches in Section 14 from a completely different measurement (mostly positive, gentle
curvature with only weak localised branching), and the two arms agreeing is part of why we
believe it.

### 7.5 Comparison to the astrophysical prior

Section 2 set an optimistic prior. Clean photometry carries roughly 2 to 5 effective
dimensions, galaxy spectra roughly 3 to 10, so an honest multimodal embedding plausibly
uses something like 5 to 10 axes. That prior is the green band in Figure 2. It was a
ceiling on the optimistic case, not a point prediction, and we said so when we set it.

The measured ID of about 10 to 12 sits **at or just above** the top of that 4-to-10 band.
We do not round it down to fit inside the band, and we do not call it a clean confirmation
of the most optimistic case. It is honestly a little higher than the best-case prior.
There are two readings, and our data cannot fully separate them. One: the embedding is
multimodal and fuses image, photometry, and redshift, so it can legitimately carry a few
more axes than photometry-or-spectra-alone priors suggest, which would put 10 to 12 in
line with expectation. Two: a couple of those extra axes are not physics but residual
structure the encoder happens to keep (nuisance directions, instrument or selection
imprints). We cannot tell these apart from the dimension alone. What we can say plainly is
that the embedding is genuinely low-dimensional, the compression from 1024 to about 11 is
large and real, and the dimension lands slightly above the optimistic physical prior
rather than below it.

That is the dimension result, with its uncertainty stated and its caveats owned. The next
section asks what shape those 10-to-12 dimensions form, by building a diffusion map and
reading its spectrum. The answer there, one smooth simply-connected body with no spectral
gap, is what gives these axes their physical meaning.

## 8. The shape of the manifold: diffusion maps

Section 6 and Section 7 (intrinsic dimension) told us how many independent knobs the
embedding uses, roughly 10 to 12. That is a count. It does not tell us how the cloud is
arranged: whether it is one connected body or several separate islands, whether it bends
smoothly or breaks at sharp seams. To see the arrangement we need a method that turns the
1024-number cloud into a small number of meaningful coordinates while staying faithful to
which galaxies are actually near which. Diffusion maps do exactly that, and they come with
a piece of theory we can lean on. This section explains the method in plain words and in
symbols, says why we chose it over the more familiar UMAP and t-SNE, describes two
practical fixes we had to make, and reads the eigenvalue spectrum (Figure 4). The
headline, stated up front so the rest can earn it: the spectrum decays smoothly with no
dominant gap, and the cloud is a single connected component, so the embedding is one
continuous body rather than a set of discrete clusters.

### 8.1 The idea in plain words

Imagine you scatter a drop of dye on the cloud of galaxies and let it spread, but the dye
can only hop from a galaxy to its near neighbours, not jump across empty space. Run the
spreading for a short time and the dye stays in a tight blob. Run it longer and the dye
fills out the broad shape of the cloud, flowing easily along dense, well-connected regions
and slowly across thin necks where few galaxies bridge two areas. A diffusion map reads
out the directions along which this spreading is slowest. Those slow directions are the
large-scale axes of the data: the ones you have to travel furthest, in the sense of "many
small neighbour-to-neighbour hops," to get from one end to the other. Fast directions, by
contrast, are local wiggles that the dye smooths out almost immediately. We keep the slow
directions and throw away the fast ones, and what is left is a low-dimensional coordinate
system that respects the cloud's real connectivity.

That is the intuition. The dye spreading is a random walk: a walker standing on one galaxy
steps to a nearby galaxy with a probability that is higher the closer the two are. The
geometry of the cloud is encoded in how that random walk mixes. The slow-mixing directions
are eigenvectors of the walk's transition matrix, and that is what the construction below
computes.

### 8.2 The construction, with every symbol defined

Start from the z-scored embeddings (each of the 1024 dimensions shifted to mean zero and
scaled to unit standard deviation, as set up in Section 4, so that no single high-variance
dimension dominates the distances). Let $x_i$ be the 1024-vector for galaxy $i$. We work
on the full sample, $N = 48{,}398$ galaxies.

**Step 1: affinity.** For each pair of galaxies we define an affinity $W_{ij}$, a number
that is large when the two are close and falls off smoothly to zero when they are far
apart. We use a Gaussian (bell-curve) kernel,
$$
W_{ij} = \exp\!\left(-\frac{\lVert x_i - x_j \rVert^2}{\sigma_i\,\sigma_j}\right),
$$
where $\lVert x_i - x_j \rVert$ is the ordinary (Euclidean) distance between the two
vectors and $\sigma_i$ is a per-galaxy length scale called the bandwidth, explained in
Section 8.4. The bandwidth sets "how far is near": pairs much closer than $\sigma$ get an
affinity near 1, pairs much farther get an affinity near 0. To keep the computation
tractable and the graph local, we only compute affinities to each galaxy's $k = 64$
nearest neighbours and treat all other pairs as zero. So $W$ is a sparse $N \times N$
matrix, the weighted adjacency matrix of a neighbour graph.

**Step 2: the anisotropic ($\alpha = 1$) normalisation.** Here is the step that makes
diffusion maps trustworthy on real data, where galaxies are not spread evenly. Define the
degree of galaxy $i$ as $q_i = \sum_j W_{ij}$, the total affinity flowing into it. A
galaxy sitting in a crowded region has a high $q_i$ simply because it has many close
neighbours, not because that region is special. If we ignored this, the random walk would
pile up in dense regions and our "slow directions" would partly trace where galaxies
happen to be densely sampled rather than the true shape of the manifold. The fix, from
Coifman and Lafon's original work, is to divide the affinity by the densities before
building the walk:
$$
\tilde{W}_{ij} = \frac{W_{ij}}{q_i^{\alpha}\, q_j^{\alpha}}, \qquad \alpha = 1.
$$
Setting the exponent $\alpha = 1$ has a clean mathematical payoff. As the sample grows and
neighbourhoods shrink, the resulting random walk converges to the Laplace-Beltrami
operator of the underlying manifold. In plain words, the Laplace-Beltrami operator is the
natural "spreading" or "smoothing" operator on a curved surface, the curved-space version
of how heat diffuses on a flat sheet. It depends only on the manifold's shape, not on how
thickly or thinly we sampled points on it. So with $\alpha = 1$ the recovered geometry
does not depend on sampling density: two surveys that observed the same population with
different selection functions would, in the limit, recover the same diffusion geometry.
That density-independence is exactly the property we want when we cannot fully trust our
sampling to be uniform. We always use $\alpha = 1$; this is a deliberate choice, not a
default.

**Step 3: the Markov matrix.** Now turn the density-normalised affinities into transition
probabilities. Let $d_i = \sum_j \tilde{W}_{ij}$ be the new (re-normalised) degree, and
define
$$
P_{ij} = \frac{\tilde{W}_{ij}}{d_i}.
$$
Each row of $P$ sums to 1, so $P_{ij}$ is the probability that a walker at galaxy $i$
steps to galaxy $j$ in one step. $P$ is the Markov matrix (a Markov process is one where
the next step depends only on where you are now, not on your history). Powers of this
matrix, $P^t$, describe the walk after $t$ steps: $P^t_{ij}$ is the probability of being
at $j$ after starting at $i$ and taking $t$ hops. That is the dye spreading for time $t$.

**Step 4: eigenpairs.** The slow directions of the walk are the eigenvectors of $P$. An
eigenvector $\psi_\ell$ and its eigenvalue $\lambda_\ell$ satisfy
$$
P\,\psi_\ell = \lambda_\ell\, \psi_\ell,
$$
which means: applying one step of the walk to the pattern $\psi_\ell$ just rescales it by
$\lambda_\ell$ without changing its shape. Because $P$ is a probability matrix, all
eigenvalues lie in the range $0 \le \lambda_\ell \le 1$. We order them from largest to
smallest, $1 = \lambda_0 \ge \lambda_1 \ge \lambda_2 \ge \cdots$. The largest,
$\lambda_0 = 1$, always belongs to the trivial constant eigenvector (the walk's stationary
state, where the dye has spread everywhere and stopped changing); it carries no shape
information and we discard it. After $t$ steps the eigenvector $\psi_\ell$ has decayed by
a factor $\lambda_\ell^t$. An eigenvalue close to 1 decays slowly, so its pattern persists
for many steps: that is a slow, large-scale direction. An eigenvalue well below 1 decays
fast: a local wiggle. Ranking eigenvalues from high to low ranks the directions from most
global to most local.

**Step 5: diffusion coordinates.** The new coordinate of galaxy $i$ along axis $\ell$ is
the value of the $\ell$ -th eigenvector at that galaxy, weighted by the eigenvalue:
$$
\Psi_\ell(i) = \lambda_\ell\, \psi_\ell(i)
$$
(for a single diffusion step, $t = 1$; more generally $\lambda_\ell^{t}\,\psi_\ell(i)$).
The weighting matters. It automatically down-ranks the faster directions, so when we keep
the first few diffusion coordinates we are keeping the genuinely slow, global axes and
letting the noisy local ones fade. The Euclidean distance between two galaxies in these
weighted coordinates equals the diffusion distance between them, which measures how hard
it is for the random walk to get from one to the other, averaged over every path of every
length that connects them. That averaging is what makes diffusion distance stable to
noise: a single broken or spurious edge barely changes a quantity computed over all paths
at once. We computed the top 30 eigenvectors and kept the leading diffusion coordinates
for the physics reading in Section 9.

### 8.3 Why diffusion maps, and not UMAP or t-SNE

UMAP and t-SNE are the two most common tools for drawing a high-dimensional cloud in two
dimensions, and they make beautiful pictures. We did not use them as our geometry method,
for three reasons that matter for a report meant to be defensible.

First, they are stochastic and their output depends on settings. Both methods start from
random initial positions and optimise a layout by gradient descent, so a different random
seed gives a visibly different picture, and the result shifts with knobs like the
perplexity (t-SNE) or the neighbour count and minimum-distance (UMAP). Two honest analysts
can get two different-looking maps of the same data. A diffusion map is the solution of a
fixed eigenvalue problem: the same input gives the same eigenvectors every time, up to
sign. It is deterministic.

Second, UMAP and t-SNE are built to preserve local neighbourhoods at the cost of global
structure. They will happily tear a single connected sheet into visually separated blobs,
or pull genuinely separate groups close together, because their objective rewards keeping
near points near and is almost indifferent to large-scale distances. That is fine for
exploration but dangerous for the exact question we are asking, which is a global one: is
this cloud one piece or many? Diffusion distance is a global quantity by construction, so
the diffusion spectrum speaks directly to connectivity.

Third, diffusion maps carry theory we can cite. The $\alpha = 1$ convergence to the
Laplace-Beltrami operator gives the density-independence described above, and the
eigenvalues have a clean meaning (decay rates of a random walk) that we can read as
evidence about cluster structure. UMAP and t-SNE distances do not have a comparable
closed-form interpretation. So we use diffusion maps as the measurement and reserve
UMAP-style plots, where we show them at all, as illustration only. In short:
deterministic, theory-backed, and density-robust.

A fair caveat belongs here. Diffusion maps are not magic. They still depend on the
bandwidth and the neighbour count, the kernel is still Gaussian by assumption, and the
convergence guarantee is a large-sample limit, not an exact statement at our finite $N$.
We treat the spectrum as strong evidence, not proof.

### 8.4 Two practical fixes: self-tuning bandwidth and the connectivity check

A first attempt with a single global bandwidth $\sigma$ (the same length scale for every
galaxy) failed in an instructive way. Because the cloud is denser in some regions than
others, no single $\sigma$ fits everywhere: a value tuned to the dense core makes the
sparse outskirts look disconnected, while a value tuned to the outskirts blurs the core
into mush. The global-bandwidth graph came out degenerate and fragmented, splitting the
cloud into many tiny disconnected pieces that are an artefact of the fixed scale, not real
structure.

The fix is a self-tuning, local-scaling bandwidth (the Zelnik-Manor and Perona idea).
Instead of one global $\sigma$, each galaxy gets its own length scale set by its local
crowding: we take $\sigma_i$ to be the distance from galaxy $i$ to its 7th nearest
neighbour. In a dense region the 7th neighbour is close, so $\sigma_i$ is small and the
kernel is sharp; in a sparse region the 7th neighbour is far, so $\sigma_i$ is large and
the kernel reaches farther. The product $\sigma_i \sigma_j$ in the affinity then adapts
the notion of "near" to each pair's local density. This is what makes the affinity in Step
1 carry two indices on the bandwidth. With local scaling the graph became well-connected
and the spectrum stable.

We also ran an explicit connectivity check on the final graph, every time. A diffusion map
is only meaningful on a connected graph: if the neighbour graph breaks into separate
components, each component gets its own trivial $\lambda = 1$ eigenvector and the whole
construction is reporting disconnection rather than shape. We count the connected
components of the kNN graph directly. For both embeddings the count is exactly **1
connected component** (measured), for the full multimodal set $E_{\text{full}}$ (image
plus photometry plus redshift) and for the image-only set $E_{\text{img}}$ alike. One
component is the precondition for everything that follows, and it is also, on its own, a
first piece of the "one continuous body" conclusion. Section 15 (topology) confirms this
independently through a minimum-spanning-tree analysis that finds the same single giant
piece.

### 8.5 Reading the spectrum

With a valid graph in hand, the eigenvalues themselves answer the cluster question. Here
is the logic. Suppose the cloud really were two well-separated clusters joined by only a
thin bridge. Then a random walker started in one cluster would take a very long time to
cross to the other, so there would be a near-constant pattern within each cluster that the
walk barely mixes: two slow modes instead of one. Concretely, $\lambda_1$ would sit almost
as high as $\lambda_0 = 1$, and then there would be a sudden, large drop, a spectral gap,
between that pair of cluster-counting eigenvalues and all the faster within-cluster modes
below. The size of the gap measures how cleanly separated the clusters are, and the number
of eigenvalues above the gap counts the clusters. A smooth decay with no big drop is the
signature of one connected body whose modes are a graded sequence of
larger-and-larger-scale wiggles, not a small set of cluster indicators.

So we look for a gap. The measured eigenvalues for $E_{\text{full}}$ are
$$
\lambda_0 = 1.000,\;\; \lambda_1 = 0.9989,\;\; \lambda_2 = 0.9772,\;\; \lambda_3 =
0.9607,\;\; \lambda_4 = 0.9598,\;\; \ldots,\;\; \lambda_{29} = 0.8213,
$$
a slow, steady glide down from 1 toward 0.82 across the first 30 modes. The consecutive
gaps $\lambda_k - \lambda_{k+1}$ are all small. The largest single gap is only about
**0.0217** (measured), the step from $\lambda_1$ to $\lambda_2$, and it is nowhere near
large enough to set off two or three eigenvalues as cluster counters from the rest. For
$E_{\text{img}}$ the picture is the same, a smooth decay with a largest gap of about
**0.0276** (measured). No dominant spectral gap, in either embedding. Read at face value,
that says the cloud is one continuous body, consistent with the single connected component
from the graph check.

![Figure 4. Diffusion-operator eigenvalue spectrum and consecutive gaps.](figures/04_diffusion_spectrum.png)

**Figure 4.** The diffusion spectrum for both embeddings, built on the local-scaling
neighbour graph. *Panel A (left)* plots the diffusion eigenvalue $\lambda_k$ (vertical
axis, dimensionless, ranging here from about 0.82 to 1.00) against the eigenvalue index
$k$ (horizontal axis, integer, from 0 to 29, ordered from largest eigenvalue to smallest).
Blue circles are $E_{\text{full}}$ (image plus spectra/photometry plus redshift); orange
squares are $E_{\text{img}}$ (image only). The single point pinned at $\lambda_0 = 1$ at
the top left is the trivial constant mode (annotated), which carries no shape information
and is discarded. *Panel B (right)* plots the consecutive gap $\lambda_k - \lambda_{k+1}$
(vertical axis, dimensionless) against the gap index $k$ (horizontal axis, the gap between
$\lambda_k$ and $\lambda_{k+1}$), same colour scheme. What to look for: in Panel A the
curve descends smoothly with no cliff, and in Panel B every bar is small and comparable,
none towering over the rest (the largest is about 0.022 for $E_{\text{full}}$ and about
0.028 for $E_{\text{img}}$). The two embeddings track each other closely, which says the
image-only representation already carries essentially the same large-scale geometry as the
full multimodal one. The measured fact is the absence of a dominant gap; the
interpretation written on the figure's footnote, that this points to one connected
continuum rather than discrete clusters, is the reading we adopt, and it agrees with the
connectivity and topology checks elsewhere in the report.

### 8.6 The harmonic screen: telling new axes from repeats

There is one more trap to avoid before we hand the diffusion coordinates to the physics
reading. Not every high-ranked eigenvector is a genuinely new direction. On a stretched or
elongated shape, the slow modes of the random walk come in a sequence much like the
standing waves on a guitar string: a fundamental, then its overtones. The overtones,
called harmonics, are mathematically distinct eigenvectors with their own eigenvalues, but
they describe the same underlying geometry as the lower modes, just oscillating faster
along it. If we treated a harmonic as a separate physical axis we would be double-counting
one direction and inventing structure that is not there. This is a well-known failure mode
of diffusion maps and Laplacian eigenmaps on simple shapes.

To screen for it we ask, for each higher coordinate, whether it is predictable from the
lower ones. Concretely we regress coordinate $k$ on a degree-2 polynomial of coordinates
1, 2 and 3 (that is, on those three coordinates plus their squares and pairwise products)
and record the fraction of coordinate $k$ 's variance that this polynomial explains,
written $R^2$ (1.0 means fully predictable, 0.0 means independent). A coordinate that is
well explained by a smooth function of the lower three is a harmonic, a repeat; one that
is not is a candidate new axis. For $E_{\text{full}}$ the screen flags coordinate 3 at
$R^2 = 0.903$ and coordinate 4 at $R^2 = 0.940$ (measured), meaning roughly 90% and 94% of
their variation is just a curved restatement of coordinates 1 through 3. So we treat dc3
and dc4 as harmonics, not as independent physical directions, and we exclude them from the
physics interpretation. That leaves the two leading non-trivial coordinates, dc1 and dc2,
as the physics-bearing axes, which Section 9 (reading physics off the manifold) takes up
and ties to morphology, redshift, colour, and star formation. Coordinate 0, separately,
turns out to be a weak slow global mode that correlates with no label above
$|\rho| = 0.07$; it is a broad gradient, not a physical axis either.

### 8.7 What this section establishes

Two measured facts carry the section. The neighbour graph has a single connected component
for both embeddings, and the eigenvalue spectrum decays smoothly with no dominant gap
(largest consecutive gap about 0.022 for $E_{\text{full}}$, about 0.028 for
$E_{\text{img}}$). The plain reading is that the AION-1 galaxy cloud is one continuous
body, not a collection of separate clusters that a gap would have exposed. That matches
the physics expectation from Section 2 (galaxies as a measurable shape): morphology is a
continuum and colour is a gradient, so a faithful embedding should look like a connected
continuum rather than discrete islands. It does not, by itself, say the body is
featureless. A continuous manifold can still bend, branch weakly, and carry internal
gradients, and the next sections measure exactly those things: Section 9 reads the
physical gradients along dc1 and dc2, Section 14 (curvature) tests for weak branching, and
Section 15 (topology) confirms the single piece and finds no holes. The diffusion spectrum
is the first clean evidence for the continuum, and the rest of the report builds on it
rather than against it.

## 9. Reading physics off the manifold

Section 8 (diffusion maps) gave us a clean geometric result: the embedding is one
connected body, its diffusion spectrum decays smoothly with no dominant gap, and the slow
random-walk directions become a small set of coordinates. That tells us the *shape* is a
single continuum. It does not yet tell us what the continuum is *made of*. A smooth
low-dimensional sheet could in principle encode anything. This section asks the next
question. When you walk along the manifold, what physical property of the galaxy is
changing under your feet?

We answer it the most direct way available. We take each diffusion coordinate, one number
per galaxy, and we ask how strongly it tracks each physical label we have (colour,
redshift, morphology vote fractions, stellar mass, star-formation rate). Then we colour
the manifold by those labels and look. Both steps are descriptive. The correlation is a
measured number; the picture is a measured picture. The reading ("this axis is a
morphology axis") is an interpretation laid on top, and we keep that distinction visible
throughout.

### 9.1 What a diffusion coordinate is, briefly, and how we score it

A quick reminder of the object, since Section 8 defined it in full. A diffusion map builds
a graph where nearby galaxies are strongly linked, treats that graph as a random walk (a
diffusion), and finds the directions in which the walk mixes slowly. Each such direction
is an eigenvector of the walk's transition matrix, and it assigns one real number to every
galaxy. We call these numbers the diffusion coordinates and write them
$\mathrm{dc}_0, \mathrm{dc}_1, \mathrm{dc}_2, \dots$ in order of how slowly they mix. A
slow-mixing coordinate is, by construction, a smooth large-scale gradient across the whole
cloud: points that sit close on the manifold get close values, points on opposite ends get
very different values. So if a diffusion coordinate lines up with a physical property, it
means that property varies *smoothly and globally* across the embedding, not just in some
local pocket.

To score the alignment we use the Spearman rank correlation, written $\rho$. Spearman is
the ordinary correlation coefficient computed on ranks rather than raw values: replace
every number by its position in sorted order, then correlate. We use ranks instead of raw
values for one reason. It measures whether two quantities rise and fall together in the
same order, without assuming the relationship is a straight line. A diffusion coordinate
can bend, saturate, or stretch relative to a physical label and Spearman still reports the
monotone agreement faithfully. The value runs from $-1$ (perfect opposite ordering)
through $0$ (no monotone relation) to $+1$ (perfect same ordering). We report $\rho$ with
its sign, because the sign carries meaning here: it tells us which end of the axis is the
red end or the high-redshift end. The magnitude $|\rho|$ tells us how cleanly the axis
encodes that one property.

One caution on the sign before we read numbers. The sign of a diffusion coordinate is
arbitrary: an eigenvector and its negative are equally valid, so a software change could
flip every correlation in a coordinate at once. The *relative* signs within a coordinate
are what matter (does smooth go up while featured goes down on the same axis), and those
are stable. We state the signs as measured for this run and lean on the relative pattern,
not the absolute direction.

### 9.2 Which coordinates carry physics, and which do not

Here is the measured result, the Spearman correlations between each diffusion coordinate
and the labels (all values measured, from the `diffcoords` arrays scored against the
labels; the headline figures are committed in the facts sheet).

| Coordinate | What it tracks (strongest labels, signed $\rho$) | Reading |
|---|---|---|
| $\mathrm{dc}_0$ | $\lvert\rho\rvert < 0.07$ with **every** label | weak global mode, no clear physics |
| $\mathrm{dc}_1$ | smooth $+0.50$, featured $-0.47$, redshift $+0.45$, mass $-0.27$ | morphology / redshift axis [interp] |
| $\mathrm{dc}_2$ | g$-$r $+0.57$, r$-$z $+0.47$, sSFR $-0.43$, mass $+0.33$ | colour / star-formation axis [interp] |
| $\mathrm{dc}_3,\ \mathrm{dc}_4$ | mostly explained by lower coords (harmonic $R^2 = 0.903,\ 0.940$) | harmonics, not new axes |

Read the table row by row, because each row is a different kind of statement.

The first row, $\mathrm{dc}_0$, is the slowest-mixing non-trivial direction, and it
correlates with nothing we can name (every $|\rho|$ below $0.07$, which is barely above
zero). About 94% of galaxies have a non-tiny value on it, so it is a broad slow gradient
spread across almost the whole sample, not a spike picking out a handful of outliers. We
read it as a diffuse global mode of the graph, a direction the random walk mixes along for
reasons that do not map onto any single physical label we hold. That is an honest null for
this coordinate: the geometry has a slowest axis, and our labels do not explain it. It
could encode something real we did not measure, or it could be a low-information mixing
direction of the graph. Our data cannot decide between those.

The next two rows are the heart of this section. $\mathrm{dc}_1$ and $\mathrm{dc}_2$ are
the two coordinates that *do* line up with physics, and they line up with two different
and roughly independent families of properties.

$\mathrm{dc}_1$ tracks **morphology and redshift together**. Its strongest correlations
are with the Galaxy Zoo smooth fraction ($+0.50$) and featured fraction ($-0.47$). Smooth
and featured are vote fractions: the share of citizen-science (and CNN-predicted) votes
saying a galaxy looks smooth and round (no internal structure, the look of an elliptical
or a featureless early-type) versus showing features like spiral arms or a disk. They are
near mirror images by construction, so a coordinate that runs up in smooth and down in
featured is a single morphology gradient: smooth round systems at one end, structured
disky systems at the other. The same coordinate also tracks redshift at $+0.45$. Redshift
is how far the galaxy's light has been stretched by cosmic expansion, a stand-in for
distance and look-back time, and here it is *mostly photo-z*, a redshift estimated from
colours rather than measured from spectral lines (true spectroscopic redshifts exist for
only 6,883 of the 48,398 galaxies). That a morphology axis and a redshift axis coincide is
not surprising: at fixed survey depth, more distant galaxies are fainter and smaller on
the sky, so their finer features wash out and they look smoother. So part of the
smooth-redshift tie is an observational selection effect, not a pure statement about
galaxy structure. We flag that as a caveat the morphology reading owns. There is also a
weaker stellar-mass component on this axis ($-0.27$).

$\mathrm{dc}_2$ tracks **colour and star formation together**. Its strongest correlations
are with the two colours, g $-$ r at $+0.57$ and r $-$ z at $+0.47$. A colour like g $-$ r
is the difference of brightness in two photometric bands (here the green-ish $g$ band
minus the red-ish $r$ band); a larger g $-$ r means relatively redder light. Redder
galaxies are, broadly, older and have stopped forming new stars; bluer galaxies are
actively forming young hot stars. The same coordinate anti-correlates with sSFR ($-0.43$),
the specific star-formation rate (new stellar mass formed per year divided by existing
stellar mass, the star-formation rate per unit mass), and it correlates positively with
stellar mass ($+0.33$). Put those together and $\mathrm{dc}_2$ is a colour /
star-formation gradient: blue, star-forming, lower-mass systems at one end, red, quenched,
higher-mass systems at the other. That matches the colour-mass bimodality from Section 2
(background): the blue cloud of star-forming galaxies and the red sequence of passive
ones, with the green valley sparse in between.

The key point about these two axes is that they separate two *different* physical stories.
$\mathrm{dc}_1$ is mostly about *shape and distance*; $\mathrm{dc}_2$ is mostly about
*colour and current star formation*. The labels that load on one are weak on the other
(colour is weak on $\mathrm{dc}_1$; smooth/featured are weak on $\mathrm{dc}_2$). So the
embedding has spread two distinct families of galaxy properties onto two distinct
geometric directions. We read that as the model having found, on its own and without
morphology labels, that "how a galaxy is shaped" and "what colour it is" are two separable
things to know about it. That is an interpretation; the support for it is the two clean
and largely non-overlapping correlation patterns above, and Section 11 (disentanglement)
makes the separation claim quantitative and stronger.

The last row is housekeeping, but honest housekeeping. $\mathrm{dc}_3$ and $\mathrm{dc}_4$
look like they might be new axes, but they are not. We ran a harmonic screen: regress each
higher coordinate on degree-2 polynomials (the values, their squares, and their products)
of the first three coordinates, and read the $R^2$, the fraction of the higher coordinate
explained that way. For $\mathrm{dc}_3$ the screen explains 90.3% and for $\mathrm{dc}_4$
it explains 94.0% (measured, `harmonic_r2` in `diffusionMap.json`). A coordinate that is
almost fully predictable from lower ones is a *harmonic*: a repeat of the same underlying
geometry at a higher frequency, the way a vibrating string's overtones are not new notes
but multiples of the fundamental. They carry little new physical information beyond
$\mathrm{dc}_1$ and $\mathrm{dc}_2$, so we set them aside. This is why the physics-bearing
axes are exactly two, and why the figures below plot $\mathrm{dc}_1$ against
$\mathrm{dc}_2$ and nothing else.

### 9.3 The manifold coloured by physics (full sample)

Now look at the picture. We take the same two coordinates, $\mathrm{dc}_1$ on the
horizontal axis and $\mathrm{dc}_2$ on the vertical, plot all 48,398 galaxies as one point
each, and colour them six times over by six different labels.

![Figure 5. The AION-1 diffusion embedding coloured by six full-sample physical labels.](figures/05_diffusion_full_labels.png)

Figure 5. Each of the six panels is the *same* scatter of all 48,398 galaxies in the
diffusion plane. The horizontal axis is diffusion coordinate 1 ($\mathrm{dc}_1$) and the
vertical axis is diffusion coordinate 2 ($\mathrm{dc}_2$); both are dimensionless
eigenvector values, and the printed axis ranges (roughly $-0.025$ to $0.005$ horizontally,
$-0.005$ to $0.015$ vertically) are the 0.5th-to-99.5th-percentile limits of each
coordinate, which keep about 99% of the points and crop only the extreme tails (those
extremes include stars and image artefacts, so cropping them is deliberate). All six
panels share these limits so the shapes are directly comparable. Within each panel a
point's colour is one physical property, and the colour bar at the right of that panel
gives the property and its numeric scale; colour scales are clipped to the 2nd-to-98th
percentiles so a few outliers do not flatten the gradient. The six properties are: **g $-$
r** (colour bar about $0.4$ to $1.7$, blue points are bluer galaxies, red points are
redder), **r $-$ z** (about $0.4$ to $0.9$, same sense), **redshift** (about $0$ to
$0.35$, mostly photo-z, dark to bright), **smooth** vote fraction (about $0.3$ to $0.9$),
**featured** vote fraction (about $0.1$ to $0.8$), and **merger** vote fraction (about $0$
to $0.5$). What to look for: first the overall shape, then, in each panel, whether the
colour changes smoothly along one of the two arms.

Start with the shape, because it is the same in every panel and it is the first measured
fact here. The cloud is not a blob and it is not a set of islands. It is a bent, roughly
"L"-shaped or cross-shaped figure with two arms. One arm runs almost horizontally,
extending toward negative $\mathrm{dc}_1$ (leftward) while $\mathrm{dc}_2$ stays near
zero. The other arm runs almost vertically, extending upward in $\mathrm{dc}_2$ while
$\mathrm{dc}_1$ stays near zero. The two arms meet near the origin in a dense corner where
most galaxies sit. There is no gap between the arms and no separate detached island; it is
one continuous connected figure, which is exactly what the single connected component and
the smooth no-gap spectrum from Section 8 predicted. The arms are how a continuum can
still have structure: most galaxies pile up in the common corner, and the two physical
gradients fan out along the two arms.

Now read the panels.

The two colour panels, **g $-$ r** and **r $-$ z**, both put their gradient along the
*vertical* arm. Low in $\mathrm{dc}_2$ the points are blue (low colour, star-forming), and
as you climb the vertical arm the points redden smoothly to high colour. The horizontal
arm, by contrast, stays roughly one colour. That is the picture-level version of the
measured $\mathrm{dc}_2$ correlations (g $-$ r $+0.57$, r $-$ z $+0.47$): colour is the
vertical-arm property. The two colour panels look almost identical, which is reassuring,
since g $-$ r and r $-$ z are two measurements of the same underlying red-versus-blue
character.

The **redshift** panel puts its gradient along the *horizontal* arm instead. Near the
origin and along the vertical arm redshift is low (dark), and as you move out along the
horizontal arm toward negative $\mathrm{dc}_1$ the points brighten to higher redshift.
This is the picture of the $\mathrm{dc}_1$ redshift correlation ($+0.45$, with the sign
convention of this run). So the two arms genuinely separate two properties: the vertical
arm is the colour axis, the horizontal arm is the redshift (and, as we will see,
morphology) axis.

The **smooth** and **featured** panels are near mirror images of each other, as they must
be, and their gradient also runs along the horizontal/morphology direction (and into the
corner), consistent with $\mathrm{dc}_1$ (smooth $+0.50$, featured $-0.47$). Where one
panel is bright the other is dark. The smooth-dominated region and the featured-dominated
region sit at different ends of the morphology gradient rather than forming two separate
clumps, which is the visual signature of a morphological *sequence* rather than two
discrete classes. That continuum reading matches the known physics from Section 2:
ellipticals and disks grade into one another rather than forming cleanly separated
species.

The **merger** panel is the quiet one. Merger vote fractions are low almost everywhere
(the colour bar tops out at $0.5$ and most points sit near the bottom of it), and there is
no strong clean gradient along either arm. That is honest and expected: ongoing mergers
are rare, the merger vote fraction is a noisy CNN-predicted label, and the embedding does
not lay it out as a clean global axis. We do not claim a merger direction here.

One thing to *not* over-read from this figure. The arms look thin and sharp, which can
tempt a reader into seeing two crisp one-dimensional tracks. They are not one-dimensional.
Section 6-7 measured the intrinsic dimension at about 10 to 12, far more than two. The
diffusion plane is a two-dimensional *shadow* of a roughly ten-dimensional body, chosen to
show the two slowest gradients; the thinness of the arms in this shadow does not mean the
manifold is thin in its own space. Treat Figure 5 as a faithful 2-D map of where the two
main physical gradients point, not as the full geometry.

### 9.4 Non-input physics: mass, sSFR, and Sersic index on the manifold

The labels in Figure 5 are available for the whole sample. Three more physical properties
matter a great deal for the science but exist only for a smaller cross-matched subset,
because they come from external catalogues (SED fits and structural fits) that were run on
a few thousand of these galaxies. They are stellar mass, specific star-formation rate
(sSFR), and Sersic index. Stellar mass is the total mass in stars. The Sersic index $n$ is
a single number describing how concentrated a galaxy's light is: low $n$ (around 1) is a
diffuse exponential disk, high $n$ (around 4 and above) is a centrally concentrated
spheroid, so $n$ is a quantitative structural cousin of the smooth-versus-featured
morphology. Figure 6 colours the manifold by these three.

![Figure 6. The diffusion embedding coloured by three cross-matched physical properties over the full-sample grey background.](figures/06_diffusion_sparse_physics.png)

Figure 6. The three panels again share the diffusion plane: horizontal axis
$\mathrm{dc}_1$, vertical axis $\mathrm{dc}_2$, same 0.5th-to-99.5th-percentile limits as
Figure 5, both dimensionless. In every panel the small grey points are all 48,398
galaxies, drawn only as context so you can see the full L-shape; the coloured points are
the cross-matched subset that has the property, and that subset is a few thousand
galaxies, not the whole sample. The annotations in the panel corners give the counts:
**mass** $n = 3{,}728$ (colour bar $\log_{10} M_\ast/M_\odot$, about $9.0$ to $10.5$),
**sSFR** $n = 4{,}473$ (colour bar $\log_{10}\mathrm{sSFR}$ in $\mathrm{yr}^{-1}$, about
$-12$ to $-9.5$, note the inverted sense so brighter/yellow is *more* star-forming), and
**Sersic index** $n = 3{,}730$ (colour bar Sersic $n$, about $1$ to $6$). Colour scales
are again clipped to the 2nd-to-98th percentiles. What to look for: whether the coloured
points show the same gradients as Figure 5, even though there are far fewer of them and
they sit on top of the grey full-sample shape.

Read it the same way, but with the smaller-sample caveat front of mind. The **mass** panel
shows higher stellar mass (yellow) concentrated up the vertical arm and in the upper
corner, lower mass (purple) toward the lower and outer parts. That is the same direction
as the colour gradient, which fits the $\mathrm{dc}_2$ mass correlation ($+0.33$) and the
everyday fact that redder, quenched galaxies tend to be more massive. The **sSFR** panel
is the cleanest of the three: with its inverted colour bar, the most actively star-forming
galaxies (yellow, high sSFR) sit toward the lower-colour part of the vertical arm and the
most quenched (dark, low sSFR) sit up the red end, the mirror of the mass and colour
gradients, consistent with the measured $\mathrm{dc}_2$ sSFR correlation ($-0.43$). The
**Sersic** panel shows higher $n$ (more concentrated, spheroid-like, yellow) trending
differently from the diffuse low- $n$ disks (dark), tracing the structural axis, the
quantitative echo of the smooth/featured morphology gradient on $\mathrm{dc}_1$.

Two honest qualifications belong to Figure 6. First, the count. With only three to
four-and-a-half thousand coloured points spread over a figure that has 48,398 grey ones
behind them, the gradients are visible but sparser and noisier than in Figure 5; we are
reading a thinned sample, and the cross-matched subset is not a random draw from the full
population (it is whichever galaxies happened to land in the external catalogues), so its
coverage of the manifold is uneven. Do not read the *density* of coloured points as
physical; read only the *colour trend* where points exist. Second, and this is the
scientifically important one: mass, sSFR, and Sersic index were *never given to the model
as inputs*. Unlike redshift and flux (which are inputs to the multimodal embedding and so
are partly read back rather than inferred), these three physical quantities come entirely
from outside catalogues. Seeing them lay out as smooth gradients on a geometry the model
built without them is the first sign that the embedding has organised galaxies by real
physics it was not handed. Section 10 (decodability) turns that sign into a measured
number, with the leakage-free image-only probes putting mass at $R^2 = 0.72$ and sSFR at
$0.76$.

### 9.5 What this section does and does not establish

Let us be exact about the claim, because it is easy to overstate.

What is measured: the Spearman correlations in the table ($\mathrm{dc}_1$ with smooth
$+0.50$, featured $-0.47$, redshift $+0.45$; $\mathrm{dc}_2$ with g $-$ r $+0.57$, r $-$ z
$+0.47$, sSFR $-0.43$, mass $+0.33$; $\mathrm{dc}_0$ below $0.07$ with everything;
$\mathrm{dc}_3$ / $\mathrm{dc}_4$ harmonic $R^2 = 0.903$ / $0.940$), the connected
L-shaped figure, and the smooth colour gradients visible in Figures 5 and 6. Those are
facts about this run.

What is interpreted: that $\mathrm{dc}_1$ "is" a morphology/redshift axis and
$\mathrm{dc}_2$ "is" a colour/star-formation axis. That is shorthand, and we should hold
it loosely for three reasons. First, none of these correlations is close to $\pm 1$; the
strongest, g $-$ r at $+0.57$, leaves most of the rank variance unexplained, so a
diffusion coordinate is a *blend* that leans toward a property, not a pure copy of it.
Second, the axes mix properties: $\mathrm{dc}_1$ carries both morphology and redshift, and
part of that tie is the observational selection effect (distant galaxies look smoother),
not a clean physical identity. Third, the sign of each coordinate is arbitrary, so only
the *relative* pattern within a coordinate is meaningful. The defensible statement is
narrow and strong: the embedding has at least two large-scale directions along which
nameable galaxy physics varies smoothly and monotonically, those two directions separate
shape-and-distance from colour-and-star-formation, and the separation extends even to
physical properties (mass, sSFR, Sersic) the model never saw. Whether the model
"represents morphology" as a clean internal variable is a stronger claim that this
descriptive reading cannot settle on its own. The probes of Section 10 and the
disentanglement test of Section 11 are what we lean on to push past correlation toward
decodability and genuine separation.

## 10. Decodability: how much physics the embedding carries

Section 8 showed the embedding is one smooth body, and Section 9 showed that two of its
diffusion coordinates line up with morphology, colour, and star formation. Those were
correlations: we coloured a 2D picture by a property and saw a gradient.

A correlation along one or two coordinates does not tell you how much of a property the
full 1024-number vector actually pins down. The diffusion picture used only the slowest
two directions; the embedding has a thousand more. This section uses all of them at once.

So it answers a sharper question. Given the whole embedding, how precisely can you read
off a physical property of the galaxy? And, the part that matters most, can you read off
properties the model was never given as input?

The tool for this is a probe. The idea is simple and deliberately weak. We do not let a
clever nonlinear model dig the answer out. We fit the simplest thing that could work, a
single linear map from the 1024 numbers to the property, and we measure how well that
linear map predicts the property on galaxies it never saw during fitting. If even a
straight-line readout recovers a property well, then that property is laid out in the
embedding in an easy, linear, accessible way. That is a strong statement about the
representation, not about our cleverness.

The weakness is a feature, not a limitation we tolerate. Suppose we let a deep network
probe the embedding instead. A deep network is so flexible that it could pull a property
out of almost any representation, even one that buried the property in a tangled, useless
form. Then a high score would tell us about the probe's power, not the embedding's
organisation.

By restricting the probe to a single linear map, we tie the score directly to a property
of the representation itself: is this quantity available along a direction, yes or no, and
how cleanly. A linear probe answers a sharp question about geometry. That is what we want
here.

One more framing point before the math. There is a difference between "the information is
present" and "the information is accessible". A pile of pixels contains a galaxy's mass
too, in principle, but you would need a whole physical model to extract it. When we say
the embedding "carries" mass, we mean something stronger than mere presence: the embedding
has arranged itself so that mass is readable by a flat, linear readout. That is the kind
of structure a downstream user actually benefits from, and it is the kind of structure a
faithful representation should have if the model truly organised galaxies by physics.

And one warning we will repeat. A high probe score is evidence of accessible structure,
not of understanding. Decodability tells you the property lives along a direction in the
embedding. It does not tell you the model "knows" the physics, nor that the direction is
causal, nor that the model would use that direction the way we read it. Those are
separate, harder questions. This section measures accessibility and nothing more, and we
keep that boundary visible throughout.

### 10.1 What a linear probe is, in plain words

Write the embedding of a galaxy as a vector $x \in \mathbb{R}^{1024}$, meaning a list of
1024 real numbers. Write the property we want to predict (redshift, stellar mass, a vote
fraction) as a single number $y$. A linear probe is a weight vector
$w \in \mathbb{R}^{1024}$ and an offset $b$ (a single number) such that the prediction is

$$\hat{y} = w \cdot x + b = \sum_{j=1}^{1024} w_j x_j + b.$$

In words: multiply each of the 1024 embedding numbers by its own weight, add them all up,
add a constant, and that sum is your guess for the property. The weights $w_j$ say how
much each embedding dimension should count, and their signs say in which direction.
Nothing here bends or curves. It is a flat, straight-line readout of the property from the
embedding. That weakness is the point. A linear probe can only succeed if the property is
already arranged in the embedding along some direction. It cannot manufacture structure
that is not there.

There is a clean geometric picture for $w$. The weight vector points in a direction inside
the 1024-dimensional embedding space, and the prediction $w \cdot x$ is the length of the
galaxy's embedding projected onto that direction (up to the offset $b$). So fitting a
probe is finding the direction along which the property increases most smoothly.

If such a direction exists and the property changes steadily as you move along it, the
probe scores high. If the property is scattered with no preferred direction, the
projection carries little information and the probe scores near zero. We will use exactly
this "probe direction" idea again in Section 11, where we measure the angle between the
direction for one property and the direction for another to test whether the model keeps
concepts apart.

We fit $w$ and $b$ by least squares with a penalty, which is the next idea.

### 10.2 Ridge regression and the cross-validated penalty

Plain least squares picks $w$ to make the prediction errors as small as possible on the
training galaxies. With 1024 weights and a target that has real noise, plain least squares
will happily use tiny accidental patterns in the training set that do not generalise.

That is overfitting: the probe looks great on the galaxies it was fit on and worse on new
ones. We want a measure of what the embedding genuinely carries, not of how well 1024 free
numbers can memorise a few thousand training points. Overfitting would inflate every score
and make a weak embedding look strong.

Ridge regression fixes this by adding a penalty on the size of the weights. It minimises

$$\sum_{i \in \text{train}} \left( y_i - (w \cdot x_i + b) \right)^2 \; + \; \alpha \, \lVert w \rVert^2,$$

where the first term is the usual sum of squared errors over training galaxies and the
second term, $\alpha \lVert w \rVert^2 = \alpha \sum_j w_j^2$, punishes large weights. The
number $\alpha \ge 0$ sets how hard we punish them.

A large $\alpha$ forces the weights toward zero, giving a simpler, smoother probe that
ignores small accidental patterns. A small $\alpha$ lets the probe fit harder. Ridge keeps
the probe honest by preferring small, spread-out weights unless the data clearly earns
larger ones.

There is also a practical reason ridge is the right base method here, beyond overfitting.
The 1024 embedding dimensions are not independent; many of them carry overlapping
information (Section 2 noted that just a few principal components capture most of the raw
variance).

When inputs are correlated, plain least squares becomes unstable: it can assign wildly
large positive and negative weights to correlated dimensions that nearly cancel, and tiny
changes in the data swing those weights around. The $\alpha \lVert w \rVert^2$ penalty
tames this by spreading the weight smoothly across the correlated group instead of letting
it blow up. So ridge is both a guard against overfitting and a guard against the
instability that correlated embedding dimensions would otherwise cause.

We do not pick $\alpha$ by hand. We use RidgeCV, which means ridge regression with the
penalty chosen by cross-validation.

Cross-validation works like this. Split the training galaxies into folds, hold one fold
out, fit on the rest, predict the held-out fold, and measure the error. Do this for each
fold and for a grid of candidate $\alpha$ values, here spanning $10^{-2}$ up to $10^{4}$
(a wide grid, six orders of magnitude, so the search is not artificially boxed in). The
$\alpha$ that gives the lowest average held-out error wins. So the penalty strength is
itself learned, from data the final scoring never touches.

The chosen $\alpha$ varies by target, and we report it because it is informative. Colours
land on a small penalty ($\alpha = 0.01$), which says the colour signal is clean and
strong enough that the probe can fit tightly without overfitting. Redshift sits a little
higher ($\alpha = 0.1$). The small-sample physical targets and the morphology
sub-fractions land on much larger penalties ($\alpha$ up to 100), which says the probe
needs heavy smoothing to generalise from a few thousand noisy points.

A high $\alpha$ is the cross-validation telling us, in its own units, "do not trust this
target's fine detail, smooth it". That is a useful side signal: the targets that demand
the heaviest smoothing are exactly the ones with the smallest samples and the noisiest
labels, which is what we would hope to see from an honest procedure. We did not hand-tune
any of this; the pattern fell out of the cross-validation.

### 10.3 R-squared, the 80/20 split, and the bootstrap interval

We score a probe by $R^2$ (R-squared), the fraction of the property's variance that the
probe explains on held-out galaxies. Define it on the test set:

$$R^2 = 1 - \frac{\sum_{i \in \text{test}} (y_i - \hat{y}_i)^2}{\sum_{i \in \text{test}} (y_i - \bar{y})^2},$$

where $y_i$ is the true property of test galaxy $i$, $\hat{y}_i$ is the probe's prediction
for it, and $\bar{y}$ is the mean property over the test set. The denominator is the error
you would make by always guessing the mean and ignoring the embedding entirely. The
numerator is the error the probe actually makes.

So $R^2 = 1$ means perfect prediction, $R^2 = 0$ means the probe is no better than
guessing the mean, and a negative $R^2$ means it is worse than the mean. Read $R^2$ as the
share of the property that the embedding nails down through a straight-line readout.

A small worked feel for the scale helps. If a property has a true spread (variance) of
some amount, an $R^2$ of $0.75$ means the probe's leftover errors have one quarter of that
variance, so the typical error is about half the original spread ($\sqrt{0.25} = 0.5$). An
$R^2$ of $0.96$ leaves four percent of the variance, so the typical error is about a fifth
of the original spread.

This is why the jump from $0.72$ (mass) to $0.96$ (colour) is larger than it looks. It is
the difference between recovering a property to half its spread versus to a fifth of its
spread, a factor of more than two in typical error.

Two rules keep this number honest. First, the 80/20 split. We fit the probe on a random
80% of the galaxies (with a fixed random seed, seed 0, so the split is reproducible) and
we compute $R^2$ only on the held-out 20% the probe never saw.

The penalty $\alpha$ was also chosen using only the training 80%, by cross-validation
inside it, so the test 20% is untouched by any fitting decision. Every $R^2$ in this
section is a held-out, out-of-sample number, not a training-set number. That is the
difference between measuring what the embedding carries and measuring how well 1024
weights can memorise.

Second, the uncertainty. A single test set gives one $R^2$, but that one number depends on
which galaxies happened to land in the 20%. To put error bars on it we use a bootstrap.

We resample the test set with replacement (draw $n_{\text{test}}$ galaxies at random,
allowing repeats, to make a new pretend test set of the same size), recompute $R^2$ on
that resample, and repeat 1000 times. The middle 95% of those 1000 values is the 95%
confidence interval (CI). It tells you the range in which the true held-out $R^2$
plausibly sits given the finite test set.

Wherever we quote an $R^2$ below, the bracket after it is this 1000-times bootstrap 95% CI
(measured). A wide bracket means a small or noisy test set; a tight bracket means the
number is well pinned.

Why the bootstrap rather than a textbook formula? Because $R^2$ is a ratio of sums of
squares, its sampling distribution is not a simple bell curve, especially when the test
set is only a few hundred galaxies. The bootstrap sidesteps the need for any formula: it
simulates "what if we had drawn a slightly different test set" by reusing the data we
have, and reads the spread of $R^2$ off those simulations directly.

The price is that the bootstrap can only reflect uncertainty from the finite test sample.
It does not capture uncertainty from the labels themselves being noisy (the vote fractions
carry 5 to 10 percent error, Section 4) or from the single fixed train/test split. So the
CIs are honest about sampling error and silent about those other sources. We flag that
rather than pretend the bracket is the whole story.

### 10.4 The leakage-free design: why we probe E_img

Here is the single most important design choice in this section, and the reason the
results can be trusted as real inference rather than bookkeeping.

The model has two embedding sets, defined in Section 3. $E_{\text{full}}$ is built from
image plus photometry (the g, r, z fluxes) plus redshift, all fed in as input tokens.
$E_{\text{img}}$ is built from the image only; the fluxes and redshift are not inputs at
all.

Now think about what it means to probe redshift. If you probe redshift from
$E_{\text{full}}$, redshift was handed to the model as an input. A high probe score then
proves almost nothing about the model's understanding, because the model could simply copy
the input forward into the embedding. The probe would be reading back something we put in.

That is leakage: information enters as an input and then gets "decoded" as if it were a
discovery. The same trap applies to colour, since the fluxes that define g-r and r-z are
inputs to $E_{\text{full}}$. Any probe on $E_{\text{full}}$ for redshift or colour is
therefore contaminated, and we treat it as such.

So every concept claim in this section is made on $E_{\text{img}}$, the image-only
embedding. On $E_{\text{img}}$, redshift and colour are not inputs. The only thing the
model saw was the four-band picture of the galaxy. If a linear probe can still recover
redshift or colour or stellar mass from $E_{\text{img}}$, the model genuinely inferred
that property from the appearance of the galaxy. That is a real inference about physics
from pixels, the kind of thing a representation can only do if it has organised images by
their physical content. This is why $E_{\text{img}}$ is the honest set and why we lead
with it. We will use $E_{\text{full}}$ only as a contrast in the ablation, to show the
leakage gap directly.

It helps to be precise about what "leakage" does and does not spoil. Leakage does not make
$E_{\text{full}}$ a bad embedding. A model is allowed to keep its inputs available, and
for a downstream user who wants the best possible redshift readout, $E_{\text{full}}$ is
the better tool.

Leakage spoils a specific kind of claim: the claim that a high probe score is evidence the
model learned or inferred something. You cannot infer what you were told. By probing
$E_{\text{img}}$, where the property was withheld, we convert the probe score from a
bookkeeping number into a genuine test of inference. The distinction is the whole reason
the project built two embeddings instead of one.

Note also that "image only" does not fully neutralise every shortcut. Colour is computable
from a multi-band image, so even on $E_{\text{img}}$ a property that is really a colour
proxy can be recovered through colour.

That loophole is exactly the photo-z issue we treat in 10.7. It does not affect the
cleanest targets (mass, sSFR), which is why those carry the most weight in the section's
conclusion.

### 10.5 Decodability from the image alone

With the method fixed, the result is a single ranked list: how decodable is each property
from the image-only embedding. We probe eleven targets. Five are full-sample labels with
about 48,000 galaxies each (the two colours, redshift, and the smooth, featured, and
merger vote fractions). Three are the cross-matched physical properties on a few thousand
galaxies (mass, sSFR, Sersic index). Two are the branch-conditional fine-morphology
fractions (spiral arm, strong bar) on about 3,000 disks. Reading the list top to bottom is
reading a ranking of which physical facts the image-only representation makes easy to
access.

Figure 7 reports the headline: held-out $R^2$ for each property, decoded from the
image-only embedding $E_{\text{img}}$, with bootstrap 95% CI bars.

![Figure 7. Decodability of physical labels from the image-only embedding.](figures/07_probe_decodability.png)

Figure 7. Each horizontal bar is one physical property; the bar length is the held-out
$R^2$ (x-axis, dimensionless, from 0 to about 1.0) of a linear ridge probe fit on 80% of
the galaxies and scored on the held-out 20%, using the image-only embedding
$E_{\text{img}}$. The thin black whisker on each bar is the 1000-times bootstrap 95%
confidence interval. The text at the right of each bar is the sample size $n$ used for
that property.

Colour codes the sample regime. Blue bars are full-sample targets ($n \approx 48{,}000$);
orange bars are small cross-matched or branch-conditional subsets ($n < 5{,}000$), where
the labels come from external catalogues or from the Galaxy-Zoo decision tree and exist
for only a few thousand galaxies. The vertical line at $R^2 = 0$ marks the no-skill
baseline (a probe that just predicts the mean).

What to look for: the bars are sorted top to bottom from most to least decodable. Colours
sit at the top near $R^2 \approx 0.91$ to $0.96$, redshift and the broad morphology
fractions cluster near $0.79$ to $0.80$, the non-input physical properties stellar mass
and sSFR sit at $0.72$ and $0.76$, and the small branch-conditional fine-morphology
targets (strong bar, spiral arm) fall off at the bottom with wide error bars. All
measured.

The numbers behind the bars (all measured, all from $E_{\text{img}}$, all held-out,
brackets are the bootstrap 95% CI):

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

Read the tiers from the top. Colour is decoded almost perfectly: g-r at $R^2 = 0.958$
[0.957, 0.960] and r-z at $0.911$ [0.903, 0.918], both with very tight intervals.

The colour g-r is the difference in brightness between the green (g) and red (r) bands,
and r-z between the red and near-infrared (z) bands; together they say whether a galaxy's
light is bluer (young stars, recent star formation) or redder (old stars, passive). These
fluxes were not inputs to $E_{\text{img}}$, so the model is reading the colour of a galaxy
straight off its image, which is a thing you can in fact see in a multi-band picture.

That this works at $0.95$ -plus tells us colour is encoded along a clean, near-linear
direction in the embedding, which is consistent with the dc2 colour axis we found in
Section 9. The tightness of the CI (a span of $0.003$ for g-r) is what you expect when
48,000 galaxies vote on a strong, clean signal.

It also sets a rough ceiling for everything below. If the easiest visible property tops
out near $0.96$ rather than at exactly $1.0$, some of the missing few percent is label and
measurement noise that no probe could recover. So a target scoring $0.72$ is not $0.72$
out of a possible $1.0$ of clean signal; it is $0.72$ against a practical ceiling somewhat
below $1.0$. That makes the mass and sSFR numbers look stronger, not weaker.

Redshift comes next at $R^2 = 0.800$ [0.792, 0.809]. The model never saw redshift for
$E_{\text{img}}$, yet a straight readout recovers four-fifths of its variance from the
image. Hold that number; the photo-z caveat in 10.7 qualifies what it means.

There is a physical reading of the whole ranking, and it is coherent. The properties that
decode best are the ones most directly visible in a multi-band image (colour, then coarse
shape). The middle tier is properties that are inferred from the image but still strongly
tied to visible cues (mass, sSFR, concentration). The bottom tier is fine structural
detail that needs resolution and clean conditions to see (bars, arms).

The model's accessibility ranking tracks how visible each property is, which is what an
honest visual representation should produce. We read this as a consistency check passing,
not as a separate discovery.

Morphology, as the broad smooth and featured vote fractions, sits at $R^2 \approx 0.79$
(smooth $0.792$ [0.782, 0.802], featured $0.794$ [0.783, 0.804]). A vote fraction is the
share of human (or human-trained CNN) votes that called a galaxy "smooth" or "featured"
(Section 4 defines these). These are the broad "is it a featureless blob or does it have
visible structure" judgements, and the image-only embedding tracks them well. That smooth
and featured land at almost the same $R^2$ is expected, since they are close to two sides
of one coin.

The merger fraction is lower at $0.681$ [0.658, 0.702], which fits intuition: mergers are
rarer and messier, the visual signature (tidal tails, double nuclei, disturbed shapes) is
more varied, and a single linear direction captures less of it. This lower score is also
consistent with the disentanglement result in Section 11, where the merger direction turns
out to behave differently from the colour and redshift directions.

The non-input physical properties are the most interesting, and they are in orange because
they come from a few thousand cross-matched galaxies. Stellar mass decodes at
$R^2 = 0.721$ [0.659, 0.772] and specific star-formation rate (sSFR) at $0.760$ [0.692,
0.816], both from the image alone.

We define these terms in Section 2. Stellar mass is the total mass locked in stars (in
solar masses), and sSFR is the star-formation rate divided by the stellar mass, a measure
of how fast a galaxy is building new stars relative to its size. Neither was ever an input
to $E_{\text{img}}$, and neither is directly "visible" the way colour is. A galaxy's mass
and star-forming state are inferred quantities even for a human expert, recovered from a
physical model of how light maps to physical properties.

Recovering roughly three-quarters of their variance from pixels means the visual
representation encodes the galaxy's mass and current star-forming state, not just its
surface appearance.

There is a physical reason this is plausible rather than magical. Mass correlates with
size and brightness and concentration, and star formation correlates with colour, clumpy
bright regions, and disk structure, all of which are visible in the image. The model
appears to have folded those visible cues into directions that line up with the underlying
physical quantities.

This connects to the colour-mass bimodality from Section 2 (the red sequence of passive
galaxies and the blue cloud of star-forming ones). If the embedding lays out colour, mass,
and sSFR along accessible directions, it has the raw material to reproduce that bimodality
on its own, which is what the diffusion picture in Section 9 hinted at and what the SAE in
Section 13 will probe directly. We return to why mass and sSFR are the cleaner evidence in
10.7.

The Sersic index $n$ (a number describing how steeply a galaxy's light falls off from
centre to edge; high $n$ is a concentrated bulge-like profile, low $n$ is a flatter disk)
decodes at $R^2 = 0.664$ [0.608, 0.713]. The structure is partly there but less cleanly
linear. Concentration is a real visible feature, so it makes sense the embedding tracks
it, but the single-number Sersic fit is itself a lossy summary of a galaxy's light
profile, which caps how high any probe can score against it.

The bottom two bars are the branch-conditional fine-morphology targets, defined only for
featured non-edge-on disks (the Galaxy-Zoo decision tree, hence the small $n = 3{,}034$,
explained in Section 4). Strong-bar fraction decodes weakly at $R^2 = 0.554$ [0.476,
0.625], and spiral-arm fraction barely at all, $R^2 = 0.252$ [0.153, 0.334].

The wide intervals are honest. With 3,034 galaxies and a fine, conditional label, the
probe has little to work with, and the spiral CI runs from $0.15$ to $0.33$, so we cannot
even pin its value to within a factor of two.

We read this plainly. The image-only embedding carries the coarse morphology (smooth vs
featured) strongly and the fine sub-structure (bars, arms) weakly. We do not claim the
model resolves spiral arms; the data says it mostly does not, at least not in a linearly
accessible way.

Two cautions before reading too much into the weak fine-morphology scores. First, these
labels are conditional, computed only for disks that already passed the "featured and not
edge-on" branch, so the probe is asked to do a harder, narrower job on a pre-filtered
population.

Second, spiral-arm and bar fractions are exactly the labels where the underlying CNN vote
fractions are noisiest, so part of the low $R^2$ is label noise rather than missing
structure in the embedding. A low score here is a soft "not clearly accessible", not a
hard "absent". We flag this rather than read the weak scores as proof the model is blind
to spiral structure.

### 10.6 The modality ablation: seeing the leakage directly

To make the leakage argument concrete rather than asserted, we run the same probe on both
embeddings for the three properties that are inputs to $E_{\text{full}}$ but not to
$E_{\text{img}}$: redshift and the two colours. The gap between the two scores is exactly
the leakage we warned about.

This doubles as a check on the method itself. If our probe were broken or our setup leaky
in some hidden way, the $E_{\text{img}}$ and $E_{\text{full}}$ scores might come out the
same, or in the wrong order. Seeing $E_{\text{full}}$ score higher on exactly the
quantities it was fed, and by an amount that shrinks as the image already carries the
signal, is the behaviour a correct leakage-aware setup should show. So the ablation
confirms both the leakage story and that the pipeline is doing what we think.

![Figure 8. Decodability: image-only versus multimodal embedding.](figures/08_modality_ablation.png)

Figure 8. Grouped bars comparing held-out $R^2$ (y-axis, dimensionless, 0 to just above
1.0) for three properties (x-axis: redshift, g-r colour, r-z colour) decoded two ways. The
blue bar in each pair is $E_{\text{img}}$, the image-only embedding (the honest set, where
these three were never inputs). The grey bar is $E_{\text{full}}$, the multimodal
embedding, which ingested the g, r, z fluxes and the redshift as inputs. Each bar is
labelled with its $R^2$ value on top. What to look for: the grey bars are always higher
than the blue bars, and the gap is largest for redshift. That gap is the leakage.
Measured.

The numbers (measured): redshift goes from $E_{\text{img}} = 0.800$ to
$E_{\text{full}} = 0.976$; g-r from $0.958$ to $0.989$; r-z from $0.911$ to $0.968$. In
every case the multimodal embedding scores higher on exactly the quantities it was fed.

The redshift jump is the biggest, from $0.80$ to $0.98$, because redshift was handed to
$E_{\text{full}}$ as a scalar input that the probe can largely copy back. The colour jumps
are smaller ($0.958 \to 0.989$, $0.911 \to 0.968$) because the image already carries most
of the colour signal, so the extra flux inputs only top it up.

The size of each gap is itself a readout of how much new information the input added on
top of the image: large for redshift (the image alone gets it only to $0.80$), small for
colour (the image alone already nearly saturates). Reading the gaps this way, the ablation
is not just a warning; it quantifies how much each non-image input mattered.

The logic to take away: if we had reported the $E_{\text{full}}$ numbers as evidence that
"the model understands redshift to $R^2 = 0.98$ ", we would be inflating a real but partly
circular result. The model can score that high partly because it was told the answer. The
$E_{\text{img}}$ numbers, where the answer was withheld, are the honest measure of what
the model inferred. This is why the whole concept analysis (this section and the
disentanglement and SAE sections that follow) is anchored on $E_{\text{img}}$ unless a
property could not leak.

One nuance worth stating. For the small-sample physical properties (mass, sSFR),
$E_{\text{full}}$ also scores higher: mass $0.870$ versus $0.721$ on $E_{\text{img}}$, and
sSFR $0.848$ versus $0.760$. But here the gap is not pure leakage, because mass and sSFR
were not direct inputs to $E_{\text{full}}$ either.

$E_{\text{full}}$ does better because it additionally saw the fluxes and redshift, which
are physically informative about mass and sSFR. Brighter and redder galaxies tend to be
more massive, and redshift sets the distance and so the conversion from observed
brightness to physical luminosity.

So the $E_{\text{full}}$ advantage there is partly legitimate extra information, not
copying. We still report the $E_{\text{img}}$ numbers as the headline because they isolate
what the image alone supports.

The Sersic index is the exception that proves the rule. $E_{\text{full}} = 0.668$ and
$E_{\text{img}} = 0.664$ are essentially identical, which says structure (light
concentration) is read from the image and adding fluxes and redshift buys nothing for it.
That makes sense: concentration is a purely geometric, distance-independent feature of the
picture, so the extra inputs have nothing to add.

### 10.7 Predicted versus true, and the photo-z caveat

$R^2$ is one summary number. To see what the probe is actually doing, Figure 10 plots
predicted against true for the three properties where the stakes are highest: redshift
(the famous one), and the two non-input physical properties, mass and sSFR.

![Figure 10. Image-only probe: predicted versus true on the held-out test set.](figures/10_probe_pred_vs_true.png)

Figure 10. Three panels, one per property, all from the image-only embedding
$E_{\text{img}}$ and all on the held-out 20% test set. Left: redshift (mostly photo-z).
Middle: stellar mass $\log_{10}(M_*/M_\odot)$ (mass in solar masses, log scale). Right:
$\log_{10}$ sSFR in units of $\text{yr}^{-1}$ (inverse years, the natural unit for a rate
per unit mass).

In each panel the x-axis is the true catalogue value and the y-axis is the probe's
prediction, in the same units. The points are galaxies, drawn as a density: the colour
scale (a perceptual colour map, dark to bright) counts galaxies per bin, so bright regions
are where many galaxies pile up and dark scattered points are sparse. The red dashed
diagonal is the line $y = x$, perfect prediction; points on it are exactly right. Each
panel is annotated with its held-out $R^2$ and its test-set size $n_{\text{test}}$.

What to look for: in all three panels the bright ridge of galaxies lies along the $y = x$
diagonal, which means the probe tracks the true value across the whole range, not just on
average. The redshift panel ($R^2 = 0.800$, $n_{\text{test}} = 9{,}680$) is the tightest,
with a dense narrow ridge. Mass ($R^2 = 0.721$, $n_{\text{test}} = 746$) and sSFR
($R^2 = 0.761$, $n_{\text{test}} = 895$) are looser, with more scatter around the
diagonal, consistent with their lower $R^2$ and smaller test sets.

A useful detail to check in the plot is that the ridge stays on the diagonal at both ends,
not just in the crowded middle. A probe that only got the average right would bend away
from $y = x$ at the extremes (predicting too high for low-mass galaxies and too low for
high-mass ones, the regression-to-the-mean tilt). The mass and sSFR ridges stay reasonably
straight across their range, which says the probe captures the property itself and not
merely its average. Measured.

The slight discrepancy in the panel labels is just rounding from the same underlying
numbers: the sSFR panel prints $R^2 = 0.761$ while the table value is $0.760$ (the table
rounds $0.7603$, the figure rounds the bootstrap-resampled point estimate). They are the
same measurement.

Now the caveat this section owns, and it is the load-bearing one. The redshift labels are
mostly photometric redshifts, "photo-z" for short.

A spectroscopic redshift comes from measuring how far a galaxy's spectral lines are
shifted, which is a direct distance-and-recession measurement. A photometric redshift is
an estimate of redshift derived from the galaxy's broad-band colours, because galaxies of
different redshift have systematically different colours.

In our sample only 6,883 galaxies have a true spectroscopic redshift; the rest carry
photo-z (Section 4). So the redshift label we probe against is, for most galaxies, itself
a colour-based estimate rather than a direct measurement.

The catch is that photo-z is itself a function of colour. So when the image-only probe
"predicts redshift" at $R^2 = 0.80$, part of what it is doing is this: read the colour off
the image (which it does at $R^2 \approx 0.96$), then map colour to photo-z (which is
roughly what the photo-z pipeline did in the first place).

In short, image $\to$ redshift here is partly image $\to$ colour $\to$ photo-z. The $0.80$
is real and the model genuinely recovers a redshift-correlated quantity from pixels, but
we must not over-read it as the model independently understanding cosmological distance.
The chain may run through colour.

We cannot fully separate the two routes with the data in hand. Splitting the
spectroscopic-only subset would let us probe true redshift directly, but with only 6,883
spectroscopic galaxies the test would be far noisier and we would be comparing different
samples.

So we state the limit honestly. The redshift number is a decodability result with a known
shortcut, not clean proof of independent redshift inference. Section 20 carries this as a
named limitation, and the synthesis in Section 19 leans on mass and sSFR, not redshift,
when it argues that the embedding encodes physics.

This is exactly why mass and sSFR are the cleaner evidence, and we say so plainly.

Stellar mass and specific star-formation rate come from external SED-fitting catalogues
(fits to the galaxy's spectral energy distribution across many bands), not from a
colour-based shortcut, and they were never inputs to $E_{\text{img}}$. They are not
trivially "visible" in the image the way colour is. There is no analogue of the photo-z
chain for them: the catalogue values do not pass through a single colour bottleneck, so a
probe cannot recover them by the colour shortcut alone.

Recovering mass at $R^2 = 0.721$ and sSFR at $0.760$ from the image alone is therefore the
strongest single piece of evidence in this section that the visual representation encodes
galaxy physics it was not given, with no obvious leakage path and no colour-shortcut to
explain it away.

The intervals are wider than for the full-sample colours ([0.659, 0.772] for mass and
[0.692, 0.816] for sSFR) because the cross-matched samples are small (a few thousand
galaxies, with held-out test sizes of 746 and 895). We report that width honestly rather
than rounding it away.

But both intervals sit well above zero, so the signal is not a fluctuation. Even at the
low end of its CI, mass is recovered at $R^2 \approx 0.66$ and sSFR at $\approx 0.69$,
which is still most of the variance. The result does not depend on reading the point
estimate optimistically; the whole interval tells the same story.

### 10.8 Connecting back to the original claim

The authors of AION-1 said the model "organises objects along physically meaningful
directions" (Section 1). This section is the most direct test of that claim that the
report contains, and it largely supports it, with one sharpened qualification.

The support: physically meaningful properties really are accessible along directions in
the embedding, strongly enough that a flat linear readout recovers most of their variance,
and this holds even for properties (mass, sSFR) that the image-only model was never given.

A representation that scattered physics randomly across its 1024 dimensions could not
produce these scores. So "physically meaningful directions" is a fair description of what
we measure, at the level of linear accessibility.

The qualification: "directions" in the plural is doing real work, and decodability alone
cannot say the directions are clean or separate. That is the next question.

### 10.9 What this section establishes and what it does not

What it establishes, measured. From the image alone, a linear probe recovers colour almost
perfectly ($0.91$ to $0.96$), redshift well ($0.80$), broad morphology well ($0.68$ to
$0.79$), and, the key result, two non-input physical properties, stellar mass ($0.72$) and
sSFR ($0.76$), at roughly three-quarters of their variance. Decodability falls off only
for fine conditional morphology (bars at $0.55$, arms at $0.25$), where the labels are
sparse and the intervals are wide.

The modality ablation shows directly why we trust the image-only numbers. The multimodal
embedding scores higher on exactly the quantities it was fed (redshift $0.98$, colours
$0.97$ to $0.99$), which is leakage, so $E_{\text{img}}$ is the fair measure.

What it does not establish. A high $R^2$ says a property is linearly accessible in the
embedding; it does not say the model has an isolated, clean concept for it. A property
could be decodable while being smeared across many entangled directions. Whether the model
keeps these concepts separate is a different question, the disentanglement question, and
Section 11 takes it up by asking whether the probe directions for different properties
point in genuinely different directions or just inherit the correlations among the labels
themselves.

And the photo-z caveat above bounds the redshift claim. Read mass and sSFR, not redshift,
as the evidence that pixels encode physics the model was never told. That single sentence
is the honest summary of this section's strongest result.

## 11. Disentanglement and neighbourhood purity

Section 10 (probes) asked how much of each physical property the embedding carries. A high
$R^2$ tells us the information is there and readable by a straight-line rule. It does not
tell us whether the model keeps two different properties on separate directions, or
whether it has piled them onto the same axis and we are simply reading one through the
other. This section runs two checks that are light on modelling and aimed at that second
question. The first asks whether the model's internal directions for different concepts
are more separated than the labels themselves would force. The second, even simpler, asks
whether galaxies of one morphological type sit next to galaxies of the same type in the
raw 1024-dimensional space, with no probe at all.

Why does separation matter, beyond decodability? Because a representation that smears two
properties onto one shared axis has, in a real sense, only learned one of them. If the
model encoded "redness" and "smoothness" on the same direction, then every time it moved a
galaxy redward it would also move it smoothward, and we could never tell whether it had a
concept of morphology at all or just a concept of colour that we were reading twice.

A representation that keeps the two on separate axes can vary one while holding the other
fixed. That is the property people mean by "disentangled": independent factors live on
independent directions, so the model can describe a red disk and a blue spheroid as easily
as the common red spheroid and blue disk. The rare combinations are the test. A
representation that has truly separated colour from shape can place a blue elliptical and
a red spiral wherever those galaxies actually fall, instead of forcing every red galaxy to
look smooth.

AION-1's authors claimed the model "organises objects along physically meaningful
directions." The disentanglement test is a direct, if linear, check on the word
"directions": are they actually distinct directions, or one direction wearing several
names?

### 11.1 What a probe direction is, and what "disentangled" should mean

Recall from Section 10 that a linear probe is a ridge regression: it fits one weight
vector $\mathbf{w} \in \mathbb{R}^{1024}$ so that the dot product
$\mathbf{w} \cdot \mathbf{x}$ predicts a property of galaxy $\mathbf{x}$. That weight
vector has a direction in the embedding space. We will call it the probe direction for
that property. It is the single straight line along which the property changes fastest, as
the model sees it.

The direction comes from the same fits already reported in Section 10, one ridge probe per
label, so no new model is trained here. The weight vector $\mathbf{w}$ is what ridge
regression returns; only its orientation matters for this test, so its overall length
drops out when we normalise. Because the embedding was z-scored first (each of the 1024
dimensions shifted to mean zero and scaled to unit spread), the directions live in a space
where no single dimension dominates by accident, which is what lets us compare angles
fairly across labels.

Now take two properties, say redshift (how far away and how stretched in wavelength a
galaxy's light is, a stand-in for distance and look-back time) and smooth fraction (the
share of Galaxy Zoo voters who called the galaxy a smooth, featureless blob rather than a
structured disk). Each gives a probe direction. We measure the angle $\theta$ between the
two directions:

$$\theta = \arccos\!\left(\frac{\mathbf{w}_A \cdot \mathbf{w}_B}{\lVert \mathbf{w}_A \rVert \, \lVert \mathbf{w}_B \rVert}\right).$$

Here $\mathbf{w}_A$ and $\mathbf{w}_B$ are the two probe weight vectors, the dot in the
numerator is their inner product, and the two norms in the denominator scale the result so
it depends only on direction, not on length. The function $\arccos$ turns a cosine back
into an angle. The quantity inside the brackets is the cosine similarity of the two
directions, a number between $-1$ and $+1$ that is $+1$ when the directions coincide, $0$
when they are perpendicular, and $-1$ when they point exactly opposite. An angle of
$0^\circ$ means the two directions are identical (the model encodes both properties on the
very same axis). An angle of $90^\circ$ means they are orthogonal, fully independent
directions, the cleanest possible separation. An angle above $90^\circ$ means the
directions point partly against each other.

It helps to picture what orthogonal directions buy us. If $\mathbf{w}_A$ and
$\mathbf{w}_B$ are perpendicular, then moving a galaxy's embedding along $\mathbf{w}_A$
changes property $A$ but leaves the projection onto $\mathbf{w}_B$, and therefore the
read-out of property $B$, untouched. The two read-outs do not interfere.

If the directions are nearly parallel ($\theta$ small), then nudging $A$ drags $B$ along
with it, and the two probes are effectively reading the same coordinate. So the angle is a
clean, geometric statement about how much the model's two read-outs can be set
independently.

A bare angle is not enough, because the two properties are themselves correlated in the
real Universe. Redder galaxies tend to be smoother and more often passive; that is an
astrophysical fact, not a modelling choice. If two labels are correlated, then any honest
representation that encodes both will have probe directions that are correlated too, just
from the labels. So a small angle does not automatically mean the model failed to separate
the concepts. It might only be reflecting a real correlation in the data.

To handle that, we build a null: the angle we would expect from the label correlation
alone, with no extra separation. If two standardised labels have Pearson correlation $r$
(Pearson correlation measures how much two quantities move together on a straight line,
running from $-1$ to $+1$), then the angle implied by that correlation is

$$\theta_{\text{null}} = \arccos(r).$$

This is the baseline. It is the angle two probe directions would make if the only thing
separating them were the labels' own correlation. We then define the excess:

$$\text{excess} = \theta - \theta_{\text{null}}.$$

A positive excess means the model holds the two concepts more apart than the label
correlation requires (genuine separation, the directions are more orthogonal than they
"have to" be). An excess near zero means the model separates them exactly as much as the
labels do, no more and no less (the directions track the label correlation and nothing
extra). A negative excess means the model's directions are closer together than the labels
are, which would say the model has entangled the two concepts beyond what the data forces.
All angles are measured on the leakage-free image-only embedding $E_{\text{img}}$ where it
matters, so we are reading inference, not inputs, consistent with Section 10.

The reason the null is built from the labels and not from chance is worth stating once
more, because it is the whole point of the test. Two random directions in 1024-dimensional
space are almost always close to perpendicular; high-dimensional spaces are roomy, and a
"default" angle of nearly $90^\circ$ is what you get from noise alone. So comparing
$\theta$ to $90^\circ$ would be the wrong yardstick: it would reward the model for an
orthogonality that costs it nothing.

The label-correlation null fixes that. It asks the sharper question: given that these two
real properties genuinely move together in the Universe, is the model still keeping their
directions further apart than that shared movement demands? Only an excess above that
label baseline counts as the model doing extra work to keep the concepts apart.

A note on angles past $90^\circ$, because several pairs show them. An angle above
$90^\circ$ means the cosine similarity is negative, so the two probe directions point
partly opposite. That is not a problem, and it is not more entangled than orthogonal. For
separation, what matters is distance from parallel; both $80^\circ$ and $100^\circ$ are
well away from parallel and represent near-independent read-outs. The sign of the cosine
only reflects which way each label was oriented (smooth-up versus featured-up, say), which
is arbitrary. So we treat $93^\circ$ and $87^\circ$ as equally good separation, and we
always judge against the same pair's null, not against the raw $90^\circ$ line.

One honest limit up front. This is a linear, correlational test. It compares the angle
between two best-fit linear directions to the angle between two labels. It does not prove
the model represents a concept on exactly one axis, and it cannot establish that changing
one direction leaves the other untouched. That would need a causal intervention (steering
or ablating a direction and watching the other), which we did not run. The excess is
suggestive evidence of separation, not a guarantee of it.

### 11.2 Reading the dumbbell plot

![Figure 9. Learned probe-direction angle versus the label-correlation null, for ten label pairs.](figures/09_disentanglement_angles.png)

Figure 9 shows all ten label pairs as a dumbbell plot. The horizontal axis is the angle
between directions in degrees, running from about $20^\circ$ to past $100^\circ$. Each row
is one pair of labels, named on the left (for example "redshift - smooth"). Within a row
there are two dots joined by a bar, which is why it is called a dumbbell.

The grey dot is the null angle $\theta_{\text{null}} = \arccos(r)$, the separation the
labels' own correlation would produce. The coloured dot is the measured probe-direction
angle $\theta$. The bar between them is the excess, and the number printed beside it is
the excess in degrees. The bar and dot are coloured blue when the excess is positive (the
model is more disentangled than the null) and red when the excess is at or below zero (no
extra separation). The vertical dashed line at $90^\circ$ marks exact orthogonality, the
cleanest separation a pair of directions can have. Rows are sorted from the largest
positive excess at the top to the most negative at the bottom, so the figure reads as a
ranking: the strongest separation is on top, the entangled or undecided pairs at the
bottom.

What to look for: the horizontal position of the coloured dot relative to its grey twin.
If the blue dot sits to the right of the grey dot, the learned directions are more
separated than the labels force. The further right, the more separation. Read the figure
together with the table below.

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

The redshift-smooth pair is the headline. The labels are correlated enough to imply a null
angle of $73.7^\circ$, which corresponds to a label correlation of
$r = \cos(73.7^\circ) \approx 0.28$: smoother galaxies do tend to sit at the redshifts
where the red sequence is well populated, so the labels lean together by about a quarter.
Yet the model's two probe directions sit at $93.0^\circ$, just past orthogonal, for an
excess of $+19.4^\circ$ (measured). So the model holds its redshift axis and its
smoothness axis nearly at right angles even though the labels themselves lean together.

We read that as the model keeping distance (a property tied to look-back time and apparent
size) on a different axis from morphology (a property tied to internal structure), more
cleanly than the raw label correlation would force (interpreted). That is exactly the
separation you would hope a faithful representation makes: how far away a galaxy is and
what shape it has are independent facts about it, and the model treats them as
independent. The redshift-r-z pair tells a similar story at $+18.5^\circ$, with
$\theta = 69.2^\circ$ against a null of $50.7^\circ$. Here the measured angle stays below
$90^\circ$, so the two directions are not orthogonal, but they are pulled $18.5^\circ$
further apart than the labels alone would put them, which is the second-largest excess in
the set.

The two colour channels, $g-r$ and $r-z$, are strongly correlated with each other (both
rise as a galaxy's light reddens), so their null angle is small, $25.2^\circ$. The model's
directions sit at $40.3^\circ$, an excess of $+15.1^\circ$. They stay partly distinct
rather than collapsing onto one colour axis.

This is a meaningful result on its own. The symbols $g-r$ and $r-z$ are two colours, the
difference in brightness between the green and red bands and between the red and
near-infrared bands. They carry overlapping but not identical information: $g-r$ is most
sensitive to recent star formation and the strength of the $4000$ -angstrom break in a
galaxy's spectrum, while $r-z$ reaches redder and picks up older stellar populations and
dust. A model that kept a single "redness" axis would set their probe directions nearly on
top of each other. That AION keeps them $15^\circ$ further apart than even their tight
label correlation forces says it has resolved colour into more than one channel
(interpreted).

The redshift- $g-r$ pair adds $+13.1^\circ$ ($\theta = 52.2^\circ$, null $39.1^\circ$).
These four pairs, all involving redshift or colour, carry the strongest disentanglement
signal in the set, and they are also the four properties the probes decoded best in
Section 10 ($g-r$ at $R^2 = 0.958$, $r-z$ at $0.911$, redshift at $0.800$). Strong
decoding and clean separation travel together here, which is the pattern we would want if
the directions are real.

The morphology-colour pairs are positive but smaller: $g-r$ -smooth at $+11.7^\circ$
($\theta = 90.5^\circ$ against null $78.8^\circ$) and $r-z$ -smooth at $+9.0^\circ$
($\theta = 93.3^\circ$ against null $84.3^\circ$). Both measured angles sit right at or
just past the $90^\circ$ orthogonality line, so the model is holding colour and smoothness
on essentially independent axes.

The excess over the null is more modest than for the redshift and colour pairs, but it is
still positive and in the right direction. Physically this is the cleanest kind of
separation to ask for: colour (a property of the stellar populations) and smoothness (a
property of the spatial structure) are genuinely different things about a galaxy, and the
model treats them as such.

Now the honest part.

The pairs involving merger fraction (the share of voters who flagged a galaxy as a merging
or disturbed system) cluster near zero, and two of them go slightly negative. The $r-z$
-merger and $g-r$ -merger excesses are small and positive ($+5.7^\circ$ and $+2.9^\circ$).
The redshift-merger excess is $-2.0^\circ$ ($\theta = 82.5^\circ$, null $84.5^\circ$) and
the smooth-merger excess is $-5.6^\circ$ ($\theta = 90.4^\circ$, null $95.9^\circ$), both
negative. A negative excess means the model's two directions are marginally closer
together than the labels are. So for merger we see no extra separation, and a hint of the
opposite. These two rows are the ones drawn in red in Figure 9, with the coloured dot
sitting to the left of its grey twin rather than to the right.

We do not over-read this. The merger labels are the weakest in the set. From Section 10,
merger was the least decodable full-N property ($R^2 = 0.681$, the lowest of the
morphology labels), and merger is a rare, noisy, hard-to-define class: a galaxy is
"merging" only fleetingly, and the visual signs are subtle and easy to confuse with
ordinary disturbed structure or projection.

A weakly decodable label gives a noisy probe direction, and a noisy direction gives a
noisy angle. The small negative excesses ($-2.0^\circ$ and $-5.6^\circ$) are well inside
the range we would expect from that noise alone.

The honest statement is: the model cleanly disentangles redshift, colour, and smoothness
from one another, while for merger the test is inconclusive and shows no extra separation.
We do not claim the model entangles merger with anything; we claim the data cannot decide
it here.

So the ten pairs fall into three tiers. The top tier (excess above $+13^\circ$) is
redshift and colour separated from each other and from smoothness, the model's clearest
disentanglement. The middle tier (excess between $+9^\circ$ and $+12^\circ$) is colour
separated from smoothness, still positive and orthogonal but with a smaller margin over
the null. The bottom tier is everything touching merger, where the excess sits within a
few degrees of zero on either side and the test cannot decide. The signal lives in the top
two tiers, and it is consistent: every property the probes decoded well also separates
well.

Two caveats belong to this whole subsection. First, $\theta_{\text{null}} = \arccos(r)$
uses the linear (Pearson) label correlation, so it is the right null only to the extent
that the relationships are linear; strong nonlinearity between labels would bias the null.
Second, the excess has no confidence interval attached in our results, so we lean on the
size of the effect: $+19^\circ$ and $+18^\circ$ are large and consistent across the
redshift and colour pairs, whereas the $\pm 2$ to $\pm 6$ degree merger numbers are small
enough to be noise. Read the big positive excesses as the signal and the near-zero ones as
undecided.

And the deepest caveat is the one from the start of the section, which we restate because
it bounds every positive number above. This test is linear and correlational. A large
excess says the two best-fit linear read-out directions are far apart; it does not prove
the model stores each concept on a single axis, and it does not show that moving one
concept leaves the other physically unchanged inside the network. Establishing that would
take a causal intervention we did not run.

So the right summary is measured-and-modest: the model's linear directions for redshift,
colour, and smoothness are more orthogonal than the labels force, which is real and
consistent evidence of separation, and falls short of a proof of true disentanglement.

### 11.4 Neighbourhood purity: a check with no probe at all

The disentanglement test still fits probes. The second check throws even that away and
asks a blunt question directly in the embedding: do galaxies of one morphological type sit
among their own kind in the raw space?

This is model-free in the sense that no regression, no fitted weight vector, and no
learned direction is involved. We only use distances between the embedding vectors
themselves, on the full multimodal $E_{\text{full}}$ set. Where the disentanglement test
asked whether the model's read-out axes are distinct, this asks whether the raw geometry
already groups galaxies by type before any axis is drawn. The two are complementary: one
looks at directions, the other at neighbourhoods.

The measure is $k$ -nearest-neighbour purity, with $k = 20$. For a chosen class of
galaxies, we take each galaxy in that class, find its 20 nearest neighbours in the
embedding (the 20 closest other galaxies by distance), and ask what fraction of those 20
belong to the same class. Average that fraction over every galaxy in the class.

A purity of $1.0$ means every neighbour of every class member is also in the class
(perfect local segregation). A purity near the class's overall share of the sample means
neighbours are no more likely to be same-class than random (no local structure). High
purity says "like sits near like" in the native geometry, with no probe needed to see it.

The right yardstick is again not zero but a baseline, and here the baseline is the class's
own prevalence. If a class makes up a fraction $p$ of the sample and points were scattered
without any structure, then a typical neighbour would be in the class with probability
$p$, so the expected purity would be about $p$. Purity well above $p$ is the signal that
the class clusters.

This baseline is what makes the two numbers below interpretable, and it is also what makes
them not directly comparable, because the two classes have very different $p$.

We define the two classes by a vote-fraction cut. A galaxy is counted as smooth if its
smooth fraction exceeds $0.7$ (more than 70% of voters called it smooth), and as featured
if its featured fraction exceeds $0.7$. The cut at $0.7$ keeps only confident cases on
each side.

The results (measured):

| Class | $n$ | $k$NN purity ($k=20$) |
|---|---|---|
| smooth ($> 0.7$) | 34,289 | 0.991 |
| featured ($> 0.7$) | 2,224 | 0.714 |

Smooth galaxies are almost perfectly surrounded by other smooth galaxies: 99.1% of the 20
neighbours of a confident smooth galaxy are themselves confident smooth galaxies. Featured
galaxies are also surrounded by their own kind, but less tightly: 71.4% of neighbours
match.

Put these against the prevalence baseline. The confident smooth set has $n = 34{,}289$
members; the confident featured set has $n = 2{,}224$. Of the galaxies that fall into one
of these two confident classes, smooth is the overwhelming majority, so its prevalence is
high and its purity of $0.991$ is close to the ceiling.

The featured prevalence is far lower, roughly one part in sixteen of the two-class pool. A
purity of $0.714$ against a prevalence near $0.06$ is a factor of about twelve above what
random scattering would give. So even though $0.714$ looks weaker than $0.991$ on the
page, relative to its own much harder baseline it is a strong signal of clustering. The
two numbers measure the same thing on two very different scales.

The reading is straightforward and the asymmetry is informative. Confident smooth galaxies
form a dense, clean region of the embedding; you can stand on almost any smooth galaxy and
your 20 nearest neighbours will all be smooth too (interpreted).

The lower featured purity has a simple structural cause we should state plainly. Smooth
galaxies are by far the larger population here ($n = 34{,}289$ versus $n = 2{,}224$, about
fifteen to one). A rarer class has fewer same-class points to be near, so even with real
local clustering its purity is mechanically pulled down, because some of any galaxy's 20
neighbours will be drawn from the dominant smooth population at the boundary between the
two regions.

So $0.714$ for featured is still well above what random mixing would give for a class that
is a small minority of the sample, and we read it as real local structure, just softer and
more boundary-exposed than the smooth core (interpreted). The class-imbalance caveat is
the one to keep: the two purity numbers are not directly comparable as if the classes were
the same size.

There is a second, gentler caveat. The class labels are themselves vote fractions
predicted by a separate network and accurate only to about 5 to 10 percent (Section 4
(data and preprocessing)). A galaxy near the $0.7$ cut could fall on either side by label
noise alone, which softens purity at the boundary even if the embedding geometry were
perfect.

So the small shortfall of the featured number from $1.0$ is partly label noise, not only
embedding structure. The smooth purity of $0.991$ is high enough that this barely matters
there; it matters more for the rarer, fuzzier featured class.

One thing the purity result does NOT say, and we should be careful here. High
neighbourhood purity is fully consistent with the single continuous body that the
diffusion map found (Section 8 (diffusion maps)) and that the topology arm confirmed (one
connected piece, no holes). Purity is a local statement: near a smooth galaxy, things are
smooth.

It is not a claim that smooth and featured galaxies form two separate islands. They occupy
different regions of one continuous cloud, and the smooth region happens to be large,
dense, and pure. Like sits near like along a continuum, not in disconnected clusters.

Taken together, the two checks point the same way as the probes and the diffusion map
(Section 8 (diffusion maps), Section 9 (reading physics off the manifold)). The model does
not just carry the information that a galaxy is smooth or featured; it arranges the
embedding so that smoothness, colour, and redshift live on largely separate directions,
and so that galaxies of a kind cluster near their own kind. The next two sections move
from probing for human-named properties to letting the embedding name its own axes, using
a sparse autoencoder to ask whether the model carries concepts we never labelled.

## 12. Concept discovery I: the sparse autoencoder

The probes in Section 10 (decodability) answered one half of question C by asking, for
each human label we
already had, whether the embedding carries it. That is a confirmatory test. We hand the
model a concept and
check whether it is there. It cannot find a concept we did not think to name. The other
half of question C
is the open-ended part: does AION-1 organise galaxies along directions that go beyond the
human taxonomy of
colour, morphology, and redshift? To answer that we need a method that lets the embedding
speak first and
propose its own axes, rather than one where we supply the axis and the model only confirms
or denies. The
sparse autoencoder (SAE) is that method. This section explains what an SAE is, the exact
construction we
used, the one number that controls the reconstruction-versus-sparsity trade-off, and why
we set that number
where we did. The concepts it actually found, and the candidate concepts it found that no
human label
explains, are the subject of Section 13 (concepts and alien candidates).

### 12.1 The idea in plain words

Start from the problem the SAE is built to solve. Each galaxy is a point in a
1024-dimensional space (the
ambient embedding, Section 3). A single dimension of that space is almost never a clean
concept. Real
networks store information in a distributed way: the activation for "this galaxy is red"
might be spread
across hundreds of the 1024 coordinates, mixed together with the activation for "this
galaxy is a spiral"
and "this galaxy is at high redshift". Several distinct meanings ride on top of each other
in the same
coordinates. This mixing has a name in the interpretability literature, superposition, and
it is the reason
you cannot just read concept number 7 off coordinate number 7. The directions that carry
single, clean
meanings are tilted at odd angles to the coordinate axes, and there can be more such
directions than there
are coordinates.

A sparse autoencoder untangles this. The plan is to learn a large dictionary of
directions, far more than
1024, and to insist that any one galaxy is rebuilt from only a handful of them.
"Dictionary" here means a
fixed set of unit directions in the 1024-space, each one a candidate concept. "Sparse"
means few-at-a-time:
for any given galaxy, almost all dictionary entries are switched off and only a small set
fire. If the
method succeeds, each dictionary entry comes to stand for one human-interpretable thing,
because the
pressure to explain every galaxy with only a few active entries forces the network to
spend each entry on a
recurring, meaningful pattern rather than on noise. The galaxy's full 1024-vector is then
approximated as a
short weighted sum of active dictionary directions. We then look at which entries fire for
which galaxies
and ask what each one means. The entries we can name (they fire for red galaxies, or for
mergers) confirm
the concepts we expected. The entries we cannot name but that are stable across training
runs are the
candidates for structure beyond the human taxonomy.

That is the whole idea. An autoencoder that compresses a galaxy into a few active
"concept" units and then
reconstructs it, where the units are the things we read off afterward. The rest of this
section makes each
of those words precise.

### 12.2 The TopK construction, symbol by symbol

An autoencoder has two halves. The encoder maps the input vector to a set of latent
activations (the
dictionary "switches", one number per dictionary entry, mostly zero). The decoder maps
those activations
back to a reconstructed input vector. Training pushes the reconstruction to match the
original. We used the
TopK variant of the sparse autoencoder, which enforces sparsity in the most direct way
possible: keep only
the $k$ largest latent activations for each galaxy and set every other latent to exactly
zero.

Write the input galaxy vector as $x \in \mathbb{R}^{1024}$. This is the z-scored full
embedding
$E_\text{full}$ (image + photometry + redshift fused; z-scoring per dimension is defined
in Section 4
(data)). Let $m$ be the dictionary size, the number of latents. The encoder is

$$ f = \mathrm{TopK}\big(W_\text{enc}\,(x - b_\text{pre}) + b_\text{enc}\big), \qquad f \in \mathbb{R}^{m}. $$

Reading this left to right: $b_\text{pre} \in \mathbb{R}^{1024}$ is a learned pre-bias
subtracted from the
input, which roughly centres the data so the dictionary does not waste entries
representing the cloud's mean
position. $W_\text{enc} \in \mathbb{R}^{m \times 1024}$ is the encoder weight matrix; each
of its $m$ rows is
a direction in the 1024-space, and the dot product of that row with the centred input
gives one pre-activation
score, "how strongly does this galaxy point along dictionary direction $i$ ".
$b_\text{enc} \in \mathbb{R}^{m}$
is a per-latent bias added to those scores. Then $\mathrm{TopK}(\cdot)$ is the operation
that does the
sparsifying: it looks at the $m$ scores, finds the $k$ largest, keeps those values, and
zeroes the rest. So
$f$ has at most $k$ non-zero entries. Those non-zero entries are the dictionary entries
that "fire" for this
galaxy, and their values are the firing strengths.

The decoder rebuilds the input from the few active latents:

$$ \hat{x} = f\,W_\text{dec} + b_\text{pre}, \qquad \hat{x} \in \mathbb{R}^{1024}. $$

Here $W_\text{dec} \in \mathbb{R}^{m \times 1024}$ is the decoder weight matrix; its rows
are the dictionary
directions themselves (the actual concept vectors we interpret later). Because $f$ has
only $k$ non-zero
entries, $\hat{x}$ is literally a sum of $k$ dictionary directions, each scaled by its
firing strength, plus
the pre-bias added back. The same $b_\text{pre}$ that was subtracted in the encoder is
added back here, so
the network only ever has to model how a galaxy differs from the centre.

Training minimises the squared reconstruction error $\lVert x - \hat{x} \rVert^2$ averaged
over all 48,398
galaxies, with $k$ held fixed. There is no separate penalty term coaxing the latents
toward zero, which is the
appeal of TopK over older SAE recipes: the sparsity is exact and is set by one integer,
$k$, not by tuning a
soft penalty weight against the reconstruction loss.

### 12.3 Two control terms: $L_0$ and the AuxK revival

The sparsity level has a standard name, $L_0$. The $L_0$ "norm" of a vector counts its
non-zero entries (it
is not really a norm, but the name is conventional). For a TopK SAE, $L_0 = k$ exactly and
by construction:
every galaxy uses exactly $k$ active latents, never more, never fewer. So when we say the
operating point is
$k = 32$, we are saying each galaxy's 1024-vector is rebuilt from exactly 32 active
dictionary entries out of
the thousands available. $L_0 = k = 32$ is the headline sparsity of this study.

A practical failure mode of any large-dictionary autoencoder is dead latents: entries
that, after some
training, never make the top- $k$ for any galaxy, get no gradient, and stay dead forever.
A dead latent is
wasted capacity; it is a dictionary slot that learned nothing. To counter this we used the
AuxK term, an
auxiliary loss that revives dead units. The mechanism: take the reconstruction residual
$x - \hat{x}$ (the
part of the galaxy the main top- $k$ latents failed to explain), and ask the current set
of dead latents,
specifically the top 512 of them by pre-activation, to reconstruct that residual on their
own. This auxiliary
reconstruction is added to the loss with a small weight ($1/32$ in our run). The effect is
gentle and
targeted. Latents that have gone dead get a reason to become useful again (they are
rewarded for explaining
what the live latents missed), without disturbing the main reconstruction. Without AuxK, a
large fraction of
the dictionary tends to die early and the effective dictionary shrinks; with it, more
entries stay in play.

### 12.4 Expansion factor and dictionary size

The dictionary size $m$ is set by the expansion factor $R$, defined as the ratio of
dictionary entries to
ambient dimensions:

$$ m = R \times 1024. $$

$R$ controls how over-complete the dictionary is. $R = 1$ would give exactly as many
entries as dimensions
(no room for superposition to be unpacked); $R > 1$ gives more entries than dimensions,
which is what lets
the SAE assign separate entries to the many tilted concept directions that share the same
coordinates. We ran
two settings:

| Expansion $R$ | Dictionary size $m = R \times 1024$ | FVU (mean over 5 seeds) | Fraction of latents alive |
|---|---|---|---|
| 4 | 4,096 | $\approx 0.035$ | $\approx 0.65$ to $0.70$ |
| 8 | 8,192 | $\approx 0.034$ | $\approx 0.33$ |

Both values are measured, from the `health` block of `results/sae.json`, averaged over
five random
initialisation seeds (the per-seed FVU values for $R=4$ run from 0.03502 to 0.03599; for
$R=8$ from 0.03411
to 0.03469). Two things stand out, and both are expected. First, doubling the dictionary
from 4,096 to 8,192
barely improves reconstruction (FVU falls only from about 0.035 to 0.034). The extra
capacity is not buying
much fidelity, which says 4,096 entries are already enough to capture nearly all the
structure at this
sparsity. Second, the fraction of entries that stay alive roughly halves, from about
two-thirds at $R=4$ to
about one-third at $R=8$. That is the expected behaviour. With twice as many slots
competing for the same
fixed 32 firing opportunities per galaxy, more slots end up unused. Because the larger
dictionary gains
almost no reconstruction quality while wasting two-thirds of its entries, we did the
concept analysis of
Section 13 on the $R=4$ dictionary ($m = 4{,}096$), which is the better-occupied and more
economical choice.

### 12.5 FVU and "alive", defined

Two diagnostics appear above and run through the rest of the SAE work, so define them
cleanly now.

FVU is the fraction of variance unexplained. It measures how badly the reconstruction
misses, on a scale where
0 is perfect and 1 is useless:

$$ \mathrm{FVU} \;=\; \frac{\sum_i \lVert x_i - \hat{x}_i \rVert^2}{\sum_i \lVert x_i - \bar{x} \rVert^2}, $$

where the sum runs over all galaxies $i$, $\hat{x}_i$ is the SAE reconstruction of galaxy
$i$, and $\bar{x}$
is the mean embedding vector. The numerator is the total squared reconstruction error; the
denominator is the
total variance of the data around its mean (the error you would get from the trivial
"predict the average
galaxy" baseline). So FVU is the residual error expressed as a fraction of the variance
you started with.
$\mathrm{FVU} = 0.036$ means the reconstruction leaves 3.6% of the variance unexplained,
equivalently it
explains $1 - 0.036 = 96.4\%$ of the variance. We also write that as "96.5% variance
explained" following the
rounded health figure; the two phrasings are the same quantity. Lower FVU is better
reconstruction.

"Alive" is the fraction of dictionary entries that actually fire for at least some
galaxies over the course of
training, the complement of the dead-latent fraction from Section 12.3. An alive fraction
of 0.65 at
$m = 4{,}096$ means roughly 2,660 of the 4,096 entries are in use; the rest never make the
top-32 and carry no
information. Only alive, used entries are eligible to be interpreted as concepts in
Section 13, which is why
the concept count there starts from $n_\text{active} = 2{,}657$ entries rather than the
full 4,096. The alive
fraction matters because a dictionary that looks large on paper but is mostly dead has a
smaller true capacity,
and reporting the nominal size alone would overstate how many concepts the SAE could
possibly hold.

### 12.6 The reconstruction-versus-sparsity frontier, and why $k = 32$

The single most consequential choice in this method is $k$, the number of active latents
per galaxy. It sets
a trade-off. Small $k$ means very sparse, very interpretable codes (each galaxy explained
by a few entries),
but a poor reconstruction (too few entries to capture the galaxy faithfully). Large $k$
means an excellent
reconstruction but codes so dense that "sparse" stops meaning anything and the entries
blur back toward the
superposed mess we were trying to undo. We need a value that reconstructs the embedding
well while keeping the
code genuinely sparse. To find it, we swept $k$ from 4 to 128 on the $R=4$ dictionary and
recorded the FVU at
each value. That sweep is the frontier.

![Figure 11. SAE reconstruction error (FVU) versus sparsity k, with the k=32 operating point.](figures/11_sae_frontier.png)

Figure 11 plots the trade-off. The horizontal axis is $k$, the number of active latents
per galaxy (the
sparsity level $L_0$), running 4, 8, 16, 24, 32, 48, 64, 96, 128 from left to right; it is
the dial we are
turning. The vertical axis is FVU, the fraction of variance unexplained (Section 12.5),
running from 0 at the
bottom to about 0.14 at the top; lower is a better reconstruction. The blue curve with
filled markers is the
measured FVU at each swept $k$, one point per value, joined to show the trend. The
vertical dashed red line
marks $k = 32$, our chosen operating point, and the red annotation states its value, FVU
$= 0.036$. Every
plotted point is measured directly from the `frontier` block of `results/sae.json`.

Read the curve as a classic diminishing-returns elbow. The measured values are:
$k=4 \to \mathrm{FVU}\,0.142$,
$k=8 \to 0.092$, $k=16 \to 0.058$, $k=24 \to 0.043$, $k=32 \to 0.036$, $k=48 \to 0.026$,
$k=64 \to 0.023$,
$k=96 \to 0.018$, $k=128 \to 0.015$. The left part of the curve is steep: going from $k=4$
to $k=8$ cuts FVU
from 0.142 to 0.092 (about a third of the remaining error gone for 4 more latents), and
$k=8$ to $k=16$ cuts
it again to 0.058. Each early latent buys a large chunk of fidelity. Then the curve bends.
Past $k=32$ it
flattens hard: doubling the budget from 32 to 64 latents moves FVU only from 0.036 to
0.023, and doubling
again to 128 reaches just 0.015. Those extra latents buy little reconstruction but cost a
lot of sparsity, and
the more entries fire per galaxy the harder each one is to read as a single clean concept.

$k = 32$ sits right at the knee. At that point the reconstruction already explains 96.4%
of the variance
(FVU 0.036), which is enough to say the dictionary faithfully represents the embedding,
while the code stays
tight: 32 active entries out of about 2,660 alive ones means roughly 99% of the dictionary
is silent for any
given galaxy. Pushing to $k=64$ would recover only a further 1.3 percentage points of
variance (FVU 0.036 to
0.023) at the price of doubling the active-set size and muddying interpretation. Pulling
back to $k=16$ would
keep the code sparser but at FVU 0.058 the reconstruction starts to lose real structure.
So $k=32$ is the
honest balance point, good fidelity at genuine sparsity. This is an engineering choice
read off the elbow of a
measured curve, not a physical constant; a reader who wanted sparser codes at some
reconstruction cost could
defensibly pick $k=16$, and the qualitative concept results would survive. We state $k=32$
as our operating
point and carry it into Section 13.

### 12.7 Why this method answers question C

It is worth being explicit about why we went to the trouble of an SAE at all, when Section
10 already showed
that physics is strongly decodable from the embedding. The probes and the SAE answer
different questions.

A probe is supervised and targeted. You name a property in advance (redshift, stellar
mass), fit a linear map
from the 1024 numbers to that property, and read off how well it works. A high probe $R^2$
tells you the
embedding contains that property, recoverably and linearly. But the probe can only ever
find what you put in
its mouth. It tests a hypothesis you already had. If AION-1 organises galaxies along some
axis that no entry in
our label set corresponds to, a probe will never reveal it, because there is no label to
probe for. That is the
ceiling of the confirmatory approach, and it is exactly the part of question C ("does the
model organise by
concepts beyond the human taxonomy?") that probes cannot reach.

The SAE inverts the direction of inquiry. It is unsupervised: it never sees a single label
during training. It
is handed only the raw embeddings and the demand to rebuild each galaxy from a few active
dictionary entries.
Whatever directions it settles on are chosen by the structure of the embedding itself, by
what recurs across
the 48,398 galaxies, not by what we asked for. So the dictionary it produces is the
model's own proposed list
of axes. We can then sort that list two ways. Entries whose firing pattern lines up with a
human label
(Section 13 measures this alignment against a shuffle null) are the model rediscovering
colour, morphology,
and redshift on its own, an unsupervised confirmation of question C's confirmatory half.
Entries that fire
stably across training seeds but match no label we have are candidates for the genuinely
open half: structure
the model represents that our taxonomy does not name. We label these "alien candidates"
and treat them with
heavy caution. They are correlational only, defined by what they fail to correlate with,
and no causal or
ablation test was run on them in this study. Section 13 reports both halves, the named
concepts and the alien
candidates, with their counts and their null thresholds.

The SAE, then, is the instrument that lets the embedding reveal its own organising axes
instead of only
echoing back the ones we supplied. That is the precise sense in which it, and not the
probes, addresses the
open part of question C.

## 13. Concept discovery II: the concepts and the alien candidates

Section 12 built the tool: a TopK sparse autoencoder (SAE) that rewrites each galaxy's
1024-number embedding as a short list of active "concept" units drawn from an
over-complete dictionary of $m = 4{,}096$ latents, with $k = 32$ of them switched on per
galaxy. This section reads the output. We ask three questions of every latent the SAE
learned. Does it line up with a property we already have a name for? Is it real, in the
sense that it reappears when we retrain from a different random start? And does it carry
signal that none of our named properties can account for? The third question is the one
that motivated this whole arm, because a foundation model trained with no morphology
labels might organise galaxies along axes that our human taxonomy never wrote down.

Throughout, keep one fact in front of you. Everything here is correlational. An SAE latent
that tracks colour is a direction in embedding space whose activation rises and falls with
colour; we never intervened on the model, never ablated a latent, never steered the
embedding to see what changes downstream. So when we call 59 latents "alien candidates"
below, read "candidate" as a literal promissory note, not a finding. We will repeat this.

### 13.1 What "alignment" means and how we score it

A latent is one number per galaxy: how strongly that concept unit fires for that galaxy. A
label is also one number per galaxy: its redshift, its g-r colour, its smooth vote
fraction, and so on. To ask whether a latent "is about" a label, we measure how tightly
the two numbers move together across all the galaxies where the latent is active.

We use the Spearman rank correlation $\rho$. Spearman replaces each value by its rank (1st
smallest, 2nd smallest, and so on) and then correlates the ranks. We chose rank
correlation, not the ordinary Pearson correlation, for a physical reason: SAE activations
are heavy-tailed and often near zero, and several labels (redshift, vote fractions) are
bounded or skewed, so a measure that only cares about ordering is the honest choice. It
does not assume a straight-line relationship; it asks the weaker, safer question of
whether high goes with high.

We define the alignment of a latent as

$$\text{align} = \max_{\ell \in \{6 \text{ labels}\}} \; \big| \rho(\text{latent}, \ell) \big|.$$

In words: take the six full-sample labels (redshift, the two colours g-r and r-z, and the
three Galaxy-Zoo vote fractions smooth, featured, merger), correlate the latent against
each, take the absolute value (a latent that fires for blue galaxies and one that fires
for red galaxies are both "colour" latents, so sign does not matter for naming), and keep
the largest. That single number is how strongly the latent matches its best-matching human
label. The label that wins the max is the latent's name.

The scoring ran on the $R=4$ dictionary ($m = 4{,}096$ latents) trained on the z-scored
multimodal embedding E_full. Of the 4,096 dictionary slots, $n_{\text{active}} = 2{,}657$
actually fire on the data (the rest are dead or near-dead, expected at this expansion).
All counts below are out of those 2,657 active latents (measured).

### 13.2 The label-shuffle null: how big an alignment is "real"

A correlation of, say, 0.04 sounds small, but with $N \approx 48{,}000$ galaxies even pure
noise produces non-zero correlations, and because we take the *maximum* over six labels we
get a free upward bias (the best of six noisy draws is larger than any single one). We
need a baseline that bakes in both effects. We use a label-shuffle null.

The recipe is simple and assumption-free. We randomly permute each label across galaxies,
breaking any true link to the embedding while keeping each label's own distribution
exactly intact, then recompute every latent's alignment against the shuffled labels.
Repeating the shuffle builds a distribution of alignment values that pure chance can
manufacture. From that distribution we read two cutoffs (measured):

$$\text{thr95} = 0.0119, \qquad \text{thr99} = 0.0129.$$

A latent whose real alignment exceeds 0.0119 sits above the 95th percentile of chance; one
above 0.0129 clears the stricter 99th percentile. So the bar for "significantly aligned"
is a Spearman of about 0.012. That is a low bar in absolute terms, which is the point: at
this sample size, even a weak but genuine correlation is detectable, and the null tells us
where genuine starts.

Two counts follow directly (measured):

- $\text{aligned\_sig} = 717$ latents clear thr95.
- $\text{aligned\_strong} = 690$ latents clear the stricter thr99.

That the two numbers are so close (717 versus 690) tells you something quiet but useful.
Almost every latent that beats the 95% bar also beats the 99% bar; there is no thick band
of borderline cases sitting between the two thresholds. The aligned latents are not
marginal. They are well clear of the null or not in the running at all.

The largest alignment anywhere is

$$\text{max\_align} = 0.279 \quad (\text{measured}),$$

which is about 23 times thr95. We write "about 23x" because $0.279 / 0.0119 = 23.4$; the
figure rounds it to 23x. So the single most label-locked latent in the dictionary still
only reaches a Spearman of 0.28. We read that as a real but modest ceiling (interpreted):
the SAE does find directions that track human labels, but no single sparse latent *is* a
label. The strongest match explains only a slice of the label's variation. This matters
for how you picture the model's code. Physics is not stored in one neuron per concept; it
is smeared across many.

### 13.3 Seed-stability: is the latent a property of the model or of the random seed?

An SAE is trained from a random initialisation, so any single run can manufacture a latent
that is an accident of that particular start. The fix is to retrain. We trained five SAEs
from five different random seeds and asked, for each latent, whether the same direction
comes back.

The comparison is on the decoder vector. Each latent has a decoder vector
$W_{\text{dec}}$, the 1024-number direction it writes back into the embedding when it
fires; this is the latent's identity in embedding space. To compare a latent in one seed
to a latent in another, we use cosine similarity, the cosine of the angle between the two
decoder vectors. Cosine of 1 means the same direction, 0 means orthogonal (unrelated), and
we set the match bar at

$$\cos \ge 0.6,$$

meaning the two directions point within about 53 degrees of each other. A latent is called
seed-stable if its decoder direction reappears with $\cos \ge 0.6$ in at least half of the
other four seeds. The "half of the others" rule guards against a single lucky coincidence:
a stable concept should show up again and again, not just once.

Across all active latents, the median best cross-seed cosine is 0.460 (measured). Read
that carefully. The *typical* latent's best match in another seed sits below the 0.6 bar,
so most individual SAE latents are partly seed-specific; the dictionary as a whole is not
perfectly reproducible unit-for-unit. That is the honest caveat for this method. What
survives the bar is a minority, and it is exactly that minority we trust. The fraction of
active latents that are seed-stable is 0.148 (measured), about one in seven.

Combining the two filters gives the headline count of dependable, human-named directions:

$$\text{aligned\_and\_stable} = 335 \quad (\text{measured}).$$

These 335 latents both beat the shuffle null (so they track a real label) and reappear
across seeds (so they are a property of the model, not the seed). This is the number to
quote for "physics-named concepts the model reliably carries." It is the intersection of
two independent screens, which is why it is the most defensible count in this section.

### 13.4 The named concepts

Naming each aligned latent by its top-correlated label and tallying them gives the concept
inventory in Figure 12.

![Figure 12. SAE features aligned to physical labels, per concept.](figures/12_sae_named_concepts.png)

Figure 12. The horizontal axis is a count: the number of active SAE latents (out of 2,657)
whose best-matching label is the concept on the vertical axis. The six concepts on the
vertical axis are the six full-sample labels, sorted top to bottom by total count. Each
concept has two stacked bars: the light bar is the number of latents that clear the
label-shuffle significance bar (thr95 = 0.012), and the darker bar is the seed-stable
subset of those (the latents that also reappear across seeds). The text label at the end
of each light bar, "max |rho| = ...", gives that concept's single strongest alignment, the
largest absolute Spearman correlation any latent reached against that label. The two
footnote lines restate the definitions and the load-bearing caveat: "aligned" is a
correlational match to a shuffle null, naming is just the top-correlated label, and
redshift in particular is spread over many weak latents rather than concentrated in one
strong feature. What to look for: colour (g-r) and the morphology pair (smooth, featured)
carry the strongest single matches (max |rho| around 0.25 to 0.28), while redshift has
many aligned latents but each is individually weak (max |rho| only 0.13).

The per-concept counts (light bar $n$, dark bar $n_{\text{stable}}$, and the strongest
alignment) are all measured:

| Concept | Aligned latents $n$ | Seed-stable $n_{\text{stable}}$ | Max alignment |
|---|---|---|---|
| g-r (colour) | 204 | 80 | 0.272 |
| redshift | 166 | 87 | 0.126 |
| featured (morphology) | 114 | 52 | 0.245 |
| r-z (colour) | 89 | 41 | 0.210 |
| smooth (morphology) | 81 | 37 | 0.279 |
| merger | 63 | 38 | 0.164 |

Two readings, both interpreted, follow from the table.

First, the clearest single concepts are colour and morphology. The g-r colour wins on
count (204 aligned latents, the most of any label) and the smooth vote fraction wins on
strength (0.279, the highest single alignment in the whole dictionary, and the same number
as the global max\_align). Featured sits just behind at 0.245. So when the SAE carves out
a direction that lines up tightly with a human label, that label is usually colour or a
smooth/featured morphology axis. This is consistent with what every other arm of the
report found: colour and the smooth-versus-featured morphology split are the model's
primary organising axes (the diffusion coordinates of Section 9, the strong probes of
Section 10).

Second, redshift behaves differently, and the difference is informative. Redshift has
plenty of aligned latents (166, second only to g-r) and the most seed-stable ones (87),
yet its single strongest alignment is only 0.126, roughly half the colour and morphology
peaks. We read this as redshift being a *global, diffuse* property of the embedding rather
than a sparse code (interpreted). It is written faintly into many directions at once, not
loudly into a few. That fits the diffusion-map picture, where redshift loaded onto a broad
smooth gradient (dc1) rather than a sharp axis, and it fits the intuition that redshift, a
distance-and-time coordinate, touches almost everything about how a galaxy looks (its
apparent size, brightness, and observed colours all shift with it) instead of being one
isolated feature. Merger is the weakest concept on both count (63) and strength (0.164),
which is unsurprising: mergers are rare, the vote fraction is noisy, and there is simply
less consistent signal for the SAE to latch onto.

One caution on the table. The counts are not a partition of "how much the model knows
about each property," because a latent can correlate with several labels and we assign it
to only its single best one. Colour and redshift, for instance, are themselves correlated
in the data (redshift is mostly photo-z, which is colour-derived, per Section 4 and
Section 10), so some latents could plausibly have gone either way. The names are a useful
summary, not a clean decomposition.

### 13.5 Novelty: signal the labels cannot explain

Now the question this arm was built for. Does the SAE find structure that our six labels
miss?

We define novelty per latent as the fraction of the latent's activation variance that the
labels *cannot* linearly account for. Concretely, we regress the latent's activations on
the six labels (a linear least-squares fit, latent as the target, the six labels as
predictors) and measure how much of the latent's variance the fit explains. Call that
explained fraction $\text{rec}$ (for "recovered"). Then

$$\text{novelty} = 1 - \text{rec}.$$

Novelty near 0 means the labels reconstruct the latent almost perfectly, so the latent is,
in effect, a relabelled combination of things we already named. Novelty near 1 means the
labels are nearly useless at predicting the latent, so it carries variation orthogonal to
our entire human vocabulary. It is a regression residual, plain and simple: what is left
of a latent after you subtract everything the labels can linearly say about it.

We then defined an alien candidate by three conditions held at once (measured):

1. Seed-stable (decoder direction reappears in at least half the other seeds at $\cos \ge 0.6$), so it is a real property of the model, not a seed accident.
2. High novelty, $\text{novelty} > 0.7$, so the six labels explain less than 30% of its activation variance.
3. Not label-aligned (its best alignment does not clear the thr95 shuffle null), so it is not just a weak echo of a named concept.

A latent that is stable, mostly unexplained by any label, and not even weakly tied to one,
is a direction the model reliably uses for *something* we have no name for. There are

$$\text{alien\_candidates} = 59 \quad (\text{measured}).$$

Figure 13 shows both the alignment distribution and where these 59 sit.

![Figure 13. SAE alignment versus the shuffle null, and alignment versus novelty with the alien candidates.](figures/13_sae_alignment_novelty.png)

Figure 13, panel A (left). The horizontal axis is alignment, each active latent's best
absolute Spearman correlation to a physical label, from 0 to about 0.28. The vertical axis
is a count of latents on a logarithmic scale (note: each step up is a factor of ten, so
the tall leftmost bar holds well over a thousand latents). The red dashed vertical line is
the shuffle null at thr95 = 0.012 and the black dotted line is thr99 = 0.013; latents to
the left of these lines are statistically indistinguishable from chance. The red arrow on
the right marks the single most-aligned latent at 0.279, annotated "max alignment = 0.279
(~23x the thr95 null)." What to look for: the distribution is hugely front-loaded, with
most latents piled just above zero (below or near the null) and a long thin tail of
genuinely aligned latents stretching out to 0.28. The shape says the same thing as the
close 717-versus-690 counts above: a minority of latents carry real label signal, and they
are clearly separated from the noise floor rather than blending into it.

Figure 13, panel B (right). The horizontal axis is again alignment (best |Spearman| to a
label). The vertical axis is novelty, the residual activation-variance fraction with the
six labels regressed out, from roughly 0.5 to 1.0. Each dot is one active latent. Colour
encodes seed-stability: faint grey dots are not seed-stable (cross-seed reconstruction
below 0.5), blue dots are seed-stable (at or above 0.5), and the orange-ringed markers are
the 59 alien candidates. The shaded orange box in the lower region near the left is the
alien selection region: low alignment (left of the thr95 line, the red dashed vertical)
and high novelty (above 0.7). The inset at the top zooms into that alien region so the 59
orange points are legible against the crowd. The footnote restates the definition and
confirms the count: "alien candidates: seed-stable, high-novelty, not label-aligned (align
<= thr95) -- CORRELATIONAL only (no causal test). JSON-reported alien\_candidates = 59;
reproduced here = 59." What to look for: most latents sit high on the novelty axis (the
labels never explain a latent fully, because each latent is sparse and idiosyncratic), but
the orange points are the ones that combine high novelty with near-zero alignment *and*
survive the seed test. They are the model's reliable, unnamed directions.

### 13.6 What the alien candidates are, and what they are not

Here is the careful reading, because this is the most interesting and the least certain
result in the whole report.

What we can say (measured). There exist 59 directions in AION-1's embedding that (a) the
model reconstructs consistently across five independent random retrainings, so they are
not artefacts of one optimisation run, and (b) our six physical labels cannot linearly
predict, explaining less than 30% of their variance, and (c) do not even weakly track any
single label above the chance floor. By construction they are stable, novel, and unnamed.

What we cannot say (the caveat, repeated because it governs every word here). These are
correlational candidates only. We ran no causal test. We did not ablate any alien latent
and check whether reconstruction or any downstream prediction degrades. We did not steer a
latent and watch the embedding or any rendered galaxy change. So we do not know whether an
alien candidate encodes a genuine, physically meaningful galaxy property that our labels
happen to omit, or whether it encodes something with no clean physical reading at all: an
imaging artefact, a survey or instrument systematic, a sky-position effect, a residual of
the masked-modelling training objective, or a correlated nuisance we did not measure.
"Alien" names the gap in our labels, not a discovery about galaxies. The honest status is:
a stable, unexplained direction worth a future causal test, no more.

There is also a built-in limit on novelty's reach. We only regressed against six labels,
and only linearly. A latent could be perfectly explained by some property we did not have
(stellar mass, sSFR, and Sersic index exist for only a few thousand galaxies, so they were
not in the six-label full-sample regression) or by a nonlinear combination of labels we
did have. So novelty over 0.7 means "not linearly explained by these six," which is
narrower than "genuinely beyond all known physics." A fraction of the 59 could collapse
into "known" the moment a richer label set or a nonlinear probe is applied. We flag this
rather than hide it.

### 13.7 What this arm establishes

Stepping back, the SAE answers the concept-discovery question (question C from Section 1)
in two parts. First, the model's reliable, human-nameable code is real but modest and
smeared: 335 seed-stable, label-aligned latents, dominated by colour and the
smooth/featured morphology split, with redshift present everywhere but concentrated
nowhere, and a strongest single match of only 0.279. No single latent is a concept;
concepts live in populations of weakly aligned directions. Second, beyond the named axes
the model reliably uses at least 59 stable directions our labels cannot account for. Those
59 are the candidates a full study should target first, with the causal tools this static,
correlational analysis could not bring to bear: ablate them and steer them, and see what
moves. Section 19 folds this concept picture into the geometric one, and Section 21 lists
the causal follow-up as the first thing a complete study should do.

## 14. Curvature and tree-likeness

We now ask a different question about the shape of the embedding. Section 6 and Section 7
(intrinsic dimension) counted how many independent knobs the manifold uses, about 10 to
12. Section 8 (diffusion maps) showed the cloud is one continuous body with no spectral
gap. But knowing the dimension and the connectivity does not tell you the *bend* of the
surface. A flat sheet and a sphere can both be two-dimensional. A flat sheet and a
branching tree can both be connected. Curvature is the missing piece: it says how the
manifold turns as you move across it, and whether it pinches into branch points. This
section measures that.

The reason curvature matters here is physical, not just geometric. Section 2 laid out two
competing expectations for how galaxies should organise. One is the smooth morphological
continuum (ellipticals shade into lenticulars shade into spirals, with no hard wall
between them), which would show up as a gently curved continuous surface. The other is the
idea of distinct quenching channels (a fast disk-to-spheroid route versus a slow one),
which, if the model encoded it as a real bifurcation, would show up as branch points:
places where one population splits into two and the geometry locally looks like a fork in
a tree. So the curvature measurement is a direct test of whether AION-1's geometry is a
continuum, a tree, or a continuum with a few weak forks in it.

### 14.1 What curvature and tree-likeness mean, in plain words

Start with curvature on an ordinary surface, then carry it to a manifold of points.

**Positive curvature** is the curvature of a sphere or the top of a hill. If you stand at
a point and walk out in every direction, the surface curves the same way under your feet,
and the area around you closes in on itself. In a point cloud, positive curvature shows up
as *clustering*: the neighbours of a point are also neighbours of each other, so a small
ball around the point is densely interconnected. Triangles close up. Local neighbourhoods
look like tight, well-connected blobs.

**Negative curvature** is the curvature of a saddle, a mountain pass, or a Pringle chip.
Walk out in one direction and the surface falls away; walk out at right angles and it
rises. Space spreads apart faster than flat space does, so there is "more room" as you
move out. In a point cloud, negative curvature shows up as a *bridge* or *bottleneck*: a
point whose neighbours fan out into two or more groups that are not neighbours of each
other. The point sits at a saddle between regions. If you imagine two dense blobs joined
by a thin neck, the points in the neck have negative curvature. This is exactly what a
branch point in a tree looks like locally.

**Zero curvature** is flat space, like a plane or a sheet of paper rolled into a cylinder
(a cylinder is flat in the intrinsic sense, because you can unroll it without stretching).

**Tree-likeness** is a global cousin of negative curvature. A tree is a graph with no
loops, where every pair of points is joined by exactly one path, and the whole thing
branches outward from a root. Trees are the extreme case of negative curvature everywhere:
at every junction the space forks, and distances grow the way they do in a saddle. The
question "is this manifold tree-like?" is the question "does it branch repeatedly, the way
an evolutionary tree or a river delta branches, rather than forming a single connected
sheet?" A continuum is the opposite of a tree: it is one body you can slide across
smoothly, with neighbourhoods that reconnect rather than fork.

We measure two things, at two scales. A **global** tree-likeness score
(delta-hyperbolicity) asks whether the whole cloud, taken together, branches like a tree.
A **local** signed curvature (Ollivier-Ricci) asks, edge by edge, whether each small
neighbourhood clusters (positive) or bridges (negative). They answer different questions,
and we report both.

### 14.2 Delta-hyperbolicity: a global tree-likeness score

The global measure is **Gromov's four-point delta**, also called delta-hyperbolicity. The
intuition: in a tree, distances satisfy a very strict relationship, and the more a space
behaves like a tree, the closer it comes to satisfying that relationship exactly. Delta
measures how far the space is from perfect tree behaviour. **Lower delta means more
tree-like. Delta = 0 means a perfect tree.**

Here is the rule it checks. Pick any four points $w, x, y, z$. You can pair them up three
ways and add the distances within each pairing:

$$S_1 = d(w,x) + d(y,z), \quad S_2 = d(w,y) + d(x,z), \quad S_3 = d(w,z) + d(x,y).$$

Sort these three sums so that the largest is $L$ and the second-largest is $M$. The
four-point delta for this quadruple is half the gap between them:

$$\delta = \frac{L - M}{2}.$$

In a perfect tree, the two largest of those three sums are always equal, so $L = M$ and
$\delta = 0$ for every quadruple. The more the largest sum exceeds the second, the further
the four points are from sitting on a tree, and the larger $\delta$. We compute this for a
huge number of random quadruples (one million of them, sampled with a fixed seed) and take
the median over all of them. We divide by the diameter of the cloud (the largest pairwise
distance) so the number is dimensionless and comparable across datasets of different
scale. So the reported $\delta$ is a *diameter-normalised median four-point delta*: a pure
shape number between 0 (perfect tree) and roughly 0.25 (the value you get from a maximally
non-tree-like, sphere-like arrangement).

All of this runs on the z-scored embedding under Euclidean distance, with cosine distance
reported as a secondary check.

#### Why the absolute number is useless without anchors

Here is the honest catch, and it is the reason this measure is built around comparison
rather than a single readout. A raw delta of, say, 0.0136 means nothing on its own. Is
that tree-like? Compared to what? Any finite cloud of points in high dimension has *some*
nonzero delta just from sampling and from the geometry of high-dimensional space, even if
there is no real branching at all. The number only becomes interpretable when you bracket
it between two reference clouds whose tree-likeness you already know. We use two anchors,
both computed with the identical procedure (same quadruple count, same normalisation, same
distance):

1. A **matched-covariance Gaussian cloud**. We draw points from a multivariate normal distribution with the same mean and covariance as the real embedding, the same number of points. This is "what the delta would be if the cloud had AION's overall spread and correlations but *no* real branching structure, just a smooth random blob." It is the null: a featureless ellipsoidal cloud. Its median delta is **0.0177** (measured).

2. A **synthetic tree**. We build a point cloud that genuinely branches like a tree and run the same measurement. This is the floor: "what does a real tree score under this estimator at this sample size?" Its median delta is **0.0079** (measured).

These two anchors turn the abstract delta into a ruler. Below the Gaussian (0.0177) means
"more tree-like than a structureless matched blob." Down near the tree (0.0079) would mean
"genuinely tree-like." The real embedding's number lives somewhere on that ruler, and
where it falls is the result.

### 14.3 Ollivier-Ricci curvature: a trustworthy signed local measure

The global delta gives one number for the whole cloud. To find *where* the geometry
branches, and to get a curvature whose sign we can trust, we use **Ollivier-Ricci
curvature** on the neighbour graph.

The idea uses optimal transport, which is worth a plain-words definition. **Optimal
transport** asks: if you have a pile of sand shaped one way and you want to reshape it
into a pile shaped another way, what is the least total "sand times distance moved" to get
from the first shape to the second? That minimum total cost is the **earth-mover
distance** (also called the Wasserstein distance) between the two shapes. It is a way of
measuring how far apart two *distributions* (two clouds of mass) are, accounting for how
far each bit of mass has to travel.

Ollivier-Ricci applies this to a graph edge. Take two points $A$ and $B$ that are linked
by an edge. Around $A$, spread a little probability mass over $A$ and its nearest
neighbours; do the same around $B$. (We keep a fraction $\alpha = 0.5$ of the mass on the
centre point and spread the rest over its $k = 10$ neighbours.) Now ask: what is the
earth-mover distance $W$ between $A$ 's neighbour cloud and $B$ 's neighbour cloud,
compared to the plain distance $d(A,B)$ between the two centres? The Ollivier-Ricci
curvature of the edge is

$$\kappa(A, B) = 1 - \frac{W\big(m_A, m_B\big)}{d(A, B)},$$

where $m_A$ and $m_B$ are the neighbour distributions around $A$ and $B$, and $W$ is the
earth-mover distance between them.

The sign of $\kappa$ is what we read, and it is meaningful:

- **Positive $\kappa$ (clustered).** It costs *less* to move $A$'s neighbourhood onto $B$'s than the centre distance, because the two neighbourhoods overlap heavily. The neighbours of $A$ are also near the neighbours of $B$. This is the clustered, positively curved case: a tight, well-connected local blob. Triangles close up, mass is shared.

- **Negative $\kappa$ (bridge).** It costs *more* to move $A$'s neighbourhood onto $B$'s than the centre distance, because the two neighbourhoods point away from each other into different regions. The edge spans a bottleneck. The neighbours fan apart. This is the saddle, the branch point, the bridge between two otherwise separate parts of the cloud.

So Ollivier-Ricci gives us a signed number per edge, and its sign directly answers "is
this spot clustered or is it a bridge?" We compute it on a 2,000-point subsample (the
optimal-transport solve is expensive) with $k = 10$ neighbours, and summarise the
distribution of edge curvatures: the mean, the fraction negative, and the 5th and 95th
percentiles. This is the *trustworthy signed measure* in our battery.

#### Why Forman-Ricci is only a ranking tool, not a sign

There is a second, much cheaper curvature called **Forman-Ricci** that we computed on the
full neighbour graph (k = 15, augmented with triangle counts). We mention it only to be
complete and to say plainly why we do *not* read its sign. Forman-Ricci is a combinatorial
formula that mostly counts node degrees (how many edges meet at each end of an edge). On a
k-nearest-neighbour graph, every node has roughly the same high degree by construction,
and that degree term dominates the formula. The result is that Forman-Ricci comes out
negative on essentially every edge no matter what the real geometry is: in our run, 99.7%
of edges are "negative" with a mean of -32.5 (measured). That is a structural artefact of
the kNN graph, not a finding about the manifold. So we use Forman-Ricci for one narrow job
only: as a fast score to *rank* which edges are the most bridge-like candidates, to be
confirmed by Ollivier. We never claim "the manifold is negatively curved" from Forman.
Ollivier carries all the sign interpretation.

To keep the three measures and their jobs straight:

| Measure | Scale | What it answers | Sign trustworthy? | Role here |
|---|---|---|---|---|
| Gromov delta-hyperbolicity | global | Is the whole cloud tree-like? | n/a (lower = more tree-like) | headline, read against anchors |
| Ollivier-Ricci | local, per edge | Clustered or bridge, and where? | yes | trustworthy signed measure |
| Forman-Ricci | local, per edge | (degree-dominated) | no | rank bridge candidates only |

### 14.4 What the numbers say

Figure 14 shows both measures.

![Figure 14. Curvature of the AION-1 galaxy embedding: delta-hyperbolicity against anchors, and Ollivier-Ricci local curvature.](figures/14_curvature.png)

Figure 14. Curvature of the AION-1 galaxy-embedding geometry. **Panel A (left): Gromov
delta-hyperbolicity, lower = more tree-like.** The vertical axis is the
diameter-normalised median four-point delta (dimensionless, 0 = a perfect tree, larger =
less tree-like). Each bar is one cloud. The solid dark portion of each bar is the median
delta (the headline statistic), printed above the bar; the lighter portion stacked on top
shows the 95th-percentile upper spread of the per-quadruple deltas, so the full bar height
conveys how heavy the upper tail is. The horizontal dashed line marks the Gaussian-anchor
median, 0.0177, which is the null reference. Left to right: the synthetic tree (validation
floor) at 0.0079; AION E_full under Euclidean distance at 0.0136; AION E_img (image-only)
under Euclidean at 0.0138; the matched-covariance Gaussian anchor at 0.0177; and AION
E_full under cosine distance at 0.0273. What to look for: both AION Euclidean bars
(0.0136, 0.0138) sit just *below* the dashed Gaussian line and well *above* the
synthetic-tree bar. Measured fact: AION is slightly more tree-like than a structureless
matched random cloud, but nowhere near a real tree. The cosine bar (0.0273) is higher
because cosine distance reshapes the cloud; we report Euclidean as primary. **Panel B
(right): Ollivier-Ricci edge curvature on a 2,000-point subsample with k = 10.** The
vertical axis is the signed edge curvature $\kappa$ (dimensionless; positive = locally
clustered, negative = a bridge/saddle). The filled circle is the mean over edges, +0.155,
with a whisker spanning the 5th percentile (+0.006) to the 95th percentile (+0.319) of the
per-edge distribution. The horizontal red line at $\kappa = 0$ is the sign boundary: above
it is positive (clustered), below it is negative (bridge). The annotation reports that
4.2% of edges fall below the red line (the negative-curvature bridge edges). What to look
for: the mean and almost the entire 5th-to-95th-percentile band sit *above* the red zero
line. Measured fact: local curvature is overwhelmingly positive (clustered
neighbourhoods), with only a small minority of edges (about one in twenty-four) being
negative bridges.

Read the two panels together.

**Globally (Panel A), the embedding is barely tree-like.** AION's diameter-normalised
median delta is **0.0136** for the full multimodal embedding E_full and **0.0138** for the
image-only embedding E_img, both under Euclidean distance (measured). The two embeddings
agree to within 1.5%, which is reassuring: the global shape does not depend on whether
redshift and flux were fed in as inputs. Both numbers fall just below the
matched-covariance Gaussian anchor at **0.0177** (measured), so AION is *mildly* more
tree-like than a featureless random cloud with the same spread and correlations. But both
numbers are nearly double the synthetic-tree floor at **0.0079** (measured). On the ruler
from "structureless blob" (0.0177) to "real tree" (0.0079), AION sits close to the blob
end and far from the tree end. The honest reading: there is a faint global tree tendency,
just enough to clear the random null, and nothing like the signature of an actual
branching tree.

**Locally (Panel B), the embedding is mostly clustered, with a few bridges.** The
Ollivier-Ricci mean edge curvature is **+0.155** (measured), firmly positive. The middle
of the distribution, from the 5th percentile (**+0.006**) to the 95th percentile
(**+0.319**), stays on the positive side of zero (measured). Only **4.2%** of edges have
negative curvature (measured): a small population of genuine bridge edges, the places
where the local geometry forks instead of clustering. So the typical neighbourhood is a
tight, positively curved blob, and branching is the exception, concentrated in a thin set
of bridge edges rather than spread everywhere.

### 14.5 What this means, and the honest caveats

Putting the global and local pictures together gives a clean, consistent statement, and we
are careful about which parts are measured and which are interpreted.

**Measured.** Median delta-hyperbolicity 0.0136 (E_full) and 0.0138 (E_img), below the
Gaussian anchor 0.0177 and above the tree floor 0.0079. Ollivier-Ricci mean +0.155, with
4.2% of edges negative and the 5th-to-95th-percentile band running +0.006 to +0.319.
Forman-Ricci negative on 99.7% of edges (a kNN-graph artefact we do not interpret as
curvature).

**Interpreted.** The manifold is mostly *locally positively curved*, meaning
neighbourhoods cluster the way they would on a curved continuous surface rather than
fanning out the way they would on a tree. Branching is real but **weak and localised**: it
lives in the roughly 4% of edges that are negative bridges, not in the bulk. Globally the
cloud is only *mildly* more tree-like than a matched random blob and is nowhere near an
actual tree. So the one-line verdict for this section: **AION-1's galaxy embedding is a
mostly-clustered continuum with weak, localised branching, not a tree.**

This lines up with the physics from Section 2 without overreaching it. A smooth
morphological continuum and a smooth colour-mass sequence would produce exactly this: a
continuous, positively curved body. The small set of negative-curvature bridge edges is
*consistent with* the idea of a few distinct quenching channels meeting at narrow necks,
but consistency is not proof. We measured where the geometry forks; we did not test what
physical process put a fork there. With only about 4% of edges negative, and with the
global delta only barely below the random null, the branching signal is weak enough that
we will not hang any strong evolutionary claim on it. The data shows a continuum with
hints of forks, and that is exactly as far as we take it.

There is one important caveat that this section owns. The Ollivier-Ricci result is
computed on a 2,000-point subsample, because solving an optimal-transport problem at every
edge of the full 48,398-point graph is expensive. A subsample can smear out the rarest,
sharpest bridges (a real bottleneck that only a handful of galaxies pass through might be
undersampled and read as merely flat rather than negative). So the 4.2% negative-edge
fraction is best treated as an estimate of the *typical* prevalence of bridges, not a
complete census of every branch point. The delta-hyperbolicity numbers, by contrast, use
the full sample via a million random quadruples, so the global statement is on firmer
ground than the precise local fraction.

Finally, this curvature result connects directly to the intrinsic-dimension finding in
Section 7, and the two reinforce each other. There we noted a *small global-minus-local
dimension gap*: the nonlinear intrinsic dimension measured at large scale (about 10) sits
close to the linear PCA participation ratio (about 11), and we read that small gap as
evidence of *weak curvature at manifold scale*. The logic is that strong curvature would
make the nonlinear (manifold-following) dimension fall well below the linear
(straight-line) one, because a tightly curled surface needs many linear directions to
contain it but few intrinsic ones to move along it. A small gap means the surface does not
curl much over the scale of the whole cloud. The curvature measurement here is the
independent confirmation of that same conclusion from a completely different method:
mostly positive but *gentle* local curvature (mean +0.155, not a large value), a
barely-tree-like global delta, and only a thin set of negative bridges. Two unrelated
estimators, the ID gap and the direct curvature, both say the manifold is curved but only
weakly so. That is the kind of cross-method agreement that turns a single measurement into
a finding.

The next section moves from how the manifold *bends* to whether it has *holes*: the
topology, the pieces and loops, which closes the geometric description of the cloud.

## 15. Topology: pieces and holes

The last two sections measured how the manifold is shaped.

Section 7 counted how many independent knobs it uses (intrinsic dimension around 10 to
12). Section 14 asked whether it curls or branches (mostly positive curvature, with weak
local branching).

This section asks a coarser but more basic question. Forget the fine shape for a moment.
How many separate pieces is the embedding made of, and does it have any holes in it?

Those two counts have names. They are the first two Betti numbers, written $\beta_0$ and
$\beta_1$.

They come from topology, the branch of math that studies the properties of a shape that
survive any smooth stretching or bending, as long as you do not tear or glue. A coffee mug
and a doughnut are the same to topology because each has exactly one hole; you can mould
one into the other without cutting. The Betti numbers are the simplest such invariants.

In plain words:

- $\beta_0$ is the number of connected pieces. A single blob has $\beta_0 = 1$. Two blobs that never touch have $\beta_0 = 2$. It counts how many parts the shape falls into.
- $\beta_1$ is the number of independent loops, or holes, in the sense of a ring or the hole of a doughnut. A solid disk has $\beta_1 = 0$ (no hole). A ring, an annulus, a circle drawn on paper has $\beta_1 = 1$. A figure-eight has $\beta_1 = 2$.

Why do we care for a galaxy embedding? Because the answer separates two very different
pictures of what the model learned.

If the embedding broke into two clean pieces ($\beta_0 = 2$), that would say the model
sees galaxies as belonging to two genuinely disconnected families, with a true gap between
them and no objects in the middle. A natural guess would be the red sequence and the blue
cloud (Section 2) sitting as two islands.

If instead $\beta_0 = 1$, the model sees one continuous body where you can walk from any
galaxy to any other without ever leaving the cloud, even if the density thins out in
places.

And $\beta_1$ asks a second question. Does that one body have a ring-shaped hole, some
forbidden region in the middle that real galaxies avoid while the cloud wraps all the way
around it?

Both questions have crisp, falsifiable answers, and we measured them.

### What persistent homology is

The tool for measuring holes from a finite set of points is persistent homology. The idea
is simple to picture.

You have a scatter of points sampled from some unknown shape. Put a ball of radius $r$
around every point. Start with $r = 0$, then grow $r$ slowly.

At $r = 0$ every point is its own island, so you have as many pieces as points. As the
balls grow they start to overlap and merge, and the number of pieces drops. At some larger
$r$ a ring of overlapping balls can close up and surround an empty middle, which creates a
loop. Grow $r$ more and the loop fills in and dies.

Persistent homology records, for every topological feature, the radius at which it is born
and the radius at which it dies. The lifetime, death minus birth, is called the
persistence.

A feature that is born early and dies late (long persistence) reflects real structure in
the shape. A feature that is born and dies almost immediately (short persistence) is the
kind of accidental loop you always get from a finite, noisy sample; it is statistical
noise, not geometry.

So the rule of persistent homology is plain: long bars are real, short bars are noise. We
count only features whose persistence clears a threshold.

It helps to know that short noise loops are not a sign that something went wrong. They are
guaranteed. Any finite sample of points, even one drawn from a perfectly solid shape with
no holes at all, produces thousands of tiny transient loops as the balls grow, simply
because the points are scattered unevenly. The job of persistence is to separate those
expected, short-lived loops from the rare long-lived one that would mark a true hole. A
method that reported every loop as real would call every dataset a sieve.

This is the honest way to read topology from data, because it never forces a single scale
on you. It watches the whole movie of growing balls and keeps the features that survive a
wide range of scales.

### $\beta_0$: how many pieces (exact, on the full sample)

For the piece count we do not need subsampling or a persistence threshold. We can compute
it directly on all 48,398 galaxies, and we did, two ways, one cross-checking the other.

First, the nearest-neighbour graph. We connected each galaxy to its $k = 15$ nearest
neighbours (in the z-scored space, the standardised coordinates defined in Section 4) and
asked how many connected components that graph has, meaning how many groups of galaxies
you can reach from one another by hopping along edges.

The result is a single number: knn_components $= 1$ (measured). Every galaxy is reachable
from every other. There is no second island.

Second, the minimum spanning tree, or MST. The MST is the cheapest possible set of links
that still ties all $N$ points into one connected web. Take the $N$ points, and add edges
(each edge weighted by the distance between its two endpoints) one at a time, cheapest
first, skipping any edge that would make a loop, until everything is joined. The result is
a tree (no loops) with $N - 1$ edges and the smallest possible total length.

The MST is useful for $\beta_0$ because of a clean fact. If you cut the single longest
edge of the MST, the tree falls into exactly two pieces, and those two pieces are the same
two clusters you would get from single-linkage clustering at that distance. Cutting the
next-longest edge splits off a third piece, and so on.

So the sorted list of the longest MST edges is a direct readout of how the cloud wants to
break apart. The sizes of the resulting pieces tell you whether a cut found a real
division or just lopped off a stray point.

This is the test that matters. A real $\beta_0 = 2$ split (two genuine families) would cut
the cloud into two large, comparable pieces. An outlier (one weird galaxy, an artefact, a
star) would cut off a piece of size one. Those two outcomes look completely different.

Here is what we measured. The longest MST edges have lengths 65.0, 61.3, 60.9, 58.9, and
on down. Cutting them in order gives these piece sizes:

| Cuts | Resulting component sizes |
|------|---------------------------|
| 0 | [48,398] |
| 1 | [48,397, 1] |
| 2 | [48,396, 1, 1] |
| 3 | [48,395, 1, 1, 1] |

Read the table. After one cut, the cloud is one giant component of 48,397 galaxies plus a
single isolated point. After two cuts, the giant has 48,396 and there are two lone points.
After three cuts, 48,395 and three lone points.

Every cut peels off exactly one outlier and leaves the bulk untouched. The giant component
never splits into two comparable halves. This is the signature of one connected body with
a few far-flung stragglers, not of two families.

Notice the edge lengths themselves. The longest edge is 65.0 and the next ones are 61.3,
60.9, 58.9, with no sudden jump that would mark a true seam between two clusters. The gaps
between consecutive sorted edges are small and irregular, which is what you see when the
longest links are just bridges to lone outliers rather than the single long span that
would join two otherwise-separate populations. There is no one dominant edge whose removal
halves the cloud.

Measured result: $\beta_0 = 1$.

We want to be precise about what this does and does not say.

It says there is no clean topological gap that separates galaxies into disconnected groups
at the resolution of this sample. It does not say the density is uniform.

There can be (and we expect there are) dense regions and thin regions inside the one body.
The red sequence and blue cloud can be two density peaks joined by a sparser bridge, the
way two hills are joined by a valley without the land ever stopping. Topology counts
pieces, not peaks.

The density story is the diffusion-map story (Section 8), and the two agree: a single
connected body with smooth internal variation.

### $\beta_1$: how many holes (subsampled, three metrics)

The loop count is harder to compute, and here we made a deliberate trade-off that the
reader should know about.

Full persistent homology of loops ($\beta_1$) on tens of thousands of points is expensive,
because the computation grows steeply with the number of points. The honest fix is to work
on random subsamples and require any real loop to show up again and again.

So instead of one expensive run, we did 10 independent runs, each on a fresh random
subsample of 2,000 galaxies, and asked which loops recur. A genuine ring in the manifold
should appear in subsample after subsample. A loop that shows up once and never again is
sampling noise.

We also ran each subsample under three different distance metrics, because a real hole
should be a hole no matter how you measure distance, while a noise loop is an artefact of
one particular metric:

- Euclidean: ordinary straight-line distance in the standardised space (our control).
- Fermat: a density-weighted geodesic, where the distance between two points is the shortest path that prefers to travel through dense regions, which makes it outlier-resistant and is our primary metric for geometry (Section 5). Fermat has a published guarantee that it converges to the right underlying geometry as the sample grows, which is exactly the property you want for a topology claim.
- Diffusion: a distance that averages over all paths through the neighbour graph (Section 8), so it is smooth and density-aware in a different way.

All distances are diameter-normalised, meaning we divide every distance by the largest
distance in that subsample so the cloud always has diameter 1. That puts the persistence
on a fixed 0-to-1 scale across runs and metrics, which is what lets us set one threshold
for everybody.

We call a loop significant if its persistence exceeds 0.1, that is, if it survives across
at least a tenth of the full diameter of the cloud. A persistence of 0.1 is already
generous; a real doughnut hole would persist over a large fraction of the diameter.

Here is what we measured, averaged over the 10 subsamples per metric:

| Metric | Mean short noise bars per run | Max loop persistence | Mean significant loops ($>0.1$) | Range over 10 runs |
|--------|------------------------------:|---------------------:|--------------------------------:|:------------------:|
| Euclidean | 2,174.8 | 0.105 | 0.1 | 0 to 1 |
| Fermat | 2,516.5 | 0.138 | 0.2 | 0 to 1 |
| Diffusion | 1,094.1 | 0.081 | 0.0 | 0 to 0 |

Read the table carefully, because it has two faces.

The middle column shows thousands of loop bars per subsample (around 2,175 under
Euclidean, 2,517 under Fermat, 1,094 under diffusion). That sounds like a lot of holes. It
is not. Those are all short bars, born and dead almost at the same radius, exactly the
accidental loops a finite point cloud always produces.

The differences between those counts are themselves a small consistency check rather than
a signal. Diffusion produces the fewest noise bars (1,094) because it is the smoothest
metric: averaging over many paths blurs the fine-grained gaps between points that would
otherwise spawn transient loops. Fermat produces the most (2,517) because its
density-weighted paths sharpen local structure. None of that bears on whether a real hole
exists; it only sets the size of the noise floor each metric carries. The point that
matters is the same in all three columns: every one of those thousands of bars is short.

The honest column is the last three. The largest persistence any loop ever reached was
0.105 (Euclidean), 0.138 (Fermat), and 0.081 (diffusion). Two of those three barely clear
the 0.1 line, and one (diffusion) never reaches it at all.

And the count of significant loops, averaged over 10 runs, is 0.1, 0.2, and 0.0. In plain
numbers: across all 30 runs (3 metrics times 10 subsamples), a loop crossed the threshold
a small handful of times at most, never more than once in any single run, and never once
in the diffusion metric.

That is the definition of a non-result for holes. A real ring would recur in nearly every
subsample and survive every metric. What we see instead is a rare, metric-dependent,
never-repeating blip that sits right at the threshold.

We read this as noise that happens to clip the line, not as a hole. Measured result:
$\beta_1 = 0$.

### Why two different method choices

The two counts are measured very differently, and the reason is a deliberate honesty
choice, not a shortcut.

The piece count $\beta_0$ is cheap. Connected components and a minimum spanning tree both
scale gently with the number of points, so we ran them on the full 48,398 galaxies with no
approximation. That is why we can state $\beta_0 = 1$ flatly: it is an exact property of
the whole sample, confirmed by two independent constructions.

The loop count $\beta_1$ is expensive. The computation of one-dimensional persistent
homology grows fast enough that running it on the full sample is impractical, so some form
of subsampling is forced on us. The question is how to subsample without fooling
ourselves.

Our answer was to demand recurrence and metric-invariance. Ten independent draws of 2,000
points each give ten independent chances for a real loop to appear; three metrics give
three independent ways for it to survive. A loop that is genuine should pass nearly all 30
of those checks. A loop that is noise should pass almost none and never repeat. By
reporting the recurring-loop count rather than a single run, we turned an unavoidable
approximation into a confidence statement: not "there are no loops" but "no loop survives
our recurrence-and-metric test, and the strongest candidate barely touches the threshold
once". That is the most we can honestly claim at this sample size, and it is what the
numbers support.

### The figure

![Figure 15. Topology of the AION-1 manifold: one connected component and no robust loops.](figures/15_topology.png)

Figure 15 shows the two measurements side by side.

**Panel A (left), the piece count $\beta_0$. ** The vertical axis lists four states of the
cloud, from top to bottom: the uncut nearest-neighbour graph, then the cloud after 1, 2,
and 3 of the longest MST edges have been cut. The horizontal axis is component size in
number of galaxies, from 0 to about 50,000.

Each row is a stacked horizontal bar. The blue segment is the giant component (its size
labelled inside, "giant: 48,398" down to "giant: 48,395"), and the thin red segment on the
right is the outliers that each cut peels off, labelled "+1 x size-1 outlier", "+2 x
size-1 outliers", "+3 x size-1 outliers". The outlier fragments are drawn at minimum width
so you can see them at all; their true size is 1 galaxy each.

What to look for in Panel A: the blue giant bar barely changes length as you go down the
rows, while the only thing the cuts add is one more tiny red sliver each time. The cloud
is not splitting into two comparable halves; it is shedding single points. That is the
visual proof of $\beta_0 = 1$ (measured), one connected continuum, not a red-versus-blue
$\beta_0 = 2$ split.

**Panel B (right), the loop count $\beta_1$. ** The vertical axis is the maximum loop
persistence found in a subsample, diameter-normalised so it runs on the same 0-to-1 scale
for every metric (the panel shows roughly 0.06 to 0.18). The horizontal axis is the three
metrics: Euclidean, Fermat, diffusion.

Each coloured bar is the single longest-lived loop seen across the 10 subsamples for that
metric, with its height labelled on top: 0.105 (Euclidean, blue), 0.138 (Fermat, purple),
0.081 (diffusion, green). The red dashed horizontal line at 0.1 is the significance
threshold; any bar reaching above it is a loop we would have to take seriously.

The grey annotation inside each bar records two things: how many short noise bars that
metric produced per subsample (around 2,175, 2,517, and 1,094) and the recurring-loop
count across the 10 runs (0.1/10, 0.2/10, and 0/10, each in the range 0 to 1).

What to look for in Panel B: even the tallest bar (Fermat, 0.138) is only barely above the
line, the diffusion bar sits below it, and the recurring-loop counts are essentially zero
everywhere. No loop is both tall and repeatable. That is the visual proof of $\beta_1 = 0$
(measured): the thousands of noise bars never condense into a single hole that survives
resampling or a change of metric.

### What it means, and the honest caveats

Put the two numbers together. $\beta_0 = 1$ and $\beta_1 = 0$ describe a simply-connected
continuum: one piece, no holes.

Topologically, the AION-1 galaxy embedding is a single solid body, like a filled lump of
clay rather than a doughnut, a chain of beads, or two separate stones. You can travel
continuously from any galaxy to any other, and you never have to go around an empty ring
to do it.

This is not an isolated finding; it is the same conclusion the diffusion map reached by a
completely different route (Section 8). There, the eigenvalue spectrum decayed smoothly
with no dominant gap, and the cloud had a single connected component.

A spectral gap would have meant discrete clusters; its absence meant one continuous body.
Here, persistent homology says one piece and no holes. Two methods, one with a random-walk
operator and one with growing balls, agree.

When independent tools agree, the reading is on firmer ground. Both point at a continuum,
which matches the physics laid out in Section 2: morphology is a sequence, not a set of
boxes, and colour-mass structure is two density peaks joined by a sparse green valley, not
two disconnected worlds.

This connection deserves a beat, because it is the single cleanest agreement between
geometry and known physics in the whole report. Astronomers have long described galaxies
as living on a continuum: the morphological sequence runs smoothly from smooth ellipticals
through lenticulars to spirals, and the colour-mass diagram has a thinly populated green
valley rather than an empty void between red and blue. If AION-1 had learned a faithful
internal map of galaxies, that map should also be one continuous body with no hard breaks.
It is. The model was never told this; it learned to fill in masked tokens, with no
morphology labels and no instruction to make its space connected. The continuum is an
emergent property of the representation, and topology is the tool that confirms it without
any reference to the labels.

We are careful not to overstate it. Topology being a continuum is consistent with the
physics; it does not prove the model understands the physics, and it cannot tell apart a
faithful continuum from any other continuous arrangement of the same points. What it does
rule out is the alternative we worried about at the start: that the model carved galaxies
into disconnected discrete classes with gaps between them. It did not.

Before the caveats, one point about falsifiability. Both of these tests could have come
out the other way, and we would have reported that. A single dominant MST edge whose
removal split the cloud into two large halves would have given $\beta_0 = 2$ and a
different story entirely. A loop that recurred across most subsamples and survived all
three metrics would have given $\beta_1 = 1$ and forced us to ask what physical region
galaxies were avoiding. Neither happened. The continuum reading is what the measurements
returned, not an assumption we built in.

Now the caveats this section owns, stated plainly.

The $\beta_0$ result is strong. It is exact on the full sample, computed two ways, and the
MST cut sizes are unambiguous: the giant stays whole and only single points peel off. We
are confident there is no clean two-family split at this resolution.

The $\beta_1$ result is a confidence-set result, not an exhaustive proof. We did not
compute loop homology on all 48,398 points at once; we used 10 subsamples of 2,000 and
required recurrence across them and across three metrics.

This is the standard and honest way to bound $\beta_1$ at this scale, but it has a real
limit. A hole that is small relative to the cloud, or a hole that only appears at a
density we under-sample with 2,000 points, could in principle hide below our detection.

So the careful statement is this: we find no evidence of any persistent loop, across three
metrics and ten subsamples, and the strongest candidate barely touches the 0.1 line and
never repeats. That is good evidence for $\beta_1 = 0$, not a theorem. Our data cannot
rule out a hole much finer than the 2,000-point resolution can see.

One more honest point about the metrics. The three metrics disagree slightly on the
tallest noise bar (Fermat reaches 0.138, diffusion only 0.081), and that disagreement is
itself informative.

If there were a real loop, the three metrics should agree that it is there. They do not
agree, which is exactly what you expect when the tallest bar in each case is just the
luckiest noise loop for that particular metric. The lack of cross-metric agreement is part
of why we read $\beta_1 = 0$ rather than $\beta_1 = 1$.

So the topology is settled in the direction the rest of the report keeps finding: one
smooth, simply-connected body.

The next section takes that one body and asks a sharper question within it. If the whole
cloud is a single continuum, do different kinds of galaxies, passive and star-forming,
still sit on sub-sheets of different intrinsic dimension inside that continuum?

## 16. Heterogeneity: different types, different dimension?

So far we have treated the embedding as one object and asked how many knobs describe it as
a whole. Section 6 and Section 7 (intrinsic dimension) gave a single answer for the full
cloud: about 10 to 12 effective dimensions at manifold scale. But a single number for
everything can hide structure. Galaxies are not one kind of thing. There are red, passive,
mostly-spheroidal systems that long ago stopped forming stars, and there are blue,
star-forming, mostly-disky systems that are still building themselves. Section 2
(background) called this the colour-mass bimodality: a red sequence and a blue cloud with
a thin green valley between them. The natural question for this section is narrow and
concrete. Do those two families sit on sub-manifolds of different intrinsic dimension? In
plain words: does one family need more independent knobs to describe than the other?

Intrinsic dimension here means the same thing it meant in Section 6: the number of
independent coordinates you would need to describe a galaxy's embedding vector if you
could lay the data out flat, which can be far below the ambient 1024. We are now asking
whether that count differs between two physically defined groups.

### Splitting the sample by colour, the data-driven way

We need a clean rule to call a galaxy "passive" or "star-forming". The obvious physical
axis is colour. We use $g - r$, the difference in brightness (in magnitudes) between the
$g$ band (greener, more sensitive to young hot stars) and the $r$ band (redder, more
sensitive to old cool stars). A large $g - r$ means the galaxy is red, which usually means
old and passive. A small $g - r$ means blue, which usually means actively forming stars.
Magnitudes run backwards: a larger magnitude is fainter, so a larger $g - r$ means the
galaxy is comparatively faint in the blue, hence red.

We did not pick the dividing line by hand. Picking a threshold by eye would bake in our
prior and make any later claim circular. Instead we fit a two-component Gaussian mixture
model (GMM) to the one-dimensional distribution of $g - r$ across the sample. A Gaussian
mixture assumes the data are drawn from a small number of bell curves (here two) of
unknown centre, width, and weight, and it finds the centres and widths that best explain
the histogram. The fit returns two component means (measured): $0.78$ and $1.24$ in
$g - r$ magnitude. The lower mean is the blue cloud, the higher mean is the red sequence.
The crossover point where a galaxy becomes more likely to belong to the red component than
the blue one falls at $g - r = 1.012$ (measured). We use that crossover as the cut.
Galaxies with $g - r > 1.012$ are labelled passive (red); galaxies below it are
star-forming (blue).

This split is honest in one specific way: the line comes from the shape of the colour
distribution itself, not from our expectation. The bimodality is genuinely there in the
data (two separated bumps), so a two-bump model is the right tool, and the valley between
the bumps is where the cut lands.

The resulting groups, as used for the dimension estimate, are passive $n = 22{,}614$ and
star-forming $n = 25{,}782$ (measured). The two subsamples are close in size, which
matters: intrinsic-dimension estimators are sensitive to sample size, so comparing two
groups of similar $n$ keeps that nuisance roughly matched and lets the comparison be about
geometry rather than about how many points we threw at the estimator.

### Why we trust the difference, not the absolute numbers

We estimated the intrinsic dimension of each group separately with the TwoNN estimator
from Section 6. As a one-line reminder: TwoNN looks at each point's first and second
nearest neighbours, forms the ratio $\mu = r_2 / r_1$ of those two distances, and uses the
fact that for a $d$ -dimensional manifold the distribution of $\mu$ follows a fixed law,
which it inverts to a maximum-likelihood estimate of $d$. It is a small-scale estimator:
it reads dimension off the very closest pair of neighbours.

That small scale is exactly why we are careful here. In Section 7 we showed that the
small-scale TwoNN reading on the full AION cloud (about $16.5$) sits well above the
large-scale plateau (about $10$ to $12$) that three independent estimators agree on. The
reason is that the first and second neighbour distances pick up sampling noise, and noise
inflates the apparent dimension. So a TwoNN number read at the smallest scale is not the
honest absolute intrinsic dimension. It is biased high.

The per-group TwoNN values below carry that same upward bias. Passive reads $17.19$ and
star-forming reads $15.53$, both above the true manifold-scale ID of the full cloud. We do
not claim either of those is the real dimension of that sub-manifold. We never report them
as absolute IDs.

What we do trust is the difference between the two groups. Here is the logic. Both
estimates were made the same way, at the same scale, with the same estimator, on
subsamples of similar size drawn from the same embedding. Whatever upward bias the
small-scale TwoNN carries, it is largely shared by both groups, so it cancels when we
subtract. The relative quantity

$$\Delta\mathrm{ID} = \mathrm{ID}_{\text{passive}} - \mathrm{ID}_{\text{star-forming}}$$

is far more reliable than either term alone, because the common-mode error drops out. This
is the same trick used throughout experimental physics: when an absolute measurement is
contaminated by a bias you cannot fully remove, you measure a difference where the bias is
common to both arms and cancels. The price is that we can only speak about which group is
higher, not by how much in true-dimension units.

### The result

![Figure 16. Colour split and stratified intrinsic dimension.](figures/16_stratified_id.png)

Figure 16 has two panels. Panel A (left) is the histogram of $g - r$ colour, in magnitudes
on the horizontal axis, with the number of galaxies per colour bin on the vertical axis.
The blue bars are galaxies the cut assigns to the star-forming group; the red bars are the
passive group. The three vertical lines are the outputs of the Gaussian-mixture fit: the
dashed black line is the GMM cut at $g - r = 1.012$ (the divider), the blue dotted line
marks the star-forming component mean at $0.78$, and the red dotted line marks the passive
component mean at $1.24$. Look at the overall shape: one broad distribution with a clear
shoulder, which the two-component model resolves into a blue bump centred near $0.78$ and
a red bump centred near $1.24$. The cut sits in the dip between them. (A note on
bookkeeping: the per-group counts printed in panel A come from a slightly different pass
than the dimension estimate and read $25{,}099$ and $23{,}297$; the counts that go with
the dimension numbers, and that we quote as measured, are $25{,}782$ star-forming and
$22{,}614$ passive. The small difference does not change the result.)

Panel B (right) is the measurement. The vertical axis is the TwoNN intrinsic-dimension
estimate (the maximum-likelihood value). The two points are the two groups: star-forming
on the left, passive on the right. The error bars are 95% confidence intervals from the
estimator's Gamma posterior. The measured values are:

| Group | $n$ | TwoNN ID (MLE) | 95% interval |
|---|---|---|---|
| Star-forming ($g - r <$ cut) | 25,782 | 15.53 | [15.39, 15.73] |
| Passive ($g - r >$ cut) | 22,614 | 17.19 | [17.01, 17.43] |

The passive point sits clearly above the star-forming point, and the two intervals do not
come close to overlapping. The annotation in the panel reports the difference directly:
$\Delta\mathrm{ID} = 1.66$ with a 95% interval of $[1.38, 1.96]$ (measured). That interval
excludes zero. Excluding zero is the formal statement that the gap is unlikely to be an
accident of sampling: if the two groups truly had the same intrinsic dimension, we would
not expect to see a difference this large with an interval sitting entirely on the
positive side.

So the measured finding is simple. Passive galaxies sit on a slightly higher-dimensional
sub-manifold of the embedding than star-forming galaxies do, by about one and a half
effective dimensions in this biased-but-shared estimate. The footnote printed on the
figure says the same thing we have been careful to say in prose: the absolute IDs (around
15 to 17) are small-scale, first-and-second-neighbour numbers and are noise-inflated; the
relative $\Delta\mathrm{ID}$ is the signal.

### What it might mean, and what it cannot

We read this (interpreted, not measured) as the model needing modestly more independent
directions to describe the red, passive population than the blue, star-forming one. A
plausible physical story is that the passive class is a mixed bag: galaxies arrive at "red
and dead" through more than one route and over a wide range of stellar masses, ages, and
structural shapes, so the family spreads across more axes. The star-forming population,
dominated by disks that are still building stars along a fairly tight scaling relation,
may be more constrained and so thinner. That is a reasonable reading. It is not proven by
this measurement. The data can tell us the dimensions differ; it cannot tell us the cause.

There is also a result in the literature that points the other way, and we report it
plainly. Cadiou et al. (2025) studied the intrinsic dimension of galaxy populations and
found the opposite ordering, with the star-forming side higher, when working from raw
photometry. We are not measuring the same object. Their input is photometry directly; ours
is a learned, multimodal AION-1 embedding that has folded images, fluxes, and redshift
into 1024 numbers and reshaped the geometry in the process. A foundation-model
representation can and does redistribute variance across its axes in ways raw photometry
does not (we saw exactly this in Section 5, where the intrinsic metrics give very
different distance contrast than raw Euclidean). So our finding and theirs are not a clean
head-to-head contradiction; they are two different representations of overlapping physics,
and they happen to order the two populations differently. The honest statement is that the
ordering of intrinsic dimension between passive and star-forming galaxies depends on the
representation you measure, and in the AION-1 embedding the passive side reads higher.

One last caveat that belongs to this section. The split is on colour alone, and colour and
star-formation rate are tightly linked but not identical (dust can redden a star-forming
galaxy; a green-valley object straddles the line). The cut is a clean statistical divider,
not a perfect physical one, so a small number of galaxies near $g - r = 1.012$ are
assigned to the group they least resemble. With $\Delta\mathrm{ID} = 1.66$ and an interval
that clears zero by a comfortable margin, a little contamination at the boundary does not
erase the effect, but it does mean the true dimensional gap between cleanly passive and
cleanly star-forming galaxies could be somewhat larger than the one we measure across a
colour line.

The signal here is real, modest, and slightly surprising. We carry it into the synthesis
in Section 19 as the one piece of evidence that the embedding is not perfectly uniform in
dimension across the populations it contains.

## 17. The geometry of concepts: an honest null

Every other arm of this report measured something positive. This one did not, and we
report it anyway, because a negative result you actually ran is more informative than a
positive result you wished into existence.

The question is about how concepts sit inside the embedding. Throughout the report we have
treated a "concept" as a direction in the 1024-dimensional space: a single unit vector
such that moving along it changes one human-named property (Section 10 found these as
ridge-probe weight vectors, Section 13 as sparse-autoencoder decoder vectors). A natural
next idea, from Park et al. and the broader work on the linear geometry of concepts in
large models, is that concepts might be organised in a hierarchy that you can read off the
directions themselves.

### The hierarchy idea, in plain words

Some concepts are children of others. "Spiral" is a kind of "featured" galaxy: a galaxy
cannot be spiral without first being featured (having visible structure rather than being
a smooth blob). Featured is the parent, spiral is the child. The Park-style claim is that
this parent-child relationship should show up geometrically in a specific way: the child's
direction should equal the parent's direction plus an extra piece that is orthogonal to it
and carries only the child-specific information.

Write $\mathbf{u}_{\text{featured}}$ for the parent direction and
$\mathbf{u}_{\text{spiral}}$ for the child direction, each a unit vector. The
clean-hierarchy prediction is

$$\mathbf{u}_{\text{spiral}} = \alpha\,\mathbf{u}_{\text{featured}} + \mathbf{r},\qquad \mathbf{r}\perp\mathbf{u}_{\text{featured}},$$

where $\alpha$ is some positive weight and $\mathbf{r}$ is the spiral-specific residual
that is at right angles to the featured direction. In words: to become "spiral" you first
move in the "featured" direction (because spirals are featured), then add a sideways step
that is pure spiral and has nothing to do with featuredness. If this holds, the embedding
has literally encoded the taxonomy in its geometry: the child is built out of the parent
plus an independent correction.

The thing you measure to test this is the alignment between the parent direction and the
spiral residual, written as the cosine of the angle between them. Cosine of an angle
between two unit vectors runs from $-1$ (pointing opposite) through $0$ (exactly
perpendicular) to $+1$ (pointing the same way). Under a clean hierarchy you want the
residual, after you have already accounted for the parent, to still carry a leftover trace
of the parent that you can detect, which shows up as a small but real positive cosine
between the featured direction and the parent-component recovered from the spiral
direction. A cosine indistinguishable from what you would get by chance means the geometry
does not encode the hierarchy.

### Fixing a test that was cheating

Our first version of this test was tautological, and we corrected it. The earlier
construction effectively built the spiral residual out of the featured direction and then
measured how much of the featured direction was in it, which is circular: by construction
the answer had to be positive, so it proved nothing. We caught this and replaced the
baseline.

The fix is a permutation null. A null is the distribution of the test statistic you would
see if there were no real effect, and a permutation null builds that distribution by
destroying the structure you are testing for while keeping everything else fixed. Here we
shuffle the labels that define the concepts (so the "spiral" and "featured" assignments
are scrambled relative to the galaxies), recompute the directions and the parent-residual
cosine from scratch many times, and collect the cosines those scrambled runs produce. That
gives us the range of cosine values consistent with no hierarchy at all. The real,
unshuffled cosine is significant only if it lands above almost all of the shuffled values.
We use the 95th percentile of the null as the threshold: the real value must beat the
cosine that 95% of shuffled runs fall below.

This is the same null logic we used for the sparse-autoencoder alignment thresholds in
Section 13, and for the same reason: a test of "is this direction real" is only honest
against a baseline of "what would a fake direction score".

### The result

The measured alignment is

$$\cos\big(\mathbf{u}_{\text{featured}},\,\text{spiral-residual}\big) = 0.072,$$

and the permutation null's 95th percentile is $0.084$ (both measured). Since
$0.072 < 0.084$, the observed value falls below the threshold. It is not significant. The
hierarchy signal we measured is weaker than what shuffled, structure-free data produces 5%
of the time by chance.

So the honest finding is a null: our data does not establish a clean linear
featured-to-spiral hierarchy in the AION-1 embedding. We do not say the hierarchy is
absent. We say we could not detect it, which is a different and weaker claim. The value
$0.072$ is not zero, and it is not far below the $0.084$ line, so the test is consistent
with a small real effect that we simply lack the resolution to confirm, and it is equally
consistent with no effect at all. The data cannot decide between those, and we will not
pretend it can.

### Why this null is fine to report, and what limits it

Two reasons this belongs in the report rather than in a drawer. First, negative results
matter: a reader deciding whether to build on the linear-concept-hierarchy idea in galaxy
embeddings should know that our one direct test of it came back empty. Reporting only the
arms that worked would give a falsely tidy picture of a model that is, in most respects,
very legible.

Second, the test is genuinely exploratory and its weakness is partly a sample problem we
can name. The spiral and bar labels exist only for featured, non-edge-on disk galaxies,
because that is where the Galaxy-Zoo decision tree even asks the spiral-arm question
(Section 4 explained this conditionality). That restricts the spiral sample to a few
thousand galaxies (the spiral-fraction probe in Section 10 ran on $n = 3{,}034$), and the
spiral probe itself was the weakest in the whole probe battery ($R^2 = 0.252$). A concept
direction estimated from a small, noisy label is itself noisy, and a noisy child direction
makes the residual geometry hard to read. With a sharper spiral label on more galaxies,
the cosine might rise above the null, or it might not. This test does not settle it.

There is a cleaner way to read this null against the rest of the report. The model clearly
carries strong, separable concept directions: colours and morphology decode with high
$R^2$ (Section 10), the probe directions are more orthogonal than the label correlations
force (Section 11, disentanglement), and the sparse autoencoder finds hundreds of
seed-stable, physics-aligned features (Section 13). What this section adds is a boundary
on those claims. The model encoding a concept as a usable direction is one thing, and we
have strong evidence for it. The model arranging its concepts into a specific linear
parent-child algebra is a stronger, more particular claim, and for the one hierarchy we
tested, the evidence is not there. Strong concept directions, no demonstrated concept
hierarchy. That is the honest line, and we carry it forward as a limit on how far the
geometry-of-concepts reading should be pushed.

## 18. The manifold made visible

Every result so far has been a number: a dimension, a correlation, a curvature sign.
Numbers are the honest currency, but they hide the thing a reader of an astronomy paper
most wants to see, which is the galaxies themselves. This short section closes that gap.
We pull real image cutouts from the sample, lay them out along two of the axes the model
discovered on its own, and let the reader check by eye whether those axes mean what the
correlations say they mean.

The two axes we walk along are the ones Section 9 (reading physics off the manifold)
flagged as the physics-bearing directions. The first is plain colour, the $g-r$ index (the
difference between brightness in the green $g$ band and the red $r$ band, in magnitudes;
larger means redder). The second is the first nontrivial diffusion coordinate, dc1, which
Section 8 (diffusion maps) built as a slow-mixing direction of the random walk over the
embedding and which correlates with morphology and redshift (measured Spearman: smooth
$+0.50$, featured $-0.47$, redshift $+0.45$). Colour is a label we measured directly. The
diffusion coordinate is a direction the model laid out without ever being told about
morphology. Showing both side by side tests two different claims with the same kind of
picture.

### How the galaxies were picked: representative, not curated

The sampling rule matters, because it's the difference between an honest figure and a
cherry-picked one. We did not hand-select pretty examples. For each axis we sorted all
galaxies by their value on that axis, then sampled at nine evenly spaced percentiles, from
the 5th percentile up to the 95th in equal steps. A percentile is just a rank position:
the 5th-percentile galaxy is the one that 5% of the sample falls below on that axis, the
50th is the median, the 95th is near the top. Walking the percentiles means each cutout is
a typical resident of its slice of the axis, not an outlier.

We deliberately stopped at the 5th and 95th percentiles rather than the true 0th and
100th. The absolute extremes of any real imaging sample are contaminated: the very bluest,
very faintest, or very most-compact "galaxies" include foreground stars, image artefacts,
deblending failures, and cosmic-ray hits that survived the cuts. Those would make the
montage look dramatic and lie about the typical object. Trimming to the 5th-to-95th range
keeps the walk inside the body of real galaxies. This is a representativeness choice, and
it's the reason the endpoints below are a moderate blue and a moderate red rather than the
most saturated cases the catalogue contains.

### How each cutout is rendered: the z/r/g arcsinh RGB

Each panel is a three-colour composite built from three of the survey bands. We map the
$z$ band (the reddest, longest-wavelength channel here) to the red display channel, the
$r$ band to green, and the $g$ band (the bluest) to blue. So redder galaxies, which are
brighter in $z$ relative to $g$, render visibly redder on screen, and bluer star-forming
galaxies render bluer. The display colour is therefore a faithful stand-in for the
physical $g-r/r-z$ colour, not an arbitrary palette.

The brightness of each channel is passed through an arcsinh (inverse hyperbolic sine)
stretch before display. The reason is dynamic range. A galaxy's bright central pixels can
outshine its faint outer disk by factors of hundreds, so a linear brightness map would
blow out the centre to solid white and bury the spiral arms and low-surface-brightness
features in black. The arcsinh function is close to linear for small values and compresses
large ones, like a logarithm but well-behaved through zero and into faint noise. It lets
one image show both the bright core and the faint structure at once. This is a standard
astronomical display transform; it changes how the light is shown, not what was measured.

### Reading the figure

![Figure 18. Real galaxy cutouts sampled along two discovered manifold axes.](figures/18_manifold_galaxy_montage.png)

Figure 18 has two rows of nine cutouts each. Both rows sweep an axis from left (low value)
to right (high value) at the nine evenly spaced percentiles described above. Every panel
is a $z/r/g$ arcsinh RGB composite, so on-screen red means physically red and on-screen
blue means physically blue. The small title above each cutout gives that galaxy's value on
the row's axis: the top row is titled by $g-r$ colour (in magnitudes), the bottom row by
Galaxy-Zoo smooth vote fraction (the share of human-style votes calling the galaxy
smooth/featureless, between 0 and 1).

The top row is the colour axis, and it does exactly what colour should do (measured).
Reading left to right, the $g-r$ titles climb from $0.46$ to $0.64$, $0.79$, $0.90$,
$1.00$, $1.09$, $1.21$, $1.35$, and $1.58$. The leftmost galaxies are visibly bluer and
more diffuse; the rightmost are visibly redder and more compact, the classic look of
passive early-type galaxies. The colour change you see by eye and the colour change in the
titles move together. That's the trivial direction of the check: colour is a label, so it
had better track itself, and it does, confirming the rendering is faithful.

The bottom row is the more interesting one, because it walks the diffusion coordinate dc1,
an axis the model built with no morphology labels at all. The panels are titled by smooth
fraction, which rises along the row from about $0.62$ at the left to $0.84$ at the right
(with a non-monotone wobble in the middle: the printed values run
$0.62, 0.49, 0.64, 0.68, 0.77, 0.81, 0.74, 0.79, 0.84$). Reading the cutouts left to
right, the early panels show more elongated, more featured, edge-on or disky systems, and
the later panels trend toward rounder, smoother, more concentrated blobs. So moving along
a purely geometric direction of the embedding moves you, on average, from featured to
smooth galaxies. The model's own slow-mixing axis lines up with human morphology
(interpreted, supported here by the eye and by the $+0.50/-0.47$ Spearman correlations
from Section 9).

One honest caveat owns this row. The morphology panel is noticeably noisier than the
colour panel, and the smooth titles are not perfectly monotone. That's expected and not a
flaw in the model. We are sampling at percentiles of a single coordinate, dc1, while
smooth fraction is only one of several properties that coordinate carries (it also
co-varies with redshift, so some of the apparent morphology drift is redshift drift, and
fainter higher-redshift galaxies are intrinsically harder to classify). Real cutouts at a
fixed percentile are a random draw from everything else that varies at that slice, so
individual panels scatter around the trend. The signal here is the direction of the
gradient across the row, not any single pair of neighbours. Treat the bottom row as a
visual sanity check that agrees with the measured correlations, not as a measurement in
its own right.

What the figure buys us is modest but real. It shows that the two axes the geometry
singled out are not statistical artefacts of the labels: they correspond to changes a
human can see in the raw imaging, a redward colour shift along one axis and a
featured-to-smooth shift along the other. That sets up the synthesis, where we argue these
same two gradients, colour and morphology, are the visible face of the single continuous
manifold the rest of the report has been measuring.

## 19. Synthesis: the coherent picture

We measured the same object many ways: counted its independent directions, mapped its
slow-mixing axes, probed how much physics rides on it, asked an autoencoder to name its
own features, took its curvature, counted its pieces and holes, and split it by colour to
see if the dimension changes. Each arm used a different mathematical tool with different
failure modes. The point of this section is to step back and ask whether they tell one
story or several. They tell one. The picture that holds across every arm is a
low-dimensional, single-piece, mostly-positively-curved continuum that strongly encodes
galaxy physics, with only weak localised branching and a small set of candidate concepts
beyond the human labels. No new numbers appear here; every value was reported and
qualified in its own section. What follows is the argument that the parts fit.

### The geometric skeleton: small, simple, smooth

Start with size. The intrinsic dimension is the count of independent knobs you need to
place a galaxy in the embedding, and Section 7 (intrinsic dimension) put it at roughly 10
to 12, far below the 1024 ambient numbers and below the resolution ceiling the sample size
sets. Three estimators that fail in different ways agreed near that plateau. The honest
qualifier is that this sits at or just above the optimistic astrophysical prior of 4 to 10
effective axes from Section 2 (galaxy background), so "low-dimensional" is the right
phrase, but not as low as the best-case prior, and self-supervised encoders are known to
under-use their nominal width, so a small number was partly expected.

Now shape. Section 8 (diffusion maps) found the eigenvalue spectrum decays smoothly with
no dominant gap. A gap would have meant the model split galaxies into discrete kinds; its
absence means the model laid them out as one graded body. Section 15 (topology) made the
same statement two more ways: the number of separate pieces is one (cutting the longest
links peels off only single outliers while the main body of tens of thousands of galaxies
stays whole), and the number of independent loops is zero (rare loops never recur across
subsamples or survive a change of metric). Simply-connected continuum is the joint
reading, and it's the same conclusion the no-gap spectrum reached from a different
direction. Three methods, one answer.

Now curl. Section 14 (curvature) showed the manifold is mostly locally positively curved,
meaning neighbourhoods cluster like patches of a sphere rather than spreading like a
saddle, with only a small fraction of negative-curvature bridge edges, and only a mild
tree tendency versus a matched random cloud. This dovetails with a quieter result from
Section 7: the nonlinear dimension at large scale came out close to the linear PCA
participation ratio, and that small gap between a curved (geodesic) measure and a flat
(linear) one means the manifold doesn't fold sharply at its own scale. Weak curvature at
manifold scale and mostly-positive local curvature are two views of the same gentle
geometry. The body bends, but softly, and it does not branch into a tree.

So the skeleton is consistent: about a dozen knobs, one connected piece with no holes,
gently and mostly positively curved, not tree-like. Each of those four claims came from a
tool that could have said otherwise, and none of them did.

### The physical reading, and where it's earned

A skeleton is just shape until you ask what the axes mean. Here the physics arms take
over. Section 10 (decodability) showed that a single linear map from the embedding
recovers galaxy properties at high accuracy, and the part that carries weight is that
stellar mass and specific star-formation rate are decodable from the image alone,
properties the model was never given as inputs in that set. Sections 8 and 9 then located
that physics on the manifold: the first discovered axis tracks morphology and redshift,
the second tracks colour and star formation, and Section 18 (the montage) let a reader see
those two gradients in raw cutouts, blue-to-red along one and featured-to-smooth along the
other.

Map this onto the astronomy. A continuous, gap-free body is exactly what the morphological
sequence looks like in nature: ellipticals and spirals are not separate species but ends
of a graded series, and the colour-mass diagram is a continuum with a dense red sequence
and a dense blue cloud joined by a thinner green valley, not two disconnected islands. The
model's smooth continuum and its two physics-bearing axes are the same structure
astronomers already draw by hand. This is the strongest part of the synthesis, because
it's supported from both sides: the geometry says continuum, the physics says continuum,
and the cutouts show it. We read the manifold's primary axes as the colour/star-formation
gradient and the morphology/redshift gradient (interpreted, but well anchored by the
measured correlations and the visible cutouts).

Two honest limits sit on this reading. First, redshift decodes well, but the redshift
labels are mostly photometric (estimated from colour), so image-to-redshift partly routes
through image-to-colour; the cleaner evidence that the model encodes physics it wasn't
handed is mass and star-formation rate, not redshift. Second, that the manifold's axes
correlate with these properties does not make the axes identical to them; they are
directions that carry the physics, not exact physical coordinates. We flagged both where
they arose and they bound the strength of the claim without overturning it.

### Where the data is silent: branching and quenching

The most tempting overreach is the branch points. Galaxies stop forming stars (quench)
through more than one route: a fast channel where a disk is transformed into a spheroid,
and a slow channel where star formation fades gradually. A representation that captured
those channels might show negative-curvature saddles where one path forks into two. We did
measure a small fraction of negative-curvature bridge edges and a mild tree tendency, so
the ingredients of weak, localised branching are present in the geometry. The honest
statement is that this is consistent with distinct quenching channels but does not prove
them. We never traced a galaxy along a path, never tied a specific bridge edge to a
specific physical transition, and a static present-day embedding shows a density of
populations, not a movie of evolution (the load-bearing caveat from Section 2). The
geometry permits the quenching-channel story; it cannot, on its own, establish it. Here
the data is silent and we say so.

The heterogeneity result from Section 16 (stratified dimension) is the one genuinely
surprising signal, and it deserves a careful word rather than a headline. Splitting the
sample by colour, passive (red) galaxies sit on a slightly higher-dimensional sub-manifold
than star-forming (blue) ones, and the difference excludes zero. The absolute dimensions
there are small-scale and noise-inflated, so only the relative difference is trustworthy,
and prior work on raw photometry found the opposite ordering. Our representation is a
different, multimodal object, so that's not a clean contradiction, but it does mean we
should hold this as a real, modest, and not-yet-explained effect rather than fit a tidy
physical story to it. Calling it surprising is more honest than calling it understood.

### The concepts, and the candidates we can't yet judge

The last arm asked whether the model organises galaxies by anything beyond the human
taxonomy. Section 12 and Section 13 (the sparse autoencoder) found a large set of
seed-stable directions that align with the known physical labels, which is reassuring: the
model's private vocabulary substantially overlaps the one astronomers use, with colour and
morphology the clearest single concepts and redshift smeared across many weak units rather
than coded in one place. Sitting alongside those are a smaller set of candidate directions
that are stable across training seeds yet not explained by any label we have. These are
the most interesting result in the report and the least certain. They are correlational
only. No causal test was run, nothing was ablated or steered, and a stable-but-unlabelled
direction can be a real new concept or an artefact of the data we simply haven't matched
to a catalogue. We list them as candidates, not discoveries, and the next study has to
test them by intervention before anyone calls them concepts the model invented.

### One object, told consistently

Put the arms in one sentence and they don't fight. AION-1's galaxy embedding is a compact
manifold of about a dozen dimensions, one connected piece with no holes, gently and mostly
positively curved rather than tree-like, whose principal axes carry colour, star
formation, morphology, and redshift strongly enough to read mass and star-formation rate
off the image alone, with weak localised branching that's compatible with but not proof of
distinct quenching channels, a small and surprising dimension gap between passive and
star-forming populations, and a handful of stable unlabelled directions worth a causal
follow-up. Every clause in that sentence is a measurement from a named arm, paired with
the physics it matches and the caveat it carries. The agreement across so many independent
tools is the real result. Not any single number, but the fact that geometry, decodability,
curvature, topology, and concept discovery all describe the same continuous,
physics-bearing body. Where they speak, they agree; where they're silent, we've marked the
silence. The next section turns that silence into a list of specific threats to the claims
we have made.

## 20. Limitations and threats to validity

Section 19 ended on a promise: turn the places where the data is silent into a concrete
list. That's this section. Every result in this report was paired with its own caveat
where it appeared, but a reader deserves them collected in one place, ranked by how badly
each one bounds the claims, and with the reasoning spelled out rather than hand-waved. So
here is the frank version. For each item we say what the limitation is, why it matters
mechanically (not just "it could be wrong" but the specific way it weakens a specific
claim), and whether the affected result is solid or only suggestive. We sort them roughly
from most to least binding.

A note on what "solid" and "suggestive" mean here, because we use the words throughout.
Solid means the measurement is stable under the checks we ran (multiple estimators,
synthetic validation, bootstrap intervals, or recurrence across subsamples) and the
limitation shifts the interpretation at the margin without overturning the result.
Suggestive means the measurement is real as a number but the limitation reaches into the
core of what we'd want to conclude from it, so it should be read as a lead, not a finding.
We try to be honest about which bucket each result falls in, even when the suggestive ones
are the most interesting.

### The redshift labels are mostly photometric, so one decodability number is partly circular

The cleanest threat first, because it touches the headline that "the model encodes physics
from the image alone." Redshift is how far a galaxy's light has been stretched by cosmic
expansion, and it doubles as a distance. We have a true spectroscopic redshift (measured
from spectral lines) for only 6,883 of the 48,398 galaxies; for the rest the redshift
label is a photometric redshift, or photo-z, which is itself an estimate derived from the
galaxy's colours by a separate model. That matters for one specific claim. When we report
(Section 10, decodability) that a linear probe reads redshift off the image-only embedding
at $R^2 = 0.800$ (95% bootstrap interval $[0.792, 0.809]$, measured), part of that success
is the probe learning image-to-colour and then colour-to-photo-z, because the label it's
being scored against was built from colour in the first place. So image-to-redshift is
partly a loop back to a colour-derived quantity, not a pure inference of physical
distance.

Why this doesn't sink the broader claim: the load-bearing evidence that the embedding
carries physics it was never handed is stellar mass at $R^2 = 0.721$ ($[0.659, 0.772]$,
$n=3{,}728$) and specific star-formation rate at $R^2 = 0.760$ ($[0.692, 0.816]$,
$n=4{,}760$), both from the image-only set and both drawn from external catalogues, not
from colour-fit photo-z (measured). Those two are clean. The redshift number is solid as a
number but should be read with the photo-z asterisk attached, and we lean on mass and
sSFR, not redshift, whenever the point is "the model knows something it wasn't told."

### The "alien" SAE candidates are correlational only, with no causal test

This is the most interesting result in the report and the one most exposed by its method.
The sparse autoencoder (Sections 12 and 13) found 59 candidate directions that are stable
across training seeds (their decoder vectors reappear in at least half of the other four
seeds at cosine $\geq 0.6$) and have high novelty (more than 70% of their activation
variance is not linearly explained by any of our six labels), yet align with none of those
labels. We called them alien candidates. The word candidate is doing heavy lifting and
it's deliberate.

The limitation is structural, not a detail we could have tidied up with more compute.
Every claim about these 59 directions is correlational. We never intervened: nothing was
ablated (zeroed out to see what breaks), nothing was steered (pushed up or down to see
what changes in the reconstruction or in a downstream readout). A direction that is
seed-stable and unlabelled can be three different things, and our measurements cannot tell
them apart. It could be a genuine new concept the model uses that our six human labels
simply don't name. It could be a real physical property that exists in some catalogue we
never cross-matched (environment, a structural parameter, an emission-line ratio), in
which case it's "alien" only to our label set, not to astronomy. Or it could be a stable
artefact of the data or the pipeline, for example a detector or survey systematic that
recurs across seeds because it's in the inputs every seed sees. Correlation with stability
rules out random noise; it does not rule out the second and third options. So this result
is suggestive, firmly in that bucket, and the only honest framing is that these are leads
for a causal follow-up, not concepts the model invented. The 335 seed-stable,
label-aligned directions are on firmer ground, because their meaning is pinned by a
measured correlation with a known property (max alignment $0.279$, about 23 times the
label-shuffle 95th-percentile null of $0.0119$, measured); but the aligned set is the
reassuring result, not the surprising one.

### The intrinsic dimension sits at or just above the optimistic prior band

The headline geometric number is an intrinsic dimension of roughly 10 to 12 (Section 7),
where three mechanistically independent estimators agree near the plateau: the large-scale
Gride ratio lands near 10 ($9.89$ for image-only at neighbour rank 256, $10.10$ for the
full set, measured), the local maximum-likelihood estimator settles near 11.4 ($11.16$
image-only at $K=226$), and the linear PCA participation ratio on z-scored data is about
11 ($11.93$ image-only, $11.18$ full). All sit below the resolution ceiling that the
sample size imposes, $\log_2(N) = \log_2(48{,}398) = 15.56$ (measured), so we are not
bumping against an artefact of having too few points.

Two limitations qualify how low "low-dimensional" really is. First, the astrophysical
prior from Section 2 (galaxy background) put the optimistic effective-axis count for an
honest multimodal embedding at roughly 4 to 10, and our 10 to 12 sits at the top of that
band or just over it. So "low-dimensional" is the correct phrase against the ambient 1024,
but the embedding is not as compact as the best-case prior, and a reader should not
picture a 4- or 5-axis object. Second, the small-scale TwoNN estimate of about 16.5
($16.05$ image-only, measured) is not the dimension and we do not report it as such; the
first-to-second-neighbour distance ratio that TwoNN uses picks up sampling noise at the
smallest scale and inflates, which is exactly why we trust the larger-scale plateau
instead. We validated this reasoning on synthetic manifolds of known dimension at the same
$N$ before trusting AION (a sphere of true dimension 5 recovered at TwoNN $4.96$, a swiss
roll of true dimension 2 at $1.95$, measured), and a two-blob control with no manifold
inflated TwoNN to about 212, confirming the estimator's failure mode. There is a deeper
caveat too: self-supervised encoders are known to under-use their nominal width (a
phenomenon sometimes called dimensional collapse), so a smallish intrinsic dimension is
partly expected from the training method and is not, by itself, proof of a clean physical
manifold. The ID result is solid as a measurement (independent estimators, synthetic
validation, tight bootstrap intervals) but the interpretation "very low-dimensional" is
bounded: true against 1024, only modestly below the astro prior.

### The loop count uses 2,000-point subsamples, not the full cloud

The topology result has two halves with very different strength. The number of connected
pieces, $\beta_0 = 1$, is computed on the full 48,398-galaxy sample: single-linkage on the
k-nearest-neighbour graph gives one component, and cutting the longest
minimum-spanning-tree edges peels off only single outliers (sizes $[48{,}397, 1]$ after
one cut, $[48{,}396, 1, 1]$ after two, measured) while the giant body stays whole. That
half is solid and exact.

The number of independent loops or holes, $\beta_1 = 0$, is weaker by construction and we
want to be plain about why. Persistent homology, the tool that counts loops, is expensive:
its cost grows fast with the number of points, so running it on all 48,398 galaxies at
once was not feasible here. Instead we ran it on 10 independent 2,000-point subsamples,
under three metrics (Euclidean, Fermat, diffusion), and asked whether any significant loop
(persistence greater than 0.1, meaning a hole that survives a wide range of scales)
recurs. It essentially never does: the mean count of significant loops per subsample is
0.1 under Euclidean, 0.2 under Fermat, and 0.0 under diffusion, with maximum persistences
of $0.105$, $0.138$, and $0.081$ respectively (measured), all at or barely above the
threshold and none recurring. So the honest statement is a confidence claim, not an
exhaustive one: across 10 random 2,000-point views and three metrics we found no loop that
recurs, which is strong evidence against any large, stable hole but cannot rule out a
fine-scale loop that a 2,000-point subsample is too sparse to resolve. We read
$\beta_1 = 0$ as well-supported and call the cloud simply-connected, while flagging that
it's a subsample-based confidence set rather than a single computation on the whole
manifold. Solid in its conclusion, suggestive in the sense that a sufficiently small hole
could hide below the subsample resolution.

### This is an image-only run with no spectra, a deviation from the full multimodal plan

AION-1 is a multimodal model: it was trained to fuse images, spectra, and catalogue
scalars. The geometry we studied uses two embedding sets, but neither includes spectra.
The full set $E_\text{full}$ fuses the four-band image with broadband photometry (g, r, z
flux) and redshift; the leakage-free set $E_\text{img}$ is image only. Spectra, the fine
wavelength-by-wavelength fingerprint that carries emission-line ratios, velocity
dispersion, and detailed stellar-population information, were not fed in for this run.

Why this bounds the claims: a spectrum encodes physics that a four-band image cannot
resolve, so some axes of the model's full representation are plausibly missing from the
object we measured. If a real concept lives mostly in the spectral channel, our
intrinsic-dimension count would miss it, our diffusion axes would not show it, and the
sparse autoencoder could not find it. So statements like "the embedding uses about a dozen
axes" and "the principal axes are colour/star-formation and morphology/redshift" are true
of the image-plus-photometry representation we analysed, and may understate the dimension
and richness of the full multimodal embedding the model is capable of. This is a scope
limitation rather than an error: the results are valid for what we measured, but the
title's "native representational geometry" should be read as the geometry of the
image-and-photometry embedding, with the spectral dimension left for future work (Section
21 lists it as the first thing a full study should add). We mark this as a
solid-but-scoped result.

### The sample carries a selection function we did not correct for

Every measured manifold is the manifold of the galaxies we put in, and our galaxies are
not a fair slice of the universe. The sample is drawn from Galaxy Zoo DESI, restricted to
redshift $0 < z < 1$, and required usable four-band imaging (96.8% of the 50,000 drawn
galaxies passed, giving $N = 48{,}398$, measured). Galaxy Zoo itself selects objects
bright and large enough on the sky to be classified by volunteers, which biases toward
nearby, luminous, well-resolved galaxies and against faint, distant, or compact ones. The
cross-matched physical subsets (mass, sSFR, Sersic index) add a second selection layer,
because a galaxy only enters that subset if it also appears in an external SED-fit or
structural catalogue, which has its own magnitude and quality cuts.

Why this matters for the geometry: the shape, density, and even the dimension of a
manifold depend on which points populate it. If a whole population (say, very faint dwarfs
or very distant red galaxies) is under-represented, a real branch or a real extra
dimension carried by that population would be thinned out or absent from our cloud, and
we'd report a simpler geometry than the true one. The continuum we found is the continuum
of this selected sample, and we cannot claim it's the continuum of all galaxies at
$0 < z < 1$. We did not reweight or model the selection function, so this caveat is
uncorrected. It bounds the generality of every geometric claim, though it does not make
any internal comparison (passive versus star-forming dimension, full versus image-only
embedding) wrong, since those comparisons are within the same selected sample. Solid for
within-sample claims, a real limit on extrapolation to the wider galaxy population.

### The morphology labels are CNN predictions with 5 to 10% error

The morphology labels we compare the geometry against (smooth fraction, featured fraction,
merger fraction, and the branch-conditional spiral and bar fractions) are vote fractions:
the share of classifiers who answered a given question about a galaxy. But these are not
raw human votes. They are produced by a separate convolutional neural network (a
Zoobot/EfficientNet model) trained to predict the human vote shares, and that network is
accurate to roughly 5 to 10% (measured as the stated label accuracy, from Section 4). So
there's a layer of model between the pixels and the number we treat as ground truth.

Why this matters: any probe $R^2$, any Spearman correlation, and any sparse-autoencoder
alignment that uses a morphology label is bounded above by the label's own noise. If the
smooth-fraction label is itself 5 to 10% uncertain, a perfect probe could not score
$R^2 = 1$ against it, so our morphology decodabilities (smooth $0.792$, featured $0.794$,
merger $0.681$, all measured) are slight underestimates of how well the embedding actually
tracks true morphology, and the gap between a decodability of, say, 0.79 and a
hypothetical 0.85 may be label noise rather than a real ceiling in the embedding. It also
means a sparse-autoencoder latent that "should" align perfectly with morphology will fall
short of alignment 1 partly because the target is noisy. This caveat mostly makes our
morphology numbers conservative rather than inflated, which is the safer direction to err,
but it does cap how much weight any single morphology comparison can bear. We treat the
morphology results as solid in direction (the gradients and correlations are real and
recur) and only approximate in magnitude.

### The passive-versus-star-forming dimension gap is small-scale in absolute terms

The heterogeneity result (Section 16) is the one genuinely surprising signal, and its
limitation is specific enough to state precisely. We split the sample by g-r colour using
a two-component Gaussian mixture (cut at $1.012$, component means $0.78$ and $1.24$,
measured) into passive (red, $n = 22{,}614$) and star-forming (blue, $n = 25{,}782$)
populations, then measured the intrinsic dimension of each with TwoNN. The passive
population reads $17.19$ ($[17.01, 17.43]$) and the star-forming $15.53$
($[15.39, 15.73]$), with a difference $\Delta\text{ID} = 1.66$ ($[1.38, 1.96]$, measured)
that excludes zero.

The catch is in those absolute numbers. They are TwoNN values, which means they're the
small-scale, first-to-second-neighbour estimates that we argued above are noise-inflated
and are not the true dimension. So $17.19$ and $15.53$ should not be read as "passive
galaxies live on a 17-dimensional sheet." Only the relative difference is trustworthy:
both populations were measured the same way at the same scale with the same estimator, so
the noise inflation is common to both and largely cancels in the difference, leaving
$\Delta\text{ID}$ as a clean comparative signal. There's a second honest qualifier: prior
work on raw photometry (Cadiou et al. 2025) found the opposite ordering, with star-forming
galaxies higher-dimensional. Our embedding is a different, multimodal object, not raw
photometry, so this is not a one-to-one contradiction, but it does mean we should hold the
result as real, modest, and unexplained rather than wrap a tidy physical story around it.
Suggestive, and we mark it so: the sign and significance of the gap are solid, the
absolute dimensions and the physical cause are not.

### Smaller threats, stated for completeness

A few limitations are real but lower-stakes, and we group them rather than give each its
own headline. The diffusion-map and curvature subsample sizes (2,000 points for
Ollivier-Ricci curvature and the metric-concentration diagnostic) mean those numbers carry
subsample variance we did not bootstrap as tightly as the full-sample probes; we report
them as point readings with the understanding that they would wobble at the few-percent
level under resampling. The Forman-Ricci curvature is structurally negative on a
k-nearest-neighbour graph because node degree dominates the formula, so we used it only to
rank candidate bridge edges and never to claim the manifold is negatively curved; the
trustworthy signed curvature is Ollivier-Ricci (mean $+0.155$, fraction of negative edges
$4.2\%$, measured). The geometry-of-concepts test (Section 17) returned an honest null
(cosine $0.072$ versus a permutation-null 95th percentile of $0.084$, so not significant),
and the limitation there is sample size as much as anything: the spiral sample is small
($n = 3{,}034$) by the design of the Galaxy-Zoo decision tree, so a real but weak linear
hierarchy could be present below our power to detect it. We report the null as a null and
do not over-read it. Finally, the whole analysis rests on mean-pooling the per-token
encoder outputs into one vector per galaxy; pooling is a choice, and a different pooling
(or using the token vectors directly) could expose structure we averaged away. None of
these overturn a headline, but each trims the confidence on a specific number.

### Reading the report with these in hand

Put together, the limitations sort cleanly. The geometric skeleton (intrinsic dimension
about 10 to 12, one connected piece, no recurring holes, mostly positive curvature) is
solid: it comes from multiple estimators, synthetic validation, and recurrence checks, and
the caveats shift interpretation at the margin without breaking the result. The physical
reading (the embedding carries colour, star formation, mass, and morphology, decodable
from the image) is solid for mass and sSFR and merely qualified for redshift by the
photo-z route. The two most exciting results, the 59 alien candidates and the
passive-minus-star-forming dimension gap, are exactly the two we file under suggestive,
because the limitations there reach into the conclusion itself: one has no causal test,
the other has no physical explanation. That alignment is not an accident. The strongest
claims are the ones built from converging tools, and the weakest are the ones a single
clever measurement reached on its own. The next section states the defensible bottom line
and lays out the specific experiments that would move the suggestive results into the
solid column.

## 21. Conclusions and what a full study should do next

We set out to ask two things about AION-1's galaxy embedding that its authors never
measured: what is the native geometry of the representation (its dimension, its curvature,
its topology), and does the model organise galaxies by anything beyond the human taxonomy.
We measured the geometry many ways and the measurements agree. Here is the defensible
bottom line, then the honest verdict, then a concrete list of the experiments that would
settle what we could not.

### The four findings, with their status

First, the embedding is low-dimensional. Three mechanistically independent estimators put
the intrinsic dimension at roughly 10 to 12 (large-scale Gride near 10, local
maximum-likelihood near 11.4, linear PCA participation ratio near 11, all measured, all
below the $\log_2(N) = 15.56$ resolution ceiling). That is a large compression from the
ambient 1024 numbers. It is solid as a measurement, with the honest qualifier that 10 to
12 sits at the top of, or just above, the optimistic astrophysical prior of 4 to 10
effective axes, so "low-dimensional" is true against 1024 but not as low as the best case.

Second, the embedding is a single smooth simply-connected continuum, not a set of discrete
clusters. The diffusion spectrum decays with no dominant gap, the number of connected
pieces is one (cutting the longest links peels off only single outliers while the body of
tens of thousands stays whole), and the number of recurring loops is zero across 10
subsamples and three metrics ($\beta_0 = 1$, $\beta_1 = 0$, measured). Three different
tools reached the same statement. This is solid, and it matches the physics: the
morphological sequence and the colour-mass diagram are continua in nature, not islands.

Third, physics is strongly decodable from the image alone. A single linear probe on the
image-only embedding reads colours at $R^2 \approx 0.91$ to $0.96$, morphology at $0.68$
to $0.79$, and (the load-bearing part) stellar mass at $0.721$ and specific star-formation
rate at $0.760$, properties the model was never given as inputs in that set (measured).
The concept directions are also more orthogonal than the labels' own correlations require
(most disentanglement excesses positive, redshift-smooth $+19.4^\circ$, measured). This is
solid, with redshift carrying the photo-z asterisk from Section 20 (decodability of
redshift partly routes through colour because the label is colour-derived), which is why
we lean on mass and sSFR.

Fourth, the model carries a private vocabulary that mostly overlaps the human one, plus a
handful of candidates that don't. A sparse autoencoder found 335 seed-stable directions
that align with the known physical labels (max alignment $0.279$, about 23 times the
shuffle null, measured), with colour and morphology the clearest single concepts and
redshift smeared across many weak units. Alongside them sit 59 seed-stable directions that
no label explains. These 59 are the most interesting result and the least certain: they
are correlational only, with no causal test, and we list them as candidates, not
discoveries.

### The honest verdict

AION-1's galaxy embedding is a compact, simply-connected, mostly positively curved
continuum of about a dozen dimensions whose principal axes carry colour, star formation,
morphology, and redshift strongly enough to read stellar mass and star-formation rate off
the image alone. It shows weak, localised branching (a small fraction of
negative-curvature bridge edges, a mild tree tendency over a matched random cloud) that is
consistent with, but does not prove, the distinct quenching channels astronomers expect.
It hides a modest and genuinely surprising dimension gap between passive and star-forming
populations, and a small set of stable, unlabelled directions worth a causal follow-up.
The confirmatory question we treated only as a control, whether the model rediscovered the
broad morphological and colour structure astronomers already draw, is answered yes: the
continuum and its two physics-bearing axes are the same structure, recovered without
morphology labels. The real result is not any single number. It is that geometry,
decodability, curvature, topology, and concept discovery, five tools with five different
failure modes, all describe the same physics-bearing body. Where they speak, they agree.
Where they are silent (does branching mean quenching channels, what are the 59 candidates,
why is the passive sheet higher-dimensional), we have marked the silence rather than
filled it.

### What a full study should add

The gap between what we measured and what we'd want to conclude points to five concrete
next experiments, roughly in order of how much they would move the suggestive results into
the solid column.

Causal tests of the alien concepts come first, because that single missing experiment is
the entire difference between "59 correlational candidates" and "59 concepts the model
uses." The test is intervention. Ablate each candidate latent (zero its activation,
reconstruct, and measure what degrades) and steer it (push the activation up and down and
watch the decoded embedding, then the downstream property predictions, move). A latent
that, when steered, changes a coherent visual or physical attribute is a real concept; one
that changes nothing, or changes noise, is an artefact. Cross-matching the candidates
against catalogues we did not use here (local environment, structural parameters,
emission-line ratios) would, in parallel, tell us which "alien" directions are alien only
to our six-label set.

Second, a Riemannian pullback metric, which we flag as the flagship upgrade. Everything in
this report measured distance with off-the-shelf metrics (Euclidean, plus the intrinsic
Fermat, Isomap, and diffusion distances) chosen because the raw Euclidean distance
concentrates badly in high dimensions. A pullback metric replaces that choice with the
geometry the model itself implies: it carries distances back through the encoder so that
"near" means near in the model's own sense, and it exposes where the manifold stretches
(where small changes in a galaxy move you far in the embedding) and where it compresses.
That would turn the curvature and branching results from descriptive statistics into
honest geodesics, and it would let us trace an actual path across a candidate quenching
bridge instead of only counting negative-curvature edges.

Third, spectra and a true multimodal embedding. This run used images and broadband
photometry only, so any concept living mainly in the spectral channel is invisible to us,
and our dimension count likely understates the full model's. Feeding spectra in and
re-measuring the geometry would test whether the intrinsic dimension rises, whether new
axes appear in the diffusion map, and whether the sparse autoencoder finds spectral
concepts the image-only run could not.

Fourth, cross-model checks. We measured one model. The interesting question behind the
whole interpretability lineage is whether large models converge on shared structure, and a
single model cannot answer it. Running the same battery on a second, independently trained
galaxy foundation model would tell us which of our findings (the dimension, the continuum,
the concept set) are properties of galaxies as seen through deep learning and which are
quirks of AION-1.

Fifth, per-population curvature at the branch points. We measured curvature over the whole
cloud and found mostly positive curvature with weak localised branching, but we never
measured curvature separately along the passive and star-forming sub-manifolds, or in the
neighbourhoods of the specific negative-curvature bridges. Doing so, paired with the
pullback metric and a physical label for what changes across each bridge, is how the
"consistent with distinct quenching channels" reading would become a tested claim instead
of a permitted one.

### The forward step

If we had to pick one experiment to run next, it is the causal SAE test, and the reason is
the same logic that runs through this whole report: the strongest claims are the ones
built from converging evidence, and the 59 alien candidates are the one result still
standing on a single leg. Steer them. Whatever moves when we push those latents, or fails
to, is the next real measurement, and it is the one that decides whether AION-1 sees
galaxies in ways we have not yet learned to name.

## 22. Glossary of terms and symbols

This glossary collects every technical term and symbol the report uses, each defined in
one or two plain sentences. It is meant to be read out of order: jump here whenever a word
in an earlier section was unfamiliar. Where a term was introduced in a specific section,
that section is named so you can find the full treatment. Definitions are kept short and
faithful to how the term is actually used in this study, not to the most general textbook
meaning.

A note on two recurring tags used throughout the report. "Measured" marks a number read
directly off the data or a results file. "Interpreted" (or "we read this as") marks an
inference we drew from measured numbers; it is a judgement, not a readout. The glossary
itself defines terms and does not assign these tags, but the distinction matters when you
carry a term back into the results sections.

### Model and representation

**Foundation model.** A large neural network trained once, on a broad task, so that its
internal representation can be reused for many downstream problems without retraining.
AION-1 is the foundation model studied here.

**AION-1.** The astronomical foundation model whose embedding geometry this report
measures. The variant used is AION-1-Large, an 800-million-parameter encoder-decoder
transformer trained by self-supervised multimodal masked modelling, with no morphology
labels (Section 3).

**Transformer.** A neural-network architecture that processes a set of input pieces
("tokens") and lets every piece attend to every other piece, so each output vector is
informed by the whole input.

**Token.** One input piece the transformer sees: a patch of an image, a chunk of a
spectrum, or a single catalogue number, each turned into a vector. AION-1 tokenises
images, photometry, and scalars and learns to fill in masked tokens.

**Encoder.** The half of the model that turns inputs into internal vectors. We keep the
encoder and discard the decoder (the half that would reconstruct masked inputs).

**Masked modelling (self-supervised).** A training scheme where parts of the input are
hidden and the model learns to predict them from the rest. "Self-supervised" means the
supervision comes from the data's own structure, not from human labels.

**Frozen model.** A model whose weights are held fixed (no fine-tuning) while we study it,
so the geometry we measure is the pretrained representation itself, not something we
reshaped (Section 3).

**Embedding.** The fixed-length vector of numbers the model assigns to a galaxy: a learned
summary. Here each galaxy gets one 1024-number embedding.

**Mean pooling.** Reducing the encoder's many per-token vectors to one vector per galaxy
by averaging them element by element. This is how we go from a variable number of tokens
to a single 1024-vector (Section 3).

**E_full.** The embedding set built from image (g, r, i, z bands) plus photometry (g, r, z
flux) plus redshift, all fused as inputs. Used for the geometry arms (Section 3).

**E_img.** The image-only embedding set, where redshift and colour are not inputs. This is
the leakage-free set: probing it for redshift or colour is a real inference, not reading
back an input (Section 3).

**Ambient dimension ($d$). ** The number of coordinates each embedding vector has, here
$d = 1024$. It is the size of the space the points live in, not how many directions they
actually use.

**Intrinsic dimension (ID).** The number of independent knobs you need to describe where a
point sits on the data's underlying surface, which can be far below the ambient dimension.
If galaxies really vary along about a dozen axes, the ID is about a dozen even though
$d = 1024$ (Sections 6 and 7).

### Preprocessing and basic spread

**z-score (standardise).** For each of the 1024 dimensions, subtract that dimension's mean
and divide by its standard deviation, so every dimension has mean 0 and spread 1. We
z-score before all geometry because per-dimension spread varies by about 23 times, and
without it a few high-variance dimensions would dominate every distance (Section 4).

**Standard deviation (std).** A measure of spread: the typical distance of values from
their mean. A larger std means the values are more spread out.

**Norm.** The length of a vector, $\lVert x\rVert = \sqrt{\sum_i x_i^2}$. The embedding
norms are tight (about 48 for E_full, about 49.6 for E_img, roughly 5% spread), so the
cloud sits near a thin spherical shell (Section 4).

**PCA (principal component analysis).** A linear method that finds the orthogonal
directions of greatest variance in the data, ordered from most to least. The directions
are the principal components; how much variance each carries is its eigenvalue.

**Eigenvalue and eigenvector.** For a matrix $M$, an eigenvector $v$ is a direction that
$M$ only stretches, $Mv = \lambda v$, and the eigenvalue $\lambda$ is the stretch factor.
In PCA the eigenvalues are variances along the component directions; in diffusion maps
they measure how slowly a walk mixes along each mode.

**Participation ratio (PCA-PR).** A linear estimate of how many directions matter,
$\big(\sum_i \lambda_i\big)^2 / \sum_i \lambda_i^2$, where $\lambda_i$ are the PCA
eigenvalues. It counts the effective number of comparable-variance directions: if a few
dominate it is small, if many share the variance it is large. We use it as the linear
baseline for intrinsic dimension (Sections 6, 7).

### Metric and distance battery

**Metric / distance.** A rule for how far apart two points are. The report stresses that
the choice of metric is not a detail, because in high dimensions different metrics tell
different stories (Section 5).

**Distance concentration.** The high-dimensional effect where, as dimension grows, all
pairwise distances become nearly equal, so the contrast between near and far points
collapses and nearest-neighbour structure gets unreliable (Section 5).

**RDR (relative distance range).** A concentration diagnostic,
$\text{RDR} = (D_{\max} - D_{\min})/D_{\min}$ over all pairwise distances. Higher means
more contrast between near and far, so less concentration (Section 5).

**NN/mean.** A second concentration diagnostic: the mean nearest-neighbour distance
divided by the mean pairwise distance. Lower means points sit relatively closer to their
nearest neighbour than to the crowd, so more concentration (Section 5).

**Euclidean distance.** Ordinary straight-line distance, $\sqrt{\sum_i (x_i - y_i)^2}$.
Used as the control metric; it was the most concentrated (lowest contrast) in our battery
(Section 5).

**Cosine distance.** Distance based on the angle between two vectors rather than their
straight-line gap. Here it did not concentrate like Euclidean, so it acts as a middling
control rather than a failure case (Section 5).

**Isomap (geodesic distance).** Distance measured as the shortest path along a graph that
links each point to its near neighbours, so it follows the curved surface instead of
cutting across empty space (Section 5).

**Geodesic.** The shortest path that stays on a curved surface, the surface-bound analogue
of a straight line.

**Fermat distance (density-weighted geodesic).** A shortest-path distance where steps
through dense regions are cheaper, so paths prefer to travel through where the data
actually is; it is outlier-stable and has a published convergence guarantee for topology,
which is why it is the primary metric for the topology and curvature arms (Sections 5, 14,
15).

**Diffusion distance.** A distance that averages over all paths between two points via a
random walk on the neighbour graph, so it is smooth and density-stable (Sections 5, 8).

### Diffusion maps

**Diffusion map.** A method that builds a graph where near points are strongly linked,
treats it as a random walk, and uses the slow directions of that walk as new coordinates.
The slow directions are the large-scale shape of the cloud (Section 8).

**Affinity ($W_{ij}$). ** The link weight between points $i$ and $j$ in the graph, large
when they are close and small when far; it sets how easily the random walk hops between
them (Section 8).

**Bandwidth.** The scale that decides how quickly affinity falls off with distance. We
used a self-tuning local-scaling bandwidth (each point's scale is its distance to its 7th
neighbour) after a single global bandwidth gave a fragmented graph (Section 8).

**Laplace-Beltrami normalisation (anisotropic $\alpha = 1$). ** A reweighting of the
affinity that removes the effect of how densely galaxies happen to be sampled, so the
recovered geometry reflects the manifold's shape and not the sampling density (Section 8).

**Markov matrix.** The row-normalised affinity matrix whose entries are the one-step hop
probabilities of the random walk; its eigenvalues and eigenvectors define the diffusion
coordinates (Section 8).

**Diffusion coordinate (dc).** A coordinate built from an eigenvector $\psi_k$ scaled by
its eigenvalue, $\psi_k \lambda_k$. We label them dc0, dc1, dc2, and so on; dc1 and dc2
are the physics-bearing axes, dc0 is a weak global mode (Sections 8, 9).

**Spectral gap.** A large drop between consecutive diffusion eigenvalues. A clear gap
would mean discrete clusters; our spectrum decays smoothly with no dominant gap, which we
read as one continuous body (Section 8).

**Harmonic.** A higher diffusion coordinate that is mostly a repeat (a polynomial
function) of lower ones rather than a genuinely new axis. The harmonic screen flagged dc3
and dc4 as harmonics, so they carry no new geometry (Section 8).

**Connected component.** A piece of the graph in which every point can reach every other
by hops. Both embedding sets form a single connected component.

### Probes and decodability

**Linear probe.** A single linear map fit from the 1024 embedding numbers to a target
property, used to ask how much of that property the embedding linearly carries (Section
10).

**Ridge regression (RidgeCV).** Linear regression with a penalty on large weights to
control overfitting; "CV" means the penalty strength was chosen by cross-validation over a
grid of values (Section 10).

** $R^2$ (coefficient of determination).** The fraction of held-out (test-set) variance
the probe explains: 1 is perfect prediction, 0 is no better than always guessing the mean
(Section 10).

**Train/test split.** Holding out 20% of galaxies the probe never sees during fitting, so
$R^2$ measures genuine prediction, not memorisation. We used an 80/20 split with a fixed
seed.

**Bootstrap confidence interval (CI).** A range for an estimate found by resampling the
data with replacement many times (here 1000 times) and reading off the middle 95% of the
resulting values. It states our uncertainty about the number, not a spread in the data
(Sections 7, 10).

**Leakage.** When a model is asked to predict something it was effectively given as an
input, making the prediction trivially easy and the "concept" circular. Using E_img
(image-only) for colour and redshift probes avoids leakage, so those probes are real
inferences (Sections 3, 10).

**Modality ablation.** Comparing image-only against multimodal embeddings to see how much
of the decodability of an input property (like redshift) is leakage versus genuine
inference (Section 10).

**Disentanglement.** Whether two concepts occupy more nearly separate (orthogonal)
directions than their own correlation would force. We measure the angle $\theta$ between
two probe weight vectors, set a null angle from the labels' correlation, and call the
excess $\theta - \text{null}$; positive excess means genuine extra separation (Section
11).

**kNN purity.** A model-free check: among a galaxy's $k$ nearest neighbours (here
$k = 20$), what fraction share its label. High purity means like sits near like (smooth
galaxies are 99.1% pure, featured 71.4%) (Section 11).

**Spearman correlation ($\rho$). ** A rank-based correlation that measures whether two
quantities rise and fall together, even nonlinearly; it runs from $-1$ to $+1$. Used to
relate diffusion coordinates and SAE latents to labels (Sections 8, 9, 13).

### Sparse autoencoder (concept discovery)

**Sparse autoencoder (SAE).** A network that learns an over-complete dictionary so each
embedding is rebuilt from only a few active "concept" units, letting the embedding reveal
its own axes rather than us probing for human-named ones (Section 12).

**TopK SAE.** The specific SAE used: the encoder keeps only the $k$ largest latent
activations per galaxy,
$f = \mathrm{TopK}\big(W_{\text{enc}}(x - b_{\text{pre}}) + b_{\text{enc}}\big)$, and the
decoder rebuilds $\hat{x} = f W_{\text{dec}} + b_{\text{pre}}$ (Section 12).

**Latent.** One unit in the SAE's hidden dictionary, with an activation strength per
galaxy and a decoder direction (the vector it writes back into the embedding).

**L0.** The number of latents allowed to be active per galaxy, here $L0 = k = 32$. It is
the sparsity budget: how many concept units describe one galaxy at a time.

**Dictionary size ($m$) and expansion ($R$). ** The number of latents,
$m = R \times 1024$, where the expansion factor $R$ (here 4 or 8) sets how over-complete
the dictionary is.

**FVU (fraction of variance unexplained).** The share of embedding variance the SAE fails
to reconstruct: 0 is perfect, smaller is better. At the $k = 32$ operating point, FVU is
about 0.036, so 96.5% of variance is explained (Section 12).

**Dead and alive latents.** A latent is "dead" if it (almost) never activates and "alive"
if it is used. The AuxK term revives dead latents by having the top dead units reconstruct
the leftover error (Section 12).

**Alignment.** For one latent, the largest absolute Spearman correlation between its
activations and any of the six labels. High alignment means the latent tracks a known
property (Section 13).

**Label-shuffle null.** A baseline made by permuting the labels and recomputing alignment,
giving the alignment a meaningless latent would reach by chance. Its 95th and 99th
percentiles set the significance thresholds (Section 13).

**Seed-stability.** Whether a latent's decoder direction reappears across independent
training seeds, here counted as present in at least half of the other four seeds with
cosine $\geq 0.6$. A stable feature is a property of the model, not of one random run
(Section 13).

**Novelty.** The fraction of a latent's activation variance that the six labels cannot
linearly explain, computed as a regression residual. High novelty plus stability plus
non-alignment defines an "alien" candidate (Section 13).

**Alien candidate.** A seed-stable, high-novelty, non-label-aligned latent: a direction
the model uses consistently that none of our labels accounts for. These are correlational
only; no causal or ablation test was run (Section 13).

### Curvature and topology

**Curvature.** How a surface bends. Positive curvature curls like a sphere and makes
neighbourhoods cluster; negative curvature opens like a saddle and marks a branch or
bridge; zero curvature is flat (Section 14).

**Gaussian and sectional curvature.** Standard differential-geometry measures of bending
at a point. We do not compute them directly on the embedding; we use discrete, graph-based
curvature notions instead (Ollivier-Ricci, Forman-Ricci) (Section 14).

**delta-hyperbolicity (Gromov 4-point delta).** A tree-likeness score from four-point
comparisons, divided by the cloud diameter; lower means more tree-like and 0 is a perfect
tree. We read it against anchors, not in absolute terms (Section 14).

**Anchor (baseline cloud).** A reference cloud with a known character used to calibrate a
measurement: a matched-covariance Gaussian (structureless) and a synthetic tree (perfectly
tree-like) bracket the delta-hyperbolicity reading (Section 14).

**Ollivier-Ricci curvature.** A signed graph curvature from the optimal-transport distance
between the neighbour distributions of two linked points; positive means clustered,
negative means a bridge or saddle. This is the trustworthy signed local measure (Section
14).

**Optimal transport (earth-mover distance).** The least total cost to move one probability
distribution onto another, where cost is distance times mass moved. It underlies the
Ollivier-Ricci curvature (Section 14).

**Forman-Ricci curvature.** A combinatorial graph curvature based on degrees and triangle
counts. On a kNN graph it is structurally negative (degree dominates), so we use it only
to rank candidate bridge edges, never to claim negativity (Section 14).

**Betti number ($\beta_0$, $\beta_1$). ** Counts of topological features: $\beta_0$ is the
number of separate pieces, $\beta_1$ is the number of independent loops or holes. We find
$\beta_0 = 1$ and $\beta_1 = 0$ (Section 15).

**Persistent homology.** A method that grows balls around each point and tracks when
topological features are born and die as the ball radius increases; long-lived features
are real, short-lived ones are noise (Section 15).

**Persistence.** The lifetime of a topological feature (death radius minus birth radius),
diameter-normalised here. A loop counts as significant only if its persistence exceeds 0.1
(Section 15).

**Minimum spanning tree (MST).** The lightest set of edges that connects all points with
no loops. Cutting its longest edges is how we probe whether the cloud splits into pieces;
cutting peeled off only single outliers (Section 15).

### Heterogeneity and concept hierarchy

**Gaussian mixture model (GMM).** A model that treats the data as a blend of Gaussian
(bell-curve) components. A two-component GMM on g-r colour split the sample at a cut of
1.012 into red (passive) and blue (star-forming) (Section 16).

**Stratified ID / delta-ID.** Intrinsic dimension measured separately for two populations,
with delta-ID their difference. The absolute small-scale IDs are noise-inflated, so the
relative delta-ID = 1.66 (which excludes zero) is the trustworthy signal (Section 16).

**TwoNN.** An intrinsic-dimension estimator using each point's ratio of second- to
first-neighbour distance, $\mu = r_2/r_1$, with cumulative law $P(\mu > x) = x^{-d}$ and
unbiased estimate $\hat{d} = (N-1)/\sum_i \log \mu_i$; it is small-scale and
noise-inflated on AION (Sections 6, 7).

** $\mu = r_2/r_1$. ** The TwoNN ratio: a point's distance to its second nearest neighbour
divided by its distance to its first. Its distribution across points encodes the intrinsic
dimension (Section 6).

**Gride.** A generalised-ratio ID estimator that uses farther neighbour ranks to trace an
ID-versus-scale curve, separating true dimension at larger scale from small-scale noise
inflation, with no subsampling needed (Sections 6, 7).

**Levina-Bickel MLE (local MLE).** A maximum-likelihood ID estimate computed per point
from its $K$ nearest-neighbour distances and swept over $K$, used to map where dimension
is locally higher (Sections 6, 7).

**Concept direction.** A unit vector in the embedding such that moving along it changes
one human-named property; recovered as a probe weight vector (Section 10) or an SAE
decoder vector (Section 13).

**Linear concept hierarchy (Park et al. test).** The idea that a child concept's direction
equals the parent's direction plus an orthogonal child-specific part. Our one test
(featured to spiral) returned a null: measured cosine 0.072 below the permutation-null
95th percentile 0.084 (Section 17).

**Permutation null.** A no-effect baseline built by scrambling the labels and recomputing
the statistic many times, keeping everything else fixed; the real value is significant
only if it beats almost all scrambled ones (Sections 13, 17).

### Astronomy: galaxies as a measurable shape

**Redshift ($z$). ** How much a galaxy's light is stretched to longer wavelengths by
cosmic expansion, which tracks distance and look-back time. Our sample spans $0 < z < 1$
(Sections 2, 4).

**Photo-z (photometric redshift).** A redshift estimated from broad-band colours rather
than a spectrum, so it is cheaper but noisier and partly colour-derived. Most of our
redshifts are photo-z, which is why image-to-redshift partly routes through
image-to-colour (Sections 4, 10).

**Spectroscopy vs photometry.** Spectroscopy splits light into a detailed wavelength
spectrum (rich but expensive); photometry sums light into a few broad bands (g, r, i, z
here). We have spectroscopic redshift for only 6,883 galaxies (Sections 2, 4).

**Morphology.** A galaxy's visual structure. It forms a continuum from smooth blobs to
featured disks rather than discrete species (Section 2).

**Early-type / late-type.** Historical morphology terms: early-types are smooth
ellipticals and lenticulars, late-types are featured spirals. The names do not imply an
evolutionary order (Section 2).

**Passive vs star-forming.** Passive galaxies have largely stopped forming stars and are
red; star-forming galaxies are actively making stars and are blue (Sections 2, 16).

**Quenching.** The process by which a galaxy stops forming stars. There are at least two
distinct channels: a fast disk-to-spheroid transformation and a slow fade (Section 2).

**Stellar mass ($\log M_*/M_\odot$). ** The total mass in stars, in units of solar masses
on a log scale. Decodable from the image alone at $R^2 = 0.72$ (Sections 2, 10).

**sSFR (specific star-formation rate).** The star-formation rate divided by stellar mass:
how fast a galaxy is growing relative to its size. Decodable from the image alone at
$R^2 = 0.76$ (Sections 2, 10).

**Sersic index ($n$). ** A single number describing how concentrated a galaxy's light
profile is, with low $n$ for disky and high $n$ for bulgy/elliptical light (Sections 2,
10).

**Colour (g-r, r-z).** A difference of magnitudes between two bands, which measures the
slope of a galaxy's spectrum; redder colours (larger g-r) indicate older or dustier
stellar populations. These are the strongest single concepts in the embedding (Sections 2,
9, 13).

**Magnitude.** A logarithmic, inverted brightness scale: smaller magnitudes are brighter.
Differences of magnitudes give colours.

**Red sequence, blue cloud, green valley.** The three regions of the colour-mass diagram:
a dense red sequence of passive early-types, a dense blue cloud of star-forming
late-types, and a sparse green valley of transitioning galaxies between them (Section 2).

**Vote fraction.** The share of human classifiers (or, here, a CNN trained to mimic them)
who assigned a galaxy a given morphology label (smooth, featured, merger), accurate to
roughly 5 to 10% (Section 4).

**Galaxy Zoo / decision tree.** The crowd-sourced morphology project and its branching
question structure. Some questions (spiral arms, bars) are only asked of featured,
non-edge-on disks, which is why those labels exist for only about 3,034 galaxies (Section
4).

This glossary closes the definitional debt of the report. Section 23 turns to the numbers
themselves: every threshold and decision rule that any of these terms relied on, gathered
in one place with its value and its meaning.

## 23. Appendix: thresholds, statistical claims, reproducibility

Every headline in this report rests on a decision rule: a threshold a number had to beat,
an interval that had to exclude zero, a sign convention, or a choice of which metric to
trust. Scattered through the sections, these rules are easy to lose track of. This
appendix gathers them in one place so a reader can audit the logic without re-reading the
whole report. Each entry gives the rule, its value, and what it decides. Every value here
is measured and matches the committed results files; none is introduced for the first
time.

### 23.1 Thresholds and decision rules

The table below is the full set of fixed cutoffs and decision rules the analysis used.
Read each row as: this is the number, this is what passing or failing it means.

| Rule | Value | What it decides | Section |
|---|---|---|---|
| Resolution ceiling $\log_2 N$ | $\log_2(48{,}398) = 15.56$ | The largest intrinsic dimension a sample of $N$ points can resolve. Any ID estimate near or above this is suspect; our plateau at $\sim$10-12 sits safely below it. | 6, 7 |
| Astrophysical prior band | 4 to 10 effective axes | The optimistic expected range for an honest multimodal galaxy embedding, from photometry ($\sim$2-5) and spectra ($\sim$3-10). It is a band to compare against, not a prediction. Our ID sits at or just above its top. | 2, 7 |
| ID estimator agreement | Gride-large $\approx 10$, local-MLE $\approx 11.4$, PCA-PR $\approx 11.2$ (E_full) | Three mechanistically-independent estimators must agree near a plateau for the ID headline to hold. They do, which is why the small-scale TwoNN $\approx 16.5$ is read as noise-inflated rather than the answer. | 7 |
| Headline ID and CI (E_full) | TwoNN MLE $16.56$ $[16.42, 16.71]$; plateau $\sim$10-12 | The TwoNN Gamma-posterior 95% interval is reported, but the trustworthy estimate is the larger-scale plateau, not this small-scale value. E_img is similar (TwoNN $16.05$ $[15.91, 16.19]$). | 7 |
| SAE alignment null thr95 | $0.0119$ | A latent's max absolute Spearman with a label must exceed this (the 95th percentile of the label-shuffle null) to count as aligned-significant. Count passing: 717. | 13 |
| SAE alignment null thr99 | $0.0129$ | The stricter 99th-percentile cutoff for aligned-strong. Count passing: 690. The max alignment, $0.279$, is about 23 times thr95. | 13 |
| SAE seed-stability cosine | $\geq 0.6$ | A latent's decoder direction must reappear in at least half of the other four seeds at cosine $\geq 0.6$ to be called seed-stable. Aligned-and-stable count: 335. Median best cross-seed cosine across active latents: $0.460$. | 13 |
| SAE novelty cut | $> 0.7$ | Fraction of a latent's activation variance the six labels cannot linearly explain. Above $0.7$, plus seed-stable, plus not label-aligned, defines an alien candidate. Count: 59 (correlational only). | 13 |
| $\beta_1$ persistence threshold | $> 0.1$ | A diameter-normalised loop must persist beyond this to be a significant hole, not noise. Significant loops essentially never recur across 10 subsamples (mean $\leq 0.2$, max persistence $0.105$ Euclidean / $0.138$ Fermat / $0.081$ diffusion), so $\beta_1 = 0$. | 15 |
| $\beta_0$ split rule | Cut longest MST edges; piece must exceed size 1 | Cutting the longest minimum-spanning-tree edges peels off only single outliers ($[48{,}397, 1]$, then $[48{,}396, 1, 1]$), so the giant body stays whole and $\beta_0 = 1$. | 15 |
| delta-ID CI excludes zero | $\Delta\text{ID} = 1.66$ $[1.38, 1.96]$ | The passive-minus-star-forming intrinsic-dimension difference is real only if its CI excludes zero. It does, so passive galaxies sit on a slightly higher-dimensional sub-manifold (passive $17.19$ $[17.01, 17.43]$ vs star-forming $15.53$ $[15.39, 15.73]$, small-scale absolute values). | 16 |
| GMM colour cut | g-r $= 1.012$ (means $0.78$, $1.24$) | The two-component Gaussian-mixture split point on g-r that defines passive ($> 1.012$, $n = 22{,}614$) versus star-forming ($n = 25{,}782$). | 16 |
| Ollivier-Ricci sign | sign is meaningful; mean $= +0.155$ | Positive curvature means locally clustered neighbourhoods, negative means a bridge or saddle. Mean is positive with only 4.2% negative edges (p5 $+0.006$, p95 $+0.319$), read as mostly-positive curvature with weak localised branching. | 14 |
| Forman-Ricci role | rank-only (not signed) | On a kNN graph Forman is structurally negative (degree dominates), so its sign is not a clean curvature indicator; it is used only to rank candidate bridge edges. Ollivier is the trustworthy signed measure. | 14 |
| delta-hyperbolicity anchoring | AION $0.0136$ vs Gaussian anchor $0.0177$, tree $0.0079$ | Tree-likeness is read against anchors, not in absolute terms. AION falls below the matched-covariance Gaussian (mildly more tree-like than random) but well above the synthetic tree (not a tree). | 14 |
| Geometry-of-concepts significance | measured $0.072$ vs permutation null p95 $0.084$ | The featured-to-spiral hierarchy cosine must beat the 95th percentile of the permutation null to count. It does not ($0.072 < 0.084$), so the test returns an honest null. | 17 |
| Metric battery roles | Fermat primary, Euclidean control | Intrinsic metrics (Fermat, isomap, diffusion) give several times more distance contrast than Euclidean (Fermat RDR $\sim$260-281 vs Euclidean $\sim$14-15), so Fermat is the primary metric for topology and curvature, with Euclidean kept as a control and cosine as a middling control. | 5 |

A few rows deserve a plain-language reminder of why they are framed as a threshold rather
than a point estimate. Three kinds of decision rule appear here. The first kind is a null
comparison: a measured number is meaningful only if it beats what scrambled, structureless
data would produce (the SAE alignment thresholds and the geometry-of-concepts permutation
null both work this way). The second kind is an interval test: a difference is real only
if its confidence interval misses zero (the delta-ID rule). The third kind is a sign
convention: the number's sign carries the physics (Ollivier-Ricci positive versus
negative), so we state which measure's sign we trust and which we do not (Forman is
rank-only). Keeping these three logics separate is what lets the report say "measured" and
"interpreted" honestly: the thresholds are where measurement stops and reading begins.

The next four paragraphs walk through the load-bearing rules in more detail, because a
reader auditing a headline needs to know not just the cutoff but why that cutoff and not
another.

The $\log_2 N$ ceiling and the prior band together bracket the intrinsic-dimension claim
from above and from the optimistic side. The ceiling is a hard fact about resolution: with
$N$ points you cannot reliably tell apart structure finer than about $\log_2 N$
independent directions, because the number of points needed to populate a $d$ -dimensional
neighbourhood grows exponentially in $d$. At $N = 48{,}398$ that ceiling is $15.56$. Our
plateau estimate of about 10 to 12 sits below it, so the sample can in principle resolve
the dimension we report; the small-scale TwoNN value of $16.56$ sits essentially at the
ceiling, which is one more reason to treat it as noise-inflated rather than as the answer.
The 4-to-10 prior band is a softer comparison: it is where astrophysics expected the
dimension to land if the embedding were honest, derived from the rough effective
dimensionality of clean photometry and galaxy spectra. Our result sits at or just above
its top edge, so "low-dimensional" is fair but "as low as the best case" is not, and we
say so rather than rounding down.

The SAE thresholds are null comparisons, and the two cutoffs do different jobs. thr95
($0.0119$) is the 95th percentile of the label-shuffle null: a latent whose best label
correlation exceeds it would be reached by chance fewer than one time in twenty, so it
counts as aligned. thr99 ($0.0129$) is the stricter 99th-percentile version for the
aligned-strong count. Both are tiny because shuffling the labels destroys almost all
correlation, so even a weak real alignment stands out; the largest observed alignment,
$0.279$, clears thr95 by roughly a factor of 23, which is why the strongest concepts
(colour and morphology) are not in doubt. The seed-stability cosine ($0.6$) and the
novelty cut ($0.7$) then layer two further filters on top of alignment: stability asks
whether a direction is a property of the model or of one lucky random initialisation, and
novelty asks whether the labels can explain the direction at all. The alien candidates
pass stability and novelty while failing alignment, which is exactly the corner of the
space where a genuinely new, model-specific axis would live, and exactly the corner where
we have the least external check, so we flag those 59 as correlational only and never
claim more.

The topology and curvature rules are about not mistaking noise for structure. For
$\beta_1$, the persistence cut of $0.1$ exists because persistent homology always produces
a swarm of short-lived "loops" that are sampling noise (thousands of them, per the
mean-bars counts), and only a loop that survives well past that cut, and recurs across
independent subsamples, and survives a change of metric, counts as a real hole. None did.
For $\beta_0$, the MST rule is conservative in the other direction: we cut the longest
connecting edges one at a time and watch what falls off, and because only single points
detach while the body of tens of thousands stays whole, the cloud is one piece and not a
hidden red/blue split. For curvature, the Ollivier sign is the whole point of using it: a
signed local curvature tells clustered neighbourhoods (positive) from bridges (negative),
and the mean of $+0.155$ with only $4.2\%$ negative edges is what licenses the "mostly
positive, weakly branching" reading. Forman is deliberately demoted to ranking only,
because its sign is forced negative by graph degree and would otherwise manufacture a
false branching signal.

The delta-ID rule is the one interval test that carries a comparative claim, and it is
framed as "excludes zero" for a reason. The absolute per-population intrinsic dimensions
($17.19$ passive, $15.53$ star-forming) are small-scale TwoNN values and share the same
noise inflation as the global TwoNN, so neither absolute number is the estimate of a
population's true dimension. What survives the noise is the difference, because both
populations are measured the same way on the same scale, so the shared bias largely
cancels. The bootstrap interval on that difference, $[1.38, 1.96]$, misses zero, so the
ordering (passive slightly higher-dimensional) is the trustworthy signal even though the
absolute heights are not.

### 23.2 Confidence intervals on the headline numbers

Every headline carries an uncertainty, and the report never presents a point estimate as
if it had none. The intervals below are the ones a reader should quote alongside each
claim. The intrinsic-dimension intervals are Gamma-posterior 95% intervals from the TwoNN
MLE (small-scale, so the plateau is the real estimate); the probe intervals are 1000-times
bootstrap 95% intervals; the delta-ID interval is a bootstrap interval over the stratified
estimate.

| Headline quantity | Estimate | 95% interval | Note |
|---|---|---|---|
| Decodable: stellar mass (E_img) | $R^2 = 0.721$ | $[0.659, 0.772]$ | Non-input; image-only. The cleaner physics evidence. |
| Decodable: sSFR (E_img) | $R^2 = 0.760$ | $[0.692, 0.816]$ | Non-input; image-only. |
| Decodable: redshift (E_img) | $R^2 = 0.800$ | $[0.792, 0.809]$ | Mostly photo-z, so partly image-to-colour-to-redshift. |
| Decodable: g-r (E_img) | $R^2 = 0.958$ | $[0.957, 0.960]$ | Strongest single decode. |
| Decodable: spiral (E_img) | $R^2 = 0.252$ | $[0.153, 0.334]$ | Weakest; $n = 3{,}034$ only. |
| delta-ID (passive minus SF) | $1.66$ | $[1.38, 1.96]$ | Excludes zero. |
| Max SAE alignment | $0.279$ | (vs null thr95 $0.0119$) | About 23x the null. |

### 23.3 Reproducibility note

The analysis is built to be re-run from committed artefacts. Three properties make it
auditable.

First, the geometry runs on the committed embeddings and labels, not on anything
regenerated on the fly. The embeddings (E_full and E_img, one 1024-vector per galaxy,
$N = 48{,}398$) and the label tables are fixed inputs, z-scored per dimension before every
geometry arm. A second run on the same inputs reproduces the same numbers up to the
documented random elements (the 80/20 probe split, the SAE seeds, the topology
subsamples), each of which uses a fixed seed where one applies.

Second, every estimator was validated on synthetic data of known answer before it was
trusted on AION. The intrinsic-dimension estimators were run at the matched $N = 48{,}398$
on a sphere (truth 5), a swiss roll (truth 2), and a plane (truth 5), and they recovered
the known dimensions (for example TwoNN $4.96$, $1.95$, $5.08$); a two-blob control was
included to show the estimator inflates on a near-zero-diameter cluster (TwoNN
$\approx 212$) and must be excluded. Curvature used a matched-covariance Gaussian anchor
and a synthetic tree as brackets. Topology and concept tests used shuffle and permutation
nulls. The validation logic is what licenses the readings; we did not trust any tool we
had not first watched recover a known answer.

Third, the figure scripts read only the committed results files. No figure recomputes a
result; each plots numbers already written to disk, so a figure cannot disagree with the
table that produced it.

The committed results files, and what each contains, are:

| File | Contents |
|---|---|
| `results/sanity.json` | NaN/Inf checks, embedding norms, per-dimension std range, raw PCA participation ratio and cumulative-variance counts for E_full and E_img. |
| `results/metric.json` | Distance-concentration diagnostics (RDR, NN/mean) across five metrics on a 2,000-point subsample, with connectivity checks. |
| `results/intrinsicDim.json` | Intrinsic-dimension estimates (TwoNN with CI, Gride scale curve, local Levina-Bickel MLE, PCA participation ratio) for AION E_full/E_img and for the synthetic validation manifolds. |
| `results/diffusionMap.json` | Diffusion-map eigenvalues, spectral gaps, connected-component count, and the harmonic screen for both embedding sets. |
| `results/diffcoords_E_full.npy`, `results/diffcoords_E_img.npy` | The per-galaxy diffusion coordinates (48,398 x 10) used for the coloured embeddings and label correlations. |
| `results/probes.json` | RidgeCV probe $R^2$ with bootstrap CIs (image-only and multimodal), the modality ablation, the cross-matched physical-subset probes, the disentanglement angles, and kNN purity. |
| `results/sae.json` | SAE health (FVU, alive fraction) per seed and expansion, the L0-vs-FVU frontier, the concept-scoring summary (null thresholds, counts, alien candidates), and the named-concept breakdown. |
| `results/sae_acts0.npy`, `sae_align.npy`, `sae_best_label.npy`, `sae_novelty.npy`, `sae_recurrence.npy` | Per-latent SAE arrays: activations (seed 0), alignment scores, best-matching label, novelty, and cross-seed recurrence, supporting the alignment-novelty and named-concept figures. |
| `results/curvature.json` | delta-hyperbolicity (median, mean, p95) for AION, the Gaussian anchor, and the tree validation; Ollivier-Ricci signed summary; Forman-Ricci ranking statistics. |
| `results/topology.json` | $\beta_0$ via MST edge cuts (split sizes) and $\beta_1$ via persistent homology across 10 subsamples under three metrics (mean bars, max persistence, significant-loop counts). |
| `results/stratifiedId.json` | The g-r GMM split (cut, component means), per-population TwoNN ID with CIs, and the delta-ID with its CI and the excludes-zero flag. |
| `results/geometryOfConcepts.json` | The featured-to-spiral parent-child cosine, the permutation-null 95th percentile, and the significance flag. |

What this appendix does not contain is any new measurement. Every threshold, interval, and
file listed here was used to support a claim earlier in the report; collecting them in one
place is meant to make the report checkable, not to add to it. A reader who wants to test
whether a headline is defensible can take its row from the tables above, open the named
results file, and confirm the number for themselves.
