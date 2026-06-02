## 12. Concept discovery I: the sparse autoencoder

The probes in Section 10 (decodability) answered one half of question C by asking, for each human label we
already had, whether the embedding carries it. That is a confirmatory test. We hand the model a concept and
check whether it is there. It cannot find a concept we did not think to name. The other half of question C
is the open-ended part: does AION-1 organise galaxies along directions that go beyond the human taxonomy of
colour, morphology, and redshift? To answer that we need a method that lets the embedding speak first and
propose its own axes, rather than one where we supply the axis and the model only confirms or denies. The
sparse autoencoder (SAE) is that method. This section explains what an SAE is, the exact construction we
used, the one number that controls the reconstruction-versus-sparsity trade-off, and why we set that number
where we did. The concepts it actually found, and the candidate concepts it found that no human label
explains, are the subject of Section 13 (concepts and alien candidates).

### 12.1 The idea in plain words

Start from the problem the SAE is built to solve. Each galaxy is a point in a 1024-dimensional space (the
ambient embedding, Section 3). A single dimension of that space is almost never a clean concept. Real
networks store information in a distributed way: the activation for "this galaxy is red" might be spread
across hundreds of the 1024 coordinates, mixed together with the activation for "this galaxy is a spiral"
and "this galaxy is at high redshift". Several distinct meanings ride on top of each other in the same
coordinates. This mixing has a name in the interpretability literature, superposition, and it is the reason
you cannot just read concept number 7 off coordinate number 7. The directions that carry single, clean
meanings are tilted at odd angles to the coordinate axes, and there can be more such directions than there
are coordinates.

A sparse autoencoder untangles this. The plan is to learn a large dictionary of directions, far more than
1024, and to insist that any one galaxy is rebuilt from only a handful of them. "Dictionary" here means a
fixed set of unit directions in the 1024-space, each one a candidate concept. "Sparse" means few-at-a-time:
for any given galaxy, almost all dictionary entries are switched off and only a small set fire. If the
method succeeds, each dictionary entry comes to stand for one human-interpretable thing, because the
pressure to explain every galaxy with only a few active entries forces the network to spend each entry on a
recurring, meaningful pattern rather than on noise. The galaxy's full 1024-vector is then approximated as a
short weighted sum of active dictionary directions. We then look at which entries fire for which galaxies
and ask what each one means. The entries we can name (they fire for red galaxies, or for mergers) confirm
the concepts we expected. The entries we cannot name but that are stable across training runs are the
candidates for structure beyond the human taxonomy.

That is the whole idea. An autoencoder that compresses a galaxy into a few active "concept" units and then
reconstructs it, where the units are the things we read off afterward. The rest of this section makes each
of those words precise.

### 12.2 The TopK construction, symbol by symbol

An autoencoder has two halves. The encoder maps the input vector to a set of latent activations (the
dictionary "switches", one number per dictionary entry, mostly zero). The decoder maps those activations
back to a reconstructed input vector. Training pushes the reconstruction to match the original. We used the
TopK variant of the sparse autoencoder, which enforces sparsity in the most direct way possible: keep only
the $k$ largest latent activations for each galaxy and set every other latent to exactly zero.

Write the input galaxy vector as $x \in \mathbb{R}^{1024}$. This is the z-scored full embedding
$E_\text{full}$ (image + photometry + redshift fused; z-scoring per dimension is defined in Section 4
(data)). Let $m$ be the dictionary size, the number of latents. The encoder is

$$ f = \mathrm{TopK}\big(W_\text{enc}\,(x - b_\text{pre}) + b_\text{enc}\big), \qquad f \in \mathbb{R}^{m}. $$

Reading this left to right: $b_\text{pre} \in \mathbb{R}^{1024}$ is a learned pre-bias subtracted from the
input, which roughly centres the data so the dictionary does not waste entries representing the cloud's mean
position. $W_\text{enc} \in \mathbb{R}^{m \times 1024}$ is the encoder weight matrix; each of its $m$ rows is
a direction in the 1024-space, and the dot product of that row with the centred input gives one pre-activation
score, "how strongly does this galaxy point along dictionary direction $i$". $b_\text{enc} \in \mathbb{R}^{m}$
is a per-latent bias added to those scores. Then $\mathrm{TopK}(\cdot)$ is the operation that does the
sparsifying: it looks at the $m$ scores, finds the $k$ largest, keeps those values, and zeroes the rest. So
$f$ has at most $k$ non-zero entries. Those non-zero entries are the dictionary entries that "fire" for this
galaxy, and their values are the firing strengths.

The decoder rebuilds the input from the few active latents:

$$ \hat{x} = f\,W_\text{dec} + b_\text{pre}, \qquad \hat{x} \in \mathbb{R}^{1024}. $$

Here $W_\text{dec} \in \mathbb{R}^{m \times 1024}$ is the decoder weight matrix; its rows are the dictionary
directions themselves (the actual concept vectors we interpret later). Because $f$ has only $k$ non-zero
entries, $\hat{x}$ is literally a sum of $k$ dictionary directions, each scaled by its firing strength, plus
the pre-bias added back. The same $b_\text{pre}$ that was subtracted in the encoder is added back here, so
the network only ever has to model how a galaxy differs from the centre.

Training minimises the squared reconstruction error $\lVert x - \hat{x} \rVert^2$ averaged over all 48,398
galaxies, with $k$ held fixed. There is no separate penalty term coaxing the latents toward zero, which is the
appeal of TopK over older SAE recipes: the sparsity is exact and is set by one integer, $k$, not by tuning a
soft penalty weight against the reconstruction loss.

### 12.3 Two control terms: $L_0$ and the AuxK revival

The sparsity level has a standard name, $L_0$. The $L_0$ "norm" of a vector counts its non-zero entries (it
is not really a norm, but the name is conventional). For a TopK SAE, $L_0 = k$ exactly and by construction:
every galaxy uses exactly $k$ active latents, never more, never fewer. So when we say the operating point is
$k = 32$, we are saying each galaxy's 1024-vector is rebuilt from exactly 32 active dictionary entries out of
the thousands available. $L_0 = k = 32$ is the headline sparsity of this study.

A practical failure mode of any large-dictionary autoencoder is dead latents: entries that, after some
training, never make the top-$k$ for any galaxy, get no gradient, and stay dead forever. A dead latent is
wasted capacity; it is a dictionary slot that learned nothing. To counter this we used the AuxK term, an
auxiliary loss that revives dead units. The mechanism: take the reconstruction residual $x - \hat{x}$ (the
part of the galaxy the main top-$k$ latents failed to explain), and ask the current set of dead latents,
specifically the top 512 of them by pre-activation, to reconstruct that residual on their own. This auxiliary
reconstruction is added to the loss with a small weight ($1/32$ in our run). The effect is gentle and
targeted. Latents that have gone dead get a reason to become useful again (they are rewarded for explaining
what the live latents missed), without disturbing the main reconstruction. Without AuxK, a large fraction of
the dictionary tends to die early and the effective dictionary shrinks; with it, more entries stay in play.

### 12.4 Expansion factor and dictionary size

The dictionary size $m$ is set by the expansion factor $R$, defined as the ratio of dictionary entries to
ambient dimensions:

$$ m = R \times 1024. $$

$R$ controls how over-complete the dictionary is. $R = 1$ would give exactly as many entries as dimensions
(no room for superposition to be unpacked); $R > 1$ gives more entries than dimensions, which is what lets
the SAE assign separate entries to the many tilted concept directions that share the same coordinates. We ran
two settings:

| Expansion $R$ | Dictionary size $m = R \times 1024$ | FVU (mean over 5 seeds) | Fraction of latents alive |
|---|---|---|---|
| 4 | 4,096 | $\approx 0.035$ | $\approx 0.65$ to $0.70$ |
| 8 | 8,192 | $\approx 0.034$ | $\approx 0.33$ |

Both values are measured, from the `health` block of `results/sae.json`, averaged over five random
initialisation seeds (the per-seed FVU values for $R=4$ run from 0.03502 to 0.03599; for $R=8$ from 0.03411
to 0.03469). Two things stand out, and both are expected. First, doubling the dictionary from 4,096 to 8,192
barely improves reconstruction (FVU falls only from about 0.035 to 0.034). The extra capacity is not buying
much fidelity, which says 4,096 entries are already enough to capture nearly all the structure at this
sparsity. Second, the fraction of entries that stay alive roughly halves, from about two-thirds at $R=4$ to
about one-third at $R=8$. That is the expected behaviour. With twice as many slots competing for the same
fixed 32 firing opportunities per galaxy, more slots end up unused. Because the larger dictionary gains
almost no reconstruction quality while wasting two-thirds of its entries, we did the concept analysis of
Section 13 on the $R=4$ dictionary ($m = 4{,}096$), which is the better-occupied and more economical choice.

### 12.5 FVU and "alive", defined

Two diagnostics appear above and run through the rest of the SAE work, so define them cleanly now.

FVU is the fraction of variance unexplained. It measures how badly the reconstruction misses, on a scale where
0 is perfect and 1 is useless:

$$ \mathrm{FVU} \;=\; \frac{\sum_i \lVert x_i - \hat{x}_i \rVert^2}{\sum_i \lVert x_i - \bar{x} \rVert^2}, $$

where the sum runs over all galaxies $i$, $\hat{x}_i$ is the SAE reconstruction of galaxy $i$, and $\bar{x}$
is the mean embedding vector. The numerator is the total squared reconstruction error; the denominator is the
total variance of the data around its mean (the error you would get from the trivial "predict the average
galaxy" baseline). So FVU is the residual error expressed as a fraction of the variance you started with.
$\mathrm{FVU} = 0.036$ means the reconstruction leaves 3.6% of the variance unexplained, equivalently it
explains $1 - 0.036 = 96.4\%$ of the variance. We also write that as "96.5% variance explained" following the
rounded health figure; the two phrasings are the same quantity. Lower FVU is better reconstruction.

"Alive" is the fraction of dictionary entries that actually fire for at least some galaxies over the course of
training, the complement of the dead-latent fraction from Section 12.3. An alive fraction of 0.65 at
$m = 4{,}096$ means roughly 2,660 of the 4,096 entries are in use; the rest never make the top-32 and carry no
information. Only alive, used entries are eligible to be interpreted as concepts in Section 13, which is why
the concept count there starts from $n_\text{active} = 2{,}657$ entries rather than the full 4,096. The alive
fraction matters because a dictionary that looks large on paper but is mostly dead has a smaller true capacity,
and reporting the nominal size alone would overstate how many concepts the SAE could possibly hold.

### 12.6 The reconstruction-versus-sparsity frontier, and why $k = 32$

The single most consequential choice in this method is $k$, the number of active latents per galaxy. It sets
a trade-off. Small $k$ means very sparse, very interpretable codes (each galaxy explained by a few entries),
but a poor reconstruction (too few entries to capture the galaxy faithfully). Large $k$ means an excellent
reconstruction but codes so dense that "sparse" stops meaning anything and the entries blur back toward the
superposed mess we were trying to undo. We need a value that reconstructs the embedding well while keeping the
code genuinely sparse. To find it, we swept $k$ from 4 to 128 on the $R=4$ dictionary and recorded the FVU at
each value. That sweep is the frontier.

![Figure 11. SAE reconstruction error (FVU) versus sparsity k, with the k=32 operating point.](figures/11_sae_frontier.png)

Figure 11 plots the trade-off. The horizontal axis is $k$, the number of active latents per galaxy (the
sparsity level $L_0$), running 4, 8, 16, 24, 32, 48, 64, 96, 128 from left to right; it is the dial we are
turning. The vertical axis is FVU, the fraction of variance unexplained (Section 12.5), running from 0 at the
bottom to about 0.14 at the top; lower is a better reconstruction. The blue curve with filled markers is the
measured FVU at each swept $k$, one point per value, joined to show the trend. The vertical dashed red line
marks $k = 32$, our chosen operating point, and the red annotation states its value, FVU $= 0.036$. Every
plotted point is measured directly from the `frontier` block of `results/sae.json`.

Read the curve as a classic diminishing-returns elbow. The measured values are: $k=4 \to \mathrm{FVU}\,0.142$,
$k=8 \to 0.092$, $k=16 \to 0.058$, $k=24 \to 0.043$, $k=32 \to 0.036$, $k=48 \to 0.026$, $k=64 \to 0.023$,
$k=96 \to 0.018$, $k=128 \to 0.015$. The left part of the curve is steep: going from $k=4$ to $k=8$ cuts FVU
from 0.142 to 0.092 (about a third of the remaining error gone for 4 more latents), and $k=8$ to $k=16$ cuts
it again to 0.058. Each early latent buys a large chunk of fidelity. Then the curve bends. Past $k=32$ it
flattens hard: doubling the budget from 32 to 64 latents moves FVU only from 0.036 to 0.023, and doubling
again to 128 reaches just 0.015. Those extra latents buy little reconstruction but cost a lot of sparsity, and
the more entries fire per galaxy the harder each one is to read as a single clean concept.

$k = 32$ sits right at the knee. At that point the reconstruction already explains 96.4% of the variance
(FVU 0.036), which is enough to say the dictionary faithfully represents the embedding, while the code stays
tight: 32 active entries out of about 2,660 alive ones means roughly 99% of the dictionary is silent for any
given galaxy. Pushing to $k=64$ would recover only a further 1.3 percentage points of variance (FVU 0.036 to
0.023) at the price of doubling the active-set size and muddying interpretation. Pulling back to $k=16$ would
keep the code sparser but at FVU 0.058 the reconstruction starts to lose real structure. So $k=32$ is the
honest balance point, good fidelity at genuine sparsity. This is an engineering choice read off the elbow of a
measured curve, not a physical constant; a reader who wanted sparser codes at some reconstruction cost could
defensibly pick $k=16$, and the qualitative concept results would survive. We state $k=32$ as our operating
point and carry it into Section 13.

### 12.7 Why this method answers question C

It is worth being explicit about why we went to the trouble of an SAE at all, when Section 10 already showed
that physics is strongly decodable from the embedding. The probes and the SAE answer different questions.

A probe is supervised and targeted. You name a property in advance (redshift, stellar mass), fit a linear map
from the 1024 numbers to that property, and read off how well it works. A high probe $R^2$ tells you the
embedding contains that property, recoverably and linearly. But the probe can only ever find what you put in
its mouth. It tests a hypothesis you already had. If AION-1 organises galaxies along some axis that no entry in
our label set corresponds to, a probe will never reveal it, because there is no label to probe for. That is the
ceiling of the confirmatory approach, and it is exactly the part of question C ("does the model organise by
concepts beyond the human taxonomy?") that probes cannot reach.

The SAE inverts the direction of inquiry. It is unsupervised: it never sees a single label during training. It
is handed only the raw embeddings and the demand to rebuild each galaxy from a few active dictionary entries.
Whatever directions it settles on are chosen by the structure of the embedding itself, by what recurs across
the 48,398 galaxies, not by what we asked for. So the dictionary it produces is the model's own proposed list
of axes. We can then sort that list two ways. Entries whose firing pattern lines up with a human label
(Section 13 measures this alignment against a shuffle null) are the model rediscovering colour, morphology,
and redshift on its own, an unsupervised confirmation of question C's confirmatory half. Entries that fire
stably across training seeds but match no label we have are candidates for the genuinely open half: structure
the model represents that our taxonomy does not name. We label these "alien candidates" and treat them with
heavy caution. They are correlational only, defined by what they fail to correlate with, and no causal or
ablation test was run on them in this study. Section 13 reports both halves, the named concepts and the alien
candidates, with their counts and their null thresholds.

The SAE, then, is the instrument that lets the embedding reveal its own organising axes instead of only
echoing back the ones we supplied. That is the precise sense in which it, and not the probes, addresses the
open part of question C.
