## 2. Background: galaxies as a measurable shape

To judge whether the embedding is faithful, we first have to know what a faithful summary of galaxies should reflect. This section lays out the physics in plain words, defines every term we use later, and ends with a quantitative prior on how many independent knobs the data really has.

It also states the one caveat that constrains every interpretation in this report: a present-day embedding shows a density of populations, not a movie of evolution.

The structure of the section is simple. First the vocabulary, every term defined once. Then the two big facts a faithful embedding should respect, the morphological continuum and the color-mass bimodality. Then quenching and its multiple channels, which is where the geometry gets interesting. Then the quantitative prior on dimension. Then the caveat.

A galaxy is a gravitationally bound system of stars, gas, dust, and dark matter. We do not observe any of that directly. We observe the light that reaches us, spread across the sky as an image and across wavelength as a spectrum or a set of broadband colors.

Almost everything astronomers say about a galaxy's physical state is inferred from that light. So when we ask whether a model's embedding reflects galaxy physics, we are really asking whether the arrangement of points lines up with the physical quantities astronomers themselves infer from the same light. Those quantities are the vocabulary of this section.

The light comes in two forms, and the difference between them runs through the whole report. An image is the light laid out on the sky: a picture, in a few broad filters, that shows shape and structure. A spectrum is the same light laid out by wavelength: a fine-grained curve that shows which atoms are present, how fast the gas moves, and exactly how far the light has been stretched. Images are cheap and exist for huge numbers of galaxies. Spectra are expensive and exist for far fewer. Most of the physical quantities below can be estimated from either form, with spectra giving the sharper answer. The model we study can in principle use both, though our run uses images plus a few catalogue numbers and no spectra (Section 4 is explicit about this).

### 2.1 The terms, defined once

**Redshift** ($z$) is how much the light from a galaxy has been stretched to longer (redder) wavelengths by the expansion of the Universe. Because the Universe expands, more distant galaxies recede faster and their light is stretched more, so a larger $z$ means a more distant galaxy seen at an earlier cosmic time. In our sample $0 < z < 1$, which reaches back several billion years.

The range $0 < z < 1$ in our sample is worth a sentence. A redshift of zero is here-and-now; a redshift of one corresponds to light that left its galaxy when the Universe was roughly half its present age. So our sample spans the recent half of cosmic history, a wide enough reach that redshift is a real source of variation the model has to handle, not a fixed background.

Redshift can be measured two ways. **Spectroscopy** disperses the light into a full wavelength spectrum and reads $z$ from the shift of known spectral lines; this is precise but expensive, so it exists for only a fraction of galaxies. **Photometry** measures the total brightness through a few broad filters and estimates $z$ from how the brightness is distributed across those filters; this is cheap and exists for almost everything, but the resulting **photometric redshift** (photo-z) is an estimate with real scatter, not a direct line measurement.

This distinction matters later. Most of our redshift labels are photo-z, so they are partly derived from color, and we treat color-related results accordingly.

The reason photo-z works at all is the same physics that makes color a useful number. The expansion of the Universe shifts a galaxy's whole spectrum toward longer wavelengths, so a more distant galaxy looks redder in the same set of filters. That shift is correlated with the galaxy's intrinsic color, so an estimate of redshift built from broadband filters is partly an estimate of color in disguise. When we later recover redshift from an image alone, we keep in mind that part of that skill could be the model reading color and inferring a color-based photo-z, rather than measuring distance from something independent.

**Photometry** more generally means brightness measured through filters. Our images use four filters named $g$, $r$, $i$, $z$. (Note: this filter $z$ is a band name and is unrelated to redshift $z$; we keep them apart in the text.) They run from bluer ($g$) to redder ($i$, and the $z$ band). A **magnitude** is an astronomer's brightness unit, defined so that a smaller number means brighter and a difference of 5 magnitudes is a factor of 100 in brightness. We will mostly use magnitudes through differences.

A **color** is the difference between two magnitudes, for example $g - r$ or $r - z$. Because magnitude is a logarithmic brightness, a color is essentially the ratio of brightness in two bands, and it tells you the shape of the light. A blue galaxy emits relatively more short-wavelength light and has a small $g - r$. A red galaxy emits relatively more long-wavelength light and has a large $g - r$.

Color is one of the most informative single numbers about a galaxy because it responds to the ages of its stars. Young, massive, hot stars are blue and short-lived; old, low-mass stars are red.

A galaxy still forming stars keeps making blue stars and stays blue. A galaxy that has stopped reddens as its blue stars die off and only the long-lived red stars remain. So color is a clock and a thermometer for star formation at once.

Two colors are better than one because they separate causes that a single color confuses. The $g - r$ color is sensitive mainly to stellar age and recent star formation. The $r - z$ color reaches farther into the red and adds sensitivity to dust and to redshift. Using both, as we do, gives the model and the probes two partly independent handles on what the light is doing, which is why we report decodability for $g - r$ and $r - z$ separately in later sections rather than collapsing color into one number.

**Stellar mass** ($M_\star$, in units of the Sun's mass $M_\odot$) is the total mass locked up in stars. We report it as $\log_{10}(M_\star/M_\odot)$ because it ranges over many factors of ten. The galaxies in samples like ours typically span roughly $\log_{10}(M_\star/M_\odot)$ from about 9 to 11.5, that is, from about a billion to a few hundred billion solar masses, so the log scale is not a convenience but a necessity.

Stellar mass is inferred by fitting models of stellar populations to the observed light, a procedure called **SED fitting** (spectral energy distribution fitting, where the SED is just the brightness as a function of wavelength). It is the single best proxy for "how big is this galaxy" in a physical sense.

Mass matters for geometry because it is one of the strongest organizing variables in the galaxy population. More massive galaxies are more likely to be passive and red; less massive ones are more likely to be star-forming and blue. So mass, color, and morphology are all braided together, and a faithful embedding should reflect that braiding rather than treating them as independent. The disentanglement question in Section 11, whether the model separates these correlated properties more than their natural correlation forces, is sharpened by exactly this braiding.

A galaxy is called **star-forming** (or active) when it is currently making new stars at a healthy rate, and **passive** (or quiescent) when it has largely stopped. These are the two physical states that the red and blue populations below correspond to. The cleanest single number to tell them apart is the specific star-formation rate, defined next; the color $g - r$ is a cheaper proxy for the same split. We use "passive" and "star-forming" throughout as the names for these two states.

**Star-formation rate** is how fast a galaxy is currently turning gas into new stars, in solar masses per year. Dividing it by stellar mass gives the **specific star-formation rate** (**sSFR**), in units of inverse time.

The sSFR answers a sharper question than the raw rate: not "how many stars per year" but "how fast is this galaxy growing relative to the stars it already has." A high sSFR means a galaxy is actively building itself. A low sSFR means it has largely stopped. The sSFR is the cleanest single number for the active-versus-passive distinction below, and like stellar mass it comes from SED fitting, so it exists for only a subset of our galaxies.

**Morphology** is the visual shape and structure of a galaxy: smooth and round, or featured with spiral arms, bars, and clumps. Astronomers compress shape into a sequence.

At one end are **early-type** galaxies, smooth featureless ellipsoids with little gas and few young stars. At the other are **late-type** galaxies, disks with spiral arms, gas, and ongoing star formation. The words "early" and "late" are historical labels for position along this sequence and carry no claim about which formed first; we use them only as names for the two ends.

Hubble arranged this sequence in his fork-shaped diagram, the tuning fork mentioned in Section 1, with smooth ellipticals on the handle and the spiral types fanning out along the tines.

A compact numerical handle on morphology is the **Sersic index** ($n$). Fit a galaxy's light profile, meaning how surface brightness falls off from the center outward, with a standard family of curves controlled by one number $n$. A small $n$ (around 1) describes an exponential disk, the light profile of a spiral. A large $n$ (around 4) describes a steep, centrally concentrated profile, the signature of an elliptical or a bulge-dominated galaxy.

So $n$ is a one-number summary that slides from disky to spheroidal, a structural cousin of the visual early-to-late sequence.

Like stellar mass and sSFR, the Sersic index comes from fitting a model to the image, so it exists for only a few thousand of our galaxies. We use it as one of the cleaner physical labels precisely because it is structural: it describes the shape of the light, not the colors that were partly fed to the model. Recovering a structural quantity like $n$ from the image is some of the better evidence that the embedding encodes real galaxy structure rather than echoing its inputs.

The Sersic index is worth keeping separate from color in your mind, because the two can disagree, and the disagreement is informative. Color reports on the stars' ages and the gas. Sersic index reports on the arrangement of light in space. A galaxy can be red yet still disky, or blue yet centrally concentrated. When color and structure agree, the galaxy sits cleanly at one end of both sequences. When they disagree, the galaxy is doing something interesting, and the quenching channels below are partly a story about which of the two changes first.

We also use **vote fractions** from a citizen-science classification (Galaxy Zoo), where many people answer questions like "is this galaxy smooth or featured?" and the fraction voting each way becomes a soft label. In our sample these fractions are predicted by a separate trained network from the images, accurate to roughly five to ten percent.

We use three at full sample size, **smooth**, **featured**, and **merger** (a galaxy in the act of colliding or merging with another), and two conditional ones (spiral-arm and bar fractions) defined only for galaxies already judged to be featured, non-edge-on disks. Section 4 covers these labels and their coverage in detail; here we only need them as named physical properties the embedding might track.

The conditional ones deserve a flag now because they shape what we can say later. A spiral-arm fraction only makes sense for a galaxy that is already a disk seen face-on; asking whether an elliptical has spiral arms is meaningless. So these labels exist for only a few thousand galaxies by design, not by accident, and any result that leans on them carries a wider error bar. When we test the "featured to spiral" concept hierarchy in Section 17 and report a null, the small spiral sample is part of why that test has limited power, and we say so there.

A last word on what these labels are. A vote fraction is a soft, human-flavored quantity: the share of people (here, the share predicted by a network trained on people) who would call a galaxy smooth. It is not a clean physical measurement like a flux. It carries the roughly five-to-ten-percent error of the predicting network and the fuzziness of human judgment. We use vote fractions because they are the best morphology labels available at full sample size, and we keep their softness in mind whenever a morphology result looks marginal.

### 2.2 Morphology is a continuum, not a set of boxes

The first physical fact a faithful embedding should reflect is that galaxy shapes form a continuum. Ellipticals and spirals are not two disconnected categories with a gap between them. They blend.

Between a pure smooth elliptical and a grand spiral sit lenticulars (disk galaxies with little gas and no arms), weak spirals, barred spirals, and every gradation in between. The Sersic index slides continuously from about 1 to about 4 with no forbidden values. The smooth and featured vote fractions trade off smoothly rather than splitting the sample into two islands.

This matters for what we expect to measure. If morphology were a small set of discrete boxes, a faithful embedding might show separate clusters with empty space between them, and a geometry measurement would find several disconnected pieces. If morphology is a continuum, a faithful embedding should be one connected body with smooth gradients across it.

The known physics predicts the second picture. Section 8 (diffusion maps) and Section 15 (topology) test directly whether the embedding is one continuous body or many islands. The morphological continuum is the physical reason we expect, and the standard against which we read, a single connected cloud.

There is a deeper reason the continuum picture is the right default, and it is worth stating because it guards against a common mistake. The human categories (elliptical, lenticular, spiral, irregular) are names we impose for convenience, not seams in the data. Nature did not draw boundaries between them. A model trained without those names has no reason to insert gaps where we drew lines. So if the embedding turned out to be cleanly clustered along the human categories, that would actually be surprising and would suggest the model had somehow absorbed the human scheme. A smooth continuum is both what the physics predicts and what an unsupervised model should produce, and finding one is therefore consistent rather than remarkable. We treat it as a passed expectation, not a discovery.

### 2.3 Color and mass come in two populations: the bimodality

The second fact is sharper and, at first sight, in tension with the first. When you plot galaxies by color and stellar mass, they do not spread out evenly. They pile up into two populations with a relatively empty zone between them. This is the **color-mass bimodality** ("bi-modality" meaning two peaks).

The **red sequence** is a tight band of red galaxies. These are mostly the early-type spheroidals: massive, gas-poor, with old stellar populations and little ongoing star formation. They are red because their blue stars have died and not been replaced. They are called passive (or quenched, defined below) because they have largely stopped forming stars. The red sequence is narrow, which tells you these galaxies share a fairly uniform old-stellar makeup.

The **blue cloud** is a looser group of blue galaxies. These are mostly late-type disks: gas-rich, actively forming stars, with young blue stars keeping them blue. It is called a cloud rather than a sequence because it is broader and more scattered, reflecting a wider range of ongoing star-formation activity.

Between them lies the **green valley**, a sparsely populated zone of intermediate color. Few galaxies sit here at any given moment. The standard reading is that galaxies cross the green valley relatively quickly as they shut down star formation, so you catch few of them mid-transition, the way few cars are parked in the middle of a one-way bridge.

The valley is not perfectly empty, and it is the region where the active-to-passive transition is happening.

Why two modes and not a smooth spread? Because the physics of star formation has a near-switch quality. While a galaxy has cold gas it keeps making stars and stays blue; once the gas is gone or stabilized, star formation drops sharply and the galaxy reddens over a comparatively short time. A process with a fast transition produces few objects caught mid-way, which is exactly the sparse green valley. The bimodality is therefore a fingerprint of how quenching works, not an arbitrary split, and that is why the next subsection is about quenching.

So there is no contradiction with the continuum picture, but there is a subtlety we have to hold. Morphology runs smoothly, yet color and star formation pile into two modes with a thin bridge.

A faithful embedding could plausibly show both: one connected body (because the populations are joined by the green-valley bridge and by the morphological continuum) that nonetheless has a denser red region and a denser blue region with a thinner zone between. Whether the model shows a single continuum with internal density structure, or genuinely separate clusters, is something we measure rather than assume.

The bimodality is the physical reason we look closely for any sign of a split. Section 16 (stratified intrinsic dimension) splits the sample at the color boundary on purpose to ask whether the two populations even have the same geometric dimension.

It is worth being precise about what bimodality is and is not, because the two-peaks picture can be over-read. Bimodality is a statement about density: two crowded regions and a thinner one between them. It is not a statement that the two regions are disconnected. A mountain range with two peaks and a low pass between them is bimodal in height yet fully connected as a surface. The red and blue populations are like that. The green valley is a low-density pass, not a chasm, and the morphological continuum and the transitioning galaxies in the valley keep the whole surface joined. Holding density and connectedness apart is what lets us reconcile the smooth-continuum fact of Section 2.2 with the two-population fact here: the embedding can be one connected body that is denser in two places. Which of those it actually is, we measure, and the answer (Sections 8 and 15) is the connected-but-uneven one.

### 2.4 Quenching, and the at-least-two channels

The process by which a galaxy stops forming stars is called **quenching**. A star-forming galaxy is quenched when its gas supply is removed, heated, or stabilized so that it can no longer collapse into new stars. After quenching, the galaxy reddens and joins the red sequence.

Quenching is the physical event that moves a galaxy from the blue cloud, through the green valley, to the red sequence.

What removes or stabilizes the gas is itself a list of distinct physical processes. Gas can be heated by energy released near a galaxy's central black hole. It can be stripped away as a galaxy falls through the hot gas of a larger group or cluster. It can be used up faster than it is replenished. It can be locked into a state stabilized by the galaxy's own deep gravity. We do not need the details for this report. We need only the consequence: because the causes differ, the routes to the red sequence differ, and that is the multi-channel picture.

The important point for geometry is that quenching is not believed to happen by a single mechanism. There appear to be at least two distinct channels, and they differ in both speed and accompanying structural change.

One channel is fast and is accompanied by a transformation of shape, a disk being disrupted into a spheroid. A violent event such as a major merger between two galaxies, or a strong burst that drives gas out, can shut down star formation quickly and rebuild a disky galaxy into a centrally concentrated spheroidal one. In this channel morphology and color change together and fast: the galaxy crosses the green valley quickly and arrives on the red sequence with a high Sersic index.

The other channel is slow and need not destroy the disk. A galaxy can run low on fresh gas, or have its gas held in a state that resists collapse, and fade in star formation gradually while keeping much of its disk structure. Here color reddens over a long time while morphology changes little, so the galaxy can arrive in the red population while still looking disky.

These are physical hypotheses from the galaxy-evolution literature, stated here as the prior context for our interpretations, not as something this report proves.

The reason they matter geometrically is specific. If there really are distinct routes from blue to red, an embedding that captures galaxy physics might show weak branching: a mostly connected body that splits, near the transition region, into separable strands corresponding to different quenching paths. That is a precise geometric prediction, and it connects to curvature.

A branch point in a manifold is a place of locally negative curvature, a saddle where the body divides. Section 14 (curvature) measures exactly this and asks whether the embedding shows the thin negative-curvature bridges that branching would produce. We will find weak, localized branching rather than a clean tree, and the two-channel picture is the physical hypothesis we hold that result up against.

We are careful, there and here, not to claim the geometry proves the channels exist. Consistency is not proof.

There is a second reason the channels matter for what we measure, and it concerns dimension rather than branching. If passive galaxies are assembled by more than one route, and from a wider range of starting structures, the passive population could occupy a sub-region with more independent degrees of freedom than the star-forming one, which is a younger and more uniform set still on its first build-up. That is a prediction about the relative intrinsic dimension of the two populations, and Section 16 tests it directly by measuring the dimension of the red and blue halves separately. We will find a small but real difference, with the passive half sitting slightly higher, and we hold the multi-channel picture up against that result as one possible reading among several.

### 2.5 How many knobs? The intrinsic-dimension prior

The last piece of background is a number, or rather a range, that sets expectations for the central geometry result. The question is how many independent quantities you need to describe a galaxy's light. This is the physical version of intrinsic dimension, the count of independent knobs, which Sections 6 and 7 measure directly for the embedding.

Start from photometry. Clean broadband photometry of galaxies, the handful of magnitudes and colors, is known to be highly redundant. The colors of normal galaxies trace out a thin, low-dimensional locus rather than filling the available color space, because the underlying drivers (stellar age, the amount of dust, the metal content, the redshift) are few and tightly correlated.

Studies that estimate the effective dimension of galaxy photometry land low, in roughly the **2 to 5** range. In words: once you know a few numbers about a galaxy's light, the rest is largely predictable.

Now add spectra. A galaxy spectrum has thousands of wavelength channels, but those channels are not independent either. The spectrum is generated by a modest number of physical drivers (the mix of stellar ages, the metal content, the dust, the gas emission, the velocity broadening), so its effective dimension is far below its channel count.

Analyses of large spectroscopic samples find that a few principal components, on the order of **3 to 10**, reconstruct most galaxy spectra. The spectrum carries more independent structure than broadband photometry, but still only a handful of effective axes.

A principal component, used loosely here, is a single combined direction that captures as much of the variation as possible; saying "a few principal components reconstruct most spectra" means a few combined directions explain most of how spectra differ. Section 6 makes this precise and turns it into one of our four dimension estimators. For now the takeaway is that both the cheap measurement (photometry) and the rich one (spectra) point to a small number of real degrees of freedom in a galaxy's light.

Put these together for a model that, in principle, fuses images, photometry, and spectra. An honest multimodal embedding of galaxies plausibly needs on the order of **5 to 10** independent axes to capture the real variation, with maybe a few more for structural and morphological detail that pure photometry misses.

We stress what this number is and is not. It is an optimistic ceiling assembled from prior work, a plausibility band, not a point prediction and not a target. The true value could sit above it if the model encodes structure the photometric and spectroscopic studies did not, or if some axes carry observational nuisances rather than physics.

We carry the **4 to 10** band into Sections 6 and 7 as a reference line on the plots. And we report honestly that our measured intrinsic dimension, near 10 to 12, sits at or just above the top of that optimistic band rather than comfortably inside it. The prior frames the result; it does not decide it.

Why might the measured value land above the optimistic band? A few honest reasons. The model encodes morphology, the detailed arrangement of light, which pure photometry throws away and which plausibly adds a handful of real axes. The image carries observational nuisances (orientation, the point at which the galaxy sits in the survey, instrumental effects) that a faithful representation might keep, inflating the count above the purely physical degrees of freedom. And the prior band itself was assembled for cleaner, more curated measurements than a raw multimodal embedding. So a value modestly above the band is not a contradiction of the physics; it is the kind of result the physics leaves room for. What would be a contradiction is a value near the ambient 1024, which would say the model failed to compress at all. We are far from that.

### 2.6 The load-bearing caveat: a density, not a movie

One caution governs every physical reading in this report, and it is easy to forget, so we state it as plainly as possible.

Our embedding is a snapshot. Each galaxy is observed once, at one moment in cosmic time, and contributes one point. The cloud of points is therefore a **density of populations**: it shows where galaxies of various kinds sit and how common each kind is right now.

It is not a **movie**. No single galaxy is tracked as it ages, reddens, and quenches. We never see a point move.

A density map and a movie can look deceptively alike. Both can show a smooth path from blue to red. But one is a count of where galaxies sit today, and the other would be a record of individual histories, and only the first is in our data.

This sounds obvious but it has teeth, because the language of evolution is tempting and wrong here. When we find a smooth bridge between the blue and red regions, it is tempting to say the model "learned" that galaxies travel across the green valley. It did not, and we cannot show that.

What we can say is that the present-day population has galaxies of intermediate color in between, and the embedding places them in between. That is a statement about where populations sit, not about any galaxy's history.

Likewise, when we find weak branching near the transition region and connect it to the two quenching channels, we are matching a static geometric feature to a dynamical hypothesis. The match is suggestive and worth reporting. It is not evidence that the model represents time.

So throughout, "the manifold connects the blue and red populations" is a measured statement about a static density. "Galaxies move along the manifold as they quench" is a story we are forbidden from claiming, because nothing in a single-epoch embedding can establish motion. We keep that line bright. Where we slip toward evolutionary language for readability, read it as shorthand for population structure, never as a claim that the model captured evolution.

The same caution applies to the multi-channel quenching picture. When we find weak branching and connect it to distinct quenching routes, the branching is a measured static feature and the routes are a dynamical story. The embedding could show that geometry for reasons that have nothing to do with quenching dynamics. We report the match because it is genuine and interesting, and we refuse to upgrade it to proof.

With the physics fixed and the prior stated, the next thing to pin down is the instrument: what AION-1 is, how it turns a galaxy into 1024 numbers, and which exact version of the embedding we measure. Section 3 sets out the model and the representation, including why we study two embedding sets rather than one and why the image-only set is the honest place to test for genuine inference.
