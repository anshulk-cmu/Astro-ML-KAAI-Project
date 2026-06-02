## 22. Glossary of terms and symbols

This glossary collects every technical term and symbol the report uses, each defined in one or two plain sentences. It is meant to be read out of order: jump here whenever a word in an earlier section was unfamiliar. Where a term was introduced in a specific section, that section is named so you can find the full treatment. Definitions are kept short and faithful to how the term is actually used in this study, not to the most general textbook meaning.

A note on two recurring tags used throughout the report. "Measured" marks a number read directly off the data or a results file. "Interpreted" (or "we read this as") marks an inference we drew from measured numbers; it is a judgement, not a readout. The glossary itself defines terms and does not assign these tags, but the distinction matters when you carry a term back into the results sections.

### Model and representation

**Foundation model.** A large neural network trained once, on a broad task, so that its internal representation can be reused for many downstream problems without retraining. AION-1 is the foundation model studied here.

**AION-1.** The astronomical foundation model whose embedding geometry this report measures. The variant used is AION-1-Large, an 800-million-parameter encoder-decoder transformer trained by self-supervised multimodal masked modelling, with no morphology labels (Section 3).

**Transformer.** A neural-network architecture that processes a set of input pieces ("tokens") and lets every piece attend to every other piece, so each output vector is informed by the whole input.

**Token.** One input piece the transformer sees: a patch of an image, a chunk of a spectrum, or a single catalogue number, each turned into a vector. AION-1 tokenises images, photometry, and scalars and learns to fill in masked tokens.

**Encoder.** The half of the model that turns inputs into internal vectors. We keep the encoder and discard the decoder (the half that would reconstruct masked inputs).

**Masked modelling (self-supervised).** A training scheme where parts of the input are hidden and the model learns to predict them from the rest. "Self-supervised" means the supervision comes from the data's own structure, not from human labels.

**Frozen model.** A model whose weights are held fixed (no fine-tuning) while we study it, so the geometry we measure is the pretrained representation itself, not something we reshaped (Section 3).

**Embedding.** The fixed-length vector of numbers the model assigns to a galaxy: a learned summary. Here each galaxy gets one 1024-number embedding.

**Mean pooling.** Reducing the encoder's many per-token vectors to one vector per galaxy by averaging them element by element. This is how we go from a variable number of tokens to a single 1024-vector (Section 3).

**E_full.** The embedding set built from image (g, r, i, z bands) plus photometry (g, r, z flux) plus redshift, all fused as inputs. Used for the geometry arms (Section 3).

**E_img.** The image-only embedding set, where redshift and colour are not inputs. This is the leakage-free set: probing it for redshift or colour is a real inference, not reading back an input (Section 3).

**Ambient dimension ($d$).** The number of coordinates each embedding vector has, here $d = 1024$. It is the size of the space the points live in, not how many directions they actually use.

**Intrinsic dimension (ID).** The number of independent knobs you need to describe where a point sits on the data's underlying surface, which can be far below the ambient dimension. If galaxies really vary along about a dozen axes, the ID is about a dozen even though $d = 1024$ (Sections 6 and 7).

### Preprocessing and basic spread

**z-score (standardise).** For each of the 1024 dimensions, subtract that dimension's mean and divide by its standard deviation, so every dimension has mean 0 and spread 1. We z-score before all geometry because per-dimension spread varies by about 23 times, and without it a few high-variance dimensions would dominate every distance (Section 4).

**Standard deviation (std).** A measure of spread: the typical distance of values from their mean. A larger std means the values are more spread out.

**Norm.** The length of a vector, $\lVert x\rVert = \sqrt{\sum_i x_i^2}$. The embedding norms are tight (about 48 for E_full, about 49.6 for E_img, roughly 5% spread), so the cloud sits near a thin spherical shell (Section 4).

**PCA (principal component analysis).** A linear method that finds the orthogonal directions of greatest variance in the data, ordered from most to least. The directions are the principal components; how much variance each carries is its eigenvalue.

**Eigenvalue and eigenvector.** For a matrix $M$, an eigenvector $v$ is a direction that $M$ only stretches, $Mv = \lambda v$, and the eigenvalue $\lambda$ is the stretch factor. In PCA the eigenvalues are variances along the component directions; in diffusion maps they measure how slowly a walk mixes along each mode.

**Participation ratio (PCA-PR).** A linear estimate of how many directions matter, $\big(\sum_i \lambda_i\big)^2 / \sum_i \lambda_i^2$, where $\lambda_i$ are the PCA eigenvalues. It counts the effective number of comparable-variance directions: if a few dominate it is small, if many share the variance it is large. We use it as the linear baseline for intrinsic dimension (Sections 6, 7).

### Metric and distance battery

**Metric / distance.** A rule for how far apart two points are. The report stresses that the choice of metric is not a detail, because in high dimensions different metrics tell different stories (Section 5).

**Distance concentration.** The high-dimensional effect where, as dimension grows, all pairwise distances become nearly equal, so the contrast between near and far points collapses and nearest-neighbour structure gets unreliable (Section 5).

**RDR (relative distance range).** A concentration diagnostic, $\text{RDR} = (D_{\max} - D_{\min})/D_{\min}$ over all pairwise distances. Higher means more contrast between near and far, so less concentration (Section 5).

**NN/mean.** A second concentration diagnostic: the mean nearest-neighbour distance divided by the mean pairwise distance. Lower means points sit relatively closer to their nearest neighbour than to the crowd, so more concentration (Section 5).

**Euclidean distance.** Ordinary straight-line distance, $\sqrt{\sum_i (x_i - y_i)^2}$. Used as the control metric; it was the most concentrated (lowest contrast) in our battery (Section 5).

**Cosine distance.** Distance based on the angle between two vectors rather than their straight-line gap. Here it did not concentrate like Euclidean, so it acts as a middling control rather than a failure case (Section 5).

**Isomap (geodesic distance).** Distance measured as the shortest path along a graph that links each point to its near neighbours, so it follows the curved surface instead of cutting across empty space (Section 5).

**Geodesic.** The shortest path that stays on a curved surface, the surface-bound analogue of a straight line.

**Fermat distance (density-weighted geodesic).** A shortest-path distance where steps through dense regions are cheaper, so paths prefer to travel through where the data actually is; it is outlier-stable and has a published convergence guarantee for topology, which is why it is the primary metric for the topology and curvature arms (Sections 5, 14, 15).

**Diffusion distance.** A distance that averages over all paths between two points via a random walk on the neighbour graph, so it is smooth and density-stable (Sections 5, 8).

### Diffusion maps

**Diffusion map.** A method that builds a graph where near points are strongly linked, treats it as a random walk, and uses the slow directions of that walk as new coordinates. The slow directions are the large-scale shape of the cloud (Section 8).

**Affinity ($W_{ij}$).** The link weight between points $i$ and $j$ in the graph, large when they are close and small when far; it sets how easily the random walk hops between them (Section 8).

**Bandwidth.** The scale that decides how quickly affinity falls off with distance. We used a self-tuning local-scaling bandwidth (each point's scale is its distance to its 7th neighbour) after a single global bandwidth gave a fragmented graph (Section 8).

**Laplace-Beltrami normalisation (anisotropic $\alpha = 1$).** A reweighting of the affinity that removes the effect of how densely galaxies happen to be sampled, so the recovered geometry reflects the manifold's shape and not the sampling density (Section 8).

**Markov matrix.** The row-normalised affinity matrix whose entries are the one-step hop probabilities of the random walk; its eigenvalues and eigenvectors define the diffusion coordinates (Section 8).

**Diffusion coordinate (dc).** A coordinate built from an eigenvector $\psi_k$ scaled by its eigenvalue, $\psi_k \lambda_k$. We label them dc0, dc1, dc2, and so on; dc1 and dc2 are the physics-bearing axes, dc0 is a weak global mode (Sections 8, 9).

**Spectral gap.** A large drop between consecutive diffusion eigenvalues. A clear gap would mean discrete clusters; our spectrum decays smoothly with no dominant gap, which we read as one continuous body (Section 8).

**Harmonic.** A higher diffusion coordinate that is mostly a repeat (a polynomial function) of lower ones rather than a genuinely new axis. The harmonic screen flagged dc3 and dc4 as harmonics, so they carry no new geometry (Section 8).

**Connected component.** A piece of the graph in which every point can reach every other by hops. Both embedding sets form a single connected component.

### Probes and decodability

**Linear probe.** A single linear map fit from the 1024 embedding numbers to a target property, used to ask how much of that property the embedding linearly carries (Section 10).

**Ridge regression (RidgeCV).** Linear regression with a penalty on large weights to control overfitting; "CV" means the penalty strength was chosen by cross-validation over a grid of values (Section 10).

**$R^2$ (coefficient of determination).** The fraction of held-out (test-set) variance the probe explains: 1 is perfect prediction, 0 is no better than always guessing the mean (Section 10).

**Train/test split.** Holding out 20% of galaxies the probe never sees during fitting, so $R^2$ measures genuine prediction, not memorisation. We used an 80/20 split with a fixed seed.

**Bootstrap confidence interval (CI).** A range for an estimate found by resampling the data with replacement many times (here 1000 times) and reading off the middle 95% of the resulting values. It states our uncertainty about the number, not a spread in the data (Sections 7, 10).

**Leakage.** When a model is asked to predict something it was effectively given as an input, making the prediction trivially easy and the "concept" circular. Using E_img (image-only) for colour and redshift probes avoids leakage, so those probes are real inferences (Sections 3, 10).

**Modality ablation.** Comparing image-only against multimodal embeddings to see how much of the decodability of an input property (like redshift) is leakage versus genuine inference (Section 10).

**Disentanglement.** Whether two concepts occupy more nearly separate (orthogonal) directions than their own correlation would force. We measure the angle $\theta$ between two probe weight vectors, set a null angle from the labels' correlation, and call the excess $\theta - \text{null}$; positive excess means genuine extra separation (Section 11).

**kNN purity.** A model-free check: among a galaxy's $k$ nearest neighbours (here $k = 20$), what fraction share its label. High purity means like sits near like (smooth galaxies are 99.1% pure, featured 71.4%) (Section 11).

**Spearman correlation ($\rho$).** A rank-based correlation that measures whether two quantities rise and fall together, even nonlinearly; it runs from $-1$ to $+1$. Used to relate diffusion coordinates and SAE latents to labels (Sections 8, 9, 13).

### Sparse autoencoder (concept discovery)

**Sparse autoencoder (SAE).** A network that learns an over-complete dictionary so each embedding is rebuilt from only a few active "concept" units, letting the embedding reveal its own axes rather than us probing for human-named ones (Section 12).

**TopK SAE.** The specific SAE used: the encoder keeps only the $k$ largest latent activations per galaxy, $f = \mathrm{TopK}\big(W_{\text{enc}}(x - b_{\text{pre}}) + b_{\text{enc}}\big)$, and the decoder rebuilds $\hat{x} = f W_{\text{dec}} + b_{\text{pre}}$ (Section 12).

**Latent.** One unit in the SAE's hidden dictionary, with an activation strength per galaxy and a decoder direction (the vector it writes back into the embedding).

**L0.** The number of latents allowed to be active per galaxy, here $L0 = k = 32$. It is the sparsity budget: how many concept units describe one galaxy at a time.

**Dictionary size ($m$) and expansion ($R$).** The number of latents, $m = R \times 1024$, where the expansion factor $R$ (here 4 or 8) sets how over-complete the dictionary is.

**FVU (fraction of variance unexplained).** The share of embedding variance the SAE fails to reconstruct: 0 is perfect, smaller is better. At the $k = 32$ operating point, FVU is about 0.036, so 96.5% of variance is explained (Section 12).

**Dead and alive latents.** A latent is "dead" if it (almost) never activates and "alive" if it is used. The AuxK term revives dead latents by having the top dead units reconstruct the leftover error (Section 12).

**Alignment.** For one latent, the largest absolute Spearman correlation between its activations and any of the six labels. High alignment means the latent tracks a known property (Section 13).

**Label-shuffle null.** A baseline made by permuting the labels and recomputing alignment, giving the alignment a meaningless latent would reach by chance. Its 95th and 99th percentiles set the significance thresholds (Section 13).

**Seed-stability.** Whether a latent's decoder direction reappears across independent training seeds, here counted as present in at least half of the other four seeds with cosine $\geq 0.6$. A stable feature is a property of the model, not of one random run (Section 13).

**Novelty.** The fraction of a latent's activation variance that the six labels cannot linearly explain, computed as a regression residual. High novelty plus stability plus non-alignment defines an "alien" candidate (Section 13).

**Alien candidate.** A seed-stable, high-novelty, non-label-aligned latent: a direction the model uses consistently that none of our labels accounts for. These are correlational only; no causal or ablation test was run (Section 13).

### Curvature and topology

**Curvature.** How a surface bends. Positive curvature curls like a sphere and makes neighbourhoods cluster; negative curvature opens like a saddle and marks a branch or bridge; zero curvature is flat (Section 14).

**Gaussian and sectional curvature.** Standard differential-geometry measures of bending at a point. We do not compute them directly on the embedding; we use discrete, graph-based curvature notions instead (Ollivier-Ricci, Forman-Ricci) (Section 14).

**delta-hyperbolicity (Gromov 4-point delta).** A tree-likeness score from four-point comparisons, divided by the cloud diameter; lower means more tree-like and 0 is a perfect tree. We read it against anchors, not in absolute terms (Section 14).

**Anchor (baseline cloud).** A reference cloud with a known character used to calibrate a measurement: a matched-covariance Gaussian (structureless) and a synthetic tree (perfectly tree-like) bracket the delta-hyperbolicity reading (Section 14).

**Ollivier-Ricci curvature.** A signed graph curvature from the optimal-transport distance between the neighbour distributions of two linked points; positive means clustered, negative means a bridge or saddle. This is the trustworthy signed local measure (Section 14).

**Optimal transport (earth-mover distance).** The least total cost to move one probability distribution onto another, where cost is distance times mass moved. It underlies the Ollivier-Ricci curvature (Section 14).

**Forman-Ricci curvature.** A combinatorial graph curvature based on degrees and triangle counts. On a kNN graph it is structurally negative (degree dominates), so we use it only to rank candidate bridge edges, never to claim negativity (Section 14).

**Betti number ($\beta_0$, $\beta_1$).** Counts of topological features: $\beta_0$ is the number of separate pieces, $\beta_1$ is the number of independent loops or holes. We find $\beta_0 = 1$ and $\beta_1 = 0$ (Section 15).

**Persistent homology.** A method that grows balls around each point and tracks when topological features are born and die as the ball radius increases; long-lived features are real, short-lived ones are noise (Section 15).

**Persistence.** The lifetime of a topological feature (death radius minus birth radius), diameter-normalised here. A loop counts as significant only if its persistence exceeds 0.1 (Section 15).

**Minimum spanning tree (MST).** The lightest set of edges that connects all points with no loops. Cutting its longest edges is how we probe whether the cloud splits into pieces; cutting peeled off only single outliers (Section 15).

### Heterogeneity and concept hierarchy

**Gaussian mixture model (GMM).** A model that treats the data as a blend of Gaussian (bell-curve) components. A two-component GMM on g-r colour split the sample at a cut of 1.012 into red (passive) and blue (star-forming) (Section 16).

**Stratified ID / delta-ID.** Intrinsic dimension measured separately for two populations, with delta-ID their difference. The absolute small-scale IDs are noise-inflated, so the relative delta-ID = 1.66 (which excludes zero) is the trustworthy signal (Section 16).

**TwoNN.** An intrinsic-dimension estimator using each point's ratio of second- to first-neighbour distance, $\mu = r_2/r_1$, with cumulative law $P(\mu > x) = x^{-d}$ and unbiased estimate $\hat{d} = (N-1)/\sum_i \log \mu_i$; it is small-scale and noise-inflated on AION (Sections 6, 7).

**$\mu = r_2/r_1$.** The TwoNN ratio: a point's distance to its second nearest neighbour divided by its distance to its first. Its distribution across points encodes the intrinsic dimension (Section 6).

**Gride.** A generalised-ratio ID estimator that uses farther neighbour ranks to trace an ID-versus-scale curve, separating true dimension at larger scale from small-scale noise inflation, with no subsampling needed (Sections 6, 7).

**Levina-Bickel MLE (local MLE).** A maximum-likelihood ID estimate computed per point from its $K$ nearest-neighbour distances and swept over $K$, used to map where dimension is locally higher (Sections 6, 7).

**Concept direction.** A unit vector in the embedding such that moving along it changes one human-named property; recovered as a probe weight vector (Section 10) or an SAE decoder vector (Section 13).

**Linear concept hierarchy (Park et al. test).** The idea that a child concept's direction equals the parent's direction plus an orthogonal child-specific part. Our one test (featured to spiral) returned a null: measured cosine 0.072 below the permutation-null 95th percentile 0.084 (Section 17).

**Permutation null.** A no-effect baseline built by scrambling the labels and recomputing the statistic many times, keeping everything else fixed; the real value is significant only if it beats almost all scrambled ones (Sections 13, 17).

### Astronomy: galaxies as a measurable shape

**Redshift ($z$).** How much a galaxy's light is stretched to longer wavelengths by cosmic expansion, which tracks distance and look-back time. Our sample spans $0 < z < 1$ (Sections 2, 4).

**Photo-z (photometric redshift).** A redshift estimated from broad-band colours rather than a spectrum, so it is cheaper but noisier and partly colour-derived. Most of our redshifts are photo-z, which is why image-to-redshift partly routes through image-to-colour (Sections 4, 10).

**Spectroscopy vs photometry.** Spectroscopy splits light into a detailed wavelength spectrum (rich but expensive); photometry sums light into a few broad bands (g, r, i, z here). We have spectroscopic redshift for only 6,883 galaxies (Sections 2, 4).

**Morphology.** A galaxy's visual structure. It forms a continuum from smooth blobs to featured disks rather than discrete species (Section 2).

**Early-type / late-type.** Historical morphology terms: early-types are smooth ellipticals and lenticulars, late-types are featured spirals. The names do not imply an evolutionary order (Section 2).

**Passive vs star-forming.** Passive galaxies have largely stopped forming stars and are red; star-forming galaxies are actively making stars and are blue (Sections 2, 16).

**Quenching.** The process by which a galaxy stops forming stars. There are at least two distinct channels: a fast disk-to-spheroid transformation and a slow fade (Section 2).

**Stellar mass ($\log M_*/M_\odot$).** The total mass in stars, in units of solar masses on a log scale. Decodable from the image alone at $R^2 = 0.72$ (Sections 2, 10).

**sSFR (specific star-formation rate).** The star-formation rate divided by stellar mass: how fast a galaxy is growing relative to its size. Decodable from the image alone at $R^2 = 0.76$ (Sections 2, 10).

**Sersic index ($n$).** A single number describing how concentrated a galaxy's light profile is, with low $n$ for disky and high $n$ for bulgy/elliptical light (Sections 2, 10).

**Colour (g-r, r-z).** A difference of magnitudes between two bands, which measures the slope of a galaxy's spectrum; redder colours (larger g-r) indicate older or dustier stellar populations. These are the strongest single concepts in the embedding (Sections 2, 9, 13).

**Magnitude.** A logarithmic, inverted brightness scale: smaller magnitudes are brighter. Differences of magnitudes give colours.

**Red sequence, blue cloud, green valley.** The three regions of the colour-mass diagram: a dense red sequence of passive early-types, a dense blue cloud of star-forming late-types, and a sparse green valley of transitioning galaxies between them (Section 2).

**Vote fraction.** The share of human classifiers (or, here, a CNN trained to mimic them) who assigned a galaxy a given morphology label (smooth, featured, merger), accurate to roughly 5 to 10% (Section 4).

**Galaxy Zoo / decision tree.** The crowd-sourced morphology project and its branching question structure. Some questions (spiral arms, bars) are only asked of featured, non-edge-on disks, which is why those labels exist for only about 3,034 galaxies (Section 4).

This glossary closes the definitional debt of the report. Section 23 turns to the numbers themselves: every threshold and decision rule that any of these terms relied on, gathered in one place with its value and its meaning.
