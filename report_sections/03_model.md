## 3. The model and the representation we study

Everything in this report is a measurement of one specific object: the set of vectors that AION-1 produces when it looks at our galaxies. Before we can say anything about the shape of that set, we have to be exact about what the model is, what a "vector" means here, and which version of the vectors we use for which question. This section does that. It does not cover how the model was trained at scale or where the computation ran. It covers only what you need to read the geometry honestly.

### 3.1 What AION-1 is, in plain words

AION-1 is a foundation model for astronomical objects. "Foundation model" means a single large neural network trained once on a broad pile of data with no task-specific labels, then reused as a fixed feature extractor for many downstream questions. The variant we study is AION-1-Large: about 800 million tunable numbers (parameters), trained by self-supervision. Self-supervised means the training signal comes from the data itself, not from human annotations. Nobody told the model which galaxies are spirals and which are ellipticals. It never saw a morphology label. That fact is what makes the rest of this report interesting: any morphology structure we find in its vectors was not handed to it.

The model is a transformer. A transformer is a neural network built around attention, an operation that lets every part of the input look at and mix information from every other part. The input to a transformer is not a raw image or a raw number. It is a sequence of tokens. A token is a small chunk of the input turned into a vector, the network's basic unit of "a thing to attend to." AION-1 tokenises several kinds of astronomical data into one shared sequence: image patches, spectra, and individual catalogue scalars (single measured numbers like a flux or a redshift) each become one or more tokens. The point of putting different data types into one token stream is that attention can then relate, say, an image patch to a brightness measurement directly.

The training objective is masked modelling. During training the model hides (masks) a random fraction of the tokens and is asked to fill them back in from the tokens it can still see. To predict a masked image patch from the surviving patches and the surviving catalogue numbers, the network has to learn the real correlations among brightness, shape, colour, and distance, because those correlations are the only thing that makes the missing token predictable. This is the same trick that lets a language model learn grammar and facts by predicting hidden words. Here the "language" is the joint statistics of how galaxies actually look and measure.

Architecturally AION-1 is an encoder paired with a decoder. The encoder reads the visible tokens and produces a rich internal representation of each one. The decoder uses that representation to reconstruct the masked tokens. For our purposes the decoder is scaffolding. It exists to create a training signal, and once training is done we have no use for it. We keep the encoder.

### 3.2 From a galaxy to a single 1024-number vector

We use the model frozen. Frozen means we never update its parameters: no fine-tuning, no task-specific retraining, no gradient ever flows back into the weights. We feed a galaxy in, read numbers out, and that is all. Freezing matters scientifically. We want to characterise the representation the model already built on its own, not a representation we bent toward our labels. If we had fine-tuned, any concept axis we later "discovered" could just be an axis we trained in. The frozen model removes that loop.

When a galaxy passes through the frozen encoder, the encoder emits one vector per token. A galaxy is described by many tokens (many image patches, plus its catalogue tokens), so the raw output is a whole sequence of vectors, not one. We want a single fixed-length summary per galaxy so that every galaxy lives in the same space and we can compare them. We get it by mean pooling: take the per-token output vectors and average them, element by element, into one vector. Mean pooling is the simplest order-independent way to collapse a variable-length sequence into a fixed-length summary. It treats the galaxy's representation as the average of its parts. Other choices exist (for example using a special summary token, or attention-weighted pooling), and they could change fine detail; we note this as a method choice, not a tested variable. The pooled vector has length $d = 1024$. We call $d$ the ambient dimension: the number of coordinate axes of the space the vectors literally live in.

So the object we study is a cloud of points. Each galaxy is one point in a 1024-dimensional space. We have $N = 48{,}398$ such points (Section 4 covers the sample). A central question of this whole report is how many directions in that 1024-dimensional space the cloud actually uses, which is almost always far fewer than 1024. That is the intrinsic dimension, and Sections 6 and 7 measure it. For now the thing to hold onto is the data type: a fixed table of 48,398 rows by 1024 columns, one row per galaxy, produced by a frozen self-supervised model that never saw a morphology label.

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

Here is the design choice that the rest of the report leans on. We do not study one embedding per galaxy. We study two, built from the same frozen model but fed different inputs.

The first set is $E_{\text{full}}$. To build it we give the model the full multimodal input it expects: the four-band image (the $g$, $r$, $i$, $z$ filters, explained in Section 4), the photometric fluxes ($g$, $r$, $z$ brightnesses), and the galaxy's redshift (a distance proxy, also defined in Section 4). Image, photometry, and redshift are fused as input tokens, the encoder runs, and we mean-pool. $E_{\text{full}}$ is the richest representation the model can form for each galaxy, because it gets to use every measurement we have. We use $E_{\text{full}}$ for the pure-geometry arms of the report: intrinsic dimension, the diffusion map, curvature, topology. Those arms ask "what shape is the cloud," and for that question we want the model's best, most complete picture of each galaxy.

The second set is $E_{\text{img}}$. Here we feed the model only the image. No flux tokens, no redshift token. Same frozen encoder, same mean pooling, just a thinner input. $E_{\text{img}}$ is the image-only representation: whatever the model can say about a galaxy from its picture alone.

Why keep both? Because of leakage. Leakage is when the thing you are trying to predict was, in effect, an input, so predicting it proves nothing. Suppose you take $E_{\text{full}}$ and ask "can a simple linear readout recover the galaxy's redshift from this vector?" The redshift was fed in as an input token. A decodable redshift then tells you only that the model did not throw its own input away. That is a trivial, circular result. The same circularity hits colour, because the photometric fluxes that define colour were also fed in. Any "concept direction" you draw for redshift or colour in $E_{\text{full}}$ might be nothing more than the model echoing its input.

$E_{\text{img}}$ closes that loophole. Redshift and colour are not inputs to $E_{\text{img}}$; only the image is. So if a linear probe recovers redshift or colour from $E_{\text{img}}$, that is a genuine inference the model drew from pixels, not a readback. This is why we call $E_{\text{img}}$ the leakage-free set, and why every concept-probing arm (decodability in Section 10, disentanglement in Section 11) runs on $E_{\text{img}}$ when the property in question could have been an input. The cleanest evidence of all comes from properties that were never inputs to either set, like stellar mass and star-formation rate (Section 4): recovering those from $E_{\text{img}}$ cannot be leakage under any reading, because the model never received them in any form.

We can also turn the gap between the two sets into a measurement. By probing the same property in both $E_{\text{full}}$ and $E_{\text{img}}$ and comparing, we read off how much of the model's apparent skill is honest image inference versus input echo. Section 10 reports that ablation directly: for redshift the image-only probe reaches $R^2 = 0.80$ (measured) while the full multimodal probe reaches $0.98$ (measured), and the difference is the leakage we deliberately excluded by working in $E_{\text{img}}$. We flag the epistemic split now and quantify it there.

One reassurance about treating the two sets as the same kind of object. They sit at almost the same scale. The pooled vectors of $E_{\text{full}}$ have an average length (Euclidean norm) of about 48.4, and those of $E_{\text{img}}$ about 49.6 (measured, Section 2 facts). Both spreads are tight, only a few percent, so neither set is dominated by a handful of giant outlier vectors, and switching inputs did not blow the representation up or collapse it. The two clouds are comparable in size and live in the same 1024-dimensional space, which is what lets us read the geometry of one against the other.

The short version, carried forward into every later section: $E_{\text{full}}$ for the shape of the manifold, $E_{\text{img}}$ for honest concept claims, and the difference between them as a leakage meter.
