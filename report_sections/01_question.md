## 1. The question, and why it is still open

Start with the object at the center of this report: an embedding. When a large neural network looks at a galaxy, it does not store the picture. It produces a list of numbers. In our case that list has 1024 entries, so each galaxy becomes a single point in a 1024-dimensional space.

That point is the model's compressed opinion about everything it can tell from the inputs it was given. Two galaxies the model judges to be similar land close together. Two it judges different land far apart. The collection of all these points, one per galaxy, is the embedding, and it is the only thing we study here.

Why 1024 and not some other number? Because that is the width the model's designers chose for its internal vectors. The width is a capacity, not a count of meaningful features. A model can have a wide internal vector and still use only a small part of it, the way a spreadsheet with a thousand columns might have real data in only ten. Distinguishing the capacity (1024) from the part actually used (the intrinsic dimension) is one of the first things this report does.

We never ask the model to label anything. We ask where it puts things and what the arrangement looks like.

This is a different exercise from the usual way machine-learning models are judged. The common question is "how well does it perform on a task," scored by accuracy on some benchmark. Our question is "what shape did it build," which is prior to any task. A model can score well and still have an arrangement that is opaque, tangled, or higher-dimensional than the data warrants. By measuring the shape directly, we get at something the benchmark numbers hide: not whether the representation works, but what it is.

### 1.1 What an embedding is, in plain words

A foundation model is a single large model trained once on a large pile of unlabeled data, then reused for many tasks without retraining. The phrase comes from the idea that the model is a shared foundation you build on top of. The model we study, AION-1, is one of these.

It was trained on astronomical survey data by a self-supervised objective, meaning it learned from the data alone with no human-supplied answer key. Nobody told it which galaxies are spirals and which are ellipticals. Nobody handed it a star-formation rate. It learned by filling in deliberately hidden pieces of its inputs, and to make those guesses it had to build an internal summary of each object. That summary, read out as a 1024-number vector, is the embedding we measure.

So the embedding is a learned 1024-number summary of a galaxy. Hold onto that plain definition.

Everything in this report is an attempt to describe the shape of the cloud those summaries make, and to ask what the shape means. The number 1024 is the ambient dimension, the count of coordinate axes the vectors literally live along. It is not the number of axes the cloud actually uses, which is almost always far smaller, and which Sections 6 and 7 measure.

One more word on why a cloud of points is the right way to think about this. We are not studying the model's weights, the millions of internal numbers fixed during training. We are studying its outputs, one vector per galaxy. The weights would tell us how the machine works; the outputs tell us how it sees the data. Those are different questions, and this report is firmly about the second one. The shape of the output cloud is a fingerprint of how the model has chosen to arrange galaxies, and that arrangement is what we can compare against known physics.

The size of the sample matters too. We analyze $N = 48{,}398$ galaxies (Section 4 gives the selection). That count is not just a detail of scale. It sets a hard ceiling on how fine a geometry we can even resolve, because you cannot measure structure with more independent directions than your data can populate. We make that ceiling explicit later as $\log_2(N) \approx 15.56$, and every dimension estimate is read against it.

### 1.2 The claim the authors made, and the gap they left

The team that built AION-1 reported that the model organizes objects along directions that line up with physically meaningful properties. In plain words: move along one direction in the embedding and a real galaxy property changes smoothly. That is a strong and interesting claim, and our own probes in later sections give it real support.

But it leaves a gap. Saying that some directions are meaningful is not the same as describing the whole shape.

It does not tell you how many independent directions the cloud actually uses. It does not tell you whether the cloud is one connected body or several separate islands. It does not tell you whether the body is flat, curved, or branched like a tree. It does not tell you whether the model found structure that no human label names.

The authors measured that the embedding is useful. They did not measure its geometry. That gap is the opening for this work.

To be fair to them, usefulness is the harder thing to demonstrate and the more immediately valuable one. A model that lets you read off galaxy properties cheaply is worth a great deal regardless of its shape. Our point is not that the shape question is more important. It is that the shape question is separate and unanswered, and that answering it tells you something the usefulness numbers cannot: whether the model's view of galaxies is simple or tangled, continuous or broken, and whether it contains structure no one has named.

### 1.3 Geometry, in a concrete and measurable sense

We treat "geometry" here as something measurable, not as a metaphor. A cloud of points in high-dimensional space has a shape, and that shape has properties you can estimate with numbers and report with error bars.

Five properties cover most of what we mean by shape.

How many directions does the cloud really fill, its intrinsic dimension? Does distance behave sensibly inside it, or does every point look equally far from every other, the concentration problem? Is it one piece or many, its connectedness? Does it have loops or holes, its topology? Does it curve like a ball, bend like a saddle, or split like a branch, its curvature?

Each maps to a section. Dimension is Sections 6 and 7, concentration is Section 5, connectedness and topology are Sections 8 and 15, and curvature is Section 14. Reading them together is how we assemble a single coherent description of the shape rather than five disconnected numbers.

These are not vague questions. Each has a definition, an estimator, a validation on data with a known answer, and a confidence interval. The body of this report is the set of those answers.

A word on the word "manifold," which recurs throughout. A manifold is a shape that looks flat if you zoom in close enough, even when it curves on a large scale, the way the ground under your feet looks flat although the Earth is a ball. When we say the embedding may live on a low-dimensional manifold inside the 1024-dimensional space, we mean the points may sit on a thin curved sheet with only a handful of independent directions, not scattered through the full volume. Whether that picture holds, and how thin the sheet is, is the first thing we measure. We do not assume it. A self-supervised model could in principle produce a messy blob with no clean low-dimensional structure, and part of the work in Sections 6 to 8 is checking that it did not.

### 1.4 The two open questions we ask

We organize the work around two questions, and we name them so later sections can point back.

Question B is the native-geometry question. What is the intrinsic shape of the embedding? Concretely: its intrinsic dimension, its connectedness and topology, its curvature, and whether distance is even reliable inside it.

We call this the native geometry because we want the shape the model itself imposes, read with as few of our own assumptions as we can manage. Where a choice has to be made, which distance to use, which estimator to trust, we make it explicitly, test it on synthetic data with a known answer, and report what the choice does.

The word "native" is doing real work in that sentence. A high-dimensional cloud does not hand you a distance for free. You have to choose how to measure how far apart two points are, and that choice can change the answer to nearly every geometric question. Straight-line distance is the obvious option and often the wrong one in high dimensions. Section 5 makes this concrete and explains why we adopt a distance that follows the cloud's own density rather than cutting across empty space. "Native" means we work hard to read the geometry the model built, not an artifact of a careless distance choice.

We label them B and C, rather than starting at A, on purpose. There is an implicit Question A behind both, the confirmatory control discussed just below, and we keep the numbering to remind ourselves that the control is prior to and separate from the two questions that carry the report.

Question C is the concept question. Does the model organize galaxies by structure that goes beyond the human taxonomy? A taxonomy is just a naming scheme, the set of categories people use to sort galaxies: spiral, elliptical, merger, and so on.

A model trained without those names is free to carve the data along axes we never wrote down. Question C asks whether it did, and if so, whether those self-found axes correspond to anything we can recognize.

We answer it with a sparse autoencoder, a second small model that forces the embedding to explain each galaxy using only a few active units at a time, so we can read those units one by one (Sections 12 and 13). The appeal of this method is that it lets the embedding reveal its own axes, rather than us probing for human-named ones. When we fit a probe for "redshift," we have already decided redshift is the thing to look for. A sparse autoencoder does not start from our list of names. It carves the cloud along whatever directions are most economical, and only afterward do we ask which of those directions match something we recognize and which do not.

We flag now, and will repeat, that any "new" axis we find this way is correlational. We can show a unit tracks something, but we ran no causal test that would prove the model uses it. The honest version of the finding is "here are stable directions the model uses that no label we hold explains," not "here is a new physical concept the model discovered." That distinction is part of the answer.

### 1.5 The confirmatory question is a control, not the thesis

There is a third question floating in the background, and we keep it firmly in the background on purpose.

Call it the confirmatory question: did the model rediscover the things astronomers already know, like the smooth sequence from spirals to ellipticals (the Hubble tuning fork, named for the fork-shaped diagram Edwin Hubble drew to arrange galaxy shapes)? That is a control, not a thesis.

If the model had failed to recover well-established physics, we would distrust everything else it told us, so we check. But recovering known physics is the floor, not the finding. The finding we are after is the full shape and the candidate structure beyond the named categories.

We will be careful throughout not to dress up a passed control as a discovery.

There is one fact about the model that makes even the control interesting, and it is worth stating early because it changes how to read every result. The model never saw a morphology label. It was never told which galaxies are spirals, never given a star-formation rate, never shown the human categories. So when its embedding turns out to encode those properties, that is not the model repeating a lesson. It is the model having inferred the structure on its own from the raw light. A passed control here still says something: that the relevant physics is recoverable from the data without supervision. We hold that as the floor and look above it for the rest.

### 1.6 Where this sits in the machine-learning lineage

None of our tools are new to machine learning, and it helps to say plainly which lines of earlier work we stand on. We are not claiming to invent methods. We are applying established representation-geometry tools to an astronomical model that had not been measured this way, validating each tool on data with a known answer before we trust it on the real embedding.

There are three such lines, and we touch each one only as far as our results require.

The first line is the study of the intrinsic dimension of learned representations. A network may emit a 1024-number vector, but those numbers are usually far from independent. The data may really live on a much thinner sheet inside the big space, the way a sheet of paper is a two-dimensional thing even when it sits in a three-dimensional room.

People have repeatedly found that deep-network representations have an intrinsic dimension far below their nominal size, and that the small number is informative: it bounds how much independent structure the representation actually carries.

There is a tension in this literature that we have to respect. A low intrinsic dimension can mean two very different things. It can mean the model found the real, compact structure of the data, which is the good story. Or it can mean the model collapsed, throwing away usable directions and crowding everything onto a thin sheet, which is a known failure mode of self-supervised training called dimensional collapse. A small number alone does not tell you which. So we do not treat a low dimension as automatic proof of a clean manifold. We check the shape itself (Sections 8, 14, 15) and ask whether the low-dimensional sheet is smooth and physically organized, which is what separates genuine compression from collapse.

We measure the dimension directly for AION-1 with four independent estimators (Sections 6 and 7). Our headline, an intrinsic dimension near 10 to 12 out of 1024, is read against the $\log_2(N)$ sampling ceiling and against several synthetic controls, not asserted on faith.

The second line is the geometry of concepts: the idea that a human-understandable property can correspond to a direction in representation space, so that adding a fixed vector moves you from "without the property" to "with it."

This is the picture behind word-vector analogies in language models and behind linear-probe studies more broadly. The classic example from language is that the direction from "man" to "woman" is roughly the same as the direction from "king" to "queen," so a property like gender behaves like a fixed step in the space. We borrow the same idea and ask whether galaxy properties like color or redshift behave like fixed steps in AION-1's space.

We use it in two ways. We fit linear probes to test how much of a property a single direction can recover (Sections 10 and 11). And we test one specific structural prediction from this literature, that a child concept should equal its parent direction plus an orthogonal part, in Section 17. We report the second test as an honest null: our data does not establish that clean hierarchy here.

A related idea from this line of work is disentanglement: the notion that a good representation keeps distinct concepts on distinct, ideally separate, directions, so that changing one property does not drag others along. Two galaxy properties are correlated in nature (redder galaxies tend to be more smooth, for instance), so we cannot expect their directions to be exactly perpendicular. The sharper question is whether the model separates them more than their natural correlation forces. Section 11 measures exactly that, comparing the angle between two learned directions to the angle the labels' own correlation would imply. We mention it here only to mark it as a third use of the concept-geometry idea.

The third line is broader and more speculative.

It is the idea that large models trained on enough data may converge toward a shared internal structure, so that the geometry of one model's representation reflects the structure of the world rather than the quirks of one training run. The appeal is obvious: if true, the geometry of a representation becomes a window onto the data's own structure. The risk is equally obvious: the appeal can lead you to over-read a single model's shape as a fact about nature.

We do not test convergence across models here; we have one model. We mention the idea only to place our question.

If a representation's geometry reflects real structure in galaxies, then measuring that geometry is a way of asking what the data itself is shaped like, seen through one careful instrument. That is a motivation, not a result, and we keep it labeled as such.

The caution cuts both ways. If two models agreed on a geometry, that might reflect shared structure in galaxies, or it might reflect shared training recipes and shared biases. With one model we cannot separate those, so we make no convergence claim at all. What we can do is measure one instrument carefully and report what it shows, which is a clean and modest goal. The cross-model comparison is something we flag in Section 21 as a clear next step, not something we attempt here.

### 1.7 What counts as an answer, and what does not

Two ground rules carry through every later section, and we set them here so they are not a surprise.

First, we separate what we measured from what we read into it. A measured statement is a direct readout with an error bar: an intrinsic-dimension estimate with a bootstrap interval, a probe score with a confidence interval, a count of connected pieces. An interpreted statement is the physical story we attach: that a smoothly connected cloud matches the known smoothness of galaxy populations, or that a thin region of negative curvature is consistent with a branch point.

The measurements are the spine of the report. The interpretations are flagged as interpretations, and where two readings are both consistent with the data, we say the data cannot decide.

This is not pedantry. The whole value of a geometry study is that the geometry is measured, not asserted. The moment we let an attractive physical story stand in for a measurement, we have given up the one advantage this approach has over hand-waving.

We want this line to be visible in the prose, not just promised here. So when you read a sentence like "the cloud is one connected body," that is measured: a counted number of pieces. When you read "this matches the smooth physical sequence from spirals to ellipticals," that is interpreted: a physical reading laid over the measurement. We try never to let the second kind of sentence stand in for the first.

Second, we always show the uncertainty and the baseline. A number with no error bar and no point of comparison is not evidence.

So an intrinsic dimension is reported against the resolution ceiling set by the sample size and against synthetic manifolds whose dimension we already know. A curvature is reported against a matched random cloud and against a synthetic tree, because the absolute number means little until you see what "more curved" and "less curved" look like for known shapes. A concept-alignment score is reported against a shuffled-label null, the value you would get by chance. Throughout, the comparison is the point.

### 1.8 Validate the ruler before measuring the thing

There is a third habit that runs through the report, and it follows from the second. Before we trust any estimator on the real embedding, we run it on synthetic data whose answer we already know, at the same sample size, with the same settings.

The reason is simple. Every estimator has biases. A dimension estimator might read high on a curved sheet, or inflate on a tight cluster, or sag at small scales where sampling noise dominates. You cannot tell a true reading from a biased one by looking at the real data alone, because you do not know the truth for the real data. That is the whole problem.

So we build a sphere whose dimension is exactly 5, a rolled-up sheet whose dimension is exactly 2, a flat plane of dimension 5, and a couple of deliberate trap shapes, and we check that each estimator returns the known answer on these before we point it at AION-1. When an estimator passes the knowns and the estimators agree with each other on the real data, the result is trustworthy. When an estimator reads oddly even on a known shape, we say so and down-weight it. Sections 6 and 7 carry this out in full, and it is why our intrinsic-dimension headline rests on agreement among independent methods that each passed their controls, not on any single number.

This is also why a negative result is a real result in this report. When the validated tool says "no significant effect," as it does for the concept-hierarchy test in Section 17, that null is as informative as a positive finding, because we have already shown the tool can detect the effect when it is present. We report the nulls plainly and do not bury them.

### 1.9 The arc of the report

It helps to know where this is going. The report moves from the instrument outward to the meaning.

After this question and the physics background, Section 3 describes the model and Section 4 the data. Section 5 settles the distance question. Sections 6 and 7 measure intrinsic dimension. Section 8 maps the shape and shows it is one continuous body. Sections 9 through 11 read physics off the manifold and test how much of it the embedding carries and how cleanly the concepts separate. Sections 12 and 13 hunt for structure beyond the human labels. Sections 14 through 16 measure curvature, topology, and whether different galaxy populations have different dimension. Section 17 reports the one test that returned a clean null, and Section 18 puts real galaxy pictures on the axes. The final sections synthesize, list the limits honestly, and say what a fuller study should do next.

Every one of those measurements is validated on known-answer data first, reported with its uncertainty, and compared to a baseline. That is the contract.

With those rules fixed, the next thing we need is the physics. Before we can say whether a geometry is faithful, we have to know what a faithful embedding of galaxies should reflect: the smooth sequence of shapes, the split between red and blue populations, and the rough number of independent knobs the data really has. Section 2 lays that out, and it ends with the load-bearing caveat that shapes the whole reading: a present-day embedding shows a density of populations, not a movie of galaxies evolving.
