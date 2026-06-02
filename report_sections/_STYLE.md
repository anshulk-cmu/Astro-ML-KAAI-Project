# SHARED STYLE & FAITHFULNESS CONTRACT (read before writing any section)

You are writing one section of a single long technical report:
"The Native Representational Geometry of an Astronomical Foundation Model (AION-1)".
The report is assembled by concatenating section files in order, so your section must read
as part of a continuous document, not a standalone essay. Do not repeat the global title.
Start with your section's own `##` heading (the outline gives the exact number and title).

## Audience and goal
A technical reader who may be an astronomer OR a machine-learning person, but not both.
So: explain every technical term the first time it appears, in plain words, then use it.
Explain the math AND the physics. Explain what we did, how we did it, and WHY we did it.
Explain every figure's axes, colour scale, legend, and annotations, and tell the reader
what to look for. Leave no ambiguity. Where a result is uncertain or our data cannot decide,
say so plainly ("our data cannot settle this"; "this is an intuition, not a measurement").

## The one rule above all: simple, clear language
Short common words over fancy ones. A bright 15-year-old should be able to follow the prose
even if the concept is advanced. Simple does not mean dumbed down; it means precise.
Technical terms (eigenvalue, intrinsic dimension, Ricci curvature, persistent homology) are
allowed and expected, but each must be defined in plain words on first use.

## Faithfulness (hard rules; this report must be defensible)
- No overclaiming. No invented facts, numbers, citations, or results. Every number you state
  must come from `_FACTS.md` or directly from the `results/*.json` files. If you are unsure of
  a number, read the JSON; do not guess.
- Tag the epistemic status: write "measured" for a direct readout, "interpreted" or "we read
  this as" for an inference. Keep the two visibly separate.
- Always report uncertainty where we have it (confidence intervals, bootstrap ranges, the null
  threshold, the anchor/baseline). Never present a point estimate as if it had no error.
- State the honest caveats that belong to your section (e.g. redshift is mostly photo-z; SAE
  "alien" features are correlational only, never causally tested here; the small-scale TwoNN
  reading is noise-inflated and is not the estimate; beta1 uses 2k subsamples; ID sits at or
  just above the optimistic prior band).
- Do NOT describe logistics (cloud, AWS, the download fleet, instance types). Write about the
  science only: data, methods, math, physics, results, meaning.

## Punctuation and vocabulary (AI-tell avoidance)
- Do NOT use em dashes ("—"). Use commas, parentheses, colons, or two sentences instead.
- Banned words/phrases (never use): tapestry, delve, realm, leverage (verb), robust (use
  "stable"/"reliable"), underscore, underpin, foster, encompass, intricate, multifaceted,
  pivotal, crucial(ly), notably, importantly, "it is worth noting", "it is important to note",
  "at its core", "in the realm of", "shed light on", "a testament to", "navigate"
  (metaphorical), "landscape" (metaphorical), "unlock", "harness", "delve into".
- Use "this is not just X but Y" at most once in the whole report (so basically avoid it).
- Contractions are fine and natural ("doesn't", "we'll", "it's"). Use them moderately.

## Rhythm and structure
- Vary sentence length. In any five sentences, the longest should be at least ~2.5x the
  shortest. Mix short punchy sentences with longer ones that carry a full idea.
- Vary paragraph length. Some one-sentence paragraphs for emphasis are good.
- Active voice by default ("we measured", "the estimator returns"), not "it was measured".
- Have a point of view. Say which evidence is strong and which is weak.
- End each section with a forward link or a consequence, not a flat restatement.

## Allowed structure for a TECHNICAL report (exceptions to the no-list default)
This is a technical document, so the following ARE allowed and encouraged where they help:
- Tables for: parameter/threshold summaries, results summaries, glossary entries, decision
  rules. Keep tables faithful to the numbers.
- Short numbered/lettered lists ONLY when listing parallel items (e.g. the four ID estimators).
  Put the actual explanation and reasoning in prose, not in fragments.
- Math: use inline LaTeX with `$...$` and display math with `$$...$$`. Define every symbol.
  Spell out what each equation means in words right after writing it.

## Figures (embed as you write)
Embed each figure assigned to your section using Markdown image syntax with a path relative to
the repo root, because the final report lives at the repo root:
`![Figure N. Short title.](figures/NN_name.png)`
After the image, write a full caption paragraph that:
1. says what is plotted on each axis (with units),
2. explains the colour scale / legend / every annotation and reference line,
3. tells the reader exactly what to look for and what it means (measured vs interpreted).
The exact filenames and what each figure shows are listed in `_OUTLINE.md` and `_FACTS.md`.

## Cross-references
Refer to other sections by number and short name (e.g. "Section 6 (intrinsic dimension)").
Do not redefine a term that an earlier section already defined; a one-line reminder is fine.

## Length
Hit at least the line target given in your task. Depth over filler: every paragraph must add
real content (a definition, a method step, a number, a physical meaning, a caveat). Do not pad.
