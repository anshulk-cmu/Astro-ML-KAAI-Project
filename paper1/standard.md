# The standard every diagnostic must meet

Binding for all nine diagnostics. Set after Diagnostic 1, which is the reference implementation. A diagnostic is not finished until every item holds. Where an item genuinely cannot apply, the diagnostic says so in the report and says why.

## 1. Substrate and model

- The encoder is frozen. Loaded from `config.MODEL`, never fine-tuned, always the same token budget and pooling.
- Headline probes run on the image-only substrate. The multimodal substrate appears only as a leakage ablation, labelled as such, never as a competing number.
- A probe is fit once on unperturbed data and evaluated on held-out rows. Nothing is ever fit to a perturbed condition, and no stratum gets its own refit as primary evidence.

## 2. Population

- State the population, the cut that defines it, and the number of objects at every stage from the full anchor down to the evaluated set.
- Establish that the label exists before using it. A column being non-null is not the same as a quantity being measured: check for placeholder values, model types that cannot carry the quantity, and sentinels.
- Verify the label's own distribution against what the sky requires, where such a requirement exists.

## 3. Nulls

- Every headline number is reported beside a named null from the seven types in the scoping document.
- Each null must itself be calibrated: state the analytic expectation and show the null reproduces it. A shuffled-label null that does not land on the chance floor is a broken null, not a weak signal.
- Report the null on every axis the result is reported on. If the result has an error and a radius, the null has an error and a radius.

## 4. Uncertainty

- Every reported statistic carries an interval, with the resampling unit stated.
- Bootstrap intervals over held-out predictions capture test-set sampling only. Where the conclusion could plausibly depend on the partition, repeat the whole fit over independent splits and report that spread alongside.
- Where a trend is claimed, report a rank correlation with an interval, not an eyeballed monotonicity.
- Where a covariate could confound a trend, report the partial correlation with the confounder held.

## 5. Measurement floors

- Establish what the label's own uncertainty is, and state whether the result is limited by the model or by the label.
- Where a model-free reference exists (raw pixels, a geometric expectation, an analytic scaling), compare against it. A result that does not beat its floor is a non-discriminating result and is marked as such.

## 6. Conventions

- Any convention the result depends on is verified, not assumed, and preferably verified numerically against the data rather than quoted from documentation.
- Conventions that govern an intervention (sign, handedness, frame) are pinned by a test that fails loudly, and are pinned before any expensive computation runs.

## 7. Consistency checks

Each diagnostic carries a block of checks that would fail if the implementation were wrong, chosen so that passing is informative:

- the null reproduces its analytic expectation;
- any identity the method implies is tested numerically rather than asserted;
- an independent implementation or an independent route to the same quantity agrees;
- results are stable under the arbitrary choices the method makes.

## 8. Artifacts and provenance

- One script per diagnostic, one results file, figures rendered by a separate script that reads only that results file.
- Every results file carries git revision, input file hashes, seed, environment, platform and wall time.
- Large regenerable arrays are not tracked; the results file and figures are.
- `runAll.py --verify` must pass before a section is written.

## 9. Writing

- The technical report is a professional academic record of method and result. Development history, defects, reruns and decisions live in the project run log and never in the report.
- Report everything measured. Do not select a headline; that choice belongs to the paper, later.
- Tag each claim measured or interpreted, and keep the two grammatically separate.
- State the evidential tier in the heading and in the master table.
- Give the mathematics that the result depends on, in full, once, where the reader meets it.
- Every section states what the result does **not** establish, including the specific alternative explanations it cannot exclude.
- Cite what was checked and how it was checked, in a per-diagnostic citations table.
- Before a section is considered complete, run an automated comparison of every number in it against its artifact.
