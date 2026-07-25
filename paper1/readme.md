# paper1

Everything for Paper 1 in one folder: the code for all nine diagnostics, their results, their figures, and the technical report they feed.

The scoping document this implements is `../paperScopingV1.md`. The report being written is `techReportPaperScopingV1.md`.

## Layout

```
paper1/
  techReportPaperScopingV1.md   the report, written diagnostic by diagnostic
  standard.md                   the standard every diagnostic must meet
  readme.md                     this file
  config.py                     paths, seeds, constants: one source of truth
  runAll.py                     run any subset, or verify every result's provenance
  lib/                          shared machinery, one implementation of each idea
    data.py                     aligned loading of substrates, labels, covariates
    probes.py                   ridge probes, held-out R2, intervals, concept directions
    circular.py                 doubled-angle readouts, circular error, loop radius
    nulls.py                    the seven named null types
    transforms.py               deterministic operations on input images
    encode.py                   frozen-encoder wrapper with a content-addressed cache
    provenance.py               git sha, input hashes, seed, environment, stamped per result
  diagnostics/                  one script per diagnostic, d1 through d9
  tests/                        convention and null-calibration tests, run before any encode
  results/                      one JSON per diagnostic, each with a provenance block
  figures/                      one or more PNG per diagnostic
  cache/                        re-encoded embeddings; regenerable, not tracked
```

## The standard

`standard.md` is binding for all nine diagnostics. It was set after Diagnostic 1, which is the
reference implementation. A diagnostic is not finished until every item in it holds.

## Rules this folder enforces

1. **The encoder is frozen.** Loaded from `config.MODEL`, never fine-tuned, always called with the same token budget and pooling.
2. **Headline probes run on the image-only substrate.** The multimodal substrate appears only in the leakage ablation of Diagnostic 6.
3. **A probe is fit once, on unperturbed data, and evaluated held out.** Nothing is fit to a perturbed condition.
4. **Every headline number is reported beside a named null.**
5. **Every result carries provenance.** Git sha, input file hashes, seed, environment. A number that is not in a results file with a provenance block does not go in the report.
6. **Conventions are tested, not assumed.** `tests/testConventions.py` verifies the catalog shape identities numerically against the data; `tests/testTransforms.py` pins the rotation sign and frame conventions before any expensive encode.
7. **Nothing is encoded twice.** `lib/encode.py` caches by content and adopts previously computed transform encodes after verifying their shape, dtype and row order.

## Reproducing

Requires the anchor data under `../data/` and the environment pinned in `../envInterp.txt` (Python 3.11, NumPy 1.26.4, SciPy 1.17.0, scikit-learn 1.8.0, PyTorch 2.10.0+cu128, polymathic-aion 0.0.2).

```
python -B paper1/runAll.py             # status and provenance of all nine
python -B paper1/runAll.py d1          # run one diagnostic and render its figures
python -B paper1/runAll.py --figures d1 # re-render figures only
python -B paper1/runAll.py --all       # run everything
python -B paper1/runAll.py --verify    # re-hash every recorded input and compare
python -B -m pytest paper1/tests -q    # convention and null-calibration tests

# -B keeps the tree free of __pycache__; the suite writes no other caches.
```

## Relationship to the older code

`code/analysis/` remains the record of the Phase 1 and Track A to D work described in `../technicalReport.md`. It is not deleted and not edited. This folder is a clean rebuild against the diagnostic structure. The technical report records method and result; the development history is kept in the project run log.
