# ANALYSIS_PLAN — AION-1 geometry prototype (measurement phase)

Locked plan for the geometry + concept measurements on the 48,398-galaxy embeddings.
Full-study methods spec: [problem.md](problem.md). Data-run plan: [prototypePlan.md](prototypePlan.md). Log: [runLog.md](runLog.md).
Everything uses ALL 48,398 galaxies (no subsampling) except where a method is combinatorially
infeasible (β1 loops, exact δ) — there we use a principled subsample-with-confidence estimator.

## Data (local + on box `/home/ubuntu/data`)
- `E_full.npy` (48398×1024 f32) = image + g,r,z flux + redshift   — geometry arms
- `E_img.npy`  (48398×1024 f32) = image only                       — leakage-free probes
- `ok_index.npy` (48398 int64)  = rows of sample.parquet kept (status==1); labels = `sample.iloc[ok_index]`
- `sample.parquet` (50000) labels; `status.npy`, `images.npy` not needed for analysis

## Environment (GPU box `/opt/pytorch`; `code/setupAnalysis.sh`)
torch (GPU), numpy, pandas, scikit-learn, scipy; cupy-cuda12x + cuml-cu12 (GPU kNN);
scikit-dimension, dadapy (ID); datafold (diffusion map); ripser, persim (topology);
GraphRicciCurvature, POT (curvature). Results → `/home/ubuntu/results`, pulled to local `results/`.

## Locked global choices
- Preprocessing: z-score each embedding dim before geometry/probes (sanity confirms norm/scale).
- Metric (chosen AFTER the diagnostic): Fermat = primary (topology/curvature), diffusion = coordinates,
  Euclidean = control; cosine = control only (it concentrates like Euclidean).
- Sensitivity: every neighborhood/scale parameter swept ≥10 geometric points; headline only k-stable ridges.
- Honest metrics passed as precomputed distance matrices downstream (never re-embed).
- Every number tagged measured-vs-interpreted; bootstrap/credible CIs everywhere.

## Scripts (`code/analysis/`, run in order; each: load → compute → save results → log)
- `common.py`        load+align, results I/O, logging, GPU kNN helper
- `synthetic.py`     known-answer sets (sphere ID5, swiss-roll ID2, two-blobs β0=2, linear null) at matched N/noise
- `sanity.py`        shapes / NaN / norms / per-dim scale / PCA participation-ratio → confirm preprocessing + linear-ID baseline
- `metric.py`        RDR + NN/mean under {Euclidean, cosine, Isomap, Fermat, diffusion} on a fixed-seed 2k → justify the metric
- `intrinsicDim.py`  TwoNN (lin + MLE + exact IG CI), Gride plateau, Bayesian-2NN, local-MLE K-sweep, PCA-PR, diffusion spectral-gap; k-sweeps + bootstrap CIs; validated on synthetic first
- `diffusionMap.py`  α=1 BGH diffusion map, sparse kNN, full 48k → 2D/3D coords + eigenspectrum; harmonic screen; bootstrap gap CI
- `probes.py`        RidgeCV on E_img (redshift, g−r, r−z, morphology fractions) held-out R² + bootstrap; disentanglement |θ−arccos ρ|; mass/sSFR/Sérsic (~4k) exploratory
- `curvature.py`     Forman (full graph) + Sinkhorn-Ollivier (2k) + global−local ID gap + sampled δ (1e6); ≥2 metrics; synthetic curvature check
- `sae.py`           TopK+AuxK, k=32, R∈{4,8}, 5 seeds; health triad; alignment + FDR alien count; recurrence fraction; L0-vs-FVU frontier
- `topology.py`      β0 EXACT full (MST) + β1 ~10×2k subsamples under Fermat/diffusion/Euclidean + Fasy √2·c_n bands + bottleneck
- `plots.py`         (local) figures from pulled results

## Go/no-go (pre-registered)
- STRONG-GO: ID ≈ 4–10 (≥3 estimators agree within CIs + pass synthetic) AND diffusion map shows
  morphology/redshift regions AND physics decodable (R²>0, morphology separated from redshift beyond
  arccos ρ) AND ≥1 seed-stable SAE concept aligned with physics.
- QUALIFIED-GO: pipeline clean but some arms noisy (reported as noisy).
- NO-GO: joint failure (ID estimators disagree + physics not decodable + SAE unstable).
- A null geometric result = "decodable but not geometrically clean", never "AION lacks physics".

## Deferred (full study)
Riemannian pullback-metric flagship; tree-fitting battery; SAE causal ablation/steering; Mapper/Reeb;
RTD + β2; spectra modality, AstroCLIP, CAMELS, N>80k; Bayesian GPLVM.
