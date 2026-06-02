# AION-1 Geometry Prototype — AWS Run Plan (6–8 hours)

**A strong, exploratory, metric-dense prototype to decide — with evidence — whether AION-1's galaxy embedding has the geometric structure worth scaling.**

This is the single source of truth for tonight's run. We execute exactly this and nothing else. Every step has a *what / how / why*. AWS provisioning is given as exact CLI commands. The science steps are given as precise specs (inputs, operations, libraries, parameters, outputs, metrics) — code is written at run time, not here.

Status going in: model `aion-large` downloaded locally (`data/models/aion-large`, 3.77 GB); all four Galaxy Zoo DESI catalogs downloaded locally (`data/data/`); AWS CLI v2 configured (acct 333650975919, region us-east-1); existing EC2 key pair `anthropic-fellows-key` (ed25519) available.

> **Operating rules (standing).**
> 1. **Local is the system of record.** All code is authored/kept locally (canonical copy). The AWS box is compute-only — push code up to run it; **pull results, run artifacts, and logs back to local and verify** *before* terminating. Nothing of record lives only on an ephemeral spot instance.
> 2. **Record everything as we go.** Maintain a running **`RUNLOG.md`** appended at every step — the exact command/decision, *why*, what happened, key numbers, observations — and capture all stdout/stderr to `~/logs/`. Goal: a complete, replayable record.
> 3. **Clean, simple, minimal code.** Short readable scripts; minimal comments (only where intent isn't obvious); no dead code, no cleverness for its own sake.
> 4. **Transparency — never overclaim.** State assumptions and limits; report every number with its uncertainty; never claim anything the data or logic cannot support. Always distinguish *measured* from *interpreted*.
> 5. **Final deliverable: an explainable report.** Produce a presentable **`REPORT.md`** — what we did, how, why, the process, the math, the results and figures, *what they mean and what we can (and cannot) interpret* — honest and transparent throughout.

---

## 0. Decisions to confirm BEFORE we launch (no silent assumptions)

These are the only open choices. Defaults are chosen and justified; veto any before the launch step (§5.6).

| # | Decision | Default (recommended) | Why |
|---|---|---|---|
| D1 | Region | **us-east-1** | Your configured region; cheapest + best GPU capacity; model/data re-download from public sources is region-agnostic. |
| D2 | GPU / instance | **g6e.8xlarge** = 1× L40S (48 GB), 32 vCPU, 256 GB RAM (~$1.63/hr spot, us-east-1b) | One L40S as requested. **Live spot check:** 8xlarge (~$1.63) is only ~$0.04/hr more than 4xlarge (~$1.59) yet doubles cores+RAM → best value, with RAM headroom for 50k geometry. Cheapest single-L40S = g6e.2xlarge (8 vCPU/64 GB, ~$1.17) if minimizing cost. A6000 is not on AWS; g6e *is* the 48 GB option. |
| D3 | Purchase mode | **Spot first, on-demand fallback** | Spot ≈ 40–65% cheaper; we checkpoint so an interruption costs minutes, not the run. Fall back to on-demand g6e.4xlarge only if spot capacity is unavailable. |
| D4 | Budget ceiling | **≤ ~$20 for the session** (g6e.8xlarge spot ~$1.63/hr × ~8 h ≈ $13 + ~200 GB gp3 EBS prorated + minimal egress) | Hard `MaxPrice` cap (2.20) on the spot request + a billing alarm + mandatory teardown (§11). |
| D5 | Model + data staging | **Re-download on the instance** from HF (`polymathic-ai/aion-large`) + Zenodo (record 8360385) | Cloud↔HF/Zenodo bandwidth ≫ your home-upload bandwidth; local copies remain as backup. |
| D6 | Sample size | **50,000 galaxies** (stretch 80k if embedding throughput allows) | Confirmed. Big enough for tight CIs, clean maps, robust topology subsampling. |
| D7 | GPU fallback if no L40S spot/quota | **g5.4xlarge** (1× A10G, 24 GB, 16 vCPU) at **N = 25k** | Half the VRAM → smaller batch + sample, still a strong prototype; keeps us moving if g6e is constrained. |

**Live AWS pre-check (us-east-1, already run — the main blocker is CLEAR):**
- G/VT quota = **48 vCPU for BOTH spot and on-demand** → no increase needed; we can launch a single L40S now (g6e.8xlarge needs 32 vCPU, g6e.4xlarge 16).
- Cheapest spot AZ = **us-east-1b**; g6e.8xlarge spot ≈ **$1.63/hr**.
- DLAMI = **`ami-012ba162b9cd2729c`** (Deep Learning PyTorch 2.7, Ubuntu 22.04) → CUDA + PyTorch preinstalled.
- The local 12 GB GPU at N≈10–15k remains the ultimate fallback only if spot *capacity* ever errors.

---

## 1. Problem & goal

**Problem.** AION-1 (Polymathic AI, 800M-param "AION-1-Large") compresses any galaxy into a 1024-number embedding, self-supervised, with no morphology labels and no notion of galaxy evolution. The authors *assert* the embedding "organizes objects along physically meaningful directions" but never measure its geometry. We measure it.

**Prototype goal (tonight).** A *go/no-go with evidence*: prove the end-to-end pipeline runs at scale (50k galaxies), and produce a dense, statistically-bounded first look at the embedding's geometry and concept structure — enough to confidently decide whether to commit to the full study, and to preview 3 of its headline arms (faithful manifold map, intrinsic-dimension triangulation, unsupervised concept discovery) on real data.

**Explicitly NOT tonight.** No final scientific claims; no "it learned galaxy evolution"; no full Riemannian pullback flagship, no CAMELS sim-vs-real, no AstroCLIP cross-model (all deferred to scale-up). We run the rich-but-bounded arm set in §8 and nothing else.

---

## 2. Scope (in / out)

**In scope (the run):**
- 50k galaxies from Galaxy Zoo DESI (morphology vote fractions) × redshift × Legacy photometry/colors.
- Multimodal embeddings: `E_full` (image + photometry + redshift) and `E_held` (image + photometry only; for probes whose target is an input).
- Metric diagnostic → faithful manifold map → intrinsic-dimension triangulation (+ heterogeneity) → decodability/concept-axis probes → curvature/tree teaser → topology (β₀/β₁) → sparse-autoencoder concept discovery → synthetic validation → figures + metrics table + readout.

**Out of scope (deferred to scale-up):** Riemannian pullback metric/geodesics; CAMELS; AstroCLIP Platonic check; full Carrière–Michel–Oudot Mapper confidence; spectra modality; >80k galaxies.

---

## 3. Success criteria & go/no-go

We pre-commit to reading the *joint* result, not any single number.

- **Strong GO** (scale it): embeddings generated for ≥40k galaxies; diffusion-map shows morphology/redshift forming visible regions; intrinsic dimension ≈ 4–10 with ≥3 estimators agreeing within CIs and passing the synthetic-sphere check; morphology + redshift linearly decodable (held-out R² clearly > 0); ≥1 seed-stable SAE concept aligned with physics; β₀ resolved with confidence.
- **Qualified GO** (scale with changes): pipeline runs and dimension/decodability are clean, but topology/curvature/SAE are noisy at this scale → scale-up with more samples/seeds.
- **NO-GO** (rethink): dimension estimators disagree wildly even after passing synthetic tests, physics not decodable beyond chance, and SAE features unstable → the embedding geometry is generic; revisit the premise. (Still a publishable negative.)

---

## 4. What we measure — the dense metrics catalog (what / how / why)

Every metric ships with a bootstrap or credible interval. Everything is computed on both `E_full` and (where the target is an input) `E_held`, and reported under ≥2 distance metrics where geometry-dependent.

**4.1 Representation quality / decodability.**
- *Linear + ridge probe* held-out R² (regression) / balanced-accuracy (classification) for: redshift, g−r and r−z colors (mass/age proxies), smooth-fraction, bar-fraction, spiral-fraction, merger-fraction. *How:* 80/20 split, 5-fold CV for λ, bootstrap CI. *Why:* establishes the embedding actually carries physics (the floor under everything else).
- *kNN purity* and *silhouette* of thresholded morphology classes in embedding space. *Why:* a model-free "are like galaxies close?" check.
- *Modality ablation:* R² for image-only vs +photometry vs +redshift. *Why:* shows what each modality contributes (and flags trivial leakage when a target is an input → use `E_held`).

**4.2 Intrinsic dimension (triangulated).**
- TwoNN (linear-fit, 10% discard) **and** unbiased MLE d̂=(N−1)/Σ log μ; Bayesian-2NN Gamma-posterior **credible interval**; **Gride** scale curve (ID vs neighbor order, to separate manifold dim from small-scale noise); **local** Levina–Bickel m̂_K(z) swept over K; **diffusion spectral-gap** dimension; **PCA participation-ratio** (linear null). *How:* `scikit-dimension`, `dadapy`, plus the diffusion eigenspectrum from §4.6; bootstrap CIs over galaxies. *Why:* one ID number is indefensible; agreement of ≥3 independent estimators is the headline.
- **Validation:** all ID estimators re-run on synthetic 1024-d sphere/swiss-roll of known dimension at matched N/noise (§8 Step E2). *Why:* certifies the numbers at our scale before trusting AION-1.

**4.3 Heterogeneity.**
- Stratified ID: split sample into passive vs star-forming (by g−r color) and report ID per population with CIs; ΔID with bootstrap. (Hidalgo Bayesian mixture if time.) *Why:* tests the novel "different galaxy types live on different-dimensional submanifolds" hypothesis (Cadiou+2025 precedent).

**4.4 Disentanglement / concept axes.**
- Ridge probe directions ŵ_y (on `E_held`), angle matrix θ_ij, compared to the label-correlation null arccos(ρ_ij); bootstrap CI on each angle. *Why:* tests whether physics axes are separated *beyond* what label correlation forces — the real disentanglement question.

**4.5 Curvature / tree-likeness teaser.**
- δ-hyperbolicity: distribution of the 4-point δ over ~10⁶ sampled quadruples, diameter-normalized (mean/median/95th pct), vs a matched-Gaussian-cloud anchor. *Why:* tree-like vs not.
- Ollivier-Ricci curvature on the kNN graph (fraction of negative-curvature "bridge" edges; where they sit vs morphology). *Why:* *localizes* branching.
- Global−local ID gap (PCA-ID minus TwoNN) as a curvature signal.

**4.6 Manifold map.**
- Diffusion map (α=1 Laplace–Beltrami, BGH bandwidth) → 2D/3D coords + eigenvalue spectrum/gap; UMAP for visual comparison only. Colored by morphology, redshift, color. *Why:* the faithful, deterministic "money figure."

**4.7 Topology.**
- Vietoris–Rips persistent homology, maxdim=1, on 5× independent 2–3k subsamples, under Euclidean **and** diffusion/Fermat metric; β₀, β₁ with Fasy-style confidence (bootstrap); bottleneck distance Euclidean-vs-intrinsic-metric diagrams. *Why:* does β₀=2 (red/blue bimodality) survive? are loops real or noise?

**4.8 Concept discovery (sparse autoencoder).**
- TopK SAE (+AuxK) on `E_full`, expansion R∈{4,8}, 2–3 seeds. Report variance-explained, L0, %-alive; per-feature max|Spearman| vs {redshift, colors, morphology fractions}; count of *aligned* features (s>0.4) and *seed-stable "alien"* candidates (high activation coherence, low R² from all labels); top-activating-galaxy montages for 2–3 features. *Why:* the novel payload — does AION-1 organize by concepts beyond the human catalog?

**Output of §4:** a single `metrics.json` (+ CSV) with every number and interval, plus ~8–10 figures (§12).

---

## 5. AWS provisioning — exact CLI steps

> Run these in order. Replace `<...>` placeholders. All read-only checks (5.1–5.3) are free and safe; the launch (5.6) is the first billable action.

### 5.1 Pre-flight (identity, region)
```bash
aws sts get-caller-identity
aws configure get region   # expect us-east-1
export AWS_REGION=us-east-1
```
*Why:* confirm the right account/region before anything billable.

### 5.2 GPU quota check (THE blocker) — do this first
```bash
# Spot vCPU quota for G/VT instances (code L-3819A6DF)
aws service-quotas get-service-quota \
  --service-code ec2 --quota-code L-3819A6DF --region $AWS_REGION \
  --query 'Quota.Value'

# On-demand vCPU quota for Running G/VT instances (code L-DB2E81BA)
aws service-quotas get-service-quota \
  --service-code ec2 --quota-code L-DB2E81BA --region $AWS_REGION \
  --query 'Quota.Value'
```
*Interpretation:* g6e.4xlarge = **16 vCPU**. Need the relevant quota ≥ 16 (≥ 32 if we upgrade to g6e.8xlarge). If a value is **0 or < 16**, request an increase immediately:
```bash
aws service-quotas request-service-quota-increase \
  --service-code ec2 --quota-code L-3819A6DF --desired-value 32 --region $AWS_REGION
# (repeat for L-DB2E81BA if using on-demand)
aws service-quotas list-requested-service-quota-change-history-by-quota \
  --service-code ec2 --quota-code L-3819A6DF --region $AWS_REGION
```
*Why / decision:* quota increases can take hours–days. **If it does not clear within ~30 min, switch to the local-GPU fallback (§10 R1) and run N≈10–15k locally tonight** rather than block.

**RESULT (live):** Spot = **48.0**, On-Demand = **48.0** vCPU → ✓ no increase needed (g6e.8xlarge needs 32; g6e.4xlarge needs 16).

### 5.3 Spot price check (pick the cheapest AZ)
```bash
aws ec2 describe-spot-price-history \
  --region $AWS_REGION \
  --instance-types g6e.4xlarge g6e.2xlarge g6e.8xlarge g5.4xlarge \
  --product-descriptions "Linux/UNIX" \
  --start-time "$(date -u -d '-1 hour' +%Y-%m-%dT%H:%M:%SZ)" \
  --query 'sort_by(SpotPriceHistory,&SpotPrice)[].[InstanceType,AvailabilityZone,SpotPrice,Timestamp]' \
  --output table
```
*Why:* choose the AZ with the lowest stable spot price; set `MaxPrice` ~20% above it (D4).

### 5.4 Pick the AMI (Deep Learning AMI to skip CUDA/PyTorch setup)
```bash
# Latest AWS Deep Learning OSS Nvidia PyTorch AMI (Ubuntu 22.04), owner = amazon
aws ec2 describe-images --region $AWS_REGION --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch *Ubuntu 22.04*" \
            "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].[ImageId,Name]' --output text
```
*Why:* the DLAMI ships CUDA + PyTorch + conda → saves ~1 h of driver/toolkit setup (the biggest setup-tax). Record the `ImageId` as `<AMI>`.

**RESULT (live):** `<AMI>` = **`ami-012ba162b9cd2729c`** — *Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04) 20260427*.

### 5.5 Networking (key pair + security group)
```bash
# Key pair: REUSE the existing Anthropic-Fellows key (confirmed via describe-key-pairs):
#   anthropic-fellows-key (ed25519, 2026-05-28).  No create-key-pair needed.
# REQUIREMENT: you must hold the matching PRIVATE key file locally to SSH. Point KEY at it:
KEY="C:/Users/worka/.ssh/anthropic-fellows-key.pem"   # confirmed present (ed25519, matches the AWS key pair)
# (Only create a fresh key pair if that private key is unavailable.)

# Security group allowing SSH only from your current public IP
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 create-security-group --region $AWS_REGION \
  --group-name aion-proto-sg --description "AION prototype SSH"
aws ec2 authorize-security-group-ingress --region $AWS_REGION \
  --group-name aion-proto-sg --protocol tcp --port 22 --cidr ${MYIP}/32
```
*Why:* lock SSH to your IP only (don't open 0.0.0.0/0).

### 5.6 Launch — SPOT first (FIRST billable action; confirm D1–D7)
```bash
aws ec2 run-instances --region $AWS_REGION \
  --image-id ami-012ba162b9cd2729c \
  --instance-type g6e.8xlarge \
  --key-name anthropic-fellows-key \
  --security-groups aion-proto-sg \
  --instance-market-options 'MarketType=spot,SpotOptions={MaxPrice=2.20,SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=500,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=proj,Value=aion-proto}]' \
  --count 1 \
  --query 'Instances[0].InstanceId' --output text
```
*On-demand fallback* (only if spot capacity errors): drop the `--instance-market-options` line.
*Why:* 200 GB gp3, `DeleteOnTermination=true` (no orphaned EBS cost), MaxPrice cap, tag for cleanup.

### 5.7 Connect
```bash
aws ec2 describe-instances --region $AWS_REGION \
  --filters "Name=tag:proj,Values=aion-proto" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].[InstanceId,PublicDnsName]' --output text
ssh -i $KEY ubuntu@<PublicDnsName>   # DLAMI default user is 'ubuntu'
nvidia-smi   # confirm the L40S (48 GB) is visible
```

---

## 6. Environment setup on the instance (~10 min)
- **Start the record (Operating-rule 2):** create `~/out/RUNLOG.md` + `~/logs/`; tee every step's stdout/stderr to `~/logs/<step>.log` and append a dated RUNLOG entry per action.
- Activate the DLAMI PyTorch env: `source activate pytorch` (or the env `nvidia-smi`/`conda env list` reports).
- `pip install` (pin versions in `requirements.txt`): the `aion` package + `huggingface_hub`, `scikit-dimension`, `dadapy`, `datafold`, `pydiffmap`, `umap-learn`, `ripser`, `persim`, `GraphRicciCurvature`, `POT`, `scikit-learn`, `pyarrow`, `astropy`, `requests`, `matplotlib`, `tqdm`.
- `python -c "import torch; print(torch.cuda.get_device_name(0))"` → confirm L40S.
- *Why:* one explicit, pinned environment so the run is reproducible and a spot-restart re-creates it fast.

## 7. Stage code, model + data on the instance (~10–15 min)
- **Code (canonical copy stays local — Operating Rule):** push our local experiment scripts to the box via `scp`/`rsync` (or `git clone` a private repo). The box only *runs* code; it never owns the source of truth.
- Model: `huggingface_hub.snapshot_download('polymathic-ai/aion-large', local_dir=~/models/aion-large)` (~4 GB).
- Catalogs: pull the four GZ DESI parquets from Zenodo record 8360385 into `~/data/` (friendly + advanced + external_catalog + volunteer) — or `aws s3 cp` if you first stash them in your bucket.
- *Why:* re-download from public mirrors (D5) is faster than uploading 14 GB from home.

---

## 8. The pipeline — ordered steps (what / how / why / output)

> Each step writes its outputs to `~/out/` and checkpoints. Steps E–J are CPU/GPU-parallel and run on the 16 vCPU + L40S.

**Step A — Build the 50k sample (~15 min).**
- *How:* load GZ DESI friendly + advanced (for branch-relevance) + external_catalog (redshift) parquets; keep r<19, SGC/DR10 footprint; require a valid redshift and branch-relevant bar/spiral entries; attach g,r,i,z fluxes + colors; sample 50k stratified across morphology + redshift so rare classes aren't lost.
- *Output:* `sample.parquet` (RA, Dec, dr8_id, vote-fractions, z, fluxes/colors). *Why:* a clean, balanced, leakage-aware label table is the backbone.

**Step B — Cutouts + embeddings (the bottleneck, ~1.5–2.5 h).**
- *How:* fetch 160-px FITS `g,r,i,z` cutouts from the Legacy Survey service with **16–32 concurrent workers + exponential backoff + write-`.tmp`-then-rename**; replicate AION's 160→96 crop + arcsinh; tokenize via `CodecManager` (build **E_full** = image+photometry+redshift, and **E_held** = image+photometry); run frozen AION-1-Large fp16 on the L40S in batches; **mean-pool** → vectors. **Checkpoint every 2k galaxies** to `embeddings.parquet` so a spot interruption loses ≤ a few minutes.
- *Smoke test first:* run 100 galaxies end-to-end and verify shapes (one 1024-vec/galaxy), before the full 50k. *Why:* the make-or-break plumbing step — fail fast, checkpoint always.
- *Output:* `E_full.npy`, `E_held.npy` (50k×1024), aligned to `sample.parquet`.

**Step C — Metric diagnostic (~5 min).**
- *How:* compute RDR = (D_max−D_min)/D_min and NN/mean-distance ratio under Euclidean, cosine, and diffusion metrics on a 5k subsample. *Why:* justify the metric choice with a measured figure (the raw 1024-d metric is provably weak).

**Step D — Manifold map (~10 min).**
- *How:* diffusion map (α=1, BGH ε, `datafold`) → 2D/3D + eigenspectrum; UMAP for comparison; color by morphology, redshift, color. *Output:* `fig_map_*.png`, `diffusion_eigs.npy`. *Why:* the headline figure + feeds spectral-gap ID and the topology metric.

**Step E — Intrinsic dimension, triangulated + validated (~15 min).**
- E1: TwoNN (lin+MLE), Bayesian-2NN CI, Gride curve, local Levina–Bickel(K), spectral-gap, PCA-PR; bootstrap CIs. (`scikit-dimension`, `dadapy`.)
- E2: re-run all on a synthetic 1024-d sphere (ID 5) + swiss-roll (ID 2) + two-blobs at matched N/noise; report estimator error. *Why:* triangulated, certified ID number — the second headline.

**Step F — Heterogeneity (~10 min).**
- *How:* split passive/star-forming by g−r; ID per population + ΔID with bootstrap. *Why:* novel "different types, different dimension" test.

**Step G — Probes / concept axes (~10 min).**
- *How:* ridge probes (CV-λ) on `E_held` for all labels → held-out R²/accuracy, bootstrap CIs; angle matrix θ_ij vs arccos(ρ_ij) null; kNN purity + silhouette; modality ablation. *Why:* decodability floor + honest disentanglement.

**Step H — Curvature / tree teaser (~15 min).**
- *How:* δ-hyperbolicity distribution (sampled quadruples, diameter-normalized) vs matched-Gaussian anchor; Ollivier-Ricci on kNN graph (`GraphRicciCurvature`, Sinkhorn) → fraction negative edges + overlay on the map; global−local ID gap. *Why:* first read on curvature/branching, localized.

**Step I — Topology (~20–30 min).**
- *How:* Vietoris–Rips maxdim=1 (`ripser`) on 5× independent 2–3k subsamples, Euclidean **and** diffusion metric; β₀/β₁ with bootstrap confidence; bottleneck distance between metrics. *Why:* does the bimodality (β₀=2) and any loop survive subsampling + metric change?

**Step J — Concept discovery / SAE (~20–30 min).**
- *How:* TopK SAE (+AuxK) on `E_full`, R∈{4,8}, 2–3 seeds (tiny, GPU-fast); score features (max|Spearman| vs labels; residual-R² novelty); seed-stability; montages for 2–3 features. *Why:* the novel payload — aligned + "alien" concepts.

**Step K — Synthesis & explainable report (~45 min).**
- *How:* assemble `metrics.json`/CSV (all numbers + CIs) and ~8–10 figures; write the explainable **`REPORT.md`** (Operating-rule 5): what we did / how / why / the math behind each metric / the results and each figure with *what it means and what we can vs cannot conclude* / honest limits + the §3 go/no-go verdict. Finalize **`RUNLOG.md`**.
- *Transparency:* every claim in `REPORT.md` is tagged *measured* vs *interpreted*; nothing is asserted beyond the data. *Output:* `REPORT.md` + `RUNLOG.md` + `figures/` + `metrics.json`.

---

## 9. Timeline (target 6–8 h, with slack)

| Block | Wall-clock | Activity |
|---|---|---|
| 0 | 0:00–0:30 | §5 provisioning: quota check → spot launch → connect → `nvidia-smi`. (If quota blocked → local fallback, restart clock.) |
| 1 | 0:30–1:00 | §6 env + §7 stage model/data + 100-galaxy smoke test (Step B). |
| 2 | 1:00–3:30 | Step A sample + Step B embeddings (50k), checkpointed; start §8 C–E on early batches. |
| 3 | 3:30–6:00 | Steps C–J metric battery (parallel across 16 vCPU + L40S). |
| 4 | 6:00–7:00 | Step E2 validation + Step K figures/metrics/readout. |
| 5 | 7:00–7:30 | §11 teardown: copy `~/out/` to local/S3, terminate instance, verify. |
| slack | 7:30–8:00 | buffer for download stalls / re-runs. |

---

## 10. Risks & mitigations

- **R1 — GPU quota — RESOLVED (live: 48 vCPU spot + on-demand).** No increase needed; launch immediately. Residual risk is only spot *capacity* in a given AZ → retry another AZ or use on-demand (still within budget); local 12 GB GPU @ N≈10–15k is the ultimate fallback.
- **R2 — Model plumbing (`aion`/`CodecManager`).** 100-galaxy smoke test before the 50k run; if it fights, shrink to image-only embeddings to get *a* signal, fix multimodal after.
- **R3 — Cutout rate-limiting (429).** Bounded concurrency (16–32) + backoff + `.tmp` rename; checkpoint every 2k; if the service throttles hard, accept partial N (embed what we have — 30k is still strong).
- **R4 — Spot interruption.** Checkpoints make a restart cost minutes; on relaunch, env + staging re-run from this plan; on-demand fallback if spot capacity vanishes.
- **R5 — Cost overrun.** `MaxPrice` cap, `DeleteOnTermination=true`, billing alarm (§11), mandatory teardown. 
- **R6 — Toy-scale noise (topology/SAE).** Bootstrap CIs + seed stability + synthetic validation already baked in; we report noisy arms as noisy (qualified-GO), not as results.

---

## 11. Cost control & teardown (MANDATORY — do not skip)
```bash
# Optional: billing alarm before launch (one-time)
# aws cloudwatch put-metric-alarm ... (Estimated Charges > $30)

# OPERATING RULE: pull EVERYTHING back to local, VERIFY, then terminate.
LOCAL=./prototype_out; mkdir -p $LOCAL
scp -i $KEY -r ubuntu@<dns>:~/out  $LOCAL/out    # results, figures, metrics.json, embeddings
scp -i $KEY -r ubuntu@<dns>:~/logs $LOCAL/logs   # all run logs
scp -i $KEY -r ubuntu@<dns>:~/runs $LOCAL/runs   # run configs/metadata + checkpoints
ls -R $LOCAL && echo "VERIFY the above is complete BEFORE terminating"

# Only after verifying the pull-back:
INSTID=$(aws ec2 describe-instances --region $AWS_REGION \
  --filters "Name=tag:proj,Values=aion-proto" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].InstanceId' --output text)
aws ec2 terminate-instances --region $AWS_REGION --instance-ids $INSTID
aws ec2 describe-instances --region $AWS_REGION --instance-ids $INSTID \
  --query 'Reservations[].Instances[].State.Name' --output text   # expect shutting-down/terminated
```
*Why:* the #1 way to waste money is a forgotten running GPU. Terminate confirms the EBS is released (DeleteOnTermination).

---

## 12. Deliverables (what lands on your laptop tomorrow)
- **`REPORT.md`** — the presentable, explainable write-up: what / how / why, process, the math, results + every figure with its meaning and honest interpretation (*measured* vs *interpreted*), limits, and the §3 go/no-go verdict. (Operating-rule 5.)
- **`RUNLOG.md`** + `logs/` — the complete step-by-step record (commands, decisions, numbers, observations) and raw stdout/stderr. (Operating-rule 2.)
- `metrics.json` + `metrics.csv` — every number with its CI / interval.
- `figures/`: (1) diffusion map ×3 colorings, (2) ID-triangulation bar chart + synthetic-validation, (3) Gride scale curve, (4) probe-R² table + disentanglement angle heatmap, (5) δ-hyperbolicity histogram, (6) Ollivier-Ricci map overlay, (7) persistence diagrams (Euclidean vs intrinsic metric) + β-bootstrap, (8) SAE feature-alignment chart + 2–3 montages.
- `embeddings.parquet` (50k×1024, `E_full`+`E_held`) — reusable for the scale-up.

All of the above are pulled back to local before teardown (Operating-rule 1).

---

## 13. Execution checklist (tick in order)
1. [ ] §5.1 identity + region confirmed.
2. [ ] §5.2 GPU quota ≥ 16 (else request increase; else local fallback R1).
3. [ ] §5.3 spot price checked; AZ + MaxPrice chosen.
4. [ ] §5.4 DLAMI AMI id recorded.
5. [ ] §5.5 reuse `anthropic-fellows-key` (private `.pem` confirmed present locally) + security group (your IP only).
6. [ ] §5.6 spot instance launched (D1–D7 confirmed).
7. [ ] §5.7 SSH in; `nvidia-smi` shows L40S 48 GB.
8. [ ] §6 env installed + CUDA confirmed.
9. [ ] §7 model + catalogs staged.
10. [ ] Step B 100-galaxy smoke test passes.
11. [ ] Step A 50k sample built.
12. [ ] Step B embeddings (≥40k) checkpointed.
13. [ ] Steps C–J metric battery complete.
14. [ ] Step E2 synthetic validation passes.
15. [ ] RUNLOG.md maintained throughout; Step K figures + metrics + **REPORT.md** written (measured-vs-interpreted; no overclaims).
16. [ ] §11 results + runs + **logs** pulled back to local and verified (Operating Rule); THEN instance TERMINATED; termination verified.

---

### Appendix — quick reference numbers
- Instance: **g6e.8xlarge** = 1× L40S 48 GB, 32 vCPU, 256 GB RAM, **us-east-1b**, **~$1.63/hr spot** (AMI `ami-012ba162b9cd2729c`). Cheapest single-L40S: g6e.2xlarge (8 vCPU/64 GB, ~$1.17). Fallback: g5.4xlarge (A10G 24 GB, ~$0.76) @ N=25k, or local 12 GB @ N=10–15k.
- Storage: **500 GB** gp3 EBS, DeleteOnTermination (sized for all downloads + cutouts + embeddings; released automatically at teardown).
- Quota codes: spot G/VT `L-3819A6DF`; on-demand G/VT `L-DB2E81BA` — **live: both = 48 vCPU ✓** (g6e.8xlarge needs 32).
- Sample: 50k (stretch 80k); homology subsample 2–3k × 5.
- Est. cost: g6e.8xlarge spot ~$1.63/hr × ~8 h + EBS ≈ **~$15**.
