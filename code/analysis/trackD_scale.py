"""
Track D cross-scale stage (Matt's diagnostic: "compare the dictionary across
model scales"): train the identical SAE protocol on AION-BASE image embeddings,
then compare against the AION-LARGE dictionary.

METHOD NOTE: base and large embeddings have different dimensionalities, so
decoder-cosine matching (Phase 1 C-3) is undefined across scales. Features are
matched by ACTIVATION-PATTERN correlation over the shared 48,398 galaxies
(max |Spearman-free| Pearson on activations; a feature "recurs at scale" if its
best cross-model activation correlation >= 0.5). Also re-runs the two headline
Track D tests at base scale: (a) is the PA loop present in the raw base
embedding (supervised probe)? (b) is it likewise fractured (distributed-K
ablation)? (c) instrument census.

Run:  python code/analysis/trackD_scale.py
Needs data/E_img_base.npy (trackD_embedBase.py). Writes results/trackD_scale.json.
"""
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, "code/analysis")
from trackD_sae import SAE, train, zscore                     # identical protocol
from trackD import topk_acts, spearman_gpu

DEV = "cuda"
DATA, RES = "data", "results"
SEED = 0


def pa_err_of(ZX, pa, elong):
    tr, te = train_test_split(np.where(elong)[0], test_size=0.2, random_state=SEED)
    mc = Ridge(alpha=100.0).fit(ZX[tr], np.cos(2 * pa[tr]))
    ms = Ridge(alpha=100.0).fit(ZX[tr], np.sin(2 * pa[tr]))
    d = np.angle(np.exp(1j * (np.arctan2(ms.predict(ZX[te]), mc.predict(ZX[te])) - 2 * pa[te])))
    return float(np.median(np.abs(np.degrees(d))) / 2)


def main():
    t0 = time.time()
    Zb = zscore(np.load(f"{DATA}/E_img_base.npy"))
    ok = np.load(f"{DATA}/ok_index.npy")
    df = pd.read_parquet(f"{DATA}/sample.parquet").iloc[ok].reset_index(drop=True)
    df["dr8_id"] = df["dr8_id"].astype(str)
    sh = pd.read_parquet(f"{DATA}/anchorShapes.parquet"); sh["dr8_id"] = sh["dr8_id"].astype(str)
    m = df.merge(sh[["dr8_id", "paDeg", "ellip"]].drop_duplicates("dr8_id"), on="dr8_id", how="left")
    pa = np.radians(m["paDeg"].to_numpy(float)); ellip = m["ellip"].to_numpy(float)
    elong = np.isfinite(ellip) & (ellip > 0.3) & np.isfinite(pa)
    out = {"base_dim": int(Zb.shape[1])}

    # (a) the loop in the RAW base embedding (scale comparison of Track A's headline)
    out["pa_err_base_raw"] = pa_err_of(Zb, pa, elong)
    out["pa_err_large_raw_same_probe"] = pa_err_of(zscore(np.load(f"{DATA}/E_img.npy")), pa, elong)
    out["baseline_note"] = ("scale comparison must use pa_err_large_raw_same_probe (identical fixed "
                            "Ridge alpha=100 probe), NOT Track A's 2.03 RidgeCV headline")
    print(f"[{time.time()-t0:.0f}s] base raw-embedding PA loop err = {out['pa_err_base_raw']:.1f} deg "
          f"(large, same probe: {out['pa_err_large_raw_same_probe']:.1f})", flush=True)

    # SAE at base scale (identical protocol), seed 0 (+ health); weights persisted
    mdl, fvu, alive = train(Zb, 4 * Zb.shape[1], 32, 0)
    out["base_sae_health"] = dict(fvu=fvu, alive=alive)
    Wd = mdl.W_dec.detach().cpu().numpy()
    np.savez(f"{RES}/trackD_sae/base_s0.npz",
             W_enc=mdl.W_enc.detach().cpu().numpy(), W_dec=Wd,
             b_enc=mdl.b_enc.detach().cpu().numpy(), b_pre=mdl.b_pre.detach().cpu().numpy())
    acts_b = topk_acts(Zb, mdl.W_enc.detach().cpu().numpy(),
                       mdl.b_enc.detach().cpu().numpy(), mdl.b_pre.detach().cpu().numpy())
    print(f"[{time.time()-t0:.0f}s] base SAE: FVU={fvu:.3f} alive={alive:.3f} (weights saved)", flush=True)

    # (b) distributed-loop ablation at base scale
    c2 = spearman_gpu(acts_b[elong], np.cos(2 * pa[elong]))
    s2 = spearman_gpu(acts_b[elong], np.sin(2 * pa[elong]))
    order = np.argsort(-np.maximum(c2, s2))
    rng = np.random.default_rng(SEED)
    activeF = acts_b.std(0) > 1e-6
    out["distributed_loop_base"] = {"baseline": pa_err_of(Zb, pa, elong)}
    for K in (15, 50, 200):
        Rk = order[:K]
        errK = pa_err_of((Zb - acts_b[:, Rk] @ Wd[:, Rk].T).astype(np.float32), pa, elong)
        Rr = rng.choice(np.setdiff1d(np.where(activeF)[0], Rk), K, replace=False)
        errR = pa_err_of((Zb - acts_b[:, Rr] @ Wd[:, Rr].T).astype(np.float32), pa, elong)
        out["distributed_loop_base"][f"K{K}"] = dict(ablate=errK, control=errR)
        print(f"  base distributed-loop K={K}: {errK:.1f} deg (control {errR:.1f})", flush=True)

    # (c) cross-scale dictionary match by ACTIVATION correlation: base seed 0 vs ALL
    # five large seeds, plus a row-shuffled chance baseline per seed
    Zl = zscore(np.load(f"{DATA}/E_img.npy"))
    Ab = torch.from_numpy(acts_b).to(DEV)
    Ab = (Ab - Ab.mean(0)) / (Ab.std(0) + 1e-9)
    rngp = torch.Generator(device=DEV).manual_seed(SEED)

    def match_stats(Al, act_l, shuffle=False):
        A2 = Ab[torch.randperm(Ab.shape[0], generator=rngp, device=DEV)] if shuffle else Ab
        best = torch.zeros(Al.shape[1], device=DEV)
        for s in range(0, A2.shape[1], 512):
            C = (Al.t() @ A2[:, s:s + 512]) / Al.shape[0]
            best = torch.maximum(best, C.abs().max(1).values)
        b = best.cpu().numpy()[act_l]
        return dict(frac_ge_0p5=float((b >= 0.5).mean()), frac_ge_0p3=float((b >= 0.3).mean()),
                    median_best_corr=float(np.median(b)))

    out["cross_scale_match_per_seed"] = {}
    for sd in range(5):
        wl = np.load(f"{RES}/trackD_sae/eimg_s{sd}.npz")
        acts_l = topk_acts(Zl, wl["W_enc"], wl["b_enc"], wl["b_pre"])
        Al = torch.from_numpy(acts_l).to(DEV)
        Al = (Al - Al.mean(0)) / (Al.std(0) + 1e-9)
        act_l = acts_l.std(0) > 1e-6
        out["cross_scale_match_per_seed"][f"s{sd}"] = dict(
            n_large_active=int(act_l.sum()),
            observed=match_stats(Al, act_l),
            shuffled_baseline=match_stats(Al, act_l, shuffle=True))
        del Al, acts_l
    obs = [v["observed"] for v in out["cross_scale_match_per_seed"].values()]
    shf = [v["shuffled_baseline"] for v in out["cross_scale_match_per_seed"].values()]
    out["cross_scale_match"] = dict(                       # seed-0 stats kept as the headline key
        n_large_active=out["cross_scale_match_per_seed"]["s0"]["n_large_active"],
        frac_matched_ge_0p5=obs[0]["frac_ge_0p5"], frac_matched_ge_0p3=obs[0]["frac_ge_0p3"],
        median_best_corr=obs[0]["median_best_corr"],
        seed_range_frac_ge_0p5=[min(o["frac_ge_0p5"] for o in obs), max(o["frac_ge_0p5"] for o in obs)],
        shuffled_median_best_corr=[min(s["median_best_corr"] for s in shf), max(s["median_best_corr"] for s in shf)],
        shuffled_frac_ge_0p5=[min(s["frac_ge_0p5"] for s in shf), max(s["frac_ge_0p5"] for s in shf)])
    cm = out["cross_scale_match"]
    print(f"[{time.time()-t0:.0f}s] cross-scale: seed0 {100*cm['frac_matched_ge_0p5']:.0f}% >=0.5 "
          f"(seeds range {[round(100*x) for x in cm['seed_range_frac_ge_0p5']]}%), median "
          f"{cm['median_best_corr']:.2f}; shuffled median {cm['shuffled_median_best_corr']}", flush=True)

    with open(f"{RES}/trackD_scale.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {RES}/trackD_scale.json", flush=True)


if __name__ == "__main__":
    main()
