"""
Track D stage 1: retrain the Phase 1 SAEs locally, SAVING THE DECODERS this time
(Phase 1 saved activations/summaries only; the Engels reducibility test needs the
dictionary matrices). Protocol is IDENTICAL to Phase 1 code/analysis/sae.py --
TopK k=32 (f = TopK(W_enc(x - b_pre) + b_enc)), R=4 (m=4096), AuxK: top-512 dead
latents, alpha=1/32, dead after 12 inactive steps, 80 epochs, batch 8192,
Adam 1e-3, decoder columns renormalized each step, b_pre init = data mean.
Faithfulness verified against the TopK formulation in the local BatchTopK paper
(papers/batchTopkSaes: L_aux = ||e - e_hat||^2 on top-k_aux dead latents, alpha=1/32).

Trains 5 seeds on z-scored E_full (continuity with the Phase 1 335+59 dictionary;
health-check vs Phase 1 logged FVU~0.035 / alive~0.70) and 5 seeds on z-scored
E_img (the leakage-free loop-bearing space Track A used).

Run:  python code/analysis/trackD_sae.py
Saves results/trackD_sae/{efull,eimg}_s{seed}.npz (W_enc, W_dec, b_enc, b_pre)
+ results/trackD_sae/health.json. Arrays stay local (gitignored like other npy).
"""
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn

DEV = "cuda"
K, R, SEEDS = 32, 4, [0, 1, 2, 3, 4]
EPOCHS, BATCH, DEAD_AFTER, KAUX, AUXC = 80, 8192, 12, 512, 1.0 / 32
OUT = "results/trackD_sae"
os.makedirs(OUT, exist_ok=True)


def zscore(E):
    return ((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32)


class SAE(nn.Module):
    def __init__(self, d, m, k):
        super().__init__()
        self.k = k
        W = torch.randn(d, m); W /= W.norm(dim=0, keepdim=True)
        self.W_dec, self.W_enc = nn.Parameter(W), nn.Parameter(W.t().clone())
        self.b_enc, self.b_pre = nn.Parameter(torch.zeros(m)), nn.Parameter(torch.zeros(d))

    def pre(self, x):
        return torch.relu((x - self.b_pre) @ self.W_enc.t() + self.b_enc)

    def topk(self, zp, k):
        v, i = zp.topk(k, dim=1)
        return torch.zeros_like(zp).scatter_(1, i, v)

    def forward(self, x):
        zp = self.pre(x)
        z = self.topk(zp, self.k)
        return z @ self.W_dec.t() + self.b_pre, z, zp


def train(X, m, k, seed):
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X).to(DEV)
    n, d = Xt.shape
    model = SAE(d, m, k).to(DEV)
    model.b_pre.data = Xt.mean(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    last = torch.zeros(m, device=DEV)
    for ep in range(EPOCHS):
        for s in range(0, n, BATCH):
            xb = Xt[torch.randperm(n, device=DEV)[s:s + BATCH]] if ep == 0 else \
                 Xt[torch.randint(0, n, (BATCH,), device=DEV)]
            xh, z, zp = model(xb)
            err = xb - xh
            loss = (err ** 2).mean()
            dead = last > DEAD_AFTER
            if dead.sum() > 0:
                zpd = zp.clone(); zpd[:, ~dead] = -1e9
                za = model.topk(zpd, min(KAUX, int(dead.sum())))
                loss = loss + AUXC * ((err.detach() - za @ model.W_dec.t()) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                model.W_dec.data /= model.W_dec.data.norm(dim=0, keepdim=True) + 1e-8
                last += 1; last[(z.abs().sum(0) > 0)] = 0
    with torch.no_grad():
        xh, z, _ = model(Xt)
        fvu = float(((Xt - xh) ** 2).sum() / ((Xt - Xt.mean(0)) ** 2).sum())
        alive = float((z.abs().sum(0) > 0).float().mean())
    return model, fvu, alive


def main():
    health = {}
    for tag, path in [("efull", "data/E_full.npy"), ("eimg", "data/E_img.npy")]:
        Z = zscore(np.load(path))
        m = R * Z.shape[1]
        for seed in SEEDS:
            t0 = time.time()
            model, fvu, alive = train(Z, m, K, seed)
            np.savez(f"{OUT}/{tag}_s{seed}.npz",
                     W_enc=model.W_enc.detach().cpu().numpy(),
                     W_dec=model.W_dec.detach().cpu().numpy(),
                     b_enc=model.b_enc.detach().cpu().numpy(),
                     b_pre=model.b_pre.detach().cpu().numpy())
            health[f"{tag}_s{seed}"] = dict(fvu=fvu, alive=alive, secs=round(time.time() - t0, 1))
            print(f"{tag} seed={seed}: FVU={fvu:.3f} alive={alive:.3f} ({time.time()-t0:.0f}s)", flush=True)
    with open(f"{OUT}/health.json", "w") as f:
        json.dump(health, f, indent=2)
    ef = [health[f"efull_s{s}"] for s in SEEDS]
    print(f"\nE_full across seeds: FVU {np.mean([h['fvu'] for h in ef]):.3f} "
          f"(Phase 1 logged ~0.035), alive {np.mean([h['alive'] for h in ef]):.3f} (Phase 1 ~0.70)")
    print(f"wrote {OUT}/*.npz + health.json")


if __name__ == "__main__":
    main()
