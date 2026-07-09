import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata, t as student_t
from common import load, save_json, save_npy, log

DEV = "cuda"
K, R_LIST, SEEDS = 32, [4, 8], [0, 1, 2, 3, 4]
KFRONT = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
EPOCHS, BATCH, DEAD_AFTER, KAUX, AUXC = 80, 8192, 12, 512, 1.0 / 32


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
            xb = Xt[torch.randperm(n, device=DEV)[s:s + BATCH]] if ep == 0 else Xt[torch.randint(0, n, (BATCH,), device=DEV)]
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
        return dict(fvu=fvu, alive=alive), z.cpu().numpy(), model.W_dec.detach().cpu().numpy().T


def rank(A, chunk=256):
    """Average-tie ranks, column-chunked to bound memory for sparse activations."""
    out = np.empty(A.shape, np.float32)
    for s in range(0, A.shape[1], chunk):
        r = rankdata(A[:, s:s + chunk], axis=0, method="average")
        r = (r - r.mean(0)) / (r.std(0) + 1e-12)
        out[:, s:s + r.shape[1]] = r.astype(np.float32)
    return out


def bh_qvalues(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(ranked)
    q[order] = np.clip(ranked, 0, 1)
    return q


def score(acts, Y):
    fin = np.isfinite(Y).all(1)
    acts, Y = acts[fin], Y[fin]
    Ra, Ry = rank(acts), rank(Y)
    sp = (Ra.T @ Ry) / len(Y)                          # feature x label Spearman
    align, best_label = np.abs(sp).max(1), np.abs(sp).argmax(1)   # strength + the label each feature names
    rr = np.clip(np.abs(sp), 0, 1 - 1e-15)
    tstat = rr * np.sqrt((len(Y) - 2) / np.maximum(1 - rr ** 2, 1e-15))
    p_label = 2 * student_t.sf(tstat, df=len(Y) - 2)
    p_feature = np.minimum(1.0, p_label.min(1) * Y.shape[1])  # Bonferroni over labels
    q_feature = bh_qvalues(p_feature)                         # BH over dictionary features
    P = np.c_[np.ones(len(Y)), (Y - Y.mean(0)) / (Y.std(0) + 1e-9)]
    beta, *_ = np.linalg.lstsq(P, acts, rcond=None)
    nov = ((acts - P @ beta) ** 2).sum(0) / np.maximum(((acts - acts.mean(0)) ** 2).sum(0), 1e-9)
    return align, nov, best_label, p_feature, q_feature


def recurrence(decs):                                  # best decoder-cosine of each seed-0 feature in each other seed
    nrm = lambda d: d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    ref = nrm(decs[0])
    return np.stack([np.abs(ref @ nrm(d).T).max(1) for d in decs[1:]], 1)


ef, ei, ok, df = load()
Z = zscore(ef)
c = lambda k: df[k].values
labs = np.column_stack([c("redshift"), c("mag_g_desi") - c("mag_r_desi"), c("mag_r_desi") - c("mag_z_desi"),
                        c("smooth-or-featured_smooth_fraction"), c("smooth-or-featured_featured-or-disk_fraction"),
                        c("merging_merger_fraction")]).astype(np.float64)
out = {"health": {}, "frontier": {}, "concepts": {}}

log("== train R in {4,8} x 5 seeds (k=32) ==")
decs4, acts0 = [], None
for R in R_LIST:
    for sd in SEEDS:
        h, acts, dec = train(Z, R * Z.shape[1], K, sd)
        out["health"][f"R{R}_s{sd}"] = h
        log(f"R={R} seed={sd} FVU={h['fvu']:.3f} alive={h['alive']:.3f}")
        if R == 4:
            decs4.append(dec)
            if sd == 0:
                acts0 = acts

LABELS = ["redshift", "g-r", "r-z", "smooth", "featured", "merger"]
log("== concept scoring (R=4): tie-aware Spearman + Bonferroni labels + BH features ==")
align, nov, best_label, p_feature, q_feature = score(acts0, labs)
best = recurrence(decs4)
active = acts0.std(0) > 1e-6
recfrac = (best >= 0.6).mean(1)                        # fraction of other seeds with a >=0.6-cosine match
q_active = np.ones_like(q_feature)
q_active[active] = bh_qvalues(p_feature[active])
q_feature = q_active
aligned, stable = q_feature <= 0.05, recfrac >= 0.5
named = {}                                             # name each aligned feature by its top physical label
for li, ln in enumerate(LABELS):
    fl = aligned & active & (best_label == li)
    if fl.sum():
        named[ln] = dict(n=int(fl.sum()), n_stable=int((fl & stable).sum()), top_align=float(align[fl].max()))
out["named_concepts"] = named
out["concepts"] = dict(
    n_features=int(len(align)), n_active=int(active.sum()),
    multiple_testing="per-feature min p Bonferroni across 6 labels; BH q<=0.05 across active dictionary",
    rank_method="average ties (true Spearman ranks)",
    max_align=float(align.max()),
    aligned_fdr=int((aligned & active).sum()),
    aligned_effect_gt_0p1=int((aligned & active & (align > 0.1)).sum()),
    aligned_and_stable=int((aligned & stable & active).sum()),
    median_best_cosine=float(np.median(best[active].max(1))),
    frac_active_stable=float(stable[active].mean()),
    stable_unexplained_not_label_aligned=int((stable & active & (nov > 0.7) & ~aligned).sum()),
    candidate_note="descriptive candidates only; novelty threshold has no confirmatory p-value")
for nm in ("sae_align", "sae_novelty", "sae_recurrence", "sae_best_label",
           "sae_pvalue", "sae_qvalue", "sae_acts0"):
    save_npy(nm, {"sae_align": align, "sae_novelty": nov, "sae_recurrence": recfrac,
                  "sae_best_label": best_label, "sae_pvalue": p_feature,
                  "sae_qvalue": q_feature, "sae_acts0": acts0}[nm])
log("named concepts: " + ", ".join(f"{k}={v['n']}({v['n_stable']} stable, max {v['top_align']:.2f})" for k, v in named.items()))
log(f"max_align={align.max():.2f} | FDR-aligned={int((aligned & active).sum())} "
    f"aligned&stable={int((aligned & stable & active).sum())} "
    f"stable-unexplained={int((stable & active & (nov > 0.7) & ~aligned).sum())}")

log("== L0-vs-FVU frontier (R=4, seed 0) ==")
for kk in KFRONT:
    h, _, _ = train(Z, 4 * Z.shape[1], kk, 0)
    out["frontier"][str(kk)] = h["fvu"]
    log(f"k={kk:3d} FVU={h['fvu']:.3f} alive={h['alive']:.3f}")

save_json("sae", out)
log("DONE")
