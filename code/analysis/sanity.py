import numpy as np
import torch
from common import load, save_json, log

ef, ei, ok, df = load()
log(f"E_full {ef.shape}  E_img {ei.shape}  labels {df.shape}")

out = {}
for name, E in [("E_full", ef), ("E_img", ei)]:
    n, d = E.shape
    norms = np.linalg.norm(E, axis=1)
    sd = E.std(0)
    X = torch.from_numpy(E).cuda().float()
    X = X - X.mean(0)
    s = torch.linalg.svdvals(X).cpu().numpy().astype(np.float64)
    var = (s ** 2) / (s ** 2).sum()
    cum = np.cumsum(var)
    pr = float((s ** 2).sum() ** 2 / (s ** 4).sum())
    pcs = {f"var{p}": int(np.searchsorted(cum, p / 100) + 1) for p in (50, 90, 95, 99)}
    out[name] = dict(
        n=n, d=d,
        nan=int(np.isnan(E).sum()), inf=int(np.isinf(E).sum()),
        norm_mean=float(norms.mean()), norm_std=float(norms.std()),
        norm_min=float(norms.min()), norm_max=float(norms.max()),
        dim_std_min=float(sd.min()), dim_std_max=float(sd.max()),
        participation_ratio=pr, pcs_for_variance=pcs,
    )
    log(f"{name}: nan={out[name]['nan']} inf={out[name]['inf']} "
        f"|x|={norms.mean():.3f}+-{norms.std():.3f} dimstd[{sd.min():.3g},{sd.max():.3g}] "
        f"PR={pr:.1f} PCs(50/90/95/99%)={[pcs[k] for k in pcs]}")

save_json("sanity", out)
log("DONE")
