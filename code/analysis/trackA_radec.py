"""
Track A: RA/Dec as their OWN loop candidates (Matt's list explicitly names these,
distinct from the RA-leakage NEGATIVE CONTROL already run on the image embedding).
AION has native Ra/Dec catalog modalities (tok_ra, tok_dec) -- feed the anchor's
true coordinates through AION's own encoder and test the SAME periodic-vs-bounded
distinction Matt draws for PA vs inclination, on an entirely different pair:
  - RA is periodic (wraps at 0/360) -> test as a k=1 loop (cos RA, sin RA).
  - Dec is bounded (-90 to +90, no wraparound) -> test as a plain scalar (arc),
    exactly like inclination, and confirm a circular treatment adds nothing.

Run: python code/analysis/trackA_radec.py   (fast: RA/Dec are scalar catalog
tokens, not images -- seconds of GPU time, all 48,398 anchor galaxies, no
subsampling.)
"""
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import Ra, Dec

import sys
sys.path.insert(0, "code/analysis")
from trackUtils import ALPHAS, SEED, ci

DATA, RES = "data", "results"
DEV = "cuda"
BATCH = 4096


def circ_recover_raw(c_true, s_true, X, tr, te, n_boot=200, seed=SEED):
    mc = RidgeCV(alphas=ALPHAS).fit(X[tr], c_true[tr])
    ms = RidgeCV(alphas=ALPHAS).fit(X[tr], s_true[tr])
    pc, ps = mc.predict(X[te]), ms.predict(X[te])
    d = np.angle(np.exp(1j * (np.arctan2(ps, pc) - np.arctan2(s_true[te], c_true[te]))))
    err = np.degrees(np.abs(d))
    rng = np.random.default_rng(seed)
    boot = [np.median(err[rng.integers(0, len(err), len(err))]) for _ in range(n_boot)]
    r2 = lambda y, p: float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return dict(med_err_deg=float(np.median(err)), ci=ci(boot),
               r2_cos=r2(c_true[te], pc), r2_sin=r2(s_true[te], ps))


def scalar_probe(y, X, tr, te, n_boot=200, seed=SEED):
    m = RidgeCV(alphas=ALPHAS).fit(X[tr], y[tr])
    p = m.predict(X[te])
    r2 = 1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        b = rng.integers(0, len(te), len(te))
        boot.append(1 - ((y[te][b] - p[b]) ** 2).sum() / ((y[te][b] - y[te][b].mean()) ** 2).sum())
    return dict(r2=float(r2), ci=ci(boot))


def main():
    ok = np.load(f"{DATA}/ok_index.npy")
    df = pd.read_parquet(f"{DATA}/sample.parquet").iloc[ok].reset_index(drop=True)
    ra = np.radians(df["ra"].to_numpy(float))
    dec = df["dec"].to_numpy(float)          # degrees, -90..90
    n = len(df)
    print(f"n = {n} (full anchor, no subsampling)", flush=True)

    model = AION.from_pretrained(f"{DATA}/models/aion-large").to(DEV).eval()
    cm = CodecManager(device=DEV)
    E_radec = np.zeros((n, 1024), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, BATCH):
            sub = slice(s, min(s + BATCH, n))
            m1 = Ra(value=torch.tensor(np.degrees(ra[sub]), device=DEV, dtype=torch.float32))
            m2 = Dec(value=torch.tensor(dec[sub], device=DEV, dtype=torch.float32))
            tok = cm.encode(m1, m2)
            E_radec[sub] = model.encode(tok, num_encoder_tokens=4).mean(1).float().cpu().numpy()
    print(f"encoded RA/Dec for {n} galaxies in {time.time()-t0:.0f}s", flush=True)

    Z = (E_radec - E_radec.mean(0)) / (E_radec.std(0) + 1e-8)
    tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=SEED)

    out = {"n": int(n)}
    out["ra_loop_k1"] = circ_recover_raw(np.cos(ra), np.sin(ra), Z, tr, te)
    out["dec_scalar"] = scalar_probe(dec, Z, tr, te)
    # control: does a CIRCULAR treatment of Dec (nonsensical, it's bounded not periodic) do any better?
    dec_rad_scaled = np.radians(dec)   # abuse as an "angle" purely as a null-shape control
    out["dec_loop_k1_control"] = circ_recover_raw(np.cos(dec_rad_scaled), np.sin(dec_rad_scaled), Z, tr, te)

    with open(f"{RES}/trackA_radec.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    r = out["ra_loop_k1"]
    print(f"RA as a loop (k=1): median err {r['med_err_deg']:.1f} deg CI{r['ci']}, "
          f"R2(cos,sin)={r['r2_cos']:.2f}/{r['r2_sin']:.2f}")
    d = out["dec_scalar"]
    print(f"Dec as a scalar (arc): R2 {d['r2']:.2f} CI{d['ci']}")
    dc = out["dec_loop_k1_control"]
    print(f"Dec forced into a circular treatment (should NOT help, bounded not periodic): "
          f"median err {dc['med_err_deg']:.1f} deg CI{dc['ci']}")
    print(f"wrote {RES}/trackA_radec.json")


if __name__ == "__main__":
    main()
