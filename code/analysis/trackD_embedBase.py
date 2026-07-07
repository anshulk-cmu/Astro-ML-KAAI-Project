"""
Track D cross-scale stage: embed the anchor's images through AION-BASE (the
smaller public model) to compare dictionaries across model scale (Matt's
diagnostic). Image-token-only embedding (E_img analogue), identical pooling to
Phase 1 (mean over encoder tokens), batch=32 (the measured local sweet spot).

Run:  python code/analysis/trackD_embedBase.py
Writes data/E_img_base.npy (48398 x dim, ok_index order).
"""
import time
import numpy as np
import torch
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage

DEV, BATCH = "cuda", 32

imgs = np.load("data/images.npy", mmap_mode="r")
ok = np.load("data/ok_index.npy")
model = AION.from_pretrained("polymathic-ai/aion-base").to(DEV).eval()
cm = CodecManager(device=DEV)
n = len(ok)
out = None
t0 = time.time()
with torch.no_grad():
    for s in range(0, n, BATCH):
        idx = ok[s:s + BATCH]
        flux = torch.from_numpy(np.ascontiguousarray(imgs[idx])).to(DEV)
        img = LegacySurveyImage(flux=flux, bands=["DES-G", "DES-R", "DES-I", "DES-Z"])
        emb = model.encode(cm.encode(img), num_encoder_tokens=600).mean(1).float().cpu().numpy()
        if out is None:
            out = np.zeros((n, emb.shape[1]), np.float32)
            print(f"embedding dim = {emb.shape[1]}", flush=True)
        out[s:s + len(idx)] = emb
        if (s // BATCH) % 100 == 0:
            print(f"{s + len(idx)}/{n} ({time.time()-t0:.0f}s)", flush=True)
np.save("data/E_img_base.npy", out)
print(f"DONE {out.shape} in {time.time()-t0:.0f}s -> data/E_img_base.npy", flush=True)
