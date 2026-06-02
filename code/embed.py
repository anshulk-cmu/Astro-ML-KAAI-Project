import time
import numpy as np
import pandas as pd
import torch
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import (
    LegacySurveyImage, LegacySurveyFluxG, LegacySurveyFluxR, LegacySurveyFluxZ, Z,
)

D, dev, B = "/home/ubuntu/data", "cuda", 256

df = pd.read_parquet(f"{D}/sample.parquet")
status = np.load(f"{D}/status.npy")
imgs = np.load(f"{D}/images.npy", mmap_mode="r")
ok = np.where(status == 1)[0]
print("embedding", len(ok), "galaxies", flush=True)


def nmgy(mag):
    return np.power(10.0, (22.5 - mag) / 2.5).astype(np.float32)


fg_all, fr_all, fz_all = nmgy(df["mag_g_desi"].values), nmgy(df["mag_r_desi"].values), nmgy(df["mag_z_desi"].values)
z_all = df["redshift"].values.astype(np.float32)

model = AION.from_pretrained("polymathic-ai/aion-large").to(dev).eval()
cm = CodecManager(device=dev)

E_img = np.zeros((len(ok), 1024), np.float32)
E_full = np.zeros((len(ok), 1024), np.float32)

t0 = time.time()
with torch.no_grad():
    for s in range(0, len(ok), B):
        idx = ok[s:s + B]
        flux = torch.from_numpy(np.ascontiguousarray(imgs[idx])).to(dev)
        img = LegacySurveyImage(flux=flux, bands=["DES-G", "DES-R", "DES-I", "DES-Z"])
        fg = LegacySurveyFluxG(value=torch.tensor(fg_all[idx], device=dev))
        fr = LegacySurveyFluxR(value=torch.tensor(fr_all[idx], device=dev))
        fz = LegacySurveyFluxZ(value=torch.tensor(fz_all[idx], device=dev))
        zz = Z(value=torch.tensor(z_all[idx], device=dev))
        tok_img = cm.encode(img)
        tok_full = {**tok_img, **cm.encode(fg, fr, fz, zz)}
        E_img[s:s + len(idx)] = model.encode(tok_img, num_encoder_tokens=600).mean(1).float().cpu().numpy()
        E_full[s:s + len(idx)] = model.encode(tok_full, num_encoder_tokens=600).mean(1).float().cpu().numpy()
        if s % (B * 20) == 0:
            print(f"{s}/{len(ok)} {time.time() - t0:.0f}s", flush=True)

np.save(f"{D}/E_img.npy", E_img)
np.save(f"{D}/E_full.npy", E_full)
np.save(f"{D}/ok_index.npy", ok)
print("DONE", E_img.shape, "time", round(time.time() - t0), "s")
