import torch
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage

dev = "cuda"
model = AION.from_pretrained("polymathic-ai/aion-large").to(dev).eval()
cm = CodecManager(device=dev)

flux = torch.rand(2, 4, 96, 96, device=dev)
img = LegacySurveyImage(flux=flux, bands=["DES-G", "DES-R", "DES-I", "DES-Z"])

with torch.no_grad():
    tokens = cm.encode(img)
    emb = model.encode(tokens, num_encoder_tokens=600)

pooled = emb.mean(dim=1)
print("token keys:", list(tokens.keys()))
print("emb shape:", tuple(emb.shape))
print("pooled shape:", tuple(pooled.shape))
print("SMOKE_OK")
