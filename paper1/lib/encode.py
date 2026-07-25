"""Frozen-encoder wrapper with a content-addressed cache.

The model is loaded from config.MODEL, never fine-tuned, and always called with
config.ENCODER_TOKENS and mean pooling so the recipe is identical across diagnostics.

Contract (to implement):
  encode(imgs, tag) -> array
      encodes in batches of config.ENCODE_BATCH, writes to CACHE/<tag>.npy, and resumes
      from a partial file rather than restarting. Re-encoding is the expensive step in the
      whole suite, measured at about 17 images per second, so nothing is ever encoded twice.
  cached(tag) -> array or None
  adopt_legacy(tag, path) -> array
      registers a pre-harness checkpoint as a cache entry after verifying its shape, dtype
      and row order against the current population mask. Used for the eight rotation and
      flip encodes that already exist and cost two and a quarter hours of GPU time.

Every cache entry records what produced it: operator, parameters, source rows, model path.
A cache hit with different parameters is an error, not a hit.
"""
