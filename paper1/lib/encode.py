"""Frozen-encoder wrapper with a content-addressed cache.

The model is loaded from config.MODEL, never fine-tuned, and always called with
config.ENCODER_TOKENS and mean pooling so the recipe is identical across diagnostics.

Every cache entry records what produced it: operator, parameters, source rows, model path.
A cache hit with different parameters is an error, not a hit.

Re-encoding is the expensive step in the whole suite, measured at about 17 images per second,
so nothing is ever encoded twice. `adopt_legacy` registers the eight rotation and flip encodes
that already exist and cost two and a quarter hours of GPU time, but only after their shape,
dtype and row order are checked against the current population.
"""
import json

import numpy as np

import config as C

BANDS = ["DES-G", "DES-R", "DES-I", "DES-Z"]
EMBED_DIM = 1024


def _meta_path(tag):
    return C.CACHE / f"{tag}.json"


def _array_path(tag):
    return C.CACHE / f"{tag}.npy"


def recipe(**params):
    """The parameters a cache entry is keyed on. Two entries agreeing here are interchangeable."""
    return {"model": str(C.MODEL), "encoder_tokens": C.ENCODER_TOKENS, "pooling": C.POOLING,
            "bands": BANDS, **params}


def read_meta(tag):
    p = _meta_path(tag)
    return json.loads(p.read_text()) if p.exists() else None


def cached(tag, params=None):
    """The array for this tag, or None. A stored entry whose recipe differs is an error."""
    m = read_meta(tag)
    if m is None or not _array_path(tag).exists() or not m.get("complete", False):
        return None
    if params is not None and m["recipe"] != recipe(**params):
        raise ValueError(f"cache entry {tag} was produced by a different recipe:\n"
                         f"  stored:    {m['recipe']}\n  requested: {recipe(**params)}")
    return np.load(m.get("source") or _array_path(tag), mmap_mode=None)


def row_alignment(reference, candidate, n_probe=200, seed=C.SEED):
    """Fraction of probe rows whose nearest neighbour in the reference is the same row.

    Recorded as a description of how far a transformation moves the embedding, NOT as a
    row-order gate. Measured values run from 0.045 at 90 degrees to 0.84 at 180 degrees:
    rotation displaces the embedding further than galaxy identity holds it, so a low value
    is the expected physics rather than evidence of a misaligned file. The authoritative
    row-order check is `reencode_check`.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidate), min(n_probe, len(candidate)), replace=False)
    a = candidate[idx].astype(np.float64)
    b = reference.astype(np.float64)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float((np.argmax(a @ b.T, axis=1) == idx).mean())


def adopt_legacy(tag, path, reference, params):
    """Register a pre-harness checkpoint as a cache entry, after checking it.

    Never adopted blind: shape, dtype and finiteness are checked against the population the
    current mask selects. Row order and operator identity are settled by `reencode_check`,
    whose result the caller is expected to record alongside.
    """
    path = str(path)
    Z = np.load(path)
    n, d = reference.shape
    checks = {"tag": tag, "path": path, "shape": list(Z.shape), "dtype": str(Z.dtype),
              "expected_shape": [n, d], "bytes": int(Z.nbytes)}
    if Z.shape != (n, d):
        raise ValueError(f"{tag}: shape {Z.shape} does not match the population {(n, d)}")
    if Z.dtype != np.float32:
        raise ValueError(f"{tag}: dtype {Z.dtype}, expected float32")
    checks["n_nonfinite"] = int((~np.isfinite(Z)).sum())
    if checks["n_nonfinite"]:
        raise ValueError(f"{tag}: {checks['n_nonfinite']} non-finite values")
    checks["row_self_match"] = row_alignment(reference, Z)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    _meta_path(tag).write_text(json.dumps(
        {"tag": tag, "recipe": recipe(**params), "adopted_from": path, "source": path,
         "complete": True, "rows": n, "checks": checks}, indent=2))
    return Z, checks


def reencode_check(cubes, operator, legacy_rows, device="cuda", batch=C.ENCODE_BATCH):
    """Re-encode a subsample under `operator` and compare it to the stored legacy rows.

    This settles two questions at once that no structural check can: whether the cached file
    was produced by the operator we now hold in lib/transforms.py, and whether its rows are
    aligned with the population the current mask selects. The rolled control reports what the
    same statistic returns when the rows are deliberately offset by one, so the check's power
    to fail is recorded beside its result rather than assumed.
    """
    got = _encode_array(operator(cubes), device=device, batch=batch)
    d = np.abs(got - legacy_rows)
    num = (got * legacy_rows).sum(1)
    den = np.linalg.norm(got, axis=1) * np.linalg.norm(legacy_rows, axis=1)
    rolled = np.abs(legacy_rows - np.roll(legacy_rows, 1, axis=0))
    return {"n": int(len(got)),
            "max_abs_diff": float(d.max()), "median_abs_diff": float(np.median(d)),
            "mean_cosine": float(np.mean(num / den)),
            "control_rolled_by_one_median_abs_diff": float(np.median(rolled)),
            "note": ("median 0 means bitwise agreement on the majority of components; a small "
                     "max reflects non-deterministic GPU reduction order, not a different "
                     "operator. The rolled control is what a one-row misalignment would give.")}


def _encode_array(arr, device="cuda", batch=C.ENCODE_BATCH):
    import torch
    from aion.modalities import LegacySurveyImage
    model, cm = _model(device)
    out = np.zeros((len(arr), EMBED_DIM), np.float32)
    for s in range(0, len(arr), batch):
        cube = np.ascontiguousarray(arr[s:s + batch], dtype=np.float32)
        with torch.no_grad():
            tok = cm.encode(LegacySurveyImage(flux=torch.from_numpy(cube).to(device), bands=BANDS))
            out[s:s + len(cube)] = (model.encode(tok, num_encoder_tokens=C.ENCODER_TOKENS)
                                    .mean(1).float().cpu().numpy())
    return out


_LOADED = {}


def _model(device="cuda"):
    """Loaded once per process: the weights cost about six seconds to bring up."""
    if device not in _LOADED:
        from aion import AION
        from aion.codecs import CodecManager
        _LOADED[device] = (AION.from_pretrained(str(C.MODEL)).to(device).eval(),
                           CodecManager(device=device))
    return _LOADED[device]


def encode(imgs, tag, params, device="cuda", batch=C.ENCODE_BATCH, log_every=20):
    """Encode a (n, 4, 96, 96) stack through the frozen model, caching to CACHE/<tag>.npy.

    Resumes from a partial file rather than restarting, since a full sweep is hours of GPU.
    """
    import time

    import torch
    from aion.modalities import LegacySurveyImage

    hit = cached(tag, params)
    if hit is not None:
        return hit

    C.CACHE.mkdir(parents=True, exist_ok=True)
    n = len(imgs)
    m = read_meta(tag) or {}
    done = int(m.get("n_done", 0)) if _array_path(tag).exists() and m.get("recipe") == recipe(**params) else 0
    out = (np.lib.format.open_memmap(_array_path(tag), mode="r+")
           if done else
           np.lib.format.open_memmap(_array_path(tag), mode="w+", dtype=np.float32,
                                     shape=(n, EMBED_DIM)))

    def write_meta(k, complete):
        _meta_path(tag).write_text(json.dumps(
            {"tag": tag, "recipe": recipe(**params), "n_done": int(k), "rows": n,
             "complete": bool(complete)}, indent=2))

    model, cm = _model(device)
    t0 = time.time()
    for s in range(done, n, batch):
        cube = np.ascontiguousarray(imgs[s:s + batch], dtype=np.float32)
        with torch.no_grad():
            tok = cm.encode(LegacySurveyImage(flux=torch.from_numpy(cube).to(device), bands=BANDS))
            out[s:s + len(cube)] = (model.encode(tok, num_encoder_tokens=C.ENCODER_TOKENS)
                                    .mean(1).float().cpu().numpy())
        if (s // batch) % log_every == 0:
            out.flush()
            write_meta(s + len(cube), False)
            print(f"    {tag} {s + len(cube)}/{n} ({time.time() - t0:.0f}s)", flush=True)
    out.flush()
    write_meta(n, True)
    return np.asarray(out)
