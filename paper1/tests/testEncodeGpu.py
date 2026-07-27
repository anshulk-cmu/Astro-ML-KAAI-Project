"""End-to-end exercise of the batched encoder, including its resume path.

`encode()` is the function Diagnostics 3, 5 and 9 will depend on to produce new embeddings,
and unlike the rest of lib/encode.py it cannot be checked without the model weights and a GPU.
It is skipped by default so the ordinary suite stays fast, and run deliberately:

    PAPER1_GPU_TESTS=1 python -B -m pytest paper1/tests/testEncodeGpu.py -q

Without this, the one piece of the cache library that actually writes embeddings would be
code that has never run.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import numpy as np
import pytest

import config as C
import data as D
import encode as E

pytestmark = pytest.mark.skipif(
    os.environ.get("PAPER1_GPU_TESTS") != "1" or not C.MODEL.exists(),
    reason="needs the model weights and a GPU; set PAPER1_GPU_TESTS=1 to run")

N = 64
PARAMS = {"operator": "identity", "rows": "gpu smoke test"}


@pytest.fixture(scope="module")
def cubes():
    df = D.anchor()
    ellip = df["ellip"].to_numpy(float)
    idx = np.where(D.pa_defined(df) & np.isfinite(ellip) & (ellip > C.ELLIP_CUT))[0][:N]
    ok = np.load(C.OK_INDEX)
    imgs = np.load(C.IMAGES, mmap_mode="r")
    return np.stack([np.asarray(imgs[ok[i]], np.float32) for i in idx]), idx


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CACHE", tmp_path)
    monkeypatch.setattr(E.C, "CACHE", tmp_path)
    return tmp_path


def test_encode_writes_a_complete_cache_entry(cache, cubes):
    stack, _ = cubes
    Z = E.encode(stack, "smoke", PARAMS)
    assert Z.shape == (N, E.EMBED_DIM) and Z.dtype == np.float32
    assert np.isfinite(Z).all()
    meta = json.loads((cache / "smoke.json").read_text())
    assert meta["complete"] is True and meta["n_done"] == N
    assert meta["recipe"] == E.recipe(**PARAMS)


def test_encode_reproduces_the_stored_untransformed_embeddings(cache, cubes):
    """The strongest available check: encoding these galaxies unchanged must reproduce the
    rows already in E_img, which were produced by a separate earlier pipeline."""
    stack, idx = cubes
    Z = E.encode(stack, "smoke", PARAMS)
    ref = np.load(C.E_IMG)[idx]
    d = np.abs(Z - ref)
    assert float(np.median(d)) == 0.0
    assert int((d.max(1) == 0).sum()) > len(idx) // 2, "most rows should be bitwise identical"
    # The max is not the right statistic: a handful of rows differ at the 0.1 level between
    # runs of the model in different processes, while their direction is unchanged.
    cos = (Z * ref).sum(1) / (np.linalg.norm(Z, axis=1) * np.linalg.norm(ref, axis=1))
    assert float(cos.min()) > 0.9999


def test_the_encoder_is_exactly_reproducible_within_one_process(cubes):
    """Distinguishes between-run drift from genuine non-determinism. The second claim is the
    one that would undermine every cached embedding, and it is false: repeats are bitwise."""
    stack, _ = cubes
    a = E._encode_array(stack[:32])
    b = E._encode_array(stack[:32])
    assert float(np.abs(a - b).max()) == 0.0


def test_a_second_call_is_served_from_cache_without_re_encoding(cache, cubes):
    stack, _ = cubes
    first = E.encode(stack, "smoke", PARAMS)
    mtime = (cache / "smoke.npy").stat().st_mtime_ns
    second = E.encode(stack, "smoke", PARAMS)
    assert np.array_equal(first, second)
    assert (cache / "smoke.npy").stat().st_mtime_ns == mtime, "the array was rewritten"


def test_encode_resumes_from_a_partial_file(cache, cubes):
    """Simulates an interrupted sweep: half the rows written, the entry marked incomplete.
    The resumed run must finish it and agree exactly with an uninterrupted one."""
    stack, _ = cubes
    full = E.encode(stack, "smoke", PARAMS).copy()

    half = N // 2
    arr = np.lib.format.open_memmap(cache / "smoke.npy", mode="r+")
    arr[half:] = 0.0
    arr.flush()
    (cache / "smoke.json").write_text(json.dumps(
        {"tag": "smoke", "recipe": E.recipe(**PARAMS), "n_done": half,
         "rows": N, "complete": False}))
    assert E.cached("smoke", PARAMS) is None, "an incomplete entry must not be served"

    resumed = E.encode(stack, "smoke", PARAMS)
    assert np.array_equal(resumed[:half], full[:half]), "resume overwrote finished rows"
    assert float(np.median(np.abs(resumed[half:] - full[half:]))) == 0.0
    assert json.loads((cache / "smoke.json").read_text())["complete"] is True


def test_a_completed_entry_requested_with_other_parameters_raises(cache, cubes):
    stack, _ = cubes
    E.encode(stack, "smoke", PARAMS)
    with pytest.raises(ValueError, match="different recipe"):
        E.encode(stack, "smoke", {**PARAMS, "operator": "rotate"})
