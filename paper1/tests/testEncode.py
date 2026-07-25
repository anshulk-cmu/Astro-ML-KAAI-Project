"""Cache-safety tests for the frozen-encoder wrapper.

`lib/encode.py` holds the rule that stops the most expensive mistake in the suite: reusing an
embedding that was produced by a different recipe or that is not row aligned with the current
population. Diagnostic 2 exercises `adopt_legacy` on real files, but the failure paths are the
part that matters and they never fire in a successful run. They are checked here on synthetic
arrays, so no GPU and no model weights are needed.

    python -m pytest paper1/tests -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import numpy as np
import pytest

import config as C
import encode as E

PARAMS = {"operator": "rotate", "phi_deg": 30.0, "rows": "elongated"}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Redirect the cache so a test can never touch the real one."""
    monkeypatch.setattr(C, "CACHE", tmp_path)
    monkeypatch.setattr(E.C, "CACHE", tmp_path)
    return tmp_path


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((64, E.EMBED_DIM)).astype(np.float32)
    moved = (reference + 0.1 * rng.standard_normal(reference.shape)).astype(np.float32)
    return reference, moved


def write(tmp_path, arr, name="leg.npy"):
    p = tmp_path / name
    np.save(p, arr)
    return p


def test_recipe_records_the_frozen_settings():
    r = E.recipe(**PARAMS)
    assert r["encoder_tokens"] == C.ENCODER_TOKENS and r["pooling"] == C.POOLING
    assert r["model"] == str(C.MODEL) and r["bands"] == E.BANDS
    assert r["operator"] == "rotate" and r["phi_deg"] == 30.0


def test_cached_returns_none_when_absent(cache):
    assert E.cached("nothing_here") is None


def test_adopt_then_cached_round_trips(cache, pair, tmp_path):
    reference, moved = pair
    Z, checks = E.adopt_legacy("t", write(tmp_path, moved), reference, PARAMS)
    assert np.array_equal(Z, moved)
    assert checks["n_nonfinite"] == 0 and checks["shape"] == [64, E.EMBED_DIM]
    assert np.array_equal(E.cached("t", PARAMS), moved)


def test_a_cache_hit_with_different_parameters_is_an_error(cache, pair, tmp_path):
    """The rule that stops a rotated encode being served as an unrotated one."""
    reference, moved = pair
    E.adopt_legacy("t", write(tmp_path, moved), reference, PARAMS)
    with pytest.raises(ValueError, match="different recipe"):
        E.cached("t", {**PARAMS, "phi_deg": 60.0})


def test_adopt_rejects_a_wrong_row_count(cache, pair, tmp_path):
    reference, moved = pair
    with pytest.raises(ValueError, match="does not match the population"):
        E.adopt_legacy("t", write(tmp_path, moved[:-1]), reference, PARAMS)


def test_adopt_rejects_a_wrong_width(cache, pair, tmp_path):
    reference, moved = pair
    with pytest.raises(ValueError, match="does not match the population"):
        E.adopt_legacy("t", write(tmp_path, moved[:, :-1]), reference, PARAMS)


def test_adopt_rejects_a_wrong_dtype(cache, pair, tmp_path):
    reference, moved = pair
    with pytest.raises(ValueError, match="dtype"):
        E.adopt_legacy("t", write(tmp_path, moved.astype(np.float64)), reference, PARAMS)


def test_adopt_rejects_non_finite_values(cache, pair, tmp_path):
    reference, moved = pair
    bad = moved.copy()
    bad[3, 7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        E.adopt_legacy("t", write(tmp_path, bad), reference, PARAMS)


def test_row_alignment_is_one_for_an_aligned_file(pair):
    reference, moved = pair
    assert E.row_alignment(reference, moved) == 1.0


def test_row_alignment_collapses_when_rows_are_permuted(pair):
    """The statistic is descriptive rather than a gate, but it must still respond to the
    failure it describes."""
    reference, moved = pair
    rolled = np.roll(moved, 1, axis=0)
    assert E.row_alignment(reference, rolled) < 0.05


def test_adopted_metadata_records_what_produced_the_entry(cache, pair, tmp_path):
    reference, moved = pair
    path = write(tmp_path, moved)
    E.adopt_legacy("t", path, reference, PARAMS)
    meta = E.read_meta("t")
    assert meta["complete"] is True and meta["rows"] == 64
    assert meta["recipe"] == E.recipe(**PARAMS)
    assert Path(meta["adopted_from"]) == path
