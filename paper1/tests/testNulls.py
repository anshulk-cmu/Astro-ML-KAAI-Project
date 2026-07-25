"""Null calibration tests. A null that is not itself calibrated is decoration.

    python -m pytest paper1/tests -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import numpy as np
import pytest

import config as C
import nulls as N
import probes as P
from circular import CircProbe, circ_error, fit_evaluate, radius


def synthetic(n=4000, d=64, k=2, r2=0.9, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, np.pi, n)
    c, s = np.cos(k * theta), np.sin(k * theta)
    Z = rng.standard_normal((n, d))
    Z[:, 0], Z[:, 1] = c, s
    noise = np.sqrt((1 - r2) / r2) * np.std(c)
    Z[:, 0] += rng.normal(0, noise, n)
    Z[:, 1] += rng.normal(0, noise, n)
    return Z.astype(np.float32), theta


def test_chance_floor_is_45_for_axial():
    """A readout carrying no information must return the analytic median of 45 degrees."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, np.pi, 200000)
    pred = rng.uniform(0, np.pi, 200000)
    err = circ_error(np.cos(2 * pred), np.sin(2 * pred), theta, 2)
    assert abs(np.median(err) - 45.0) < 0.5


def test_chance_floor_is_90_for_full_circle():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 200000)
    pred = rng.uniform(0, 2 * np.pi, 200000)
    err = circ_error(np.cos(pred), np.sin(pred), theta, 1)
    assert abs(np.median(err) - 90.0) < 1.0


def test_wrap_is_symmetric_across_the_seam():
    """179 and 1 degrees are two degrees apart on an axial scale, not 178."""
    t = np.radians(np.array([179.0]))
    p = np.radians(np.array([1.0]))
    err = circ_error(np.cos(2 * p), np.sin(2 * p), t, 2)
    assert abs(err[0] - 2.0) < 1e-6


def test_shuffled_labels_reach_chance_and_collapse_the_radius():
    Z, theta = synthetic()
    mask = np.ones(len(theta), bool)
    sh = fit_evaluate(Z, N.shuffled(theta, mask), 2, mask)[3]
    assert 40.0 < sh["med_err_deg"] < 50.0
    assert sh["loop_radius"] < 0.2
    assert sh["r2_cos"] < 0.05 and sh["r2_sin"] < 0.05


def test_signal_recovers_and_radius_approaches_one():
    Z, theta = synthetic(r2=0.95)
    out = fit_evaluate(Z, theta, 2, np.ones(len(theta), bool))[3]
    assert out["med_err_deg"] < 5.0
    assert out["loop_radius"] > 0.85


def test_loop_radius_matches_the_shrinkage_identity():
    """Expected squared radius equals the mean component R2 under the linear-Gaussian model."""
    for r2 in (0.5, 0.8, 0.95):
        Z, theta = synthetic(r2=r2)
        out = fit_evaluate(Z, theta, 2, np.ones(len(theta), bool))[3]
        predicted = np.sqrt((out["r2_cos"] + out["r2_sin"]) / 2)
        assert abs(out["loop_radius_rms"] - predicted) < 0.03


def test_two_random_directions_are_near_orthogonal_in_high_dimension():
    """A raw angle near 90 degrees is the null, not a finding."""
    V = N.matched_norm_directions(np.ones(1024), 500, seed=0)
    cos = V[:250] @ V[250:].T / (np.linalg.norm(V[:250], axis=1)[:, None]
                                 * np.linalg.norm(V[250:], axis=1)[None, :])
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    assert abs(np.median(ang) - 90.0) < 1.0


def test_matched_norm_directions_preserve_norm():
    w = np.random.default_rng(0).standard_normal(256)
    V = N.matched_norm_directions(w, 32, seed=1)
    assert np.allclose(np.linalg.norm(V, axis=1), np.linalg.norm(w))


def test_fixed_probe_applies_unchanged_to_new_rows():
    """The causal diagnostics depend on a probe fit once and never refit."""
    Z, theta = synthetic()
    tr, te = P.split(np.ones(len(theta), bool))
    pr = CircProbe(Z, theta, 2, tr)
    a = pr.predict(Z[te])
    b = pr.predict(Z[te])
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_bootstrap_interval_brackets_the_point_estimate():
    x = np.random.default_rng(0).gamma(2.0, 1.0, 5000)
    lo, hi = P.boot_stat(x, np.median, n_boot=400)
    assert lo < np.median(x) < hi


def test_split_is_deterministic():
    m = np.ones(1000, bool)
    assert np.array_equal(P.split(m)[0], P.split(m)[0])


def test_radius_of_unit_truth_is_one():
    t = np.linspace(0, np.pi, 100)
    assert np.allclose(radius(np.cos(2 * t), np.sin(2 * t)), 1.0)


@pytest.mark.parametrize("k", [1, 2])
def test_error_range_is_bounded_by_the_period(k):
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi / k, 10000)
    pred = rng.uniform(0, 2 * np.pi / k, 10000)
    err = circ_error(np.cos(k * pred), np.sin(k * pred), theta, k)
    assert err.min() >= 0.0 and err.max() <= 180.0 / k + 1e-9
