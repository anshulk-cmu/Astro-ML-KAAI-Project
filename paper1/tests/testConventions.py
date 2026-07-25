"""Catalog convention tests.

The scoping protocol treats a documented convention as an assumption until it is checked.
These tests verify the ellipticity parameterisation directly against the catalog columns we
actually use, which is a stronger check than quoting documentation: if the identities hold to
machine precision on 48,290 real sources, the convention in our data is the stated one.

    python -m pytest paper1/tests -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import numpy as np
import pytest

import config as C
import data as D

pytestmark = pytest.mark.skipif(not C.SHAPES.exists(), reason="anchor shape catalog not present")


@pytest.fixture(scope="module")
def cat():
    df = D.anchor()
    ok = D.pa_defined(df) & np.isfinite(df["ellip"].to_numpy(float))
    return df[ok].reset_index(drop=True)


def test_ellipticity_magnitude_identity(cat):
    e1, e2 = cat["shape_e1"].to_numpy(float), cat["shape_e2"].to_numpy(float)
    assert np.allclose(cat["ellip"].to_numpy(float), np.hypot(e1, e2), atol=1e-9)


def test_axis_ratio_identity(cat):
    e = cat["ellip"].to_numpy(float)
    assert np.allclose(cat["ba"].to_numpy(float), (1 - e) / (1 + e), atol=1e-9)


def test_position_angle_is_the_half_argument(cat):
    e1, e2 = cat["shape_e1"].to_numpy(float), cat["shape_e2"].to_numpy(float)
    pa = np.degrees(0.5 * np.arctan2(e2, e1)) % 180.0
    assert np.allclose(cat["paDeg"].to_numpy(float) % 180.0, pa, atol=1e-9)


def test_components_reconstruct_from_magnitude_and_doubled_angle(cat):
    """e1 = |eps| cos 2phi and e2 = |eps| sin 2phi: the catalog stores the axial angle doubled."""
    e = cat["ellip"].to_numpy(float)
    phi = np.radians(cat["paDeg"].to_numpy(float))
    assert np.allclose(cat["shape_e1"].to_numpy(float), e * np.cos(2 * phi), atol=1e-9)
    assert np.allclose(cat["shape_e2"].to_numpy(float), e * np.sin(2 * phi), atol=1e-9)


def test_inclination_follows_from_axis_ratio(cat):
    ba = cat["ba"].to_numpy(float)
    assert np.allclose(cat["inclDeg"].to_numpy(float), np.degrees(np.arccos(ba)), atol=1e-6)


def test_ellipticity_and_axis_ratio_stay_in_range(cat):
    e, ba = cat["ellip"].to_numpy(float), cat["ba"].to_numpy(float)
    assert e.min() >= 0.0 and e.max() < 1.0
    assert ba.min() > 0.0 and ba.max() <= 1.0


def test_circular_models_are_excluded_from_the_defined_population():
    """REX, PSF and DUP models carry no fitted ellipticity and cannot define an angle."""
    df = D.anchor()
    ok = D.pa_defined(df)
    e1, e2 = df["shape_e1"].to_numpy(float), df["shape_e2"].to_numpy(float)
    assert not ((e1[ok] == 0) & (e2[ok] == 0)).any()
    excluded = np.isfinite(df["paDeg"].to_numpy(float)) & ~ok
    assert set(df["type"].astype(str).to_numpy()[excluded]) <= {"REX", "PSF", "DUP"}


def test_position_angle_of_defined_sources_is_uniform_on_the_sky():
    """No preferred sky orientation should survive once placeholder angles are removed."""
    from scipy.stats import kstest
    df = D.anchor()
    pa = df["paDeg"].to_numpy(float)[D.pa_defined(df)]
    assert kstest((pa % 180) / 180.0, "uniform").pvalue > 1e-3


def test_sentinel_values_are_excluded_from_valid_labels():
    """Some catalog columns encode 'not measured' as a finite sentinel that isfinite accepts."""
    df = D.anchor()
    ok, v = D.valid(df, "total_ssfr_median")
    assert np.isfinite(v).sum() > ok.sum(), "sentinel rule removed nothing"
    assert (v[ok] > C.SENTINEL_MIN["total_ssfr_median"]).all()
    assert ok.sum() == 4473


def test_labels_without_a_declared_sentinel_keep_every_finite_value():
    df = D.anchor()
    for col in ("redshift", "elpetro_mass_log", "sersic_n"):
        ok, v = D.valid(df, col)
        assert ok.sum() == np.isfinite(v).sum()


def test_every_label_used_has_a_declared_source():
    for col in ("redshift", "total_ssfr_median", "elpetro_mass_log", "sersic_n",
                "paDeg", "ellip", "psfsize_r", "ebv"):
        assert col in C.LABEL_SOURCES and C.LABEL_SOURCES[col][0] != "undeclared"


def test_pa_uncertainty_scales_inversely_with_elongation(cat):
    """sigma_PA ~ sigma_e / (2|eps|), so the product with ellipticity should be flat."""
    sig = D.pa_uncertainty(cat)
    e = cat["ellip"].to_numpy(float)
    ok = np.isfinite(sig) & (e > 0.02)
    lo = np.median((sig * e)[ok & (e < 0.1)])
    hi = np.median((sig * e)[ok & (e > 0.4)])
    assert 0.3 < hi / lo < 3.0
