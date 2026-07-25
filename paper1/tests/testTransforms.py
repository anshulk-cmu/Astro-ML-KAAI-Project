"""Convention tests for the input-space operators.

These pin the rotation sign, the array-versus-catalog frame handedness, and the algebraic
properties of the operators before any expensive re-encoding is run. Diagnostic 2 measures a
signed shift of a readout, so a sign error here would not raise an exception: it would return
a slope of the wrong sign, and the only visible symptom would be a conclusion stated
backwards. That is why these run first and fail loudly.

The synthetic tests establish what the operators do to an angle in the array frame. The
real-catalog tests establish how the array frame relates to the catalog frame, which no
synthetic image can settle, and repeat the rotation test on real galaxies.

    python -m pytest paper1/tests -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import numpy as np
import pytest

transforms = pytest.importorskip("transforms")

pytestmark = pytest.mark.skipif(
    not hasattr(transforms, "rotate"),
    reason="input-space operators are introduced with Diagnostic 2; contract in lib/transforms.py",
)

import config as C
import data as D
from transforms import (ARRAY_TO_CATALOG_OFFSET_DEG, adaptive_moments, array_angle,
                        axial_concentration, mirror, rotate, wrap_axial)

ANGLES = [30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
START = [10.0, 55.0, 100.0, 145.0]
NEEDS_DATA = pytest.mark.skipif(not (C.IMAGES.exists() and C.SHAPES.exists()),
                                reason="anchor images or shape catalog not present")


def bar(angle_deg, n=96, length=30.0, width=3.0):
    """A bright bar whose array-frame angle is angle_deg by construction."""
    y, x = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    dx, dy = x - c, y - c
    t = np.radians(angle_deg)
    along = dx * np.cos(t) + dy * np.sin(t)
    perp = -dx * np.sin(t) + dy * np.cos(t)
    return np.exp(-0.5 * (along / length) ** 2 - 0.5 * (perp / width) ** 2).astype(np.float32)


@pytest.fixture(scope="module")
def real():
    """Bright, strongly elongated galaxies, where the moment angle is well determined."""
    df = D.anchor()
    ok = np.load(C.OK_INDEX)
    ellip = df["ellip"].to_numpy(float)
    sel = (D.pa_defined(df) & np.isfinite(ellip) & (ellip > 0.5)
           & (df["mag_r_desi"].to_numpy(float) < 18.0))
    idx = np.where(sel)[0]
    rng = np.random.default_rng(C.SEED)
    idx = idx[rng.choice(len(idx), min(250, len(idx)), replace=False)]
    imgs = np.load(C.IMAGES, mmap_mode="r")
    planes = np.stack([np.asarray(imgs[ok[i], 1], np.float32) for i in idx])
    meas = np.array([adaptive_moments(p) for p in planes], dtype=object)
    theta = np.array([m[0] for m in meas], float)
    good = np.array([bool(m[2]) for m in meas]) & np.isfinite(theta)
    return planes[good], theta[good], df["paDeg"].to_numpy(float)[idx][good]


def test_the_bar_generator_produces_the_angle_it_claims():
    """The measurement tool is checked before it is used to check anything else."""
    for a in [0.0, 20.0, 45.0, 70.0, 90.0, 110.0, 135.0, 160.0]:
        assert abs(wrap_axial(array_angle(bar(a)) - a)) < 0.5


def test_rotation_shifts_the_array_angle_by_minus_phi():
    """scipy.ndimage.rotate(+phi) turns the array frame so the measured angle DECREASES by
    phi. The sign is the whole point: the opposite convention would invert Diagnostic 2."""
    for a0 in START:
        for phi in ANGLES:
            got = wrap_axial(array_angle(rotate(bar(a0), phi)) - a0)
            assert abs(wrap_axial(got - (-phi))) < 0.5, (a0, phi, got)


def test_rotation_preserves_shape_and_dtype():
    b = bar(30.0)
    r = rotate(b, 37.0)
    assert r.shape == b.shape and r.dtype == np.float32


def test_a_full_turn_returns_an_unresampled_copy():
    """phi = 0 must not pay interpolation cost, and must not alias to a real rotation."""
    b = bar(30.0)
    assert np.array_equal(rotate(b, 0.0), b)
    assert np.array_equal(rotate(b, 360.0), b)
    assert rotate(b, 0.0) is not b


def test_rotation_is_applied_plane_by_plane_on_a_stack():
    """The rotation sense must not depend on how many leading axes the caller passed."""
    stack = np.stack([[bar(10.0), bar(80.0)], [bar(140.0), bar(35.0)]])
    got = rotate(stack, 40.0)
    assert got.shape == stack.shape
    for i in range(2):
        for j in range(2):
            assert np.array_equal(got[i, j], rotate(stack[i, j], 40.0))


def test_inverse_rotation_recovers_the_angle():
    for phi in [17.0, 40.0, 123.0]:
        assert abs(wrap_axial(array_angle(rotate(rotate(bar(65.0), phi), -phi)) - 65.0)) < 0.5


def test_mirror_is_an_exact_pixel_permutation():
    """No resampling: the multiset of pixel values is unchanged, so the mirror carries no
    interpolation error for the rotation results to be compared against."""
    for a in [0.0, 23.0, 45.0, 88.0, 131.0]:
        b = bar(a)
        m = mirror(b)
        assert np.array_equal(np.sort(b.ravel()), np.sort(m.ravel()))


def test_mirror_is_an_involution():
    b = bar(37.0)
    assert np.array_equal(mirror(mirror(b)), b)


def test_mirror_negates_the_array_angle():
    for a in [0.0, 20.0, 45.0, 70.0, 90.0, 135.0]:
        assert abs(wrap_axial(array_angle(mirror(bar(a))) - (-a))) < 0.5


def test_mirror_preserves_flux_exactly_and_rotation_does_not():
    """The contrast that separates representational error from resampling artifact."""
    b = bar(30.0)
    assert mirror(b).sum() == b.sum()
    assert not np.isclose(rotate(b, 37.0).sum(), b.sum(), rtol=0, atol=1e-6)


def test_mirror_fixed_points_and_antinodes():
    """Under theta -> -theta the fixed points are 0 and 90, and the displacement is largest
    at 45 and 135. That shape is a stronger prediction than any single number."""
    for a in [0.0, 90.0]:
        assert abs(wrap_axial(array_angle(mirror(bar(a))) - a)) < 0.5
    for a in [45.0, 135.0]:
        assert abs(abs(wrap_axial(array_angle(mirror(bar(a))) - a)) - 90.0) < 0.5


def test_composition_of_mirror_then_rotation():
    """Group composition in the plane: theta -> -theta - phi."""
    for a0 in [25.0, 70.0, 115.0]:
        for phi in [30.0, 75.0]:
            got = array_angle(rotate(mirror(bar(a0)), phi))
            assert abs(wrap_axial(got - (-a0 - phi))) < 0.5


def test_axial_wrap_is_bounded_by_the_period():
    d = wrap_axial(np.linspace(-1000, 1000, 5001))
    assert d.min() > -90.0 - 1e-9 and d.max() <= 90.0 + 1e-9


def test_axial_concentration_is_calibrated():
    """R is 1 for identical angles and 0 for uniform ones, which is what makes it able to
    decide handedness rather than merely describe a spread."""
    rng = np.random.default_rng(0)
    r_uniform, _ = axial_concentration(rng.uniform(0, 180, 200000))
    assert r_uniform < 0.01
    r_same, mu = axial_concentration(np.full(500, 33.0))
    assert r_same > 0.999 and abs(mu - 33.0) < 1e-6


@NEEDS_DATA
def test_the_array_frame_is_same_handed_with_the_catalog_frame(real):
    """Decisive test: if the frames are same-handed then array_angle - PA is a constant and
    concentrates; if they are opposite-handed then array_angle + PA is. Only one can."""
    _, theta, pa = real
    r_same, _ = axial_concentration(wrap_axial(theta - pa))
    r_opp, _ = axial_concentration(wrap_axial(theta + pa))
    assert r_same > 0.9, r_same
    assert r_opp < 0.2, r_opp


@NEEDS_DATA
def test_the_array_to_catalog_offset_is_ninety_degrees(real):
    """The cutouts are a north-up east-left projection, so east = -x and north = +y and a
    direction at PA east of north is (-sin PA, cos PA): array_angle = PA + 90. Measured here
    on real galaxies rather than taken from the cutout service documentation."""
    _, theta, pa = real
    _, offset = axial_concentration(wrap_axial(theta - pa))
    assert abs(wrap_axial(offset - ARRAY_TO_CATALOG_OFFSET_DEG)) < 2.0, offset


@NEEDS_DATA
def test_the_rotation_sign_holds_on_real_cutouts(real):
    """The synthetic bar is noise-free and centred. Real galaxies have neighbours, sky and
    asymmetry, so the sign is confirmed again where it will actually be applied."""
    planes, theta, _ = real
    for phi in [30.0, 60.0, 150.0]:
        got = np.array([wrap_axial(array_angle(rotate(p, phi)) - t)
                        for p, t in zip(planes[:80], theta[:80])])
        # rotation zeroes the corners, so the moment fit fails on a few sources that
        # converged unrotated; that is reported rather than hidden by the median
        assert np.isfinite(got).mean() > 0.9, np.isfinite(got).mean()
        assert abs(wrap_axial(np.nanmedian(got) - (-phi))) < 0.5, (phi, np.nanmedian(got))
