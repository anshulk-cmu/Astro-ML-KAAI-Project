"""Convention tests for Diagnostic 3's isolating operator and its control.

The claim these must pin is that `major_axis_flip` changes chirality and nothing else. If it
also moved the position angle or the ellipticity, the difference vector it produces would be
dominated by orientation, which is the failure the whole design exists to avoid, and no
exception would be raised to say so.

The synthetic spiral makes the claim falsifiable: flipping about the major axis maps the
object-frame angle phi to -phi, which turns a logarithmic spiral of one handedness into the
same spiral of the other. So `major_axis_flip(spiral(+1))` must reproduce `spiral(-1)`, an
image built independently rather than derived from the first.

    python -m pytest paper1/tests -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import numpy as np
import pytest

from transforms import (ARRAY_TO_CATALOG_OFFSET_DEG, array_angle, adaptive_moments,
                        axis_sandwich, major_axis_flip, rotate, wrap_axial)

PA_GRID = [0.0, 23.0, 47.0, 90.0, 118.0, 156.0]
N = 96


def source(pa_deg, handed=0, n=N, arms=2, pitch=0.45, r0=6.0, q=0.55, sigma=15.0, amp=0.8):
    """An elliptical envelope with optional logarithmic spiral arms.

    handed = 0 gives an achiral ellipse. handed = +1 and -1 give mirror-image spirals of
    identical envelope, so any difference between them is chirality alone.
    """
    y, x = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    t = np.radians(pa_deg + ARRAY_TO_CATALOG_OFFSET_DEG)
    dx, dy = x - c, y - c
    xo = dx * np.cos(t) + dy * np.sin(t)          # object frame, major axis along xo
    yo = -dx * np.sin(t) + dy * np.cos(t)
    env = np.exp(-0.5 * ((xo / sigma) ** 2 + (yo / (q * sigma)) ** 2))
    if handed == 0:
        return env.astype(np.float32)
    rho = np.hypot(xo, yo) + 1e-3
    phi = np.arctan2(yo, xo)
    pattern = 1.0 + amp * np.cos(arms * (phi - handed * np.log(rho / r0) / pitch))
    return np.clip(env * pattern, 0, None).astype(np.float32)


def corr(a, b):
    u, v = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def test_the_generator_makes_two_genuinely_different_handednesses():
    """Checked before the generator is used to check anything else."""
    for pa in PA_GRID:
        assert corr(source(pa, +1), source(pa, -1)) < 0.9
        assert corr(source(pa, +1), source(pa, +1)) == pytest.approx(1.0)


def test_the_generator_makes_the_position_angle_it_claims():
    for pa in PA_GRID:
        assert abs(wrap_axial(array_angle(source(pa, 0)) - (pa + ARRAY_TO_CATALOG_OFFSET_DEG))) < 1.0


def test_major_axis_flip_turns_a_spiral_into_its_mirror_twin():
    """The decisive test. The comparison image is built independently, not derived."""
    for pa in PA_GRID:
        got = major_axis_flip(source(pa, +1)[None], [pa])[0]
        twin = axis_sandwich(source(pa, -1)[None], [pa])[0]   # same resampling, no flip
        same = axis_sandwich(source(pa, +1)[None], [pa])[0]
        assert corr(got, twin) > 0.99, (pa, corr(got, twin))
        assert corr(got, twin) > corr(got, same) + 0.05


def test_major_axis_flip_leaves_an_achiral_ellipse_alone():
    """An ellipse is symmetric about its own major axis, so the flip must do nothing to it
    beyond what the two rotations already do. This is the interpolation floor of Diagnostic 3."""
    for pa in PA_GRID:
        flipped = major_axis_flip(source(pa, 0)[None], [pa])[0]
        control = axis_sandwich(source(pa, 0)[None], [pa])[0]
        assert corr(flipped, control) > 0.999, (pa, corr(flipped, control))


def test_major_axis_flip_preserves_the_axis_of_an_achiral_object():
    for pa in PA_GRID:
        before = source(pa, 0)
        after = major_axis_flip(before[None], [pa])[0]
        assert abs(wrap_axial(array_angle(after) - array_angle(before))) < 0.5


def test_the_flip_reflects_a_chiral_object_s_apparent_angle_about_its_envelope_axis():
    """A chiral object's moment angle is NOT its envelope axis: the arms pull it off by a few
    degrees, and the flip reflects that offset to the other side. The envelope axis is what is
    preserved, and the apparent angle moves by twice the offset.

    This is the operator behaving correctly, and it is also a confound Diagnostic 3 has to
    control: for chiral objects, and only for them, the flip does change apparent orientation.
    Whether the model's readout follows the envelope or the moments is measured on the encoded
    data, not assumed here.
    """
    for pa in PA_GRID:
        envelope = pa + ARRAY_TO_CATALOG_OFFSET_DEG
        for handed in (+1, -1):
            before = array_angle(source(pa, handed))
            after = array_angle(major_axis_flip(source(pa, handed)[None], [pa])[0])
            assert abs(wrap_axial(before - envelope)) > 1.0, "the test object is not chiral enough"
            assert abs(wrap_axial(after - (2 * envelope - before))) < 0.5, (pa, handed)


def test_major_axis_flip_preserves_the_ellipticity():
    for pa in PA_GRID:
        e_before = adaptive_moments(source(pa, 0))[1]
        e_after = adaptive_moments(major_axis_flip(source(pa, 0)[None], [pa])[0])[1]
        assert abs(e_after - e_before) < 0.02, (pa, e_before, e_after)


def test_major_axis_flip_is_an_involution():
    """Compared against the control applied twice, not against the raw source, so that four
    accumulated interpolations are not mistaken for a failure of the involution."""
    for pa in PA_GRID:
        twice = major_axis_flip(major_axis_flip(source(pa, +1)[None], [pa]), [pa])[0]
        control = axis_sandwich(axis_sandwich(source(pa, +1)[None], [pa]), [pa])[0]
        assert corr(twice, control) > 0.999, (pa, corr(twice, control))


def test_the_control_carries_the_same_interpolation_cost():
    """If the control resampled less than the operator, the comparison between them would
    measure resampling rather than chirality."""
    for pa in PA_GRID:
        im = source(pa, +1)[None]
        lost_flip = 1 - major_axis_flip(im, [pa]).sum() / im.sum()
        lost_ctrl = 1 - axis_sandwich(im, [pa]).sum() / im.sum()
        assert abs(lost_flip - lost_ctrl) < 1e-3, (pa, lost_flip, lost_ctrl)


def test_the_control_is_not_the_identity():
    """It must resample, otherwise it is not a matched control."""
    im = source(47.0, +1)
    assert not np.array_equal(axis_sandwich(im[None], [47.0])[0], im)


def test_both_operators_apply_per_object_angles():
    stack = np.stack([source(pa, +1) for pa in PA_GRID])
    out = major_axis_flip(stack, PA_GRID)
    for i, pa in enumerate(PA_GRID):
        assert np.array_equal(out[i], major_axis_flip(stack[i][None], [pa])[0])


def test_a_wrong_number_of_angles_is_an_error():
    stack = np.stack([source(pa, 0) for pa in PA_GRID])
    with pytest.raises(ValueError, match="one angle per object"):
        major_axis_flip(stack, [0.0])


def test_the_operators_work_on_multi_band_cubes():
    cube = np.stack([np.stack([source(47.0, +1)] * 4)])
    assert major_axis_flip(cube, [47.0]).shape == cube.shape
    assert axis_sandwich(cube, [47.0]).shape == cube.shape
