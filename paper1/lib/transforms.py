"""Deterministic operations on input images. These are what make a diagnostic causal.

Each operator has an exactly known consequence, verified in tests/testTransforms.py BEFORE
any of them is run at scale. Sign and frame conventions are the failure mode here: the
array frame, the catalog angle convention and the library rotation convention must be
pinned down and tested, not assumed.

Frame. Images are 96x96 four-band cutouts from the Legacy Surveys viewer, a north-up
east-left tangent projection, loaded so that axis -2 increases with declination and axis -1
increases to the west. `array_angle` measures the major axis from +x (axis -1) toward +y
(axis -2), modulo 180. With east = -x and north = +y a direction at catalog position angle
PA east of north has components (-sin PA, cos PA), so the expectation is

    array_angle = PA + ARRAY_TO_CATALOG_OFFSET_DEG   (mod 180),  same-handed

That is an expectation from the cutout provenance; the tests measure it on real galaxies
and would fail if it were wrong.

Verified consequences of the two operators implemented here:
  rotate(imgs, phi)   array_angle -> array_angle - phi  (mod 180)
  mirror(imgs)        array_angle -> -array_angle       (mod 180)

Contract (to implement):
  blur(imgs, fwhm_target, fwhm_current)   Moffat convolution to worse seeing
  add_noise(imgs, depth_target)           noise to a target point-source depth
  zero_band(imgs, band)                   band ablation and synthetic band-blanking
  scale_flux(imgs, dmag)                  zeropoint offset
  redden(imgs, ebv, rv)                   per-band multiplicative from an extinction curve
  pedestal(imgs, level)                   additive constant on all pixels
  ferengi(imgs, z_from, z_to, ...)        angular size, flux dimming, k-correction,
                                          PSF reconvolution, noise to target depth

Everything below major_axis_flip is new work. The Phase 2 realism code that implements blur
and noise lives on the Vera cluster and is not on this machine, so it cannot be lifted.
"""
import numpy as np
from scipy.ndimage import rotate as _sp_rotate

# Resampling recipe for rotate. Fixed here so no caller can silently choose its own, and
# identical to the recipe that produced the cached rotation encodes.
ROTATE_ORDER = 1
ROTATE_MODE = "constant"
ROTATE_CVAL = 0.0

# Measured on real cutouts in tests/testTransforms.py, not quoted from documentation.
ARRAY_TO_CATALOG_OFFSET_DEG = 90.0


def wrap_axial(deg):
    """Fold a difference of axial angles into (-90, 90], the signed mod-180 scale."""
    return (np.asarray(deg, float) + 90.0) % 180.0 - 90.0


def rotate(imgs, phi):
    """Rotate the last two axes by phi degrees, shape preserved.

    Applied plane by plane so the rotation sense cannot depend on how many leading axes the
    caller passed. Bilinear interpolation costs accuracy; mirror does not, which is what
    separates representational error from resampling artifact. phi a multiple of 360 returns
    an unresampled copy.
    """
    a = np.ascontiguousarray(imgs, dtype=np.float32)
    if phi % 360.0 == 0.0:
        return a.copy()
    out = np.empty_like(a)
    plane = a.reshape(-1, *a.shape[-2:])
    plane_out = out.reshape(-1, *a.shape[-2:])
    for i in range(plane.shape[0]):
        plane_out[i] = _sp_rotate(plane[i], phi, reshape=False, order=ROTATE_ORDER,
                                  mode=ROTATE_MODE, cval=ROTATE_CVAL)
    return out


def mirror(imgs):
    """Mirror about the vertical axis: an exact permutation of columns.

    No resampling, so no interpolation error and no flux change. This is the cleanest
    intervention in the suite.
    """
    return np.ascontiguousarray(np.flip(np.asarray(imgs), axis=-1))


def _per_object(imgs, angles_deg, flip_rows):
    """Rotate each object to put its major axis horizontal, optionally flip about that axis,
    then rotate back. Applied object by object because the angle differs per object."""
    a = np.asarray(imgs, dtype=np.float32)
    ang = np.asarray(angles_deg, float)
    if ang.shape != (len(a),):
        raise ValueError(f"expected one angle per object, got {ang.shape} for {len(a)} objects")
    out = np.empty_like(a)
    for i in range(len(a)):
        r = rotate(a[i], ang[i])
        if flip_rows:
            r = np.flip(r, axis=-2)
        out[i] = rotate(r, -ang[i])
    return out


def major_axis_flip(imgs, pa_deg):
    """Flip each object about its OWN major axis. Diagnostic 3's isolating operator.

    Rotate the major axis horizontal, flip vertically, rotate back. This leaves the position
    angle, the ellipticity and the whole elliptical envelope unchanged, and inverts chirality,
    so the difference it produces contains only what the model sees beyond the ellipse.

    The object's major axis sits at array angle pa + ARRAY_TO_CATALOG_OFFSET_DEG, and
    rotate(+phi) decreases the array angle by phi, so rotating by that angle brings the major
    axis to array angle 0, where a flip of the rows leaves it fixed.
    """
    return _per_object(imgs, np.asarray(pa_deg, float) + ARRAY_TO_CATALOG_OFFSET_DEG, True)


def axis_sandwich(imgs, pa_deg):
    """The same two rotations as major_axis_flip with the flip omitted.

    Diagnostic 3's matched-interpolation control. Rotation resamples and the resampling
    damages fine structure more than smooth structure, so an uncontrolled comparison of
    featured against smooth objects would attribute a resampling difference to chirality.
    This operator carries exactly the same resampling and inverts nothing.
    """
    return _per_object(imgs, np.asarray(pa_deg, float) + ARRAY_TO_CATALOG_OFFSET_DEG, False)


def adaptive_moments(im, sigma0=4.0, n_iter=40, tol=1e-4):
    """Iterated Gaussian-weighted second moments of one 2D plane.

    The weight is re-derived from the current moment matrix each pass, so the window adapts
    to the source instead of imposing a fixed aperture. Plain unweighted moments over a
    cutout are dominated by sky and neighbours and do not recover the catalog angle.

    Returns (theta_deg in [0,180), ellipticity, converged).
    """
    im = np.clip(np.asarray(im, float), 0, None)
    ny, nx = im.shape
    y, x = np.mgrid[0:ny, 0:nx]
    xc, yc = (nx - 1) / 2.0, (ny - 1) / 2.0
    mxx = myy = float(sigma0) ** 2
    mxy = 0.0
    for _ in range(n_iter):
        det = mxx * myy - mxy ** 2
        if not np.isfinite(det) or det <= 1e-6:
            return np.nan, np.nan, False
        dx, dy = x - xc, y - yc
        q = (myy * dx ** 2 - 2 * mxy * dx * dy + mxx * dy ** 2) / det
        f = im * np.exp(-0.5 * q)
        tot = f.sum()
        if tot <= 0:
            return np.nan, np.nan, False
        nxc, nyc = (f * x).sum() / tot, (f * y).sum() / tot
        ddx, ddy = x - nxc, y - nyc
        # the factor 2 undoes the shrinkage a Gaussian weight applies to second moments
        nmxx, nmyy = 2 * (f * ddx ** 2).sum() / tot, 2 * (f * ddy ** 2).sum() / tot
        nmxy = 2 * (f * ddx * ddy).sum() / tot
        moved = max(abs(nxc - xc), abs(nyc - yc),
                    abs(nmxx - mxx), abs(nmyy - myy), abs(nmxy - mxy))
        xc, yc, mxx, myy, mxy = nxc, nyc, nmxx, nmyy, nmxy
        if moved < tol:
            break
    else:
        return np.nan, np.nan, False
    theta = float(np.degrees(0.5 * np.arctan2(2 * mxy, mxx - myy)) % 180.0)
    tr = mxx + myy
    return theta, float(np.hypot(mxx - myy, 2 * mxy) / tr) if tr > 0 else np.nan, True


def array_angle(im, **kw):
    """Major-axis angle of one plane in the array frame, degrees in [0, 180), or NaN."""
    return adaptive_moments(im, **kw)[0]


def axial_concentration(deg):
    """Mean resultant length and circular mean of axial angles, via the doubled angle.

    R is 1 when the angles are identical and 0 when they are uniform, so it is the statistic
    that decides handedness: the correct pairing concentrates and the wrong one does not.
    The mean is returned on [0, 180) because an offset near 90 sits exactly on the boundary
    of the signed scale and would otherwise report as -90 or +90 at random.
    """
    d = np.asarray(deg, float)
    d = d[np.isfinite(d)]
    z = np.exp(2j * np.radians(d)).mean()
    return float(np.abs(z)), float(np.degrees(np.angle(z)) / 2.0 % 180.0)
