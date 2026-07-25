"""Deterministic operations on input images. These are what make a diagnostic causal.

Each operator has an exactly known consequence, verified in tests/testTransforms.py BEFORE
any of them is run at scale. Sign and frame conventions are the failure mode here: the
array frame, the catalog angle convention and the library rotation convention must be
pinned down and tested, not assumed.

Contract (to implement):
  rotate(imgs, phi)          bilinear, reshape preserved. Interpolating: costs accuracy.
  mirror(imgs)               exact column permutation. No resampling, no interpolation cost.
  major_axis_flip(imgs, pa)  rotate the major axis horizontal, flip vertically, rotate back.
                             Leaves position angle, ellipticity and the elliptical envelope
                             unchanged; inverts chirality. Diagnostic 3's isolating operator.
  blur(imgs, fwhm_target, fwhm_current)   Moffat convolution to worse seeing
  add_noise(imgs, depth_target)           noise to a target point-source depth
  zero_band(imgs, band)                   band ablation and synthetic band-blanking
  scale_flux(imgs, dmag)                  zeropoint offset
  redden(imgs, ebv, rv)                   per-band multiplicative from an extinction curve
  pedestal(imgs, level)                   additive constant on all pixels
  ferengi(imgs, z_from, z_to, ...)        angular size, flux dimming, k-correction,
                                          PSF reconvolution, noise to target depth

Only rotate and mirror exist anywhere in the project today. Everything below major_axis_flip
is new work. The Phase 2 realism code that implements blur and noise lives on the Vera
cluster and is not on this machine, so it cannot be lifted.
"""
