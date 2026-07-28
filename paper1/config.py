"""Single source of truth for paths, seeds and constants. Every diagnostic imports from here."""
import sys

sys.dont_write_bytecode = True

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER1 = ROOT / "paper1"
DATA = ROOT / "data"
RESULTS = PAPER1 / "results"
FIGURES = PAPER1 / "figures"
CACHE = PAPER1 / "cache"

# Frozen encoder. Never fine-tuned. The token budget is a documented deviation from
# the 256-token pretraining budget and is a property of this recipe, not of AION-1.
MODEL = DATA / "models" / "aion-large"
ENCODER_TOKENS = 600
POOLING = "mean"
ENCODE_BATCH = 32

# Substrates. E_IMG is the headline substrate for every diagnostic.
# E_FULL is image + scalar g/r/z flux + catalog redshift, used ONLY for the leakage ablation.
E_IMG = DATA / "E_img.npy"
E_FULL = DATA / "E_full.npy"
E_IMG_BASE = DATA / "E_img_base.npy"
OK_INDEX = DATA / "ok_index.npy"
IMAGES = DATA / "images.npy"

# Label and covariate tables, all keyed to the anchor row order via OK_INDEX.
SAMPLE = DATA / "sample.parquet"
SHAPES = DATA / "anchorShapes.parquet"
COVARIATES = DATA / "anchorCovariates.parquet"
MASS_ZOU = DATA / "anchorMassZou.parquet"
MASS_COLOUR = DATA / "anchorMass.parquet"
AGE_METAL = DATA / "anchorAgeMetal.parquet"
FASTSPEC = DATA / "anchorFastspec.parquet"
WISE = DATA / "anchorWise.parquet"
SPECTRA_MANIFEST = DATA / "anchorSparclManifest.parquet"

# Verified reusable encodes from the pre-harness runs: 8 files, each (15893, 1024) float32,
# rows = the elongated population in ascending anchor order. Row order independently
# re-verified 2026-07-25. Reused rather than re-encoded; see lib/encode.py.
LEGACY_ROT_CKPT = ROOT / "results" / "trackA_causal_ckpt"
LEGACY_FLIP_CKPT = ROOT / "results" / "trackA_flip_ckpt"

# Analysis constants. Fixed once so no diagnostic can silently choose its own.
SEED = 0
TEST_SIZE = 0.2
ALPHAS = np.logspace(-2, 4, 7)
ALPHA_DIRECTION = 100.0
N_BOOT = 1000
N_BOOT_ANGLE = 200
ELLIP_CUT = 0.3

# Label validity. Some catalog columns encode "not measured" as a finite sentinel, which
# np.isfinite does not catch. A label is valid only if it is finite and above its sentinel.
SENTINEL_MIN = {"total_ssfr_median": -90.0}

# The same trap in the observing-condition columns. The northern surveys (BASS and MzLS)
# observe g, r and z but not i, so psfsize_i and psfdepth_i are -9999 for every northern
# galaxy. That sentinel is therefore perfectly collinear with which telescope took the image,
# and any regression that treats it as a number is partialling on the hemisphere.
COVARIATE_SENTINEL_MIN = -9990.0
COVARIATES_ALL = ("ebv", "psfsize_g", "psfsize_r", "psfsize_i", "psfsize_z",
                  "psfdepth_g", "psfdepth_r", "psfdepth_i", "psfdepth_z")

# Declared provenance of each label column. The counts are measured; these attributions are
# declared here rather than read from the files, which carry no provenance metadata.
LABEL_SOURCES = {
    "redshift": ("GZ DESI external catalog, photometric-dominated", "sample.parquet"),
    "spec_z": ("GZ DESI external catalog, spectroscopic subset", "sample.parquet"),
    "photo_z": ("GZ DESI external catalog, photometric", "sample.parquet"),
    "mag_g_desi": ("Legacy Surveys DR8 catalog", "sample.parquet"),
    "mag_r_desi": ("Legacy Surveys DR8 catalog", "sample.parquet"),
    "mag_z_desi": ("Legacy Surveys DR8 catalog", "sample.parquet"),
    "smooth-or-featured_smooth_fraction": ("Galaxy Zoo DESI vote fractions", "sample.parquet"),
    "smooth-or-featured_featured-or-disk_fraction": ("Galaxy Zoo DESI vote fractions", "sample.parquet"),
    "disk-edge-on_yes_fraction": ("Galaxy Zoo DESI vote fractions", "sample.parquet"),
    "elpetro_mass_log": ("NSA elpetro crossmatch", "sample.parquet"),
    "total_ssfr_median": ("NSA crossmatch", "sample.parquet"),
    "sersic_n": ("NSA crossmatch", "sample.parquet"),
    "paDeg": ("derived from DR10 tractor shape_e1, shape_e2", "anchorShapes.parquet"),
    "ellip": ("derived from DR10 tractor shape_e1, shape_e2", "anchorShapes.parquet"),
    "shape_r": ("DR10 tractor half-light radius", "anchorShapes.parquet"),
    "psfsize_r": ("DR10 tractor observing conditions", "anchorCovariates.parquet"),
    "psfdepth_r": ("DR10 tractor observing conditions", "anchorCovariates.parquet"),
    "ebv": ("DR10 tractor, SFD extinction", "anchorCovariates.parquet"),
    "footprint": ("derived from Legacy Surveys release code", "anchorCovariates.parquet"),
}

# Instrument constants.
PIXEL_SCALE = 0.262
CUTOUT_PIX = 96
ZEROPOINT = 22.5
BANDS = ("g", "r", "i", "z")
