"""Single source of truth for paths, seeds and constants. Every diagnostic imports from here."""
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

# Instrument constants.
PIXEL_SCALE = 0.262
CUTOUT_PIX = 96
ZEROPOINT = 22.5
BANDS = ("g", "r", "i", "z")
