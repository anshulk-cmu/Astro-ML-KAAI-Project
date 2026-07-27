"""Diagnostic 3 - Chirality. Tier: input-space causal.

Does the model encode parity-odd structure, spiral arm handedness, or has it discarded a real
physical observable?

The naive test fails and it is worth stating why. Comparing a pool of originals against a pool
of mirror images measures nothing, because handedness is roughly even and the two pools have
identical distributions, so the test must be paired: each object against its own reflection.
And a plain mirror is dominated by orientation, since Diagnostic 2 showed the readout tracks
position angle almost exactly under reflection.

The isolating operation is a flip about the object's OWN major axis. It leaves the position
angle, the ellipticity and the whole elliptical envelope unchanged and inverts chirality, so
the difference it produces contains only what the model sees beyond the ellipse.

    d_i = E(x_i) - E(majorflip(x_i))

Two additions to the scoping document, both forced by Diagnostic 2's findings and labelled
EXTENSION where they appear in the artifact:

  1. A matched-interpolation control. The flip needs two off-axis rotations, and Diagnostic 2
     measured that off-axis rotation costs the morphology readout about 0.027 in R2 while an
     interpolation-free operation costs nothing. Resampling damages fine structure more than
     smooth structure, so featured objects lose more than smooth ones from the resampling
     alone. Using smooth objects as the noise floor therefore does not control the comparison,
     and is biased in the direction that manufactures a positive result. `axis_sandwich`
     applies the same two rotations without the flip, giving a per-object control.
  2. An orientation-leak control. `tests/testMajorAxisFlip.py` shows the flip preserves the
     envelope axis exactly but reflects a chiral object's moment angle about it, so for chiral
     objects and only for them the apparent orientation moves. Whether the model's readout
     follows the envelope or the moments is measured here with the frozen Diagnostic 1 probe
     rather than assumed.

    python paper1/diagnostics/d3Chirality.py

Writes paper1/results/d3Chirality.json with a provenance block. Every number this script
emits lands in that file; nothing is quoted into the report that did not come from here.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import numpy as np
from scipy.stats import binomtest
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd

import config as C
import data as D
import encode as E
import probes as P
import provenance
import transforms as T
from circular import fit_evaluate

NAME = "d3Chirality"
POPULATION = "pa_defined"
N_RANDOM_DIR = 20

TAGS = {"original": ("d3Original", "identity"),
        "majorflip": ("d3MajorFlip", "major_axis_flip"),
        "sandwich": ("d3Sandwich", "axis_sandwich")}


def population(df):
    """Every galaxy the operator is defined for: it needs a fitted major axis."""
    return np.where(D.pa_defined(df))[0]


def params_for(operator, n):
    return {"operator": operator, "population": POPULATION, "n": int(n)}


def ensure_encodes(device="cuda"):
    """Encode the three conditions over the full population, resuming if interrupted.

    Kept separate from the analysis so the expensive step runs once and the analysis can be
    iterated against the cache afterwards.
    """
    df = D.anchor()
    idx = population(df)
    pa = df["paDeg"].to_numpy(float)[idx]
    rows = np.load(C.OK_INDEX)[idx]
    imgs = np.load(C.IMAGES, mmap_mode="r")
    n = len(idx)
    print(f"population {n} galaxies with a defined position angle", flush=True)

    ops = {"original": None, "majorflip": T.major_axis_flip, "sandwich": T.axis_sandwich}
    out = {}
    for key, (tag, operator) in TAGS.items():
        t0 = time.time()
        fn = ops[key]
        source = E.BatchTransform(imgs, rows, fn, None if fn is None else pa)
        out[key] = E.encode(source, tag, params_for(operator, n), device=device)
        print(f"{tag}: {out[key].shape} in {time.time() - t0:.0f}s", flush=True)
    return out


def pools(df, idx):
    """Named populations. Chiral structure is visible only in resolved face-on discs, so the
    pools are morphology-defined and each carries its own count."""
    g = lambda c: df[c].to_numpy(float)[idx]
    sm = g("smooth-or-featured_smooth_fraction")
    fe = g("smooth-or-featured_featured-or-disk_fraction")
    sp = g("has-spiral-arms_yes_fraction")
    eo = g("disk-edge-on_yes_fraction")
    return {
        "spiral_armed": (np.isfinite(sp) & (sp > 0.5),
                         "has-spiral-arms vote fraction above 0.5; the chiral pool"),
        "featured": (np.isfinite(fe) & (fe > 0.7),
                     "featured-or-disk vote fraction above 0.7"),
        "smooth": (np.isfinite(sm) & (sm > 0.7),
                   "smooth vote fraction above 0.7; achiral, the population control"),
        "edge_on": (np.isfinite(eo) & (eo > 0.5),
                    "edge-on vote fraction above 0.5; discs whose handedness is not resolvable"),
        "all": (np.ones(len(idx), bool), "every galaxy with a defined position angle")}


def norm_stats(d, mask, seed=C.SEED):
    n = np.linalg.norm(d[mask], axis=1)
    return {"n": int(mask.sum()), "median_norm": float(np.median(n)),
            "median_norm_ci": P.boot_stat(n, np.median, seed=seed),
            "mean_norm": float(n.mean()), "p10_p90": [float(np.percentile(n, 10)),
                                                      float(np.percentile(n, 90))]}


def axis_structure(d, mask, n_random=N_RANDOM_DIR, seed=C.SEED):
    """Does the difference lie along a single direction?

    The singular value decomposition is taken WITHOUT centering: an encoded handedness gives
    d_i = s_i c with s_i = +/-1, whose mean is zero but whose leading singular direction is c.
    Centering would be harmless there and destructive if the mean were not zero, so the mean
    is reported separately instead of being removed.
    """
    X = d[mask]
    if len(X) < 50:
        return {"n": int(len(X)), "reported": False, "reason": "too few objects"}
    total = float((X ** 2).sum())
    U, S, Vt = randomized_svd(X, n_components=10, random_state=seed)
    frac = (S ** 2) / total
    c = Vt[0]
    proj = X @ c
    along = np.abs(proj) / (np.linalg.norm(X, axis=1) + 1e-12)
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n_random, X.shape[1]))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    rand_frac = [float(((X @ v) ** 2).sum() / total) for v in V]
    return {"n": int(len(X)), "reported": True,
            "leading_variance_fraction": float(frac[0]),
            "variance_fraction_top10": [float(v) for v in frac],
            "mean_vector_norm_over_median_norm":
                float(np.linalg.norm(X.mean(0)) / (np.median(np.linalg.norm(X, axis=1)) + 1e-12)),
            "median_fraction_along_leading_axis": float(np.median(along)),
            "sign_balance_positive": float((proj > 0).mean()),
            "projection_bimodality_frac_above_half": float((along > 0.5).mean()),
            "NULL_random_direction_variance_fraction": {
                "median": float(np.median(rand_frac)), "ci": P.ci(rand_frac),
                "analytic_expectation": 1.0 / X.shape[1],
                "note": "a random direction captures 1/d of the variance in expectation"},
            "reading": (
                "a dominant leading axis is NOT on its own evidence of encoded handedness. A "
                "constant offset shared by every object, which is what a systematic resampling "
                "residual looks like, produces an equally dominant axis. Verified on synthetic "
                "data: a true handedness axis gives sign_balance near 0.5 and "
                "mean_vector_norm_over_median_norm near 0, while a common offset gives sign "
                "balance 1.0 and a ratio near 1. The three fields must be read together"),
            "leading_axis": c.astype(float).tolist()}


def calibrate_axis_statistic(dim=1024, n=2000, seed=C.SEED):
    """What the single-axis statistic returns in three cases with known answers.

    Run on every execution rather than quoted from a notebook, because the reading of the real
    result depends entirely on it: a dominant leading axis is produced BOTH by a genuine
    handedness feature and by a constant offset shared by every object, and only the sign
    balance and the mean-to-median ratio tell them apart.
    """
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(dim)
    c /= np.linalg.norm(c)
    noise = rng.standard_normal((n, dim)) / np.sqrt(dim)
    cases = {
        "true_handedness_axis": rng.choice([-1.0, 1.0], n)[:, None] * c[None, :] * 3.0 + noise,
        "constant_offset_no_handedness": np.ones((n, 1)) * c[None, :] * 2.0 + noise,
        "no_structure": rng.standard_normal((n, dim))}
    out = {}
    for name, X in cases.items():
        a = axis_structure(X.astype(np.float32), np.ones(n, bool), seed=seed)
        out[name] = {k: a[k] for k in ("leading_variance_fraction", "sign_balance_positive",
                                       "mean_vector_norm_over_median_norm")}
        out[name]["random_direction_fraction"] = \
            a["NULL_random_direction_variance_fraction"]["median"]
    out["reading"] = ("the first two cases both give a dominant leading axis; they are "
                      "separated by the sign balance and the mean-to-median ratio, not by the "
                      "leading fraction")
    return out


def main():
    t0 = time.time()
    df = D.anchor()
    idx = population(df)
    n = len(idx)
    pa_deg = df["paDeg"].to_numpy(float)[idx]
    ellip = df["ellip"].to_numpy(float)[idx]

    enc = {}
    for key, (tag, operator) in TAGS.items():
        hit = E.cached(tag, params_for(operator, n))
        if hit is None:
            raise RuntimeError(f"encode {tag} is missing or incomplete; run ensure_encodes()")
        enc[key] = hit

    ei = np.load(C.E_IMG)
    mu, sd = ei.mean(0), ei.std(0) + 1e-8
    z = {k: ((v - mu) / sd).astype(np.float32) for k, v in enc.items()}

    # Three differences. The first is the scoping document's; the second removes the
    # resampling that the first carries; the third is that resampling on its own.
    d_spec = z["original"] - z["majorflip"]
    d_pure = z["sandwich"] - z["majorflip"]
    d_resamp = z["original"] - z["sandwich"]

    pl = pools(df, idx)
    out = {"population": {
        "n": int(n), "definition": "every anchor galaxy with a defined position angle",
        "substrate": "E_img (image tokens only)",
        "pools": {k: {"n": int(m.sum()), "definition": why} for k, (m, why) in pl.items()},
        "note": ("the operator needs a fitted major axis, so the population is pa_defined "
                 "rather than the elongated cut of Diagnostics 1 and 2")}}

    out["operator"] = {
        "isolating": "major_axis_flip: rotate the major axis horizontal, flip, rotate back",
        "control": "axis_sandwich: the same two rotations with the flip omitted",
        "pinned_by": "paper1/tests/testMajorAxisFlip.py",
        "verified_properties": [
            "turns a synthetic spiral into its independently built mirror twin",
            "leaves an achiral ellipse unchanged beyond the two rotations",
            "preserves the envelope axis and the ellipticity",
            "is an involution when compared against the control applied twice",
            "carries the same interpolation loss as the control"]}

    kinds = {"d_spec": (d_spec, "E(x) - E(majorflip(x)), the scoping document's definition"),
             "EXTENSION_d_pure": (d_pure, "E(sandwich(x)) - E(majorflip(x)); both sides carry "
                                          "identical resampling, so this is chirality alone"),
             "NULL_d_resampling": (d_resamp, "E(x) - E(sandwich(x)); the same procedure with a "
                                             "rotation instead of a flip, so it carries the "
                                             "resampling and no parity inversion")}
    mags = {}
    for name, (d, why) in kinds.items():
        mags[name] = {"definition": why,
                      "by_pool": {k: norm_stats(d, m) for k, (m, _) in pl.items()}}
    out["difference_magnitudes"] = mags

    # Does the resampling explain the pool difference the scoping document would attribute to
    # chirality? Diagnostic 2 predicts it partly does, because resampling costs fine structure
    # more than smooth structure.
    def med(d, key):
        return float(np.median(np.linalg.norm(d[pl[key][0]], axis=1)))

    out["EXTENSION_confound_budget"] = {
        "question": ("the scoping document compares spiral against smooth magnitudes and calls "
                     "the smooth value the interpolation floor. That floor is not shared: "
                     "resampling damages fine structure more, so the featured pool loses more "
                     "from the rotations alone"),
        "spiral_over_smooth_ratio": {
            "d_spec": med(d_spec, "spiral_armed") / med(d_spec, "smooth"),
            "d_pure": med(d_pure, "spiral_armed") / med(d_pure, "smooth"),
            "d_resampling": med(d_resamp, "spiral_armed") / med(d_resamp, "smooth")},
        "reading": ("a ratio above 1 in the resampling row is the confound measured directly; "
                    "the d_pure row is the same comparison with that confound removed")}

    out["axis_statistic_calibration"] = calibrate_axis_statistic()
    out["single_axis"] = {k: axis_structure(d_pure, m) for k, (m, _) in pl.items()}
    out["single_axis_NULL_resampling"] = {k: axis_structure(d_resamp, pl[k][0])
                                          for k in ("spiral_armed", "smooth")}

    # Is the chirality axis the same direction as the resampling axis? It must not be.
    a_ch = out["single_axis"]["spiral_armed"]
    a_rs = out["single_axis_NULL_resampling"]["spiral_armed"]
    if a_ch.get("reported") and a_rs.get("reported"):
        c1 = np.array(a_ch["leading_axis"])
        c2 = np.array(a_rs["leading_axis"])
        out["EXTENSION_axis_separation"] = {
            "cosine_between_chirality_and_resampling_axes":
                float(abs(c1 @ c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))),
            "expectation": ("near zero if the flip and the resampling write to different "
                            "directions; near one would mean the chirality axis is resampling")}

    # Orientation leak. The flip preserves the envelope axis but reflects a chiral object's
    # moment angle about it, so a moment-like readout would move for chiral objects only.
    Zi, _ = D.substrate("img")
    pa_rad = np.radians(df["paDeg"].to_numpy(float))
    elong_full = (D.pa_defined(df) & np.isfinite(df["ellip"].to_numpy(float))
                  & (df["ellip"].to_numpy(float) > C.ELLIP_CUT))
    pr = fit_evaluate(Zi, pa_rad, 2, elong_full)[0]

    def read(X):
        pc, ps = pr.predict(X)
        return np.degrees(0.5 * np.arctan2(ps, pc))

    r_flip, r_ctrl = read(z["majorflip"]), read(z["sandwich"])
    shift = T.wrap_axial(r_flip - r_ctrl)
    out["EXTENSION_orientation_leak"] = {
        "question": ("the operator preserves the envelope axis exactly but reflects a chiral "
                     "object's moment angle about it. If the model's angle readout follows the "
                     "moments rather than the envelope, orientation leaks into the difference "
                     "vector for chiral objects and only for them"),
        "readout_shift_deg": {k: {"n": int(m.sum()),
                                  "median_abs": float(np.median(np.abs(shift[m]))),
                                  "median_abs_ci": P.boot_stat(np.abs(shift[m]), np.median),
                                  "median_signed": float(np.median(shift[m]))}
                              for k, (m, _) in pl.items()},
        "probe": "the Diagnostic 1 doubled-angle probe, fit once on untransformed embeddings",
        "reading": ("a shift near zero everywhere means the readout follows the envelope and "
                    "no orientation leaks; a shift larger in the chiral pools than in the "
                    "smooth pool is the leak, and bounds how much of d is orientation")}

    # The operator's precision depends on the catalog angle, which degrades for round objects.
    bins = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
    grade = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = pl["spiral_armed"][0] & (ellip >= lo) & (ellip < hi)
        if m.sum() < 50:
            continue
        grade.append({"ellip_lo": lo, "ellip_hi": hi, "n": int(m.sum()),
                      "median_norm_d_pure": float(np.median(np.linalg.norm(d_pure[m], axis=1))),
                      "median_abs_readout_shift_deg": float(np.median(np.abs(shift[m])))})
    out["EXTENSION_ellipticity_grading"] = {
        "bins": grade,
        "note": ("the flip is about the CATALOG major axis, whose uncertainty grows as the "
                 "object becomes round, so the operator isolates chirality less cleanly at low "
                 "ellipticity")}

    # The pools differ in more than handedness: spiral-armed galaxies are larger, brighter and
    # more elongated than smooth ones, and all three change how far a resampling moves the
    # embedding. A raw pool comparison cannot separate those from chirality, so each spiral is
    # matched to a smooth galaxy with the same ellipticity, brightness and angular size and the
    # comparison is made pair by pair.
    feat = np.column_stack([ellip,
                            df["mag_r_desi"].to_numpy(float)[idx],
                            np.log10(df["shape_r"].to_numpy(float)[idx] + 1e-3)])
    good = np.isfinite(feat).all(1)
    fs = (feat - np.nanmean(feat[good], 0)) / (np.nanstd(feat[good], 0) + 1e-12)
    src = np.where(pl["spiral_armed"][0] & good)[0]
    pool = np.where(pl["smooth"][0] & good)[0]
    nn = NearestNeighbors(n_neighbors=1).fit(fs[pool])
    taken, pairs = set(), []
    order = np.argsort(-np.linalg.norm(fs[src], axis=1))
    for j in src[order]:
        dist, who = nn.kneighbors(fs[j][None], n_neighbors=min(40, len(pool)))
        for cand in pool[who[0]]:
            if cand not in taken:
                taken.add(cand)
                pairs.append((j, cand))
                break
    pa_i = np.array([p[0] for p in pairs])
    pb_i = np.array([p[1] for p in pairs])
    nrm = lambda d, s: np.linalg.norm(d[s], axis=1)
    delta = nrm(d_pure, pa_i) - nrm(d_pure, pb_i)
    delta_rs = nrm(d_resamp, pa_i) - nrm(d_resamp, pb_i)
    out["EXTENSION_matched_control"] = {
        "n_pairs": int(len(pairs)),
        "matched_on": ["catalog ellipticity", "r magnitude", "log angular size shape_r"],
        "balance_after_matching": {
            k: {"spiral": float(np.median(feat[pa_i, i])),
                "smooth": float(np.median(feat[pb_i, i]))}
            for i, k in enumerate(["ellip", "mag_r", "log_shape_r"])},
        "paired_delta_norm_d_pure": {
            "median": float(np.median(delta)), "ci": P.boot_stat(delta, np.median),
            "fraction_positive": float((delta > 0).mean()),
            "relative_to_smooth_median": float(np.median(delta) / np.median(nrm(d_pure, pb_i)))},
        "paired_delta_norm_resampling_null": {
            "median": float(np.median(delta_rs)), "ci": P.boot_stat(delta_rs, np.median),
            "fraction_positive": float((delta_rs > 0).mean())},
        "sign_test": {
            "n_positive": int((delta > 0).sum()), "n_pairs": int(len(delta)),
            "p_value": float(binomtest(int((delta > 0).sum()), len(delta), 0.5).pvalue),
            "null": "half the pairs positive if the flip moved both members equally"},
        "sign_test_resampling_null": {
            "n_positive": int((delta_rs > 0).sum()),
            "p_value": float(binomtest(int((delta_rs > 0).sum()), len(delta_rs), 0.5).pvalue)},
        "reading": ("the paired difference is the excess displacement of a spiral over an "
                    "otherwise identical smooth galaxy. The resampling row is the same "
                    "comparison for the operation that inverts nothing, on the same pairs, so "
                    "any residual mismatch in size or brightness would move both rows "
                    "together. It is what the chirality row has to beat")}

    # Is the excess about arms specifically, or about being featured at all?
    fs_mask = pl["featured"][0] & ~pl["spiral_armed"][0]
    out["EXTENSION_arms_versus_featured"] = {
        "spiral_armed": norm_stats(d_pure, pl["spiral_armed"][0]),
        "featured_without_recorded_arms": norm_stats(d_pure, fs_mask),
        "edge_on": norm_stats(d_pure, pl["edge_on"][0]),
        "reading": ("edge-on discs carry arms that cannot be resolved as handedness, so the "
                    "physics predicts they behave like the achiral pool despite being discs")}

    # Restricting to a well determined major axis. The flip is about the CATALOG axis, and its
    # uncertainty grows as an object becomes round, so the operator isolates chirality least
    # well exactly where spiral structure is most visible.
    clean = ellip > C.ELLIP_CUT
    out["EXTENSION_well_determined_axis"] = {
        "selection": f"catalog ellipticity above {C.ELLIP_CUT}",
        "magnitudes": {k: {"n": int((m & clean).sum()),
                           "median_norm_d_pure": (float(np.median(np.linalg.norm(
                               d_pure[m & clean], axis=1))) if (m & clean).sum() > 30 else None)}
                       for k, (m, _) in pl.items()},
        "single_axis": {k: axis_structure(d_pure, pl[k][0] & clean)
                        for k in ("spiral_armed", "smooth")},
        "orientation_leak_median_abs_deg": {
            k: (float(np.median(np.abs(shift[m & clean]))) if (m & clean).sum() > 30 else None)
            for k, (m, _) in pl.items()}}

    # How much of the difference vector is orientation rather than structure?
    out["EXTENSION_leak_correlation"] = {
        k: P.spearman(np.abs(shift[m]), np.linalg.norm(d_pure[m], axis=1))
        for k, (m, _) in pl.items()}

    out["consistency_checks"] = {
        "achiral_pool_is_the_floor": {
            "smooth_median_d_pure": med(d_pure, "smooth"),
            "spiral_median_d_pure": med(d_pure, "spiral_armed"),
            "expectation": "an ellipse flipped about its own major axis is itself"},
        "resampling_null_has_no_preferred_axis": {
            "spiral_leading_fraction_d_pure":
                out["single_axis"]["spiral_armed"].get("leading_variance_fraction"),
            "spiral_leading_fraction_resampling":
                out["single_axis_NULL_resampling"]["spiral_armed"].get(
                    "leading_variance_fraction")},
        "sign_balance_near_half": out["single_axis"]["spiral_armed"].get("sign_balance_positive"),
        "encodes": {k: {"tag": TAGS[k][0], "shape": list(v.shape), "dtype": str(v.dtype),
                        "n_nonfinite": int((~np.isfinite(v)).sum())} for k, v in enc.items()}}

    np.savez(C.RESULTS / f"{NAME}Arrays.npz",
             norm_spec=np.linalg.norm(d_spec, axis=1).astype(np.float32),
             norm_pure=np.linalg.norm(d_pure, axis=1).astype(np.float32),
             norm_resamp=np.linalg.norm(d_resamp, axis=1).astype(np.float32),
             projection=(d_pure @ np.array(out["single_axis"]["spiral_armed"]["leading_axis"],
                                           np.float32)).astype(np.float32),
             readout_shift=shift.astype(np.float32),
             ellip=ellip.astype(np.float32),
             **{f"pool_{k}": m for k, (m, _) in pl.items()})

    inputs = [C.E_IMG, C.OK_INDEX, C.SAMPLE, C.SHAPES, C.COVARIATES, C.IMAGES]
    inputs += [C.CACHE / f"{TAGS[k][0]}.npy" for k in TAGS]
    path = provenance.write(NAME, out, inputs, t0)

    print(f"n={n}  pools " + ", ".join(f"{k}={v['n']}" for k, v in out['population']['pools'].items()))
    for name in kinds:
        b = mags[name]["by_pool"]
        print(f"{name:22} spiral {b['spiral_armed']['median_norm']:7.3f}  "
              f"smooth {b['smooth']['median_norm']:7.3f}  "
              f"ratio {b['spiral_armed']['median_norm'] / b['smooth']['median_norm']:.3f}")
    a = out["single_axis"]["spiral_armed"]
    print(f"leading axis fraction spiral {a['leading_variance_fraction']:.4f} vs random "
          f"{a['NULL_random_direction_variance_fraction']['median']:.6f}; "
          f"sign balance {a['sign_balance_positive']:.3f}")
    lk = out["EXTENSION_orientation_leak"]["readout_shift_deg"]
    print(f"orientation leak: spiral {lk['spiral_armed']['median_abs']:.3f} deg, "
          f"smooth {lk['smooth']['median_abs']:.3f} deg")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
