"""
Sample B: volume-weighted-as-observed draw (importance-sampled) (preRegistration §12; plan §5.4).

Preserves ASTRID's NATIVE demographics (subject to the GZ DESI r<19 selection), so P1 (quiescent
deficit) and P3 (GSMF deficit) live here. A naive volume draw collapses after selection (high-z
admits only M*>~11, ASTRID's depleted tail), so we importance-sample: oversample the low-pass-rate
high-z shells and attach importance weights w = V_shell x (|selected|/n_drawn) so weighted
statistics recover the volume-weighted-as-observed population.

Selection probability is estimated CHEAPLY pre-rendering (plan §5.4): apparent r = catalog
intrinsic desR (at snapshot z) shifted to the assigned z by distance modulus + a constant
dust+MW dimming, cut at r<19. This is a provisional p_sel; the exact per-galaxy dust selection
+ refill loop is S3. One-entry dedup via phase-space chainRoot. Volume-weighted continuous-z
within each shell.

Deliverables: data/sampleB.parquet (+ weights), results/sampleB.json with ESS per stratum and
the headline P1 measurement (weighted quiescent fraction vs M*, M*>1e10 per the Q1 resolution
caveat).
"""
import json
import numpy as np
import pandas as pd

CAT = "vera/data/catalogs/catalog{}.parquet"
EXT = "vera/data/apertureExtra{}.parquet"
CHAINS = "data/dedupChainsPhaseSpace.parquet"
H0, OM, OL, C = 67.74, 0.3089, 0.6911, 299792.458
NSAMPLE = 50000
SEED = 0
DUST_MW_DIM = 0.12          # cheap constant r-band dimming (internal ~0.05 + MW ~0.07); S3 refines
RCUT = 19.0
SSFR_Q = 1e-11              # quiescent threshold (1/yr)
# shell: (renderSnap, catalogSnap, z_lo, z_hi) -- nearest-snapshot partition of [0.075,0.55]
SHELLS = [("771", "771", 0.075, 0.15), ("743", "743", 0.15, 0.25),
          ("743r", "743", 0.25, 0.35), ("692", "692", 0.35, 0.45),
          ("660", "660", 0.45, 0.55)]


def comov(z):
    zg = np.linspace(0, np.max(z) + 1e-3, 4000)
    Ez = np.sqrt(OM * (1 + zg) ** 3 + OL)
    dc = (C / H0) * np.concatenate([[0], np.cumsum(0.5 * (1 / Ez[1:] + 1 / Ez[:-1]) * np.diff(zg))])
    return np.interp(z, zg, dc)


def distmod(z):
    return 5 * np.log10(np.clip((1 + z) * comov(z), 1e-6, None) * 1e5)


def assignZ(zlo, zhi, n, rng):
    """continuous z in [zlo,zhi] sampled proportional to comoving shell volume dV/dz."""
    zg = np.linspace(zlo, zhi, 500)
    Ez = np.sqrt(OM * (1 + zg) ** 3 + OL)
    w = comov(zg) ** 2 / Ez
    cdf = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(zg))])
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, zg)


def loadShellPool(csnap):
    e = pd.read_parquet(EXT.format(csnap), columns=["index", "mStarAper2", "pathology"])
    c = pd.read_parquet(CAT.format(csnap),
                        columns=["index", "idMostbound", "isCentral", "desR", "sfrNew"])
    ch = pd.read_parquet(CHAINS)
    ch = ch[ch.snap == csnap][["index", "chainRoot"]]
    e = e[(e.pathology == "") & np.isfinite(e.mStarAper2)]
    df = e.merge(c, on="index").merge(ch, on="index", how="left")
    df["logM"] = np.log10(df.mStarAper2)
    df["ssfr"] = df.sfrNew / df.mStarAper2
    return df.reset_index(drop=True)


def main():
    rng = np.random.default_rng(SEED)
    zsnap = {"771": 0.1, "743": 0.2, "692": 0.4, "660": 0.49677}
    pools = {cs: loadShellPool(cs) for cs in ["771", "743", "692", "660"]}

    # 1) per shell: assign z, cheap selection, build selected pool
    shellData = []
    for rsnap, csnap, zlo, zhi in SHELLS:
        p = pools[csnap].copy()
        zass = assignZ(zlo, zhi, len(p), rng)
        rObs = p.desR.values + (distmod(zass) - distmod(zsnap[csnap])) + DUST_MW_DIM
        sel = rObs < RCUT
        vShell = comov(np.array([zhi]))[0] ** 3 - comov(np.array([zlo]))[0] ** 3
        s = p[sel].copy()
        s["zObs"], s["rObs"], s["renderSnap"] = zass[sel], rObs[sel], rsnap
        shellData.append(dict(rsnap=rsnap, csnap=csnap, zlo=zlo, zhi=zhi,
                              vShell=float(vShell), nPool=len(p), nSel=int(sel.sum()), sel=s))
        print(f"shell {rsnap} z[{zlo},{zhi}): pool={len(p)} selected(r<{RCUT})={int(sel.sum())} "
              f"({100*sel.mean():.1f}%)  Vshell~{vShell/1e6:.1f}e6")

    # 2) importance draw: take all selected at scarce (high-z) shells, subsample abundant ones
    #    to ~NSAMPLE total; weight w = vShell * (nSel/nDrawn) so weighted stats are volume-weighted
    cap = NSAMPLE  # refined below
    for _ in range(40):
        tot = sum(min(sh["nSel"], cap) for sh in shellData)
        if abs(tot - NSAMPLE) < 200 or cap > max(sh["nSel"] for sh in shellData):
            break
        cap = int(cap * NSAMPLE / max(tot, 1))

    used, rows = set(), []
    for sh in sorted(shellData, key=lambda x: -x["zlo"]):     # scarce high-z first (one-entry)
        s = sh["sel"]
        s = s[~s.chainRoot.isin(used)]
        nDraw = min(len(s), cap)
        pick = s.iloc[rng.choice(len(s), size=nDraw, replace=False)] if nDraw else s.iloc[:0]
        used.update(pick.chainRoot.tolist())
        wt = sh["vShell"] * (sh["nSel"] / max(nDraw, 1))
        for _, g in pick.iterrows():
            rows.append((sh["rsnap"], sh["csnap"], sh["zlo"], float(g.zObs), int(g["index"]),
                         int(g.idMostbound), g.chainRoot, float(g.logM), float(g.ssfr),
                         bool(g.isCentral), float(g.rObs), wt))

    sb = pd.DataFrame(rows, columns=["renderSnap", "catalogSnap", "shellZlo", "zObs", "index",
                                     "idMostbound", "chainRoot", "logM", "ssfr", "isCentral",
                                     "rObs", "weight"])
    sb["weight"] /= sb.weight.sum() / len(sb)                 # normalise to mean 1 (relative)
    sb.to_parquet("data/sampleB.parquet")

    # 3) ESS per (renderSnap) stratum + headline P1: weighted quiescent fraction vs M* (M*>1e10)
    def ess(w):
        return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0
    essByShell = {sh: ess(sb.weight[sb.renderSnap == sh].values) for sh in sb.renderSnap.unique()}
    p1 = []
    for lo in np.arange(10.0, 11.6, 0.2):
        m = (sb.logM >= lo) & (sb.logM < lo + 0.2)
        if m.sum() >= 20:
            w = sb.weight[m].values
            q = (sb.ssfr[m].values < SSFR_Q).astype(float)
            p1.append(dict(logMlo=round(float(lo), 1), nGal=int(m.sum()), ess=round(ess(w), 0),
                           fQuiescent_weighted=round(float((w * q).sum() / w.sum()), 3),
                           fQuiescent_raw=round(float(q.mean()), 3)))

    summary = dict(
        nDrawn=int(len(sb)), nTarget=NSAMPLE, cap=int(cap), chainReuse=int(sb.chainRoot.duplicated().sum()),
        essByRenderSnap={k: round(v, 0) for k, v in essByShell.items()},
        essTotal=round(ess(sb.weight.values), 0),
        shells=[{k: sh[k] for k in ("rsnap", "zlo", "zhi", "nPool", "nSel", "vShell")} for sh in shellData],
        P1_quiescentFractionVsMass=p1,
        selectionModel=f"cheap: intrinsic desR + distmod(z_assigned-z_snap) + {DUST_MW_DIM} dust/MW, r<{RCUT}; S3 refines",
        massConvention="aperture mStarAper2 (h=0.6774); ssfr=sfrNew/mStarAper2 (newSFa SFR)",
        note="volume-weighted-as-observed; P1 restricted to M*>1e10 (Q1 resolution caveat); "
             "weighted stats use 'weight'. Pre-selection candidate; r<19 refill loop = S3.",
    )
    with open("results/sampleB.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSample B: {len(sb)} drawn (cap={cap}), chainReuse={int(sb.chainRoot.duplicated().sum())}, "
          f"ESS_total={ess(sb.weight.values):.0f}")
    print("ESS by shell:", {k: round(v) for k, v in essByShell.items()})
    print("\nP1 -- weighted quiescent fraction vs M* (ASTRID, volume-weighted-as-observed):")
    print(f"{'logM':>6}{'nGal':>7}{'ESS':>7}{'fQ_wtd':>8}{'fQ_raw':>8}")
    for r in p1:
        print(f"{r['logMlo']:>6}{r['nGal']:>7}{int(r['ess']):>7}{r['fQuiescent_weighted']:>8}{r['fQuiescent_raw']:>8}")
    print("wrote data/sampleB.parquet + results/sampleB.json")


if __name__ == "__main__":
    main()
