"""
Cross-snapshot dedup chains via SubhaloIDMostbound (preRegistration §11.1 / plan §5.2(4)).

The same physical galaxy recurs across adjacent snapshots; particle IDs are conserved, so
subhalos sharing a most-bound particle ID (idMostbound) are the same object. Sample A/B may
draw at most ONE snapshot instance per linked chain (exchangeability protection for the
permutation/conformal procedures). This builds the chains from the five local catalogs and
audits them. The most-bound-swap fallback (top-N bound members + phase-space) needs particle
data on Vera; the direct-ID chains built here are the primary linkage.

Mock pool snapshots = 771/743/692/660 (817 is z=0, not used for mocks; included for context).
"""
import json
import numpy as np
import pandas as pd

CATS = "vera/data/catalogs/catalog{}.parquet"
SNAPS = ["817", "771", "743", "692", "660"]
POOL = ["771", "743", "692", "660"]          # snapshots that feed the mock samples
Z = {"817": 0.0, "771": 0.1, "743": 0.2, "692": 0.4, "660": 0.49677}


def main():
    frames = []
    for s in SNAPS:
        d = pd.read_parquet(CATS.format(s),
                            columns=["index", "idMostbound", "mStar", "nStar", "isCentral"])
        d["snap"] = s
        frames.append(d)
        # within-snapshot idMostbound must be unique (global particle ID)
        dup = d["idMostbound"].duplicated().sum()
        umax = np.iinfo(np.uint64).max
        sentinel = int(((d["idMostbound"] == 0) | (d["idMostbound"] == umax)).sum())
        print(f"snap {s}: rows={len(d):,} dupIDwithin={dup} sentinelID={sentinel} "
              f"mStar>1e9 floor nStar min={int(d.nStar.min())}")

    allc = pd.concat(frames, ignore_index=True)
    # chains: group by idMostbound across snapshots
    g = allc.groupby("idMostbound")
    nsnap = g["snap"].nunique()
    chainLen = nsnap.value_counts().sort_index()
    print(f"\ntotal catalog rows (5 snaps) = {len(allc):,}")
    print(f"unique physical galaxies (idMostbound) = {allc.idMostbound.nunique():,}")
    print("chain length (in #snapshots) distribution:")
    for k, v in chainLen.items():
        print(f"  in {k} snapshot(s): {v:,} galaxies")

    # mass consistency on multi-snap chains (same galaxy, adjacent z -> modest mStar change)
    multi = nsnap[nsnap >= 2].index
    sub = allc[allc.idMostbound.isin(multi)]
    mm = sub.groupby("idMostbound")["mStar"]
    logspread = np.log10(mm.max() / mm.min())
    print(f"\nmulti-snapshot chains = {len(multi):,}; log10(maxM/minM) "
          f"median={logspread.median():.3f} p95={logspread.quantile(0.95):.3f} "
          f"frac>0.3dex={float((logspread>0.3).mean()):.3f}")

    # POOL view: duplication the sample must dedup (771/743/692/660 only)
    pool = allc[allc.snap.isin(POOL)]
    poolNsnap = pool.groupby("idMostbound")["snap"].nunique()
    poolDupRows = int((pool.groupby("idMostbound")["snap"].transform("nunique") >= 2).sum())
    print(f"\nPOOL (771/743/692/660): rows={len(pool):,} unique={pool.idMostbound.nunique():,}")
    print(f"  rows belonging to a multi-snapshot chain (would double-count w/o dedup) = "
          f"{poolDupRows:,} ({100*poolDupRows/len(pool):.1f}%)")
    for k, v in poolNsnap.value_counts().sort_index().items():
        print(f"  pool galaxies in {k} pool-snapshot(s): {v:,}")
    # duplication is mass-dependent: high-mass galaxies persist across all snaps
    pool = pool.copy()
    pool["logM"] = np.log10(pool.mStar)
    pool["chainN"] = pool.groupby("idMostbound")["snap"].transform("nunique")
    for lo in [9.0, 10.0, 10.8, 11.5]:
        hi = {9.0: 10.0, 10.0: 10.8, 10.8: 11.5, 11.5: 99}[lo]
        m = (pool.logM >= lo) & (pool.logM < hi)
        if m.sum():
            print(f"  logM[{lo},{hi}): rows={int(m.sum()):,} "
                  f"meanChainLen={pool.chainN[m].mean():.2f} "
                  f"frac_in_chain={float((pool.chainN[m]>=2).mean()):.3f}")

    # save per-row chain membership for S2 (vectorized; joins straight onto the catalogs).
    # S2 dedups by grouping rows on idMostbound; chainLen5 = #snaps incl 817, chainLenPool
    # = #pool snaps. The draw enforces at most one row per idMostbound per sample.
    allc["chainLen5"] = allc.groupby("idMostbound")["snap"].transform("nunique")
    poolMask = allc.snap.isin(POOL)
    pl = allc[poolMask].copy()
    pl["chainLenPool"] = pl.groupby("idMostbound")["snap"].transform("nunique")
    allc = allc.merge(pl[["idMostbound", "snap", "chainLenPool"]],
                      on=["idMostbound", "snap"], how="left")
    allc[["snap", "index", "idMostbound", "mStar", "isCentral",
          "chainLen5", "chainLenPool"]].to_parquet("data/dedupChains.parquet")

    summary = {
        "snapshots": SNAPS, "poolSnapshots": POOL, "redshifts": Z,
        "totalRows5snap": int(len(allc)),
        "uniquePhysicalGalaxies": int(allc.idMostbound.nunique()),
        "chainLenDist": {int(k): int(v) for k, v in chainLen.items()},
        "multiSnapChains": int(len(multi)),
        "massConsistency_logSpread": {
            "median": float(logspread.median()), "p95": float(logspread.quantile(0.95)),
            "frac_gt_0p3dex": float((logspread > 0.3).mean())},
        "pool": {
            "rows": int(len(pool)), "unique": int(pool.idMostbound.nunique()),
            "rowsInMultiChain": poolDupRows,
            "fracRowsInMultiChain": float(poolDupRows / len(pool))},
        "note": "Sample A/B draw at most one snapshot instance per idMostbound chain; "
                "duplication is strongly mass-dependent (massive galaxies persist across all snaps).",
    }
    with open("results/dedupChains.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote data/dedupChains.parquet + results/dedupChains.json")


if __name__ == "__main__":
    main()
