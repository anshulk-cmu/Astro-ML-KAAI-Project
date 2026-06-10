import numpy as np
import bigfile

PIG = "/hildafs/datasets/Asterix/PIG2/PIG_817_subfind"
H = 0.6774
MUNIT = 1e10 / H  # code mass 1e10 Msun/h -> Msun


def main():
    f = bigfile.File(PIG)
    print(f"opened {PIG}")
    print(f"top-level blocks: {sorted(set(b.split('/')[0] for b in f.blocks))}\n")

    h = f["Header"]
    print("Header attrs:")
    for k in h.attrs:
        print(f"  {k} = {h.attrs[k]}")
    print()

    for grp in ["FOFGroups", "SubGroups", "4", "NewgalSFR"]:
        cols = sorted(b.split("/", 1)[1] for b in f.blocks if b.startswith(grp + "/"))
        print(f"{grp}/ columns ({len(cols)}):")
        for c in cols:
            blk = f[f"{grp}/{c}"]
            print(f"  {c:32s} dtype={blk.dtype.str} nmemb={blk.dtype.shape or 1} size={blk.size}")
        print()

    nsub = f["SubGroups/SubhaloMass"].size
    print(f"total subhalo count: {nsub}")

    mstar = f["SubGroups/SubhaloMassType"][:][:, 4] * MUNIT
    thr = 10**9.3
    print(f"subhalos with Mstar > 10^9.3 Msun ({thr:.3e} Msun): {(mstar > thr).sum()}")
    print(f"subhalos with Mstar > 10^10  Msun: {(mstar > 1e10).sum()}")
    print(f"max Mstar = {mstar.max():.3e} Msun\n")

    # one subhalo just above 10^10 Msun
    cand = np.where(mstar > 1e10)[0]
    idx = int(cand[np.argmin(np.abs(mstar[cand] - 1.5e10))])
    print(f"picked subhalo index {idx}: Mstar = {mstar[idx]:.3e} Msun")
    for c in ["SubhaloGroupNr", "SubhaloPos", "SubhaloHalfmassRadType", "SubhaloSFR", "SubhaloIDMostbound"]:
        print(f"  {c} = {f['SubGroups/' + c][idx : idx + 1][0]}")

    off = f["SubGroups/SubhaloOffsetType"][idx : idx + 1][0][4]
    n = f["SubGroups/SubhaloLenType"][idx : idx + 1][0][4]
    print(f"  star particles: offset={off} count={n}")

    pos = f["4/Position"][off : off + n]
    mass = f["4/Mass"][off : off + n] * MUNIT
    sft = f["4/StarFormationTime"][off : off + n]
    met = f["4/Metallicity"][off : off + n]
    sgid = f["4/SubgroupID"][off : off + n]
    gid = f["4/GroupID"][off : off + n]

    print(f"  slice check: GroupID uniq={np.unique(gid)}  SubgroupID uniq={np.unique(sgid)}")
    print(f"  Position  [ckpc/h]      n={len(pos)} min={pos.min(0)} max={pos.max(0)}")
    print(f"  Mass      [Msun]        sum={mass.sum():.3e} min={mass.min():.3e} max={mass.max():.3e}")
    print(f"  SFTime    [scale fac a] min={sft.min():.4f} max={sft.max():.4f} median={np.median(sft):.4f}")
    print(f"  Metallicity [mass frac] min={met.min():.3e} max={met.max():.3e} median={np.median(met):.3e}")
    f.close()


if __name__ == "__main__":
    main()
