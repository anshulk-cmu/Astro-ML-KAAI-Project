import numpy as np
import bigfile

PATHS = {
    "PIG_817_photometric": "/hildafs/datasets/Asterix/photometric/PIG_817_photometric",
    "PIG_724_photometric (Fatemeh/LSST)": "/hildafs/datasets/Asterix/Fatemeh/LSST/PIG_724_photometric",
}


def describe(name, path):
    print(f"########## {name}\n{path}")
    f = bigfile.File(path)
    tops = sorted(set(b.split("/")[0] for b in f.blocks))
    print(f"top-level blocks: {tops}")
    info = {}
    for grp in tops:
        cols = sorted(b.split("/", 1)[1] for b in f.blocks if b.startswith(grp + "/"))
        print(f"{grp}/ columns ({len(cols)}):")
        for c in cols:
            blk = f[f"{grp}/{c}"]
            attrs = {k: blk.attrs[k] for k in blk.attrs}
            print(f"  {c:12s} dtype={blk.dtype.str} size={blk.size} attrs={attrs or 'none'}")
            info[f"{grp}/{c}"] = (blk.dtype.str, blk.size)
        sample = f[f"{grp}/{cols[0]}"][:5]
        print(f"  sample {grp}/{cols[0]}[:5] = {sample}")
    f.close()
    print()
    return info


def main():
    infos = {n: describe(n, p) for n, p in PATHS.items()}
    (n1, i1), (n2, i2) = infos.items()
    print("########## structural comparison")
    only1 = sorted(set(i1) - set(i2))
    only2 = sorted(set(i2) - set(i1))
    print(f"blocks only in {n1}: {only1}")
    print(f"blocks only in {n2}: {only2}")
    for b in sorted(set(i1) & set(i2)):
        d1, s1 = i1[b]
        d2, s2 = i2[b]
        flag = " <-- dtype differs" if d1 != d2 else ""
        print(f"  {b:24s} {d1} n={s1}  vs  {d2} n={s2}{flag}")


if __name__ == "__main__":
    main()
