import json
import os

import numpy as np
import bigfile

PHOT = "/hildafs/datasets/Asterix/photometric"
FLSST = "/hildafs/datasets/Asterix/Fatemeh/LSST"
SUBFIND_ROOTS = {"PIG2": "/hildafs/datasets/Asterix/PIG2", "PIG3": "/hildafs/datasets/Asterix/PIG3"}
OUT_JSON = "/hildafs/projects/phy200026p/akumar45/results/photometricCensus.json"


def productDirs():
    dirs = []
    for d in sorted(os.listdir(PHOT)):
        p = os.path.join(PHOT, d)
        if not os.path.isdir(p) or d == "fsps_grids":
            continue
        if d == "restframe":
            for s in sorted(os.listdir(p)):
                dirs.append((f"photometric/restframe/{s}", os.path.join(p, s)))
        else:
            dirs.append((f"photometric/{d}", p))
    for d in sorted(os.listdir(FLSST)):
        p = os.path.join(FLSST, d)
        if not os.path.isdir(p) or "issue" in d:
            continue
        if d == "Aperture":
            for sub in sorted(os.listdir(p)):
                for s in sorted(os.listdir(os.path.join(p, sub))):
                    if "issue" in s:
                        continue
                    dirs.append((f"Fatemeh/LSST/Aperture/{sub}/{s}", os.path.join(p, sub, s)))
        else:
            dirs.append((f"Fatemeh/LSST/{d}", p))
    return dirs


def subfindCounts():
    counts = {}
    for tag, root in SUBFIND_ROOTS.items():
        for d in sorted(os.listdir(root)):
            if "subfind" not in d:
                continue
            try:
                f = bigfile.File(os.path.join(root, d))
                rec = {}
                if "SubGroups/SubhaloMass" in f.blocks:
                    rec["nsub"] = f["SubGroups/SubhaloMass"].size
                if "4/Mass" in f.blocks:
                    rec["nstar"] = f["4/Mass"].size
                f.close()
                if rec:
                    counts[f"{tag}/{d}"] = rec
            except Exception as e:
                counts[f"{tag}/{d}"] = {"error": str(e)}
    return counts


def convention(median):
    if median is None:
        return "no-data"
    if median < 0:
        return "absolute"
    if median > 5:
        return "apparent"
    return "ambiguous"


def describeGroup(f, grp):
    cols = sorted(b.split("/", 1)[1] for b in f.blocks if b.startswith(grp + "/"))
    if not cols:
        return None
    dtypes = {}
    rows = {}
    for c in cols:
        blk = f[f"{grp}/{c}"]
        dtypes[c] = blk.dtype.str
        rows[c] = blk.size
    blk = f[f"{grp}/{cols[0]}"]
    med = None
    if blk.size:
        s = np.asarray(blk[: min(1024, blk.size)], dtype="f8")
        s = s[np.isfinite(s)]
        med = float(np.median(s)) if s.size else None
    uniqDt = sorted(set(dtypes.values()))
    uniqRows = sorted(set(rows.values()))
    return {
        "columns": cols,
        "dtype": uniqDt[0] if len(uniqDt) == 1 else dtypes,
        "rows": uniqRows[0] if len(uniqRows) == 1 else rows,
        "sampleMedian": med,
        "convention": convention(med),
    }


def main():
    sf = subfindCounts()
    products = []
    for name, path in productDirs():
        entry = {"product": name}
        try:
            f = bigfile.File(path)
            tops = sorted(set(b.split("/")[0] for b in f.blocks))
            entry["topBlocks"] = tops
            for grp in tops:
                g = describeGroup(f, grp)
                if g:
                    entry[grp] = g
            f.close()
            if not tops:
                entry["note"] = "empty (no blocks)"
        except Exception as e:
            entry["error"] = str(e)
        sg = entry.get("SubGroups")
        if sg and isinstance(sg.get("rows"), int):
            entry["alignedSubfind"] = sorted(k for k, v in sf.items() if v.get("nsub") == sg["rows"])
        st = entry.get("4")
        if st and isinstance(st.get("rows"), int):
            entry["alignedSubfindStars"] = sorted(k for k, v in sf.items() if v.get("nstar") == st["rows"])
        products.append(entry)

    out = {"generated": "2026-06-10", "subfindCounts": sf, "products": products}
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {OUT_JSON}\n")

    print(f"{'product':58s} {'bands':28s} {'4/ rows':>13s} {'SG rows':>10s} {'conv4':>9s} {'convSG':>9s} aligned")
    for e in products:
        bands = e.get("SubGroups", e.get("4", {})) or {}
        cols = bands.get("columns", [])
        bandStr = ",".join(sorted(set(c.split("_")[0] for c in cols))) + f"({len(cols)})" if cols else "-"
        r4 = e.get("4", {}).get("rows", "-")
        rs = e.get("SubGroups", {}).get("rows", "-")
        c4 = e.get("4", {}).get("convention", "-")
        cs = e.get("SubGroups", {}).get("convention", "-")
        al = ";".join(e.get("alignedSubfind", [])) or e.get("error", e.get("note", "-"))
        print(f"{e['product']:58s} {bandStr:28s} {str(r4):>13s} {str(rs):>10s} {c4:>9s} {cs:>9s} {al}")

    print("\n--- subfind reference counts ---")
    for k, v in sf.items():
        print(f"{k:42s} nsub={v.get('nsub','-'):>10} nstar={v.get('nstar','-'):>13}")

    print("\n--- focus questions ---")
    for snap in ["660", "692", "743", "771"]:
        hits = [e for e in products if f"PIG_{snap}" in e["product"]]
        for e in hits:
            cols = (e.get("SubGroups") or e.get("4") or {}).get("columns", [])
            des = any(c.startswith("des_") for c in cols)
            convs = {g: e[g]["convention"] for g in ("4", "SubGroups") if g in e}
            print(f"PIG_{snap}: {e['product']:55s} des={des} conv={convs}")


if __name__ == "__main__":
    main()
