"""
Phase-space cross-snapshot galaxy matcher (preRegistration §11.1 dedup; plan §5.2(4)).

idMostbound links the same galaxy across snapshots only when the most-bound particle does NOT
swap, which over our 1-2 Gyr snapshot gaps almost never holds (measured recall ~0.25%, ~0% at
high mass). The same physical galaxies persist across the box, so we link them by phase space:
predict each galaxy's position at the next snapshot via velocity extrapolation, then mutual-
nearest match with mass consistency. The km/s -> ckpc/h velocity scale `k` is CALIBRATED
empirically per snapshot pair from confident position-only matches (no assumed convention),
then applied. Union-find over consecutive-pair links builds the dedup chains for S2.

Pool snapshots 771(z0.1) 743(z0.2) 692(z0.4) 660(z0.5); Dt from the measured temporal grid.
"""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CATS = "vera/data/catalogs/catalog{}.parquet"
BOX = 250000.0
POOL = ["771", "743", "692", "660"]
DT = {("771", "743"): 1.1661, ("743", "692"): 1.9043, ("692", "660"): 0.7560}  # Gyr


def load(s):
    d = pd.read_parquet(CATS.format(s),
                        columns=["index", "idMostbound", "mStar",
                                 "posX", "posY", "posZ", "velX", "velY", "velZ"])
    pos = np.mod(d[["posX", "posY", "posZ"]].to_numpy("f8"), BOX)
    vel = d[["velX", "velY", "velZ"]].to_numpy("f8")
    return dict(idx=d["index"].to_numpy("i8"), idmb=d["idMostbound"].to_numpy(),
                logM=np.log10(d["mStar"].to_numpy("f8")), pos=pos, vel=vel)


def minImage(dx):
    return dx - BOX * np.round(dx / BOX)


def mutualNearest(posA, posB):
    """indices (iA, iB) of mutual nearest neighbours, with separation."""
    tB = cKDTree(posB, boxsize=BOX)
    dAB, jAB = tB.query(posA, k=1)
    tA = cKDTree(posA, boxsize=BOX)
    _, iBA = tA.query(posB, k=1)
    iA = np.arange(len(posA))
    mutual = iBA[jAB] == iA
    return iA[mutual], jAB[mutual], dAB[mutual]


def matchPair(E, L, dt):
    # Stage 1: position-only confident matches (high mass) -> calibrate velocity scale k
    selE = E["logM"] > 10.5
    selL = L["logM"] > 10.5
    eIdx = np.flatnonzero(selE)
    lIdx = np.flatnonzero(selL)
    iA, iB, sep = mutualNearest(E["pos"][eIdx], L["pos"][lIdx])
    e0, l0 = eIdx[iA], lIdx[iB]
    keep = (sep < 300) & (np.abs(E["logM"][e0] - L["logM"][l0]) < 0.15)
    e0, l0 = e0[keep], l0[keep]
    disp = minImage(L["pos"][l0] - E["pos"][e0])
    v = E["vel"][e0]
    k = float((disp * v).sum() / (v * v).sum())            # scalar least squares: disp = k v
    resid0 = np.linalg.norm(disp, axis=1)
    resid1 = np.linalg.norm(disp - k * v, axis=1)
    cal = dict(nCalib=int(len(e0)), kCkpchPerKmps=k,
               residMedian_noVel=float(np.median(resid0)),
               residMedian_velCorr=float(np.median(resid1)))

    # Stage 2: velocity-extrapolated mutual-nearest over ALL galaxies, mass-consistent.
    # Tolerance grows with mass: high-mass galaxies are sparse (mean sep ~10 Mpc/h) so a wide
    # window is false-match-safe and recovers fast cluster movers; mass band widens too
    # (mergers grow mass going forward in time).
    posPredE = np.mod(E["pos"] + k * E["vel"], BOX)
    iA, iB, sep = mutualNearest(posPredE, L["pos"])
    lm = E["logM"][iA]
    tol = np.where(lm > 11.0, 600.0, np.where(lm > 10.8, 400.0, 150.0))
    mband = np.where(lm > 10.8, 0.35, 0.25)
    keep = (sep < tol) & (np.abs(lm - L["logM"][iB]) < mband)
    return iA[keep], iB[keep], cal


class UF:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def main():
    cat = {s: load(s) for s in POOL}
    uf = UF()
    links, calib = {}, {}
    for a, b in zip(POOL[:-1], POOL[1:]):
        iA, iB, cal = matchPair(cat[a], cat[b], DT[(a, b)])
        calib[f"{a}->{b}"] = cal
        idMatch = int((cat[a]["idmb"][iA] == cat[b]["idmb"][iB]).sum())
        links[f"{a}->{b}"] = dict(nLinks=int(len(iA)), idMostboundAgree=idMatch)
        for ea, lb in zip(iA, iB):
            uf.union((a, int(ea)), (b, int(lb)))
        print(f"{a}->{b}: k={cal['kCkpchPerKmps']:.3f} ckpc/h per km/s "
              f"(resid {cal['residMedian_noVel']:.0f}->{cal['residMedian_velCorr']:.0f} ckpc/h); "
              f"links={len(iA)} (idMostbound agrees on {idMatch})")

    # assemble chains
    rows = []
    for s in POOL:
        for r in range(len(cat[s]["idx"])):
            root = uf.find((s, r))
            rows.append((s, int(cat[s]["idx"][r]), f"{root[0]}:{root[1]}",
                         float(cat[s]["logM"][r])))
    df = pd.DataFrame(rows, columns=["snap", "index", "chainRoot", "logM"])
    df["chainLen"] = df.groupby("chainRoot")["snap"].transform("nunique")
    df.to_parquet("data/dedupChainsPhaseSpace.parquet")

    # recurrence by mass bin (fraction of rows in a multi-snapshot chain)
    byMass = {}
    for lo, hi in [(9.0, 10.0), (10.0, 10.8), (10.8, 11.5), (11.5, 99)]:
        m = (df.logM >= lo) & (df.logM < hi)
        byMass[f"{lo}-{hi}"] = dict(
            rows=int(m.sum()),
            fracInChain=float((df.chainLen[m] >= 2).mean()) if m.sum() else 0.0,
            meanChainLen=float(df.chainLen[m].mean()) if m.sum() else 0.0)

    dist = df.drop_duplicates("chainRoot")["chainLen"].value_counts().sort_index()
    summary = dict(
        poolSnapshots=POOL, dtGyr={f"{a}->{b}": DT[(a, b)] for a, b in zip(POOL[:-1], POOL[1:])},
        velocityCalibration=calib, pairLinks=links,
        totalRows=int(len(df)), uniqueChains=int(df.chainRoot.nunique()),
        chainLenDist={int(k): int(v) for k, v in dist.items()},
        recurrenceByMass=byMass,
        note="Phase-space chains are the S2 dedup substrate; idMostbound is a high-precision "
             "confirmer only (it agrees on a small minority of true links).")
    with open("results/dedupPhaseSpace.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nunique chains {df.chainRoot.nunique():,} of {len(df):,} rows")
    print("recurrence by mass:", json.dumps(byMass, indent=2))
    print("wrote data/dedupChainsPhaseSpace.parquet + results/dedupPhaseSpace.json")


if __name__ == "__main__":
    main()
