import pandas as pd

SEED, N = 0, 50000
D = "/home/ubuntu/data"

fr = pd.read_parquet(f"{D}/gz_desi_deep_learning_catalog_friendly.parquet")
ex = pd.read_parquet(
    f"{D}/external_catalog.parquet",
    columns=["dr8_id", "redshift", "spec_z", "photo_z", "elpetro_mass_log",
             "total_ssfr_median", "u_minus_r", "sersic_n",
             "mag_g_desi", "mag_r_desi", "mag_z_desi"],
)

df = fr.merge(ex, on="dr8_id", how="left")
df = df[df["redshift"].notna() & (df["redshift"] > 0) & (df["redshift"] < 1)]
df = df[df["ra"].notna() & df["dec"].notna()]
df = df.sample(n=min(N, len(df)), random_state=SEED).reset_index(drop=True)

keep = ["dr8_id", "ra", "dec",
        "smooth-or-featured_smooth_fraction", "smooth-or-featured_featured-or-disk_fraction",
        "has-spiral-arms_yes_fraction", "bar_strong_fraction", "bar_weak_fraction",
        "merging_merger_fraction", "disk-edge-on_yes_fraction",
        "redshift", "spec_z", "photo_z", "elpetro_mass_log", "total_ssfr_median",
        "u_minus_r", "sersic_n", "mag_g_desi", "mag_r_desi", "mag_z_desi"]
df = df[keep]
df.to_parquet(f"{D}/sample.parquet")

print("sample rows:", len(df))
print("non-null counts:")
print(df.notna().sum().to_string())
print("head:")
print(df.head(3).to_string())
