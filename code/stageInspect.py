import os
import requests
import pyarrow.parquet as pq

D = "/home/ubuntu/data"
os.makedirs(D, exist_ok=True)
files = ["gz_desi_deep_learning_catalog_friendly.parquet", "external_catalog.parquet"]
base = "https://zenodo.org/records/8360385/files/{}?download=1"

for f in files:
    p = os.path.join(D, f)
    if not os.path.exists(p):
        print("downloading", f, flush=True)
        with requests.get(base.format(f), stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(p, "wb") as out:
                for c in r.iter_content(1 << 23):
                    out.write(c)
    pf = pq.ParquetFile(p)
    print("====", f, "rows =", pf.metadata.num_rows, "====")
    print(pf.schema_arrow.names)
