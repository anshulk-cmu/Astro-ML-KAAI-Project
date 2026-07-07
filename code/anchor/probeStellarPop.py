"""
Probe NOIRLab Data Lab for stellar AGE + METALLICITY tables (for Track C's
age-metallicity degeneracy angle). Finds candidate VAC tables (Firefly, SDSS
stellar-mass VACs, MPA-JHU galSpec indices) and lists their age/metallicity/D4000
columns, so we can pick a source and write the real crossmatch.

Run from the project root:  python code/anchor/probeStellarPop.py
"""
import urllib.request, urllib.parse
from io import StringIO
import pandas as pd

TAP = "https://datalab.noirlab.edu/tap/sync"


def q(sql):
    d = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                "FORMAT": "csv", "QUERY": sql}).encode()
    return pd.read_csv(StringIO(urllib.request.urlopen(TAP, data=d, timeout=120).read().decode()))


print("== candidate tables ==")
t = q("SELECT table_name FROM tap_schema.tables WHERE "
      "table_name LIKE '%firefly%' OR table_name LIKE '%stellarmass%' "
      "OR table_name LIKE '%galspecindx%' OR table_name LIKE '%galspecextra%' "
      "OR table_name LIKE '%spec%stellar%'")
print(t.to_string())

for tn in t.table_name:
    try:
        cols = q("SELECT column_name FROM tap_schema.columns WHERE table_name='" + tn + "' "
                 "AND (column_name LIKE '%age%' OR column_name LIKE '%metal%' "
                 "OR column_name LIKE '%logz%' OR column_name LIKE '%z_h%' "
                 "OR column_name LIKE '%d4000%' OR column_name LIKE '%mwage%' "
                 "OR column_name LIKE '%lwage%')")
        if len(cols):
            print(f"\n{tn}  age/metal columns:")
            print("  " + ", ".join(cols.column_name.tolist()))
    except Exception as e:
        print(f"  ({tn}: {e})")
