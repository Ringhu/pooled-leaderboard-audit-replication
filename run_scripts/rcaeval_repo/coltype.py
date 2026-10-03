import pandas as pd, numpy as np, sys
base="$HOME/rcaeval_repo/data/sock-shop-2/carts_cpu/1/"
for f in ["data.csv","simple_data.csv"]:
    d=pd.read_csv(base+f)
    cols=[c for c in d.columns if c!="time"]
    mono=const=0
    for c in cols:
        v=pd.to_numeric(d[c],errors="coerce").ffill().fillna(0).to_numpy()
        if np.nanstd(v)==0: const+=1
        elif np.all(np.diff(v)>=-1e-9): mono+=1
    print(f"{f}: n_metric={len(cols)}  monotone_nondecreasing={mono} ({100*mono/len(cols):.0f}%)  constant={const}")
    # name families
    import collections
    suf=collections.Counter(c.split("_",1)[1].split("-")[-1] if "_" in c else "?" for c in cols)
    print("   top suffixes:", suf.most_common(5))
