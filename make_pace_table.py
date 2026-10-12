import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SNAP = "/Users/anjalisharma/.cache/huggingface/hub/datasets--amazon--AmazonQAC/snapshots/975cbfac4623ffb054c59631e06e39d0831e47fb"
files = sorted(glob.glob(f"{SNAP}/train/*.parquet"))[::7][:3]
cols = ["prefixes", "first_prefix_typed_time", "search_time"]

frames = []
for f in files:
    pf = pq.ParquetFile(f)
    got = 0
    for batch in pf.iter_batches(batch_size=35000, columns=cols):
        frames.append(batch.to_pandas())
        got += len(frames[-1])
        if got >= 70000:
            break
df = pd.concat(frames, ignore_index=True)
print("Rows read:", len(df), "from", len(files), "files")

df["n_prefixes"] = df["prefixes"].apply(len)
start = pd.to_datetime(df["first_prefix_typed_time"], format="mixed")
end = pd.to_datetime(df["search_time"], format="mixed")
df["dur_s"] = (end - start).dt.total_seconds()

ok = df[(df["dur_s"] > 0) & (df["n_prefixes"] >= 2)].copy()
lo, hi = ok["dur_s"].quantile([0.01, 0.99])
ok = ok[(ok["dur_s"] >= lo) & (ok["dur_s"] <= hi)]
ok["sec_per_prefix"] = ok["dur_s"] / ok["n_prefixes"]
print(f"Kept {len(ok)} rows; duration trimmed to [{lo:.2f}s, {hi:.2f}s]")

levels = np.linspace(0, 1, 1001)
table = pd.DataFrame({"q": levels, "sec_per_prefix": ok["sec_per_prefix"].quantile(levels).values})
table.to_csv("amazonqac_pace_quantiles.csv", index=False)
print(table.iloc[[0, 100, 250, 500, 750, 900, 1000]])
print("Saved amazonqac_pace_quantiles.csv")
