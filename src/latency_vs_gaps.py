"""Share of invocations whose result would arrive before the user's next keystroke."""
import numpy as np
import pandas as pd
from typing_sim import load_pace_quantiles, sample_pace, simulate_typing, SCENARIOS

rng = np.random.default_rng(42)
levels, values = load_pace_quantiles()

df = pd.read_csv("phase1_results_v2.csv")
df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
qwen = df[df["system"] == "qwen"]
lat = {g: grp["latency_ms"].dropna().to_numpy() / 1000 for g, grp in qwen.groupby("granularity")}

uniq = np.array([t for t in df["prefix"].dropna().unique() if len(t) >= 20], dtype=object)
texts = list(rng.permutation(uniq)[:300])
print(f"Typing {len(texts)} real prefixes under each scenario\n")

grans = ["partial_word", "next_word", "phrase", "sentence"]
print(f"{'scenario':<28}" + "".join(f"{g:>14}" for g in grans))
for sigma, b in SCENARIOS:
    gaps = []
    for t in texts:
        tl = simulate_typing(t, sample_pace(rng, levels, values), sigma, b, rng)
        gaps.extend(np.diff([x for x, _ in tl]))
    gaps = np.array(gaps)
    row = f"sigma={sigma}, boundary x{b}".ljust(28)
    for g in grans:
        draws = rng.choice(lat[g], size=len(gaps))
        row += f"{np.mean(gaps > draws):>14.1%}"
    print(row)
