"""Timing-only comparison of invocation policies (no model calls).

For each policy we simulate typing, decide when the model is invoked, and check
whether Qwen's answer (latency sampled from Phase 1) arrives before the user's
next keystroke.

  calls/100 keys   - model calls per 100 keystrokes (cost)
  in time %        - share of those calls whose answer arrives before the next key
  in-time/100 keys - answers delivered in time per 100 keystrokes (benefit)

Policies:
  always          call the model right after every keystroke
  pause X s       call only if the user has not typed again within X seconds
  word boundary   call only right after a space (word boundary)
"""
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

GRANS = ["partial_word", "next_word", "phrase", "sentence"]
# name: (wait_seconds_before_calling, only_after_space)
POLICIES = {
    "always": (0.0, False),
    "pause 0.3s": (0.3, False),
    "pause 0.6s": (0.6, False),
    "pause 1.0s": (1.0, False),
    "word boundary": (0.0, True),
}

for sigma, b in SCENARIOS:
    gaps, spaces = [], []
    for t in texts:
        tl = simulate_typing(t, sample_pace(rng, levels, values), sigma, b, rng)
        times = [x for x, _ in tl]
        gaps.extend(np.diff(times))
        spaces.extend([p.endswith(" ") for _, p in tl[:-1]])
    gaps = np.array(gaps)
    spaces = np.array(spaces)
    n_keys = len(gaps)

    print(f"\n=== typing style: sigma={sigma}, boundary x{b} ===")
    print(f"{'granularity':<14}{'policy':<16}{'calls/100 keys':>16}{'in time %':>12}{'in-time/100 keys':>18}")
    for g in GRANS:
        L = rng.choice(lat[g], size=n_keys)
        for name, (wait, need_space) in POLICIES.items():
            if need_space and g == "partial_word":
                print(f"{g:<14}{name:<16}{'n/a (mid-word)':>46}")
                continue
            invoked = (gaps > wait) & (spaces if need_space else True)
            in_time = invoked & (gaps > wait + L)
            calls = invoked.sum()
            pct = 100 * in_time.sum() / calls if calls else float("nan")
            print(f"{g:<14}{name:<16}{100 * calls / n_keys:>16.1f}{pct:>12.1f}{100 * in_time.sum() / n_keys:>18.1f}")
