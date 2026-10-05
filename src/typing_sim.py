"""Typing simulator: real per-query pace (AmazonQAC) + modelled within-query gaps.

AmazonQAC only gives one start and one end time per query, so the gaps inside
a query are an ASSUMPTION controlled by sigma (gap spread) and boundary_mult
(extra pause after a space). Results should be checked across several settings.
"""
import numpy as np
import pandas as pd


def load_pace_quantiles(path="amazonqac_pace_quantiles.csv"):
    df = pd.read_csv(path)
    return df["q"].to_numpy(), df["sec_per_prefix"].to_numpy()


def sample_pace(rng, levels, values):
    """One typist's average seconds per keystroke, drawn from the real distribution."""
    return float(np.interp(rng.random(), levels, values))


def simulate_typing(text, pace, sigma, boundary_mult, rng):
    """Return [(time_in_seconds, prefix_typed_so_far), ...], one entry per keystroke."""
    n = len(text)
    if n < 2:
        return [(0.0, text)]
    raw = rng.lognormal(mean=-sigma ** 2 / 2, sigma=sigma, size=n - 1)
    after_space = np.array([text[i] == " " for i in range(n - 1)])
    raw = np.where(after_space, raw * boundary_mult, raw)
    gaps = raw * (pace / raw.mean())          # keep the average gap equal to the real pace
    times = np.concatenate([[0.0], np.cumsum(gaps)])
    return [(float(times[i]), text[: i + 1]) for i in range(n)]


SCENARIOS = [(0.3, 1.0), (0.7, 2.0), (1.2, 3.0)]   # (sigma, boundary_mult): mild -> bursty


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    levels, values = load_pace_quantiles()
    text = "can you help me write a short email to my landlord about the broken heater"

    pace = sample_pace(rng, levels, values)
    print(f"One session (pace {pace:.2f}s/key):")
    for t, p in simulate_typing(text, pace, 0.7, 2.0, rng)[:12]:
        print(f"  {t:6.2f}s  {p!r}")

    print("\nGap statistics per scenario (2000 simulated typists):")
    for sigma, b in SCENARIOS:
        gaps = []
        for _ in range(2000):
            tl = simulate_typing(text, sample_pace(rng, levels, values), sigma, b, rng)
            gaps.extend(np.diff([t for t, _ in tl]))
        gaps = np.array(gaps)
        print(f"  sigma={sigma}, boundary x{b}: median gap {np.median(gaps):.2f}s, "
              f"90th pct {np.percentile(gaps, 90):.2f}s, gaps >1s: {np.mean(gaps > 1):.1%}")
