"""Real gap shape from the Aalto data (table made by fit_aalto_gaps.py).

Gaps are drawn from the empirical distribution of gaps relative to a typist's own
average, separately for gaps that follow a space and gaps that do not, then rescaled
so the average gap equals the pace drawn from AmazonQAC.
"""
import numpy as np
import pandas as pd


def load_gap_shape(path="aalto_gap_shape.csv"):
    df = pd.read_csv(path)
    return df["q"].to_numpy(), df["rel_gap_nonspace"].to_numpy(), df["rel_gap_space"].to_numpy()


def simulate_typing_empirical(text, pace, shape, rng):
    """Return [(time_in_seconds, prefix_typed_so_far), ...], one entry per keystroke."""
    levels, nonspace, space = shape
    n = len(text)
    if n < 2:
        return [(0.0, text)]
    u = rng.random(n - 1)
    after_space = np.array([text[i] == " " for i in range(n - 1)])
    raw = np.where(after_space, np.interp(u, levels, space), np.interp(u, levels, nonspace))
    raw = np.maximum(raw, 1e-3)
    gaps = raw * (pace / raw.mean())
    times = np.concatenate([[0.0], np.cumsum(gaps)])
    return [(float(times[i]), text[: i + 1]) for i in range(n)]
