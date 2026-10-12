import os
import sys

import numpy as np
import pandas as pd

# Local cache path used by the original author. Keep this relative/generic so a
# committed version never crashes someone else's checkout.
DEFAULT_CACHE = "data/amazonqac"

OUT = "amazonqac_pace_quantiles.csv"
LEVELS = np.round(np.linspace(0, 1, 101), 2)


def resolve_input():
    candidates = []
    if len(sys.argv) > 1:
        candidates.append(sys.argv[1])
    env = os.environ.get("AMAZONQAC_PATH")
    if env:
        candidates.append(env)
    candidates.append(DEFAULT_CACHE)
    for c in candidates:
        if os.path.isdir(c):
            files = sorted(f for f in os.listdir(c) if f.endswith((".csv", ".tsv", ".json")))
            if files:
                return os.path.join(c, files[0])
        if os.path.isfile(c):
            return c
    return None


def main():
    path = resolve_input()
    if path is None:
        print("AmazonQAC data not found - nothing to do.\n"
              "  This script is only needed to REBUILD the pace table; the repo already\n"
              "  ships amazonqac_pace_quantiles.csv, and policy_eval / policy_eval_v2 read\n"
              "  that CSV directly (no download needed).\n"
              "  To rebuild anyway:  uv run python -m src.make_pace_table /path/to/AmazonQAC\n"
              "  or set AMAZONQAC_PATH=/path/to/AmazonQAC. Leaving the committed CSV as is.")
        return 0
    df = pd.read_csv(path)
    dur = pd.to_numeric(df["duration"], errors="coerce")
    if "prefix_count" not in df:
        df["prefix_count"] = pd.to_numeric(df.get("n_prefixes", np.nan), errors="coerce")
    pace = (dur / df["prefix_count"]).dropna()
    pace = pace[(pace > 0) & (pace < 60)]          # drop degenerate sessions
    q = np.quantile(pace, LEVELS)
    out = pd.DataFrame({"q": LEVELS, "sec_per_prefix": q})
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} from {path} ({len(pace)} queries); median pace {np.median(pace):.2f}s/prefix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())