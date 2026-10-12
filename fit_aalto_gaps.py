"""Fit the SHAPE of real inter-key gaps from the Aalto 136M Keystrokes dataset.

Each gap is divided by the mean gap of its own sentence, so only the shape and the
word-boundary effect are kept; the overall pace comes from AmazonQAC.

Usage:  uv run python fit_aalto_gaps.py <folder with *_keystrokes.txt, or a zip> [participants]
Writes: aalto_gap_shape.csv   (q, rel_gap_nonspace, rel_gap_space)

Source: https://userinterfaces.aalto.fi/136Mkeystrokes/  (free for research use with attribution)
"""
import glob
import os
import sys
import zipfile

import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    sys.exit("usage: uv run python fit_aalto_gaps.py <folder or zip> [participants]")
path = os.path.expanduser(sys.argv[1])
n_part = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
MIN_KEYS = 15       # sentences with fewer keystrokes are skipped
MAX_GAP_S = 10.0    # sentences containing a longer gap (interrupted) are skipped
MODIFIERS = [16, 17, 18, 20, 91, 93]   # SHIFT, CTRL, ALT, CAPS, META: not typed characters
rng = np.random.default_rng(42)

if os.path.isdir(path):
    z = None
    all_names = glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)
else:
    z = zipfile.ZipFile(path)
    all_names = [n for n in z.namelist() if n.lower().endswith(".txt")]
names = [n for n in all_names if "keystroke" in os.path.basename(n).lower() and "readme" not in n.lower()]
if not names:
    sys.exit("no keystroke files found")
names = sorted(names)
names = [names[i] for i in rng.permutation(len(names))[:n_part]]
print(f"Reading {len(names)} participant files")

space_gaps, other_gaps = [], []
kept = skipped = 0
for i, name in enumerate(names):
    with (z.open(name) if z else open(name, "rb")) as f:
        df = pd.read_csv(f, sep="\t", encoding="latin-1", quoting=3,
                         on_bad_lines="skip", low_memory=False)
    if i == 0:
        print("Columns:", list(df.columns))
        if not {"PRESS_TIME", "TEST_SECTION_ID", "KEYCODE"} <= set(df.columns):
            sys.exit("Unexpected columns - paste the 'Columns:' line above to Claude")
    code = pd.to_numeric(df["KEYCODE"], errors="coerce")
    df = df.assign(_code=code, _t=pd.to_numeric(df["PRESS_TIME"], errors="coerce")).dropna(subset=["_t"])
    df = df[~df["_code"].isin(MODIFIERS)]
    df = df.assign(_sp=df["_code"] == 32)
    for _, g in df.groupby("TEST_SECTION_ID"):
        g = g.sort_values("_t")
        if len(g) < MIN_KEYS:
            skipped += 1
            continue
        gaps = np.maximum(np.diff(g["_t"].to_numpy() / 1000.0), 0.001)
        if gaps.max() > MAX_GAP_S:
            skipped += 1
            continue
        rel = gaps / gaps.mean()
        sp = g["_sp"].to_numpy()[:-1]          # gap that follows a space key
        space_gaps.extend(rel[sp])
        other_gaps.extend(rel[~sp])
        kept += 1

space_gaps, other_gaps = np.array(space_gaps), np.array(other_gaps)
if kept == 0 or len(space_gaps) == 0:
    sys.exit("no usable sentences - check the columns printed above")
print(f"Sentences kept {kept}, skipped {skipped}")
print(f"Gaps: {len(other_gaps)} inside words, {len(space_gaps)} after a space")

cap = np.quantile(np.concatenate([space_gaps, other_gaps]), 0.999)
space_gaps, other_gaps = np.minimum(space_gaps, cap), np.minimum(other_gaps, cap)
allg = np.concatenate([space_gaps, other_gaps])

print(f"\nWord-boundary effect: mean gap after a space = {space_gaps.mean() / other_gaps.mean():.2f}x "
      f"the mean gap inside words   (modelled styles: 1.0 / 2.0 / 3.0)")
print(f"Spread inside words (std of log gap) = {np.std(np.log(other_gaps)):.2f}   (modelled styles: 0.3 / 0.7 / 1.2)")
print(f"Share of gaps longer than 2x / 3x the typist's own average: "
      f"{np.mean(allg > 2):.1%} / {np.mean(allg > 3):.1%}")

levels = np.linspace(0, 1, 1001)
pd.DataFrame({"q": levels,
              "rel_gap_nonspace": np.quantile(other_gaps, levels),
              "rel_gap_space": np.quantile(space_gaps, levels)}).to_csv("aalto_gap_shape.csv", index=False)
print("Saved aalto_gap_shape.csv")
