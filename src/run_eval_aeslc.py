"""Stage A8: the Phase 1 evaluation of run_eval_v3.py, run on AESLC emails instead of OASST2.

Nothing else changes: same cut points (15/30/50/70/85 %), same seeds, same models, same output
columns, so policy_eval_v3.py and learned_policy.py can read the result with RESULTS=...
The first N_TRAIN (20) messages fit the baselines, the rest are evaluated.

Run from the repo root:
    LIMIT=2 OUT=smoke_aeslc uv run python -m src.run_eval_aeslc     # quick test, 2 messages
    caffeinate -i uv run python -m src.run_eval_aeslc                # full run
Settings (environment variables): N_MESSAGES (default 140 -> 120 evaluated), OUT (default
phase1_results_aeslc), LIMIT. The run is resumable: stop with Ctrl+C and start it again.
"""
import os

try:
    import src.run_eval_v3 as rv3
    from src.aeslc_prep import load_aeslc
except ImportError:  # run as a plain script from inside src/
    import run_eval_v3 as rv3
    from aeslc_prep import load_aeslc

rv3.N_MESSAGES = int(os.environ.get("N_MESSAGES", "140"))
rv3.OUT = os.environ.get("OUT", "phase1_results_aeslc")
rv3.load_oasst2_prompter_en = lambda max_messages=rv3.N_MESSAGES: load_aeslc(max_messages)

if __name__ == "__main__":
    rv3.main()