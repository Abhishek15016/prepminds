# Base vs. Instruction Fine-Tuned (SFT) Model Comparison

**Status: needs regeneration (judge heuristic + decoding tweak applied).**

The core bug (SFT echoing the prompt) is fixed and confirmed working - the
latest run produced genuine, on-topic generations. Remaining issue: this
small, lightly-trained 0.5B model still occasionally collapses into a
repetition loop. Two changes:

1. Bumped `repetition_penalty` from 1.2 to 1.3 as a further, still-soft mitigation.
2. `src/eval_questions.py`'s heuristic judge now detects degenerate/looping
   output (`_is_degenerate()`) and never scores it as the "winner" purely for
   being long or keyword-dense - see `judge_pair()`.

Re-run `notebooks/regenerate_reports.ipynb`, opened fresh from GitHub, to
overwrite this file with the real comparison table (Assignment Step 7).
