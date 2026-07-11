# Base Model Evaluation

**Status: needs regeneration (judge heuristic + decoding tweak applied).**

The core bug (SFT/DPO echoing the prompt) is fixed and confirmed working -
the latest run produced genuine, on-topic generations. Remaining issue: this
small, lightly-trained 0.5B model still occasionally collapses into a
repetition loop (e.g. a word or short phrase repeated dozens of times,
sometimes with no whitespace between repeats). Two changes:

1. Bumped `repetition_penalty` from 1.2 to 1.3 (matching what already works
   on the base-model decoding path) as a further, still-soft mitigation.
2. More importantly: `src/eval_questions.py`'s heuristic judge previously
   scored answers mainly on length and domain-keyword density, which could
   rate a degenerate loop as "denser" than a shorter, coherent answer just
   because it repeated a keyword many times. Added `_is_degenerate()` -
   detects repetition loops (including ones with no whitespace between
   repeats) - and `judge_pair()`/`judge_base_answer()` now flag or penalize
   degenerate output instead of rewarding it.

Some repetition-collapse rows may still appear in the raw model output after
this - that's an honest characteristic of a 0.5B model fine-tuned on ~100-150
examples, not a bug to keep chasing indefinitely. The judge should now at
least *label* it correctly instead of masking it as a "win."

Re-run `notebooks/regenerate_reports.ipynb`, opened fresh from GitHub, to
overwrite this file with the real Question / Base Model Answer / Problem
table (Assignment Step 5).
