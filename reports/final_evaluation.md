# Final Evaluation: Base vs. SFT vs. DPO-Aligned Model

**Status: needs regeneration (one more decoding-parameter fix applied).**

Good news from the latest Colab run: the `tokenize=True` fix worked - the
SFT/DPO models are now genuinely generating novel, on-topic text instead of
echoing the prompt. However, a few rows degenerated into character-level
gibberish (e.g. trailing off into repeated single letters/fragments). That
was a side effect of `no_repeat_ngram_size=3`, a *hard* block on repeated
3-grams: for this small, lightly-trained 0.5B model, forbidding its natural
(repetitive) continuation outright pushed it into low-probability nonsense
instead. Removed `no_repeat_ngram_size`, kept the softer `repetition_penalty=1.2`
alone (matching what already works for the base-model path), in
`notebooks/dpo_alignment.ipynb` and `notebooks/regenerate_reports.ipynb`.

Re-run `notebooks/regenerate_reports.ipynb`, opened fresh from GitHub (not a
reused Colab tab), to overwrite this file with the real three-way comparison
table (Assignment Step 10).
