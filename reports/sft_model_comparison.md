# Base vs. Instruction Fine-Tuned (SFT) Model Comparison

**Status: needs regeneration (previous runs affected by a since-fixed bug).**

Two prior Colab runs produced a "Fine-Tuned Model Answer" column that was
broken - verbatim echoes of the question or system prompt, and in some rows
a degenerate repetition loop. The root cause was `generate_chat()` rendering
the chat template to a string and re-tokenizing it separately, which could
desync the prompt-length slice used to isolate new tokens from what
`generate()` actually consumed as its prompt.

`notebooks/regenerate_reports.ipynb` now:
1. Tokenizes directly via `apply_chat_template(tokenize=True, return_tensors="pt")`, eliminating that desync.
2. Adds `repetition_penalty=1.2` / `no_repeat_ngram_size=3` to the chat-model decoding call.
3. Asserts the fix is actually present in the running kernel before generating anything, and fails loudly instead of silently reproducing the old bug - this catches the case where a Colab tab was left open across a `git pull` and is still running stale, pre-fix cell code.

Re-run `notebooks/regenerate_reports.ipynb`, but **open it fresh from GitHub**
(or fully disconnect/delete the Colab runtime and re-clone) rather than
reusing an already-open tab, so the fix actually loads. It pulls the base
checkpoint and the Stage 2/3 adapters from the Hugging Face Hub - no
retraining needed - and will overwrite this file with the real comparison
table (Assignment Step 7).
