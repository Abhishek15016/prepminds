# Base Model Evaluation

**Status: needs regeneration (previous run affected by a since-fixed bug).**

A prior Colab run of `notebooks/regenerate_reports.ipynb` used a `generate_chat()`
helper that rendered the chat template to a string and then re-tokenized it
separately. That double round-trip could produce a different token count
than what `generate()` actually consumed as its prompt, so the prompt-length
slice used to isolate new tokens sometimes cut into the prompt itself instead
of the newly generated text. The base model in this report doesn't use a chat
template (it's queried as a plain completion), so it isn't directly affected,
but this file is being regenerated alongside `sft_model_comparison.md` and
`final_evaluation.md` for a consistent, verified set produced with the fixed
notebook.

The fix (tokenizing directly via `apply_chat_template(tokenize=True,
return_tensors="pt")` instead of a render-then-retokenize round trip) is
already in `notebooks/regenerate_reports.ipynb`, `notebooks/dpo_alignment.ipynb`,
`notebooks/instruction_finetuning.ipynb`, and `src/inference.py`.

Re-run `notebooks/regenerate_reports.ipynb` top-to-bottom (pulls the base
checkpoint and the Stage 2/3 adapters from the Hugging Face Hub - no
retraining needed) to overwrite this file with the real Question / Base
Model Answer / Problem table (Assignment Step 5).
