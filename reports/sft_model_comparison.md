# Base vs. Instruction Fine-Tuned (SFT) Model Comparison

**Status: not yet generated.**

This file is auto-written by running
[`notebooks/instruction_finetuning.ipynb`](../notebooks/instruction_finetuning.ipynb)
on a GPU (Colab/Kaggle T4), after Stage 1 has produced
`models/non_instruction_adapter/`. The notebook fine-tunes the model on
`data/instruction_dataset.jsonl`, then regenerates answers for the base model
and the SFT model on the same 10 evaluation questions, and overwrites this
file with the real comparison table (Assignment Step 7).

Run that notebook top-to-bottom, then re-check this file in.
