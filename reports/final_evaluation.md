# Final Evaluation: Base vs. SFT vs. DPO-Aligned Model

**Status: not yet generated.**

This file is auto-written by running
[`notebooks/dpo_alignment.ipynb`](../notebooks/dpo_alignment.ipynb) on a GPU
(Colab/Kaggle T4), after Stage 2 has produced `models/sft_adapter/`. The
notebook runs DPO on `data/preference_dataset.jsonl`, then regenerates
answers for the base, SFT, and DPO models on the same 10 evaluation
questions, and overwrites this file with the real three-way comparison table
(Assignment Step 10).

Run that notebook top-to-bottom, then re-check this file in.
