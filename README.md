# GenAI / Agentic AI Interview-Ready Domain Assistant

A domain-specific LLM assistant built by fine-tuning **Qwen2.5-0.5B** with
**[Unsloth](https://github.com/unslothai/unsloth)** through three progressive
stages: non-instruction (domain-adaptive) fine-tuning, instruction fine-tuning
(SFT), and DPO preference alignment — the full pipeline requested in the
*Practical Fine-Tuning Assignment*.

```
Base Model (Qwen2.5-0.5B)
   |
   v
Stage 1: Non-Instruction Fine-Tuning   (raw domain text, LoRA/QLoRA)
   |
   v
Stage 2: Instruction Fine-Tuning (SFT) (101 Q&A pairs, LoRA/QLoRA)
   |
   v
Stage 3: DPO Preference Alignment      (50 chosen/rejected pairs)
   |
   v
Final Domain-Specific AI Assistant
```

## Domain selected

**GenAI / Agentic AI interview preparation.** The assistant answers questions
about transformers, tokenization, embeddings, LoRA/QLoRA, RLHF/DPO/ORPO,
quantization, RAG, and agent frameworks (LangGraph, CrewAI, AutoGen) in a
structured, analogy-driven, *interview-ready* style — not a generic Q&A bot.

## Business problem

As a GenAI Engineer, I was asked to build an internal assistant that helps
engineers and candidates get genuinely interview-ready on GenAI/Agentic AI
topics: correct terminology, production trade-offs (not just textbook
definitions), and answers structured the way a strong candidate would
actually explain them out loud. A general-purpose base model answers these
questions in a generic, unstructured way (see
[`reports/base_model_evaluation.md`](reports/base_model_evaluation.md)); this
project closes that gap through domain-adaptive and preference-aligned
fine-tuning.

## Dataset details

| File | Purpose | Size | Format |
|---|---|---|---|
| `data/non_instruction_data.txt` | Raw domain explainer passages | ~50 passages | Plain text, `====` separated |
| `data/non_instruction_dataset.jsonl` | Stage 1 training data | 50 examples | `{"text": ...}` |
| `data/instruction_dataset.jsonl` | Stage 2 training data | 101 examples | Chat `{"messages": [system, user, assistant]}` |
| `data/preference_dataset.jsonl` | Stage 3 training data | 50 pairs | `{"prompt": [...], "chosen": [...], "rejected": [...]}` |

All content was authored for this project around a standard GenAI/Agentic AI
curriculum (transformers, fine-tuning, RAG, agents, deployment), manually
reviewed for correctness, then converted into the schemas above.
The instruction and preference datasets use the chat-message list format
(richer than a flat `instruction`/`response` string) so the same data works
directly with `tokenizer.apply_chat_template` and TRL's `SFTTrainer` /
`DPOTrainer` — `messages[user]` *is* the instruction, `messages[assistant]`
*is* the response.

## Base model

**`unsloth/Qwen2.5-0.5B-bnb-4bit`** — small enough to fully fine-tune three
times over on a free Colab/Kaggle T4 GPU in minutes per stage, while still
being a real, modern instruction-capable architecture worth demonstrating the
technique on.

## Approach

### Stage 1 — Non-instruction fine-tuning
`notebooks/non_instruction_finetuning.ipynb` continues pretraining the base
model on raw domain text with plain next-token prediction (no chat
structure), shifting its vocabulary and "voice" toward GenAI/Agentic AI
before it ever sees a question/answer pair.

### Stage 2 — Instruction fine-tuning (SFT)
`notebooks/instruction_finetuning.ipynb` loads the Stage 1 adapter and trains
on the 101 instruction/response pairs using a ChatML template, with loss
masked to assistant tokens only (`train_on_responses_only`), so the model
learns to *answer*, not just continue text.

### Stage 3 — DPO alignment
`notebooks/dpo_alignment.ipynb` loads the Stage 2 SFT model and runs DPO on
50 chosen/rejected pairs, sharpening which correctly-shaped answer the model
prefers — pushing it toward complete, professional, domain-precise responses
and away from shallow or unsafe ones.

Full conceptual write-up (why LoRA/QLoRA, SFT vs. DPO, and every
hyperparameter used) is in
[`reports/fine_tuning_explanation.md`](reports/fine_tuning_explanation.md).

### LoRA / QLoRA configuration

| Param | Stage 1 | Stage 2 | Stage 3 (DPO) |
|---|---|---|---|
| rank / alpha | 16 / 16 | 16 / 16 | 16 / 16 |
| dropout | 0 | 0 | 0 |
| learning rate | 2e-4 | 2e-4 | 5e-6 |
| effective batch size | 8 | 8 | 4 |
| epochs | 3 | 3 | 2 |
| quantization | 4-bit (QLoRA) | 4-bit (QLoRA) | 4-bit (QLoRA) |

## How to run

1. Open each notebook in Google Colab or Kaggle (**Runtime > Change runtime type > T4 GPU**), either by cloning this repo first or opening directly via *File > Open notebook > GitHub*.
2. Run notebooks **in order** — each stage's deliverable feeds the next:
   1. `notebooks/non_instruction_finetuning.ipynb` -> `models/non_instruction_adapter/`, `reports/base_model_evaluation.md`
   2. `notebooks/instruction_finetuning.ipynb` -> `models/sft_adapter/`, `reports/sft_model_comparison.md`
   3. `notebooks/dpo_alignment.ipynb` -> `models/dpo_adapter/`, `reports/final_evaluation.md`
3. Query the final model:
   ```bash
   python src/inference.py -q "What is LoRA?"
   ```
   or as a library:
   ```python
   from src.inference import generate_answer
   answer = generate_answer("How can I apply for reimbursement?")
   print(answer)
   ```

### Persisting adapters across Colab sessions (Hugging Face Hub)

Each stage runs in a fresh Colab/Kaggle session, and the free tier wipes local
disk between sessions — so `models/non_instruction_adapter/` from Stage 1
won't exist by the time you open a new session for Stage 2 unless you carry
it over yourself. Every notebook has an optional **Step 0** cell for this:

1. Set `HF_USERNAME` in that cell to your Hugging Face username.
2. Add a Colab secret named `HF_TOKEN` (key icon in the left sidebar) holding
   a Hugging Face token with **write** access — this lets `login()` run
   without an interactive prompt.
3. Each notebook then automatically pushes its adapter to
   `<HF_USERNAME>/qwen2.5-0.5b-genai-agentic-stage{1,2,3}-...` (private by
   default) after training, and the *next* notebook automatically pulls from
   the Hub if no local copy is found — no manual zip/download/upload needed.

Leave `HF_USERNAME` blank to skip the Hub entirely and rely on local disk
(fine if you run all three notebooks back-to-back in one long session, or
save/reopen a single persistent Colab runtime / mounted Drive).

`src/inference.py` accepts a Hub repo id directly, so you can point it at the
final pushed model from anywhere:
```bash
python src/inference.py -q "What is LoRA?" --model <hf-username>/qwen2.5-0.5b-genai-agentic-stage3-dpo
```

The evaluation reports (`base_model_evaluation.md`, `sft_model_comparison.md`,
`final_evaluation.md`) are **auto-generated by the notebooks themselves** —
every stage runs real inference on the same 10 canonical evaluation
questions (`src/eval_questions.py`, the single source of truth used by all
three notebooks) and writes the resulting comparison table straight to
`reports/`, so the before/after numbers in this repo come from an actual
run, not hand-written examples.

## Before vs. after

See the full tables in `reports/`; the shape of the result:

- **Base model**: generic, sometimes off-topic completions — it wasn't asked to *answer*, it was asked to *continue text*.
- **SFT model**: on-topic, structured, uses correct domain terminology.
- **DPO model**: same correct shape as SFT, but consistently the more complete, professional, and precise of the two — the effect targeted by the preference dataset's chosen/rejected contrast.

## Training screenshots / logs

_Add screenshots of the `trainer.train()` output cells from each notebook
here after your first Colab run (loss should trend down across steps in all
three stages)._

## Final observations

- Even a 0.5B model picks up a distinct, consistent domain voice after
  Stage 1 + 2 — the size of the base model matters far less than the
  quality and consistency of the fine-tuning data for a narrow domain like
  this one.
- DPO's effect is more subtle than SFT's: it doesn't teach new facts, it
  reshapes *style and completeness* within what SFT already taught, which is
  exactly the role it should play in this pipeline.
- Masking loss to assistant-only tokens (`train_on_responses_only`) matters
  more than it sounds — without it, the model wastes capacity learning to
  predict its own system/user prompts.

## Challenges faced

- Keeping the same 10 evaluation questions and generation settings identical
  across all three notebooks was essential for a fair before/after
  comparison — solved by centralizing them in `src/eval_questions.py`.
- A non-instruction base model has no natural stopping point for a plain-text
  completion prompt; `clean_completion()` trims runaway generations that
  start hallucinating a new `Question:` turn.
- Free-tier GPU memory is tight enough that QLoRA (4-bit base) rather than
  16-bit LoRA was the right default even for a model this small, to leave
  headroom for the DPO stage's extra reference-model bookkeeping.

## Future improvements

- Grow the instruction and preference datasets beyond the minimums (101 /
  50) for more robust generalization.
- Add an automated LLM-as-judge pass (rather than the keyword-based
  heuristic in `src/eval_questions.py`) to score the comparison tables.
- Try ORPO as a lighter-weight alternative to DPO (no reference model) and
  compare alignment quality.
- Push the final merged model to the Hugging Face Hub and wrap
  `src/inference.py` in a small FastAPI/Gradio demo.

## Repository structure

```
domain-ai-assistant-finetuning/
├── data/
│   ├── non_instruction_data.txt
│   ├── non_instruction_dataset.jsonl
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
├── src/
│   ├── inference.py
│   └── eval_questions.py
├── models/            # adapters saved here by the notebooks (gitignored; also optionally pushed to the HF Hub)
├── README.md
└── requirements.txt
```

## Interview pitch

> I created a domain-specific AI assistant using Unsloth. I first performed
> non-instruction fine-tuning on raw domain text, then instruction
> fine-tuning on Q&A data, and finally DPO alignment using preference data. I
> compared the base model, SFT model, and DPO-aligned model on an identical
> 10-question evaluation set — with the comparison reports generated
> automatically by the training notebooks — and built a simple inference
> script.
