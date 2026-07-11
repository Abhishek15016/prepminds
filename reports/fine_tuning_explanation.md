# Fine-Tuning Explanation: LoRA, QLoRA, Non-Instruction FT, SFT, and DPO

This is a plain-language explanation of every technique used in this project's
three-stage pipeline (`Base -> Non-Instruction FT -> Instruction FT -> DPO`),
written the way I'd explain it in an interview.

## Why full fine-tuning is expensive

Full fine-tuning updates every parameter in the model. For that you need to
keep, per parameter, the weight itself, its gradient, and (with Adam) two
optimizer moment estimates — roughly 4x the raw parameter memory on top of
activations. Even for a "small" 7B model that's well over 100GB when you
include optimizer state, which is why full fine-tuning realistically needs a
multi-GPU cluster. It also produces a full new copy of the model per
fine-tuned task, which gets expensive to store and serve. For a 0.5B model
like the one used here, full fine-tuning is actually feasible on a single
GPU — but the project deliberately uses LoRA/QLoRA anyway, because that's the
technique that scales to the 7B–70B models used in real production systems.

## What LoRA does

LoRA (Low-Rank Adaptation) freezes the original weight matrix `W` completely
and instead learns a small *update* to it, expressed as the product of two
skinny matrices: `ΔW = B x A`, where `A` is `(r x d)` and `B` is `(d x r)` for
a chosen small rank `r` (16 in this project) instead of the full `d x d`.
The forward pass becomes `h = Wx + (B·A)x`. Because `r << d`, the number of
trainable parameters drops by 100–1000x — in this project, `model.print_trainable_parameters()`
in every notebook shows well under 1% of the total parameters are trainable.
Everything else about the model (weights, memory layout) stays untouched, so:
- Optimizer state only needs to track the tiny `A`/`B` matrices, not the full weight matrix.
- The saved "adapter" is a few MB instead of a full model copy.
- Multiple LoRA adapters can be trained and swapped on top of the same frozen base model.

## What QLoRA does

QLoRA adds one more idea on top of LoRA: quantize the *frozen* base model to
4-bit precision (NF4) before attaching LoRA adapters, and keep the adapters
themselves in full precision (bf16/fp16). The frozen base weights are only
ever read for a forward/backward pass and never updated, so lossy 4-bit
storage doesn't compound error the way it would in full fine-tuning.
Combined with double quantization (quantizing the quantization constants
too) and paged optimizers (spilling optimizer state to CPU RAM on memory
spikes), QLoRA cuts the base model's memory footprint roughly 4x compared to
16-bit LoRA, for a small, controlled hit to numerical precision.

## Why QLoRA is useful on limited GPU

A free Colab/Kaggle T4 GPU has 16GB of VRAM. Loading a model in 16-bit takes
roughly 2 bytes/parameter; in 4-bit it's roughly 0.5 bytes/parameter — a 4x
reduction before any LoRA adapters or activations are even accounted for.
That's what turns "needs an A100" into "runs on a free GPU tier," and it's
why QLoRA is the default recipe Unsloth uses even for a small model like the
0.5B one here: it leaves headroom for larger batch sizes, longer sequences,
and gradient checkpointing without hitting an out-of-memory error.

## What is non-instruction fine-tuning?

Non-instruction fine-tuning (a small-scale version of "continued
pretraining") trains the model on raw, unlabeled domain text with the plain
next-token-prediction objective it was originally pretrained with — no
question/answer structure, no chat format. The goal isn't to teach the model
*how to follow instructions*; it's to shift its internal representations and
vocabulary toward a specific domain (here, GenAI/Agentic AI terminology and
explanation style) before instruction tuning even begins. This is Stage 1 in
`notebooks/non_instruction_finetuning.ipynb`, trained on `data/non_instruction_data.txt`
via `data/non_instruction_dataset.jsonl`.

## What is instruction fine-tuning?

Instruction fine-tuning (SFT — Supervised Fine-Tuning) trains the model on
paired instruction/response examples formatted as a chat conversation, so it
learns the *behavior* of answering a question directly instead of just
continuing text statistically. This is Stage 2 in
`notebooks/instruction_finetuning.ipynb`, trained on `data/instruction_dataset.jsonl`
(101 Q&A pairs), with the loss masked so the model is only ever penalized for
its own assistant-turn tokens, not for the system/user prompt it was given.

## What is DPO?

DPO (Direct Preference Optimization) is a way to align a model's behavior to
human preference *without* the complexity of classic RLHF (no separate
reward model, no PPO reinforcement-learning loop). Instead, for each prompt
you give the model a `chosen` response and a `rejected` response, and DPO
directly optimizes the policy to increase the log-probability gap between
them — pushing the model toward outputs like `chosen` and away from outputs
like `rejected` — using a frozen copy of the pre-DPO model as an implicit
reference to prevent the model from drifting too far or reward-hacking.
This is Stage 3 in `notebooks/dpo_alignment.ipynb`, trained on
`data/preference_dataset.jsonl` (50 chosen/rejected pairs) starting from the
Stage 2 SFT model. (ORPO is a close cousin: it folds the same
preference signal into the SFT loss directly via an odds-ratio penalty,
skipping the reference model entirely — a lighter-weight alternative to DPO
this project's `dpo_alignment.ipynb` could be swapped to use.)

## Difference between SFT and DPO

SFT teaches the model *a* correct shape of answer by imitation — "here is a
good response, predict it token by token." It has no concept of "better" vs
"worse," only "given" vs "not given." DPO operates one level up: it teaches
the model to *prefer* one valid-looking response over another, using
contrastive chosen/rejected pairs. In practice they're complementary and
sequential, not competing — SFT gets the model into the right neighborhood of
behavior (answering questions, in-domain, roughly correct), and DPO then
sharpens *which* answer in that neighborhood the model should produce,
nudging it toward more complete, safe, and professional responses and away
from the shallow, vague, or unsafe patterns captured in the "rejected"
column.

## Hyperparameters used

| Parameter | Stage 1 (non-instruction) | Stage 2 (instruction/SFT) | Stage 3 (DPO) |
|---|---|---|---|
| LoRA rank (`r`) | 16 | 16 | 16 |
| LoRA alpha | 16 | 16 | 16 |
| LoRA dropout | 0 | 0 | 0 |
| Target modules | q/k/v/o_proj, gate/up/down_proj | q/k/v/o_proj, gate/up/down_proj | q/k/v/o_proj, gate/up/down_proj |
| Learning rate | 2e-4 | 2e-4 | 5e-6 |
| LR scheduler | linear | cosine | linear |
| Per-device batch size | 2 | 2 | 1 |
| Gradient accumulation | 4 | 4 | 4 |
| Effective batch size | 8 | 8 | 4 |
| Epochs | 3 | 3 | 2 |
| Optimizer | adamw_8bit | adamw_8bit | adamw_8bit |
| Precision | 4-bit base (QLoRA) + bf16/fp16 adapters | same | same |
| DPO beta | — | — | 0.1 |

`alpha == r` (a 1:1 scaling ratio) keeps the LoRA update's effective
magnitude close to the base weights' natural scale — a common, stable
default for small ranks. Dropout is left at 0 because these are small,
low-epoch runs on relatively small datasets, where LoRA is already a strong
regularizer by construction (few trainable parameters). DPO's learning rate
is ~40x smaller than SFT's because DPO is *fine-tuning a fine-tuned model's
preferences* — large steps here easily overcorrect and collapse response
diversity or drift off the reference model DPO is implicitly anchored to.
