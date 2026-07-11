"""
Simple inference script for the final, DPO-aligned GenAI/Agentic AI domain
assistant (Assignment Step 12).

Usage:
    python src/inference.py                                  # interactive REPL
    python src/inference.py -q "What is LoRA?"                # one-shot question
    python src/inference.py -q "What is LoRA?" --model models/sft_adapter
    python src/inference.py -q "What is LoRA?" --model <hf-username>/qwen2.5-0.5b-genai-agentic-stage3-dpo

As a library:
    from src.inference import generate_answer
    answer = generate_answer("How can I apply for reimbursement?")
    print(answer)

`--model` (or the `model_dir` argument to `generate_answer`) accepts either a
local adapter directory (e.g. `models/dpo_adapter`, saved by the notebooks)
or a Hugging Face Hub repo id (e.g. one pushed by the notebooks' optional
`push_adapter(...)` step) - useful since a local `models/` folder saved in a
Colab session doesn't survive after that session ends. Set the
`HF_DPO_REPO` env var to change the default without passing `--model` every
time.

Requires a CUDA GPU with unsloth/transformers/trl installed (see
requirements.txt) - run this on the same Colab/Kaggle instance (or a machine
with the trained model available locally or on the Hub).
"""

import argparse
import os

_LOCAL_DPO_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "dpo_adapter")
DEFAULT_MODEL_DIR = _LOCAL_DPO_DIR if os.path.isdir(_LOCAL_DPO_DIR) else (os.environ.get("HF_DPO_REPO") or _LOCAL_DPO_DIR)
MAX_SEQ_LENGTH = 2048
SYSTEM_PROMPT = (
    "You are a friendly expert tutor in Generative AI and Agentic AI. "
    "Explain concepts in depth with intuitive analogies, practical production "
    "insight, and interview-ready takeaways, in a clear and engaging style."
)

_model = None
_tokenizer = None


def _load(model_dir: str):
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    # `model_dir` may be a local adapter directory (e.g. models/dpo_adapter,
    # saved by the notebooks) or a Hugging Face Hub repo id (e.g. one pushed
    # by the notebooks' optional push_adapter(...) step) - Unsloth/Transformers
    # resolve either transparently, so we just try and give a clearer combined
    # error message if both lookups fail.
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not load '{model_dir}' as a local adapter directory or a "
            "Hugging Face Hub repo id. Run notebooks/dpo_alignment.ipynb (or "
            "instruction_finetuning.ipynb for the SFT-only model) first, or "
            "pass --model with a valid local path or '<hf-username>/<repo>' id. "
            f"Original error: {e}"
        ) from e
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    FastLanguageModel.for_inference(model)

    _model, _tokenizer = model, tokenizer
    return _model, _tokenizer


def generate_answer(question: str, model_dir: str = DEFAULT_MODEL_DIR, max_new_tokens: int = 300) -> str:
    """Generate an answer to `question` from the fine-tuned domain assistant."""
    model, tokenizer = _load(model_dir)

    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()


def main():
    parser = argparse.ArgumentParser(description="Query the fine-tuned GenAI/Agentic AI domain assistant.")
    parser.add_argument("-q", "--question", type=str, default=None, help="Ask a single question and exit.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_DIR, help="Path to the model/adapter directory.")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    if args.question:
        print(generate_answer(args.question, model_dir=args.model, max_new_tokens=args.max_new_tokens))
        return

    print("GenAI / Agentic AI domain assistant - type a question ('exit' to quit)\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        print(generate_answer(question, model_dir=args.model, max_new_tokens=args.max_new_tokens))
        print()


if __name__ == "__main__":
    main()
