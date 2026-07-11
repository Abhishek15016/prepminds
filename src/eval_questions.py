"""
Shared evaluation harness for the GenAI/Agentic AI domain assistant.

Single source of truth for the 10 canonical evaluation questions and the
lightweight heuristic "judge" used to auto-draft the Problem / Which-is-Better /
Reason columns in the markdown reports. The heuristics only draft a first pass -
review and edit the generated reports/*.md before submission.

Used by all three notebooks (so every stage is scored on the identical
question set) and by src/inference.py.
"""

import re

EVAL_QUESTIONS = [
    "If a transformer reads all words at once, how does it know that 'bank' in 'river bank' isn't a financial institution?",
    "Full fine-tuning vs LoRA: why can LoRA train a 7B model on a gaming GPU while full fine-tuning needs a cluster?",
    "QLoRA claims you can fine-tune large models in 4-bit precision without wrecking quality. What's the trick?",
    "RLHF, DPO, ORPO, GRPO - untangle this alphabet soup. What problem do they share, and how do they differ?",
    "What actually separates an 'AI agent' from a well-prompted LLM pipeline? Where's the line?",
    "What is Mixture of Experts (MoE), and why do models like Mixtral use it instead of a plain dense transformer?",
    "Bigger embedding dimensionality generally captures more nuance. So why don't production RAG systems just always use the largest embedding model available?",
    "INT8, INT4, GPTQ, AWQ - there are several quantization approaches for shrinking LLMs. How do they actually differ?",
    "FlashAttention is mentioned everywhere as a training/inference speedup. In plain terms, what problem does it solve?",
    "How does LangGraph model an agent as a state machine - what are nodes, edges, and checkpointing actually doing?",
]

_DOMAIN_KEYWORDS = [
    "attention", "token", "lora", "qlora", "quantiz", "4-bit", "4bit", "dpo", "orpo",
    "rlhf", "grpo", "agent", "moe", "mixture of experts", "embedding", "rag",
    "retrieval", "flashattention", "flash attention", "gpu", "vram", "langgraph",
    "state machine", "checkpoint", "gate", "expert", "int8", "int4", "gptq", "awq",
    "gradient", "adapter", "fine-tun", "preference",
]

_GENERIC_PHRASES = [
    "i don't know", "as an ai", "i am not sure", "i cannot help",
    "please provide more", "i'm just a language model",
]


def clean_completion(raw: str, question: str) -> str:
    """Trim a base-model free-completion at the point it starts hallucinating
    a new Q&A turn (a common failure mode for a non-chat base model)."""
    text = raw.strip()
    for cut_marker in ["\nQuestion:", "\nQ:", "\n##", "<|im_start|>", "\n\nQuestion"]:
        idx = text.find(cut_marker)
        if idx > 0:
            text = text[:idx].strip()
    return text if text else "(model produced an empty / degenerate completion)"


def _domain_hit_count(text: str) -> int:
    low = text.lower()
    return sum(1 for kw in _DOMAIN_KEYWORDS if kw in low)


def judge_base_answer(question: str, answer: str) -> str:
    """Heuristic first draft for the 'Problem' column in base_model_evaluation.md."""
    low = answer.lower()
    words = len(answer.split())
    if any(p in low for p in _GENERIC_PHRASES) or words < 8:
        return "Generic / evasive - the base model gives almost no usable domain content."
    if _domain_hit_count(answer) == 0:
        return "Off-topic or superficial - misses the core GenAI/Agentic-AI terminology the question is asking about."
    if len(set(answer.split())) < words * 0.55:
        return "Repetitive / rambling - the base model loops instead of giving a structured, direct answer."
    return "Shallow and unstructured - technically adjacent but lacks the depth, correct terminology, and interview-ready framing a domain expert would give."


def judge_pair(question: str, a_name: str, a_answer: str, b_name: str, b_answer: str):
    """Heuristic first draft for a 'Which is better / Reason' style column.
    Returns (winner_name, reason). Prefers the answer with denser correct
    domain vocabulary and less repetition; ties break toward the later stage
    model, matching the expected effect of additional fine-tuning."""
    score_a = _domain_hit_count(a_answer) + len(a_answer.split()) / 40
    score_b = _domain_hit_count(b_answer) + len(b_answer.split()) / 40
    if any(p in a_answer.lower() for p in _GENERIC_PHRASES) or len(a_answer.split()) < 8:
        score_a -= 5
    if any(p in b_answer.lower() for p in _GENERIC_PHRASES) or len(b_answer.split()) < 8:
        score_b -= 5

    if abs(score_a - score_b) < 0.75:
        winner, reason = b_name, "Comparable content; picked as the later-stage model, which had more targeted training signal."
    elif score_a > score_b:
        winner, reason = a_name, "Denser, more accurate domain vocabulary and a more complete answer."
    else:
        winner, reason = b_name, "Denser, more accurate domain vocabulary and a more complete answer."
    return winner, reason


def markdown_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = [str(c).replace("|", "\\|").replace("\n", "<br>") for c in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
