"""Adversarial prompting utilities for medical VQA."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class AdversarialPromptConfig:
    """Configuration for question prompt perturbations."""

    enabled: bool = False
    probability: float = 0.5
    mode: str = "mixed"
    seed: int = 42


PROMPT_TEMPLATES = {
    "instruction": [
        "Answer only from the image: {question}",
        "Use the image only and answer: {question}",
    ],
    "careful": [
        "Carefully inspect the image and answer: {question}",
        "Review the image closely before answering: {question}",
    ],
    "strict": [
        "Base your answer only on visible evidence: {question}",
        "Do not guess; answer using the image alone: {question}",
    ],
    "contrast": [
        "Ignore irrelevant details and answer: {question}",
        "Even if the image has distracting artifacts, answer: {question}",
    ],
    "mixed": [
        "Answer only from the image: {question}",
        "Carefully inspect the image and answer: {question}",
        "Do not guess; answer using the image alone: {question}",
        "Ignore irrelevant details and answer: {question}",
    ],
}


def _stable_offset(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def build_adversarial_prompt(question: str, mode: str = "mixed", seed: int = 42) -> str:
    """Rewrite a question into a more robust prompt without changing the answer."""
    normalized_mode = (mode or "mixed").lower()
    templates = PROMPT_TEMPLATES.get(normalized_mode, PROMPT_TEMPLATES["mixed"])
    rng = random.Random(seed + _stable_offset(question))
    template = rng.choice(templates)
    cleaned_question = question.strip()
    return template.format(question=cleaned_question)


def perturb_questions(
    questions: Sequence[str],
    config: AdversarialPromptConfig,
) -> Tuple[List[str], List[bool]]:
    """Optionally perturb a batch of questions for adversarial prompting."""
    if not config.enabled:
        return list(questions), [False for _ in questions]

    perturbed_questions: List[str] = []
    prompt_flags: List[bool] = []

    for question in questions:
        rng = random.Random(config.seed + _stable_offset(question))
        should_perturb = rng.random() < config.probability
        if should_perturb:
            perturbed_questions.append(build_adversarial_prompt(question, mode=config.mode, seed=config.seed))
        else:
            perturbed_questions.append(question)
        prompt_flags.append(should_perturb)

    return perturbed_questions, prompt_flags
