"""
Evaluation module for MemBuilder.

This module provides:
- LLM Judge for answer correctness evaluation
- Dataset loaders (LOCOMO, LongMemEval, PerLTQA)
- Unified evaluation CLI
"""

from .llm_judge import evaluate_answer, LLMJudge
from .metrics import compute_accuracy, print_accuracy_stats

__all__ = [
    "evaluate_answer",
    "LLMJudge", 
    "compute_accuracy",
    "print_accuracy_stats",
]
