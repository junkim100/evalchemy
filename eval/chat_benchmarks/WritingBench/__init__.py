"""
WritingBench benchmark implementation for evalchemy.

This module provides a comprehensive benchmark for evaluating LLMs' writing capabilities
across diverse real-world tasks, following the original WritingBench methodology.
"""

from .eval_instruct import WritingBenchBenchmark, WritingBenchConfig
from .evaluator import BaseEvaluator, GPTEvaluator

__all__ = [
    "WritingBenchBenchmark",
    "WritingBenchConfig",
    "BaseEvaluator",
    "GPTEvaluator",
]
