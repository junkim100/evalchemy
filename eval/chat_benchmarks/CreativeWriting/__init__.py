"""
Creative Writing Bench integration for evalchemy.

This module provides a complete integration of the EQ-Bench Creative Writing Benchmark v3
into the evalchemy evaluation framework, maintaining 100% compatibility with the original
evaluation methodology.
"""

from .eval_instruct import CreativeWriting

__all__ = ["CreativeWriting"]
