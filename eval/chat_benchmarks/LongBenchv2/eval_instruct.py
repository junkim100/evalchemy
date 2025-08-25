import logging
import json
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional
import math
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark


class LongBenchv2Benchmark(BaseBenchmark):
    """
    LongBench v2 Benchmark for evaluating long context understanding capabilities of LLMs.

    LongBench v2 is designed to assess the ability of LLMs to handle long-context problems
    requiring deep understanding and reasoning across real-world multitasks.

    Features:
    - Context length ranging from 8k to 2M words
    - 503 challenging multiple-choice questions
    - Six major task categories: single-document QA, multi-document QA,
      long in-context learning, long-dialogue history understanding,
      code repo understanding, and long structured data understanding
    - Multiple choice format (A, B, C, D) for reliable evaluation

    Link: https://arxiv.org/abs/2412.15204
    """

    def __init__(
        self,
        dataset_name: str = "THUDM/LongBench-v2",
        max_tokens: int = 32768,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        enable_cot: bool = False,
        filter_domain: Optional[str] = None,
        filter_difficulty: Optional[str] = None,
        filter_length: Optional[str] = None,
    ):
        """
        Initialize LongBench v2 benchmark.

        Args:
            dataset_name: HuggingFace dataset name for LongBench v2
            max_tokens: Maximum tokens for model generation
            debug: If set, only evaluate on 5 examples
            seed: Random seed for reproducibility
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
            enable_cot: Enable Chain-of-Thought prompting
            filter_domain: Filter by specific domain (optional)
            filter_difficulty: Filter by difficulty level ("easy" or "hard")
            filter_length: Filter by length category ("short", "medium", or "long")
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = dataset_name
        self.max_new_tokens = max_tokens
        self.debug = debug
        self.seed = seed
        self.enable_cot = enable_cot
        self.filter_domain = filter_domain
        self.filter_difficulty = filter_difficulty
        self.filter_length = filter_length

        # Load dataset
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        """Load LongBench v2 dataset from HuggingFace."""
        try:
            # Load the dataset
            dataset = load_dataset(self.dataset_name, split="train", trust_remote_code=True)

            # Apply filters if specified
            if self.filter_domain:
                dataset = dataset.filter(lambda x: x["domain"] == self.filter_domain)
                self.logger.info(f"Filtered by domain '{self.filter_domain}': {len(dataset)} examples")

            if self.filter_difficulty:
                dataset = dataset.filter(lambda x: x["difficulty"] == self.filter_difficulty)
                self.logger.info(f"Filtered by difficulty '{self.filter_difficulty}': {len(dataset)} examples")

            if self.filter_length:
                dataset = dataset.filter(lambda x: x["length"] == self.filter_length)
                self.logger.info(f"Filtered by length '{self.filter_length}': {len(dataset)} examples")

            if self.debug:
                dataset = dataset.select(range(min(5, len(dataset))))
                self.logger.info(f"Debug mode: using {len(dataset)} examples")

            self.logger.info(f"Loaded {len(dataset)} LongBench v2 examples")
            return dataset

        except Exception as e:
            self.logger.error(f"Error loading LongBench v2 dataset: {str(e)}")
            # Fallback: create a minimal example for testing
            self.logger.warning("Creating minimal test dataset")
            return self._create_test_dataset()

    def _create_test_dataset(self):
        """Create a minimal test dataset for development/testing."""
        test_data = [
            {
                "_id": "test_1",
                "domain": "single_document_qa",
                "sub_domain": "academic_paper",
                "difficulty": "hard",
                "length": "long",
                "question": "What is the main contribution of this paper?",
                "choice_A": "A new model architecture",
                "choice_B": "A novel training method",
                "choice_C": "A comprehensive benchmark",
                "choice_D": "An optimization technique",
                "answer": "C",
                "context": "This paper presents LongBench v2, a comprehensive benchmark for evaluating long context understanding capabilities of large language models. The benchmark consists of 503 challenging multiple-choice questions with contexts ranging from 8k to 2M words across six major task categories."
            }
        ]

        # Convert to dataset-like structure
        class TestDataset:
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
            def __iter__(self):
                return iter(self.data)
            def select(self, indices):
                return TestDataset([self.data[i] for i in indices])
            def filter(self, func):
                return TestDataset([item for item in self.data if func(item)])

        return TestDataset(test_data)

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and evaluation data,
            or None for non-primary ranks
        """
        examples = []

        # Prepare instances for model
        all_instances = []
        for idx, example in enumerate(self.dataset):
            # Construct the prompt based on LongBench v2 format
            prompt = self._construct_prompt(example)

            messages = [
                {"role": "user", "content": prompt},
            ]

            templated_messages = self._prepare_messages(messages, model)

            instance = Instance(
                "generate_until",
                example,
                (
                    templated_messages,
                    {
                        "do_sample": False,
                        "max_new_tokens": self.max_new_tokens,
                        "temperature": 0.0,
                        "seed": self.seed,
                    },
                ),
                idx,
            )

            # Add metadata for tracking
            instance.metadata = {
                "question_id": example.get("_id", str(idx)),
                "domain": example.get("domain", "unknown"),
                "difficulty": example.get("difficulty", "unknown"),
                "length": example.get("length", "unknown"),
            }

            all_instances.append(instance)

        # Generate model responses
        self.logger.info("Generating responses for LongBench v2...")
        outputs = self.compute(model, all_instances)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        # Process outputs and prepare for evaluation
        for example, output in zip(self.dataset, outputs):
            # Extract text from different output types
            if isinstance(output, str):
                response_text = output
            elif hasattr(output, 'outputs') and output.outputs:
                response_text = output.outputs[0].text
            elif hasattr(output, 'text'):
                response_text = output.text
            else:
                response_text = str(output)

            example_dict = dict(example) if hasattr(example, 'keys') else example
            example_dict["model_output"] = response_text
            examples.append(example_dict)

        return {"examples": examples}

    def _construct_prompt(self, example: Dict[str, Any]) -> str:
        """
        Construct the prompt for LongBench v2 question.

        Args:
            example: Dictionary containing question information

        Returns:
            Formatted prompt string
        """
        # Extract components
        context = example.get("context", "")
        question = example.get("question", "")
        choice_a = example.get("choice_A", "")
        choice_b = example.get("choice_B", "")
        choice_c = example.get("choice_C", "")
        choice_d = example.get("choice_D", "")

        # Construct prompt
        if self.enable_cot:
            # Chain-of-Thought prompting
            prompt = f"""Please read the following context carefully and answer the question.

Context:
{context}

Question: {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Please think step by step and then provide your answer. Your final answer should be one of A, B, C, or D.

Let me think step by step:"""
        else:
            # Direct prompting
            prompt = f"""Please read the following context carefully and answer the question.

Context:
{context}

Question: {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Please provide your answer as one of A, B, C, or D."""

        return prompt

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated responses using accuracy metrics.

        Args:
            results: Dictionary containing examples with model outputs

        Returns:
            Dictionary containing evaluation metrics
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        # Track evaluation results
        correct_answers = 0
        domain_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        difficulty_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        length_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        detailed_results = []

        for example in examples:
            model_output = example.get("model_output", "")
            ground_truth = example.get("answer", "").strip().upper()

            # Extract predicted answer from model output
            predicted_answer = self._extract_answer(model_output)

            # Check if prediction is correct
            is_correct = predicted_answer == ground_truth

            if is_correct:
                correct_answers += 1

            # Update domain statistics
            domain = example.get("domain", "unknown")
            domain_stats[domain]["total"] += 1
            if is_correct:
                domain_stats[domain]["correct"] += 1

            # Update difficulty statistics
            difficulty = example.get("difficulty", "unknown")
            difficulty_stats[difficulty]["total"] += 1
            if is_correct:
                difficulty_stats[difficulty]["correct"] += 1

            # Update length statistics
            length = example.get("length", "unknown")
            length_stats[length]["total"] += 1
            if is_correct:
                length_stats[length]["correct"] += 1

            detailed_results.append({
                "question_id": example.get("_id", "unknown"),
                "domain": domain,
                "difficulty": difficulty,
                "length": length,
                "predicted_answer": predicted_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "model_output": model_output
            })

        # Calculate overall metrics
        overall_accuracy = correct_answers / num_questions if num_questions > 0 else 0.0

        # Calculate standard error (assuming binomial distribution)
        overall_std_err = np.sqrt(overall_accuracy * (1 - overall_accuracy) / num_questions) if num_questions > 0 else 0.0

        # Calculate domain-specific accuracies
        domain_accuracies = {}
        for domain, stats in domain_stats.items():
            domain_accuracies[f"accuracy_{domain}"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

        # Calculate difficulty-specific accuracies
        difficulty_accuracies = {}
        for difficulty, stats in difficulty_stats.items():
            difficulty_accuracies[f"accuracy_{difficulty}"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

        # Calculate length-specific accuracies
        length_accuracies = {}
        for length, stats in length_stats.items():
            length_accuracies[f"accuracy_{length}"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

        evaluation_results = {
            "num_total": num_questions,
            "correct_answers": correct_answers,
            "accuracy_avg": overall_accuracy,
            "accuracy_std_err": overall_std_err,
            "detailed_results": detailed_results,
            **domain_accuracies,
            **difficulty_accuracies,
            **length_accuracies,
        }

        self.logger.info(f"LongBench v2 Evaluation Results:")
        self.logger.info(f"  Total questions: {num_questions}")
        self.logger.info(f"  Correct answers: {correct_answers}")
        self.logger.info(f"  Overall accuracy: {overall_accuracy:.3f}")

        # Log domain-specific results
        for domain, stats in domain_stats.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            self.logger.info(f"  {domain} accuracy: {acc:.3f} ({stats['correct']}/{stats['total']})")

        # Log difficulty-specific results
        for difficulty, stats in difficulty_stats.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            self.logger.info(f"  {difficulty} accuracy: {acc:.3f} ({stats['correct']}/{stats['total']})")

        return evaluation_results

    def _extract_answer(self, model_output: str) -> str:
        """
        Extract the predicted answer (A, B, C, or D) from model output.

        Args:
            model_output: Raw model output text

        Returns:
            Extracted answer or empty string if not found
        """
        # Clean the output
        output = model_output.strip().upper()

        # Look for explicit answer patterns
        import re

        # Pattern 1: "Answer: A" or "The answer is A"
        answer_patterns = [
            r'(?:ANSWER|FINAL ANSWER|THE ANSWER)(?:\s*IS)?(?:\s*:)?\s*([ABCD])',
            r'(?:^|\s)([ABCD])(?:\s*$|\s*\.|\s*\))',
            r'(?:OPTION|CHOICE)\s*([ABCD])',
            r'([ABCD])(?:\s*IS\s*(?:THE\s*)?(?:CORRECT|RIGHT|ANSWER))',
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, output)
            if matches:
                return matches[-1]  # Return the last match

        # Pattern 2: Look for single letter A, B, C, or D at the end
        end_pattern = r'([ABCD])(?:\s*$|\s*\.$)'
        match = re.search(end_pattern, output)
        if match:
            return match.group(1)

        # Pattern 3: Look for the first occurrence of A, B, C, or D
        first_letter = re.search(r'([ABCD])', output)
        if first_letter:
            return first_letter.group(1)

        # If no clear answer found, return empty string
        return ""