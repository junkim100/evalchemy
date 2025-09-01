import logging
import re
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError:
    AutoTokenizer = None
    LLM = None
    SamplingParams = None

try:
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM as EvalLM
except ImportError:
    Instance = None
    EvalLM = None

try:
    from eval.task import BaseBenchmark
except ImportError:
    BaseBenchmark = None

try:
    from .eval_utils import (
        compute_results_classifier,
        compute_results_hashing,
        compute_results_advbench,
        LLAMA2_CLS_PROMPT
    )
except ImportError:
    # Fallback imports for when running outside the HarmBench directory
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from eval_utils import (
            compute_results_classifier,
            compute_results_hashing,
            compute_results_advbench,
            LLAMA2_CLS_PROMPT
        )
    except ImportError:
        compute_results_classifier = None
        compute_results_hashing = None
        compute_results_advbench = None
        LLAMA2_CLS_PROMPT = None


class HarmBenchBenchmark(BaseBenchmark if BaseBenchmark is not None else object):
    """
    HarmBench benchmark for evaluating language model safety and robustness against harmful prompts.

    This benchmark evaluates models on their ability to refuse harmful requests across multiple
    categories including illegal activities, harmful content, misinformation, and more.

    Link: https://github.com/centerforaisafety/HarmBench
    """

    def __init__(
        self,
        behaviors_path: str = "eval/chat_benchmarks/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv",
        classifier_path: str = "cais/HarmBench-Llama-2-13b-cls",
        max_examples: Optional[int] = None,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 512,
        temperature: float = 0.0,
        include_advbench_metric: bool = True,
        functional_categories: Optional[List[str]] = None,
        semantic_categories: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize HarmBench benchmark.

        Args:
            behaviors_path: Path to the behaviors CSV file
            classifier_path: Path or name of the HarmBench classifier model
            max_examples: Maximum number of examples to evaluate (None for all)
            debug: If set, only evaluate on 10 examples
            seed: Random seed for reproducibility
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            include_advbench_metric: Whether to include AdvBench refusal metric
            functional_categories: List of functional categories to include (None for all)
            semantic_categories: List of semantic categories to include (None for all)
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        # Check for required dependencies
        missing_deps = []
        if pd is None:
            missing_deps.append("pandas")
        if np is None:
            missing_deps.append("numpy")
        if AutoTokenizer is None or LLM is None:
            missing_deps.append("transformers vllm")
        if BaseBenchmark is None:
            missing_deps.append("lm_eval")

        if missing_deps:
            raise ImportError(
                f"Missing required dependencies for HarmBench evaluation: {', '.join(missing_deps)}. "
                f"Install with: pip install {' '.join(missing_deps)}"
            )

        super().__init__(logger=logger, system_instruction=system_instruction)

        self.behaviors_path = behaviors_path
        self.classifier_path = classifier_path
        self.max_examples = max_examples
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.temperature = temperature
        self.include_advbench_metric = include_advbench_metric
        self.functional_categories = functional_categories
        self.semantic_categories = semantic_categories

        # Load behaviors dataset
        self.behaviors = self._load_behaviors()

        # Initialize classifier (lazy loading)
        self._classifier = None
        self._classifier_tokenizer = None

        # Available categories
        self.available_functional_categories = list(set(b['FunctionalCategory'] for b in self.behaviors))
        self.available_semantic_categories = list(set(b['SemanticCategory'] for b in self.behaviors))

        self.logger.info(f"Loaded {len(self.behaviors)} behaviors")
        self.logger.info(f"Available functional categories: {self.available_functional_categories}")
        self.logger.info(f"Available semantic categories: {self.available_semantic_categories}")

    def _load_behaviors(self) -> List[Dict[str, str]]:
        """Load behaviors from CSV file."""
        if not os.path.exists(self.behaviors_path):
            raise FileNotFoundError(f"Behaviors file not found: {self.behaviors_path}")

        behaviors = []
        with open(self.behaviors_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by categories if specified
                if self.functional_categories and row['FunctionalCategory'] not in self.functional_categories:
                    continue
                if self.semantic_categories and row['SemanticCategory'] not in self.semantic_categories:
                    continue
                behaviors.append(row)

        return behaviors

    def _get_classifier(self):
        """Lazy load the classifier model."""
        if self._classifier is None:
            try:
                self._classifier = LLM(
                    model=self.classifier_path,
                    tensor_parallel_size=1,
                    dtype="float16"
                )
                self._classifier_tokenizer = AutoTokenizer.from_pretrained(self.classifier_path)
                self.logger.info(f"Loaded classifier: {self.classifier_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load classifier {self.classifier_path}: {e}")
                self._classifier = None
                self._classifier_tokenizer = None

        return self._classifier, self._classifier_tokenizer

    def _create_harmful_prompt(self, behavior: str, context: str = None) -> str:
        """Create a prompt for the harmful behavior."""
        if context and context.strip():
            # Contextual behavior
            prompt = f"{context}\n\n{behavior}"
        else:
            # Standard behavior
            prompt = behavior

        return prompt

    def generate_responses(self, model) -> Dict[str, Any]:
        """
        Generate responses to harmful prompts using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and evaluation data,
            or None for non-primary ranks
        """
        # Filter and prepare examples
        examples = self.behaviors.copy()

        if self.debug:
            examples = examples[:10]
        elif self.max_examples:
            examples = examples[:self.max_examples]

        self.logger.info(f"Evaluating {len(examples)} harmful behaviors")
        if self.functional_categories:
            self.logger.info(f"Functional categories: {self.functional_categories}")
        if self.semantic_categories:
            self.logger.info(f"Semantic categories: {self.semantic_categories}")

        # Prepare instances for model
        all_instances = []
        for idx, behavior_dict in enumerate(examples):
            behavior = behavior_dict['Behavior']
            context = behavior_dict.get('ContextString', '')

            prompt = self._create_harmful_prompt(behavior, context)

            messages = [
                {"role": "user", "content": prompt},
            ]

            templated_messages = self._prepare_messages(messages, model)

            instance = Instance(
                "generate_until",
                behavior_dict,
                (
                    templated_messages,
                    {
                        "do_sample": self.temperature > 0,
                        "max_new_tokens": self.max_new_tokens,
                        "temperature": self.temperature,
                        "seed": self.seed,
                    },
                ),
                idx,
            )

            # Add metadata
            instance.metadata = {
                "behavior": behavior,
                "context": context,
                "behavior_id": behavior_dict['BehaviorID'],
                "functional_category": behavior_dict['FunctionalCategory'],
                "semantic_category": behavior_dict['SemanticCategory'],
                "tags": behavior_dict.get('Tags', ''),
            }

            all_instances.append(instance)

        # Generate model responses
        self.logger.info(f"Generating responses for HarmBench...")
        outputs = self.compute(model, all_instances)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        # Process outputs
        processed_examples = []
        for behavior_dict, output in zip(examples, outputs):
            behavior = behavior_dict['Behavior']
            context = behavior_dict.get('ContextString', '')

            # Extract response from model output
            model_response = self._extract_response(output)

            processed_examples.append({
                "behavior": behavior,
                "context": context,
                "behavior_id": behavior_dict['BehaviorID'],
                "functional_category": behavior_dict['FunctionalCategory'],
                "semantic_category": behavior_dict['SemanticCategory'],
                "tags": behavior_dict.get('Tags', ''),
                "generation": model_response,
                "model_output": output,
                "behavior_dict": behavior_dict,
            })

        return {
            "examples": processed_examples,
            "classifier_path": self.classifier_path,
            "include_advbench_metric": self.include_advbench_metric
        }

    def _extract_response(self, output: str) -> str:
        """
        Extract the response from model output.

        Args:
            output: Raw model output

        Returns:
            Extracted response text
        """
        # Handle different output formats
        if isinstance(output, str):
            text = output
        elif hasattr(output, 'outputs') and output.outputs:
            text = output.outputs[0].text
        elif hasattr(output, 'text'):
            text = output.text
        else:
            text = str(output)

        # Clean up the response - for HarmBench we want the full response
        return text.strip()

    def _evaluate_with_classifier(self, examples: List[Dict]) -> List[Dict]:
        """Evaluate examples using the HarmBench classifier."""
        classifier, tokenizer = self._get_classifier()

        if classifier is None or compute_results_classifier is None:
            self.logger.warning("Classifier not available, skipping classifier evaluation")
            return examples

        # Group examples by behavior for batch processing
        behavior_groups = defaultdict(list)
        for example in examples:
            behavior_id = example['behavior_id']
            behavior_groups[behavior_id].append(example)

        # Evaluate each behavior group
        evaluated_examples = []
        for behavior_id, behavior_examples in behavior_groups.items():
            behavior_dict = behavior_examples[0]['behavior_dict']

            # Prepare data for classifier
            data = [{"generation": ex["generation"]} for ex in behavior_examples]

            try:
                # Use the evaluation utility function
                cls_params = {
                    "model": classifier,
                    "tokenizer": tokenizer,
                    "sampling_params": SamplingParams(
                        temperature=0.0,
                        max_tokens=10,
                        stop=["\n"]
                    )
                }

                results = compute_results_classifier(behavior_dict, data, classifier, cls_params)

                # Merge results back
                for example, result in zip(behavior_examples, results):
                    example.update(result)
                    evaluated_examples.append(example)

            except Exception as e:
                self.logger.error(f"Error evaluating behavior {behavior_id} with classifier: {e}")
                # Add default labels
                for example in behavior_examples:
                    example['label'] = 0  # Default to not harmful
                    evaluated_examples.append(example)

        return evaluated_examples

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated responses for harmful content."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        classifier_path = results["classifier_path"]
        include_advbench_metric = results["include_advbench_metric"]

        if not examples:
            return {"error": "No examples to evaluate"}

        self.logger.info(f"Evaluating {len(examples)} responses for harmful content")

        # Evaluate with classifier
        evaluated_examples = self._evaluate_with_classifier(examples)

        # Handle copyright behaviors with hashing
        copyright_examples = []
        standard_examples = []

        for example in evaluated_examples:
            tags = example.get('tags', '')
            if 'hash_check' in tags:
                copyright_examples.append(example)
            else:
                standard_examples.append(example)

        # Evaluate copyright behaviors
        if copyright_examples and compute_results_hashing is not None:
            try:
                # Group by behavior for hashing evaluation
                copyright_behavior_groups = defaultdict(list)
                for example in copyright_examples:
                    behavior_id = example['behavior_id']
                    copyright_behavior_groups[behavior_id].append(example)

                for behavior_id, behavior_examples in copyright_behavior_groups.items():
                    behavior_dict = behavior_examples[0]['behavior_dict']
                    data = [{"generation": ex["generation"]} for ex in behavior_examples]

                    try:
                        hash_results = compute_results_hashing(behavior_dict, data)
                        for example, result in zip(behavior_examples, hash_results):
                            example.update(result)
                    except Exception as e:
                        self.logger.warning(f"Error evaluating copyright behavior {behavior_id}: {e}")
                        for example in behavior_examples:
                            example['label'] = 0

            except Exception as e:
                self.logger.warning(f"Error in copyright evaluation: {e}")

        # Add AdvBench metric if requested
        if include_advbench_metric and compute_results_advbench is not None:
            try:
                data = [{"generation": ex["generation"]} for ex in evaluated_examples]
                advbench_results = compute_results_advbench(evaluated_examples, data)
                evaluated_examples = advbench_results
            except Exception as e:
                self.logger.warning(f"Error computing AdvBench metric: {e}")

        # Calculate metrics
        results_dict = self._calculate_metrics(evaluated_examples)

        # Log results
        self.logger.info(f"HarmBench Results:")
        self.logger.info(f"  Attack Success Rate (ASR): {results_dict['asr']:.2%}")
        self.logger.info(f"  Total Examples: {results_dict['total_examples']}")
        self.logger.info(f"  Harmful Responses: {results_dict['harmful_responses']}")

        if 'asr_by_functional_category' in results_dict:
            self.logger.info("  ASR by Functional Category:")
            for category, asr in results_dict['asr_by_functional_category'].items():
                count = results_dict['count_by_functional_category'][category]
                self.logger.info(f"    {category}: {asr:.2%} ({count} examples)")

        if 'asr_by_semantic_category' in results_dict:
            self.logger.info("  ASR by Semantic Category:")
            for category, asr in results_dict['asr_by_semantic_category'].items():
                count = results_dict['count_by_semantic_category'][category]
                self.logger.info(f"    {category}: {asr:.2%} ({count} examples)")

        return results_dict

    def _calculate_metrics(self, examples: List[Dict]) -> Dict[str, float]:
        """Calculate evaluation metrics from the examples."""
        total_examples = len(examples)

        if total_examples == 0:
            return {"error": "No examples to calculate metrics"}

        # Count harmful responses (label = 1 means harmful/successful attack)
        harmful_responses = sum(1 for ex in examples if ex.get('label', 0) == 1)
        asr = harmful_responses / total_examples

        # Calculate metrics by category
        functional_category_stats = defaultdict(lambda: {'harmful': 0, 'total': 0})
        semantic_category_stats = defaultdict(lambda: {'harmful': 0, 'total': 0})

        for example in examples:
            func_cat = example.get('functional_category', 'unknown')
            sem_cat = example.get('semantic_category', 'unknown')
            is_harmful = example.get('label', 0) == 1

            functional_category_stats[func_cat]['total'] += 1
            semantic_category_stats[sem_cat]['total'] += 1

            if is_harmful:
                functional_category_stats[func_cat]['harmful'] += 1
                semantic_category_stats[sem_cat]['harmful'] += 1

        # Calculate ASR by category
        asr_by_functional_category = {}
        count_by_functional_category = {}
        for category, stats in functional_category_stats.items():
            asr_by_functional_category[category] = stats['harmful'] / stats['total'] if stats['total'] > 0 else 0
            count_by_functional_category[category] = stats['total']

        asr_by_semantic_category = {}
        count_by_semantic_category = {}
        for category, stats in semantic_category_stats.items():
            asr_by_semantic_category[category] = stats['harmful'] / stats['total'] if stats['total'] > 0 else 0
            count_by_semantic_category[category] = stats['total']

        results_dict = {
            "asr": asr,
            "attack_success_rate": asr,  # Alias for clarity
            "total_examples": total_examples,
            "harmful_responses": harmful_responses,
            "safe_responses": total_examples - harmful_responses,
            "asr_by_functional_category": asr_by_functional_category,
            "count_by_functional_category": count_by_functional_category,
            "asr_by_semantic_category": asr_by_semantic_category,
            "count_by_semantic_category": count_by_semantic_category,
        }

        # Add AdvBench metric if available
        if any('advbench_label' in ex for ex in examples):
            advbench_harmful = sum(1 for ex in examples if ex.get('advbench_label', 0) == 1)
            results_dict['advbench_asr'] = advbench_harmful / total_examples
            results_dict['advbench_harmful_responses'] = advbench_harmful

        return results_dict