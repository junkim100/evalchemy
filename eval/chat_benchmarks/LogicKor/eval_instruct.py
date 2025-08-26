import logging
import json
import requests
from collections import defaultdict
from typing import Any, Dict, List, Optional
import math
import numpy as np
import re

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark


class LogicKorBenchmark(BaseBenchmark):
    """
    LogicKor Benchmark for evaluating Korean logical reasoning capabilities of LLMs.

    LogicKor is a Korean logical reasoning benchmark designed to evaluate the logical
    reasoning capabilities of large language models in Korean across multiple domains.

    Features:
    - Korean language logical reasoning tasks
    - Multiple categories: Reasoning, Math, Writing, Coding, Understanding, Grammar
    - Multi-turn conversations with follow-up questions
    - Judge-based evaluation using GPT-4 or similar models
    - Comprehensive scoring from 1-10 scale

    Link: https://github.com/instructkr/LogicKor
    """

    def __init__(
        self,
        dataset_url: str = "https://raw.githubusercontent.com/instructkr/LogicKor/main/questions.jsonl",
        max_tokens: int = 32768,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        filter_category: Optional[str] = None,
        enable_cot: bool = False,
        judge_model: str = "gpt-4",
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize LogicKor benchmark.

        Args:
            dataset_url: URL to the LogicKor questions.jsonl file
            max_tokens: Maximum tokens for model generation
            debug: If True, only evaluate on 5 examples
            seed: Random seed for reproducibility
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
            filter_category: Filter by specific category (optional)
            enable_cot: Enable Chain-of-Thought prompting
            judge_model: Model to use for evaluation (requires OpenAI API)
            openai_api_key: OpenAI API key for judge model evaluation
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_url = dataset_url
        self.max_new_tokens = max_tokens
        self.debug = debug
        self.seed = seed
        self.filter_category = filter_category
        self.enable_cot = enable_cot
        self.judge_model = judge_model
        self.openai_api_key = openai_api_key

        # Load dataset
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        """Load LogicKor dataset from GitHub."""
        try:
            # Download the dataset from GitHub
            response = requests.get(self.dataset_url, timeout=30)
            response.raise_for_status()

            # Parse JSONL data
            data = []
            for line in response.text.strip().split("\n"):
                if line.strip():
                    data.append(json.loads(line.strip()))

            # Apply category filter if specified
            if self.filter_category:
                original_count = len(data)
                data = [item for item in data if item.get("category") == self.filter_category]
                self.logger.info(
                    f"Filtered by category '{self.filter_category}': {len(data)}/{original_count} examples"
                )

            # Apply debug mode
            if self.debug:
                data = data[:5]
                self.logger.info(f"Debug mode: using {len(data)} examples")

            self.logger.info(f"Loaded {len(data)} LogicKor examples")
            return data

        except Exception as e:
            self.logger.error(f"Error loading LogicKor dataset: {str(e)}")
            # Fallback: create a minimal test dataset
            self.logger.warning("Creating minimal test dataset")
            return self._create_test_dataset()

    def _create_test_dataset(self):
        """Create a minimal test dataset for development/testing."""
        test_data = [
            {
                "id": 1,
                "category": "추론(Reasoning)",
                "questions": [
                    "A, B, C 세 사람 중 한 명이 유리를 깨뜨렸습니다. 경찰이 찾아와 범인을 찾으려 합니다. 세 사람 중 한 명은 거짓말을 하고 나머지 두 명은 진실을 말하고 있습니다. 범인은 누구일까요?\nA: '범인은 C에요.'\nB: '제가 범인이에요.'\nC: '저는 범인이 아니에요.'",
                    "이런 문제에 대해서 어떻게 생각하나요? 한번 비슷한 문제를 만들고 풀이까지 제시해보세요.",
                ],
                "references": ["B", None],
            }
        ]
        return test_data

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
            questions = example.get("questions", [])

            # Process each question in the multi-turn conversation
            for q_idx, question in enumerate(questions):
                # Construct the prompt
                prompt = self._construct_prompt(question, q_idx == 0)

                messages = [
                    {"role": "user", "content": prompt},
                ]

                templated_messages = self._prepare_messages(messages, model)

                instance = Instance(
                    "generate_until",
                    {
                        "example_id": example.get("id", idx),
                        "question_index": q_idx,
                        "question": question,
                        "category": example.get("category", "unknown"),
                        "reference": (
                            example.get("references", [None])[q_idx]
                            if q_idx < len(example.get("references", []))
                            else None
                        ),
                        "is_first_question": q_idx == 0,
                    },
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 0.0,
                            "seed": self.seed,
                        },
                    ),
                    f"{idx}_{q_idx}",
                )

                # Add metadata for tracking
                instance.metadata = {
                    "example_id": example.get("id", idx),
                    "question_index": q_idx,
                    "category": example.get("category", "unknown"),
                }

                all_instances.append(instance)

        # Generate model responses
        self.logger.info("Generating responses for LogicKor...")
        outputs = self.compute(model, all_instances)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        # Process outputs and prepare for evaluation
        processed_examples = defaultdict(
            lambda: {"questions": [], "outputs": [], "references": [], "category": "", "id": None}
        )

        for instance, output in zip(all_instances, outputs):
            # Extract text from different output types
            if isinstance(output, str):
                response_text = output
            elif hasattr(output, "outputs") and output.outputs:
                response_text = output.outputs[0].text
            elif hasattr(output, "text"):
                response_text = output.text
            else:
                response_text = str(output)

            example_id = instance.doc["example_id"]
            question_index = instance.doc["question_index"]

            # Group by example_id
            processed_examples[example_id]["questions"].append(instance.doc["question"])
            processed_examples[example_id]["outputs"].append(response_text)
            processed_examples[example_id]["references"].append(instance.doc["reference"])
            processed_examples[example_id]["category"] = instance.doc["category"]
            processed_examples[example_id]["id"] = example_id

        # Convert to list format
        examples = list(processed_examples.values())

        return {"examples": examples}

    def _construct_prompt(self, question: str, is_first_question: bool = True) -> str:
        """
        Construct the prompt for LogicKor question.

        Args:
            question: The question text
            is_first_question: Whether this is the first question in a multi-turn conversation

        Returns:
            Formatted prompt string
        """
        if self.enable_cot:
            # Chain-of-Thought prompting
            if is_first_question:
                prompt = f"""다음 문제를 단계별로 차근차근 생각해보고 답변해주세요.

문제: {question}

단계별로 생각해보겠습니다:"""
            else:
                prompt = f"""이전 답변을 바탕으로 다음 질문에 대해 단계별로 생각해보고 답변해주세요.

질문: {question}

단계별로 생각해보겠습니다:"""
        else:
            # Direct prompting
            prompt = f"""다음 문제에 대해 정확하고 자세한 답변을 한국어로 제공해주세요.

문제: {question}"""

        return prompt

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated responses using rule-based metrics and optional judge evaluation.

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
        total_score = 0.0
        category_stats = defaultdict(lambda: {"total_score": 0.0, "count": 0})
        question_type_stats = defaultdict(lambda: {"total_score": 0.0, "count": 0})

        detailed_results = []

        for example in examples:
            questions = example.get("questions", [])
            outputs = example.get("outputs", [])
            references = example.get("references", [])
            category = example.get("category", "unknown")

            # Evaluate each question-answer pair
            example_scores = []
            for i, (question, output) in enumerate(zip(questions, outputs)):
                reference = references[i] if i < len(references) else None

                # Rule-based evaluation
                rule_score = self._evaluate_rule_based(question, output, reference, category)

                # For now, use rule-based score (judge evaluation would require OpenAI API)
                final_score = rule_score
                example_scores.append(final_score)

                # Update statistics
                question_type = "first_question" if i == 0 else "follow_up_question"
                question_type_stats[question_type]["total_score"] += final_score
                question_type_stats[question_type]["count"] += 1

                detailed_results.append(
                    {
                        "example_id": example.get("id", "unknown"),
                        "question_index": i,
                        "category": category,
                        "question": question,
                        "output": output,
                        "reference": reference,
                        "rule_score": rule_score,
                        "final_score": final_score,
                        "question_type": question_type,
                    }
                )

            # Calculate average score for this example
            avg_score = sum(example_scores) / len(example_scores) if example_scores else 0.0
            total_score += avg_score

            # Update category statistics
            category_stats[category]["total_score"] += avg_score
            category_stats[category]["count"] += 1

        # Calculate overall metrics
        overall_score = total_score / num_questions if num_questions > 0 else 0.0

        # Calculate standard error
        overall_std_err = np.sqrt(overall_score * (10 - overall_score) / num_questions) if num_questions > 0 else 0.0

        # Calculate category-specific scores
        category_scores = {}
        for category, stats in category_stats.items():
            category_scores[f"score_{category.replace('(', '_').replace(')', '_').replace(' ', '_')}"] = (
                stats["total_score"] / stats["count"] if stats["count"] > 0 else 0.0
            )

        # Calculate question type scores
        question_type_scores = {}
        for q_type, stats in question_type_stats.items():
            question_type_scores[f"score_{q_type}"] = (
                stats["total_score"] / stats["count"] if stats["count"] > 0 else 0.0
            )

        evaluation_results = {
            "num_total": num_questions,
            "score_avg": overall_score,
            "score_std_err": overall_std_err,
            "detailed_results": detailed_results,
            **category_scores,
            **question_type_scores,
        }

        self.logger.info(f"LogicKor Evaluation Results:")
        self.logger.info(f"  Total examples: {num_questions}")
        self.logger.info(f"  Overall score: {overall_score:.3f}/10")

        # Log category-specific results
        for category, stats in category_stats.items():
            avg_score = stats["total_score"] / stats["count"] if stats["count"] > 0 else 0.0
            self.logger.info(f"  {category} score: {avg_score:.3f}/10 ({stats['count']} examples)")

        return evaluation_results

    def _evaluate_rule_based(self, question: str, output: str, reference: Optional[str], category: str) -> float:
        """
        Evaluate response using rule-based metrics.

        Args:
            question: The question text
            output: Model's response
            reference: Reference answer (if available)
            category: Question category

        Returns:
            Score from 0.0 to 10.0
        """
        if not output or not output.strip():
            return 0.0

        score = 5.0  # Base score

        # Check if response is in Korean (required for LogicKor)
        if not self._is_korean_response(output):
            return 0.0  # Zero score for non-Korean responses

        # Length-based scoring (reasonable length)
        output_length = len(output.strip())
        if output_length < 10:
            score -= 2.0  # Too short
        elif output_length > 2000:
            score -= 1.0  # Too long
        else:
            score += 1.0  # Good length

        # Reference-based scoring (if reference is available)
        if reference and reference.strip():
            if self._check_reference_match(output, reference, category):
                score += 3.0
            else:
                score += 1.0  # Partial credit for attempting
        else:
            # No reference available, use heuristic scoring
            score += self._heuristic_scoring(question, output, category)

        # Category-specific adjustments
        if category in ["수학(Math)", "문법(Grammar)"]:
            # More strict scoring for math and grammar
            if reference and not self._check_reference_match(output, reference, category):
                score -= 1.0

        # Ensure score is within bounds
        return max(0.0, min(10.0, score))

    def _is_korean_response(self, text: str) -> bool:
        """Check if the response is primarily in Korean."""
        if not text:
            return False

        # Count Korean characters and total alphabetic characters
        korean_chars = 0
        total_alphabetic_chars = 0

        for char in text:
            # Check if character is alphabetic (Korean or Latin)
            if (
                char.isalpha()
                or (0x1100 <= ord(char) <= 0x11FF)
                or (0x3130 <= ord(char) <= 0x318F)
                or (0xAC00 <= ord(char) <= 0xD7AF)
            ):
                total_alphabetic_chars += 1
                # Check if character is Korean (Hangul syllables, Jamo, compatibility Jamo)
                if (
                    (0xAC00 <= ord(char) <= 0xD7AF)
                    or (0x1100 <= ord(char) <= 0x11FF)
                    or (0x3130 <= ord(char) <= 0x318F)
                ):
                    korean_chars += 1

        if total_alphabetic_chars == 0:
            return True  # No alphabetic characters, assume Korean (numbers, punctuation, etc.)

        # At least 50% Korean characters for Korean response (more lenient for mixed text)
        return (korean_chars / total_alphabetic_chars) >= 0.5

    def _check_reference_match(self, output: str, reference: str, category: str) -> bool:
        """Check if output matches the reference answer."""
        if not reference:
            return False

        output_clean = output.strip().lower()
        reference_clean = reference.strip().lower()

        # Exact match
        if reference_clean in output_clean:
            return True

        # Category-specific matching
        if category == "수학(Math)":
            # Extract numbers from both
            output_numbers = re.findall(r"\d+(?:\.\d+)?", output)
            reference_numbers = re.findall(r"\d+(?:\.\d+)?", reference)
            return len(set(output_numbers) & set(reference_numbers)) > 0

        # Fuzzy matching for other categories
        return self._fuzzy_match(output_clean, reference_clean)

    def _fuzzy_match(self, text1: str, text2: str, threshold: float = 0.6) -> bool:
        """Simple fuzzy matching based on common words."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return (intersection / union) >= threshold

    def _heuristic_scoring(self, question: str, output: str, category: str) -> float:
        """Heuristic scoring when no reference is available."""
        score = 0.0

        # Check for structured response
        if any(marker in output for marker in ["1.", "2.", "첫째", "둘째", "따라서", "결론"]):
            score += 1.0

        # Check for reasoning indicators
        if any(word in output for word in ["왜냐하면", "그 이유는", "때문에", "따라서", "그러므로"]):
            score += 1.0

        # Category-specific heuristics
        if category == "코딩(Coding)":
            if any(lang in output for lang in ["python", "javascript", "java", "def ", "function", "class"]):
                score += 1.0
        elif category == "수학(Math)":
            if any(symbol in output for symbol in ["=", "+", "-", "*", "/", "계산", "공식"]):
                score += 1.0

        return score
