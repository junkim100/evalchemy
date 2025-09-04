import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from eval.task import BaseBenchmark
except ImportError:
    # Try alternative import paths
    try:
        from evalchemy.eval.task import BaseBenchmark
    except ImportError:
        from ...task import BaseBenchmark
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

# Original WritingBench domain order (from calculate_scores.py)
DOMAIN1_ORDER = [
    'Academic & Engineering',
    'Finance & Business',
    'Politics & Law',
    'Literature & Arts',
    'Education',
    'Advertising & Marketing'
]

REQUIREMENT_DIMENSION = ["style", "format", "length"]


@dataclass
class WritingBenchConfig:
    """Configuration for WritingBench evaluation."""

    # Dataset configuration
    data_name: str = "writing_bench"
    dataset_version: str = "v1"
    split: str = "test"
    start_idx: int = 0
    end_idx: int = -1

    # Model configuration
    engine: str = None
    model_name: str = None
    max_words_to_eval: int = 1000
    do_sample: bool = False  # 기본값은 False, 사용자가 True로 설정 가능
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.0
    max_new_tokens: int = 2048

    # Evaluation configuration
    evaluator_model: str = "gpt"  # Only GPT evaluator supported
    eval_template: str = None
    batch_mode: bool = True
    api_parallel: int = 8

    # GPT configuration (only evaluator)
    gpt_model: str = "gpt-4o-mini"
    gpt_api_key: str = ""
    gpt_api_base: str = ""

    # Evaluation parameters
    eval_temperature: float = 1.0
    eval_top_p: float = 0.95
    eval_max_tokens: int = 2048

    # Domain weights for scoring
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        "Academic & Engineering": 1.0,
        "Finance & Business": 1.0,
        "Politics & Law": 1.0,
        "Literature & Arts": 1.0,
        "Education": 1.0,
        "Advertising & Marketing": 1.0,
    })

    def __post_init__(self):
        if self.eval_template is None:
            # Use the original WritingBench evaluation prompt
            self.eval_template = "eval/chat_benchmarks/WritingBench/prompt.py"


class WritingBenchBenchmark(BaseBenchmark):
    """
    WritingBench benchmark for evaluating LLMs' writing capabilities across diverse real-world tasks.

    This benchmark evaluates models on 1,000 writing queries spanning 6 primary domains and 100 subdomains,
    with instance-specific criteria for comprehensive assessment.
    """

    REQUIRES_OPENAI_ANNOTATOR = False  # We use critic model instead

    def __init__(
        self,
        config: Optional[WritingBenchConfig] = None,
        debug: bool = False,
        max_tokens: int = 16000,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize WritingBench benchmark.

        Args:
            config: WritingBench configuration
            debug: If True, run in debug mode on 10 samples
            max_tokens: Maximum tokens for generation
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.config = config or WritingBenchConfig()
        self.debug = debug
        self.max_new_tokens = max_tokens

        # Override config max_new_tokens if provided
        if max_tokens != 16000:
            self.config.max_new_tokens = max_tokens

        self.logger.info(f"Initialized WritingBench with config: {self.config}")

        # Initialize evaluator (lazy initialization)
        self.evaluator = None
        self._evaluator_initialized = False

    def _init_evaluator(self):
        """Initialize the evaluator based on configuration (lazy initialization)."""
        if self._evaluator_initialized:
            return

        try:
            if self.config.evaluator_model == "gpt":
                from .evaluator import GPTEvaluator
                self.evaluator = GPTEvaluator(
                    model_name=self.config.gpt_model,
                    api_key=self.config.gpt_api_key or None,
                    api_base=self.config.gpt_api_base or None,
                    logger=self.logger
                )
            else:
                raise ValueError(f"Only 'gpt' evaluator is supported. Got: {self.config.evaluator_model}")

            self._evaluator_initialized = True
            self.logger.info(f"Initialized {self.config.evaluator_model} evaluator")
        except Exception as e:
            self.logger.error(f"Failed to initialize GPT evaluator: {str(e)}")
            # GPT evaluator is required for WritingBench evaluation
            raise RuntimeError(f"WritingBench requires GPT evaluator for evaluation: {str(e)}")

    def load_dataset(self) -> Tuple[List[str], List[Dict], List[str], Dict[str, List[Any]]]:
        """Load and preprocess the WritingBench dataset."""
        try:
            # Load dataset from local JSONL file
            dataset_path = os.path.join(
                os.path.dirname(__file__),
                "data", "benchmark_all.jsonl"
            )

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"WritingBench dataset not found at {dataset_path}")

            # Load data from JSONL file
            dataset = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        dataset.append(json.loads(line))

            if self.debug:
                dataset = dataset[:min(10, len(dataset))]
                self.logger.info(f"Debug mode: using {len(dataset)} examples")

            # Initialize data structures
            id_strs = []
            queries = []
            extracted_queries = []
            metadata = {
                "domain1": [],
                "domain2": [],
                "lang": [],
                "checklist": []
            }

            # Process each item
            for item in dataset:
                id_strs.append(str(item["index"]))
                queries.append(item["query"])
                extracted_queries.append(item["query"])

                # Store metadata
                for key in metadata:
                    if key in item:
                        metadata[key].append(item[key])
                    else:
                        metadata[key].append(None)

            # Apply index limits
            if self.config.end_idx < 0:
                self.config.end_idx = len(id_strs)

            slice_range = slice(self.config.start_idx, self.config.end_idx)
            return (
                id_strs[slice_range],
                queries[slice_range],
                extracted_queries[slice_range],
                {k: v[slice_range] for k, v in metadata.items()},
            )

        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise

    def _prepare_messages(self, query: str, model: Optional[LM] = None) -> Union[List[Dict[str, str]], str]:
        """
        Prepare messages for the writing task.

        Args:
            query: The writing query/prompt
            model: Optional language model instance for applying chat template

        Returns:
            If model is provided, returns the templated string. Otherwise returns the prepared message list.
        """
        messages = [
            {"role": "user", "content": query}
        ]

        # Add system instruction if available
        if self.system_instruction:
            messages.insert(0, {"role": "system", "content": self.system_instruction})

        if model is not None:
            # Apply chat template and ensure it returns a string
            templated = model.tokenizer.apply_chat_template(messages, tokenize=False)
            if isinstance(templated, list):
                return templated[0] if templated else ""
            return templated if isinstance(templated, str) else str(templated)

        return messages

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses for WritingBench tasks.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing generation results with filepath and temp directory
        """
        try:
            # Load dataset
            id_strs, queries, extracted_queries, metadata = self.load_dataset()

            self.logger.info(f"Loaded {len(id_strs)} WritingBench queries")

            # Prepare model inputs
            model_inputs = [self._prepare_messages(query, model) for query in extracted_queries]

            # Generate responses
            all_instances = [
                Instance(
                    "generate_until",
                    None,
                    (
                        inputs,
                        {
                            "max_new_tokens": self.config.max_new_tokens,
                            "do_sample": self.config.do_sample,
                            "top_p": self.config.top_p if self.config.do_sample else 1.0,
                            "top_k": self.config.top_k if self.config.do_sample else -1,
                            "temperature": self.config.temperature if self.config.do_sample else 0.0,
                            "repetition_penalty": self.config.repetition_penalty,
                        },
                    ),
                    idx,
                )
                for idx, inputs in enumerate(model_inputs)
            ]

            self.logger.info("Generating responses for WritingBench...")
            outputs = self.compute(model, all_instances)

            # Return None early for non-primary ranks
            if model.rank != 0:
                return None

            # Prepare responses data (HLE pattern - no file saving needed)
            responses_data = []
            for i, (id_str, output, query) in enumerate(zip(id_strs, outputs, queries)):
                response_item = {
                    "index": int(id_str),
                    "response": output,
                    "query": query,
                    "domain1": metadata["domain1"][i] if i < len(metadata["domain1"]) else None,  # 대분류
                    "domain2": metadata["domain2"][i] if i < len(metadata["domain2"]) else None,  # 소분류
                    "domain": metadata["domain1"][i] if i < len(metadata["domain1"]) else None,  # 호환성을 위해 유지
                    "subdomain": metadata["domain2"][i] if i < len(metadata["domain2"]) else None,  # 호환성을 위해 유지
                    "lang": metadata["lang"][i] if i < len(metadata["lang"]) else None,
                    "criteria": metadata["checklist"][i] if i < len(metadata["checklist"]) else None,
                    "model_name": getattr(model, 'model_name', 'unknown'),
                    "timestamp": str(time.time())
                }
                responses_data.append(response_item)

            # Return data directly (HLE pattern)
            return {"examples": responses_data}

        except Exception as e:
            self.logger.error(f"Error in generate_responses: {str(e)}")
            raise



    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate generated responses using the configured evaluator.

        Args:
            results: Dictionary containing generation results

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        try:
            # Get responses data directly from results (HLE pattern)
            responses_data = results.get("examples", [])

            self.logger.info(f"DEBUG: responses_data type: {type(responses_data)}, length: {len(responses_data)}")
            if responses_data:
                sample = responses_data[0]
                self.logger.info(f"DEBUG: Sample response_data keys: {sample.keys()}")
                self.logger.info(f"DEBUG: Sample domain1: {sample.get('domain1')}, domain2: {sample.get('domain2')}")

            if not responses_data:
                self.logger.error("ERROR: No response data found in results")
                raise ValueError("No response data found in results")

            self.logger.info(f"Evaluating {len(responses_data)} responses")

            # Load original dataset with criteria
            dataset_path = os.path.join(
                os.path.dirname(__file__),
                "data", "benchmark_all.jsonl"
            )

            criteria_map = {}
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        criteria_map[item["index"]] = item.get("checklist", [])

            # Initialize evaluator if not already done
            if not self._evaluator_initialized:
                self._init_evaluator()

            # Evaluate each response
            all_scores = []
            detailed_scores = {}

            # GPT evaluator is required for evaluation
            if self.evaluator is None:
                raise RuntimeError("GPT evaluator is required for WritingBench evaluation")

            # Use batch evaluation if available
            if hasattr(self.evaluator, 'evaluate_batch'):
                # 배치 평가 사용
                batch_evaluations = []
                eval_metadata = []  # 평가 메타데이터 저장

                for response_data in responses_data:
                    index = response_data["index"]
                    query = response_data["query"]
                    response = response_data["response"]
                    criteria_list = criteria_map.get(index, [])

                    if not criteria_list:
                        self.logger.warning(f"No criteria found for index {index}")
                        continue

                    for criterion in criteria_list:
                        batch_evaluations.append({
                            "query": query,
                            "response": response,
                            "criteria": criterion
                        })
                        eval_metadata.append({
                            "index": index,
                            "criterion_name": criterion["name"]
                        })

                # 배치로 평가 실행
                try:
                    batch_results = self.evaluator.evaluate_batch(
                        batch_evaluations,
                        top_p=self.config.eval_top_p,
                        temperature=self.config.eval_temperature,
                        max_tokens=self.config.eval_max_tokens
                    )

                    # 결과 정리
                    response_scores_map = {}
                    for i, (result, metadata) in enumerate(zip(batch_results, eval_metadata)):
                        index = metadata["index"]
                        criterion_name = metadata["criterion_name"]

                        if index not in detailed_scores:
                            detailed_scores[index] = {}
                            response_scores_map[index] = []

                        score = result["score"]
                        response_scores_map[index].append(score)
                        detailed_scores[index][criterion_name] = {
                            "score": score,
                            "reason": result["reason"]
                        }

                    # 평균 점수 계산
                    for index, scores in response_scores_map.items():
                        if scores:
                            avg_score = sum(scores) / len(scores)
                            all_scores.append(avg_score)

                except Exception as e:
                    self.logger.error(f"Batch evaluation failed: {str(e)}")
                    raise RuntimeError(f"WritingBench batch evaluation failed: {str(e)}")

            # Individual evaluation when batch is not available
            else:
                for response_data in responses_data:
                    index = response_data["index"]
                    query = response_data["query"]
                    response = response_data["response"]
                    criteria_list = criteria_map.get(index, [])

                    if not criteria_list:
                        self.logger.warning(f"No criteria found for index {index}")
                        continue

                    # Evaluate against each criterion
                    response_scores = []
                    detailed_scores[index] = {}

                    for criterion in criteria_list:
                        try:
                            # GPT evaluator is required
                            if self.evaluator is None:
                                raise RuntimeError("GPT evaluator is required for WritingBench evaluation")

                            eval_result = self.evaluator.evaluate(
                                query=query,
                                response=response,
                                criteria=criterion,
                                top_p=self.config.eval_top_p,
                                temperature=self.config.eval_temperature,
                                max_tokens=self.config.eval_max_tokens
                            )

                            score = eval_result["score"]
                            response_scores.append(score)
                            detailed_scores[index][criterion["name"]] = {
                                "score": score,
                                "reason": eval_result["reason"]
                            }

                        except Exception as e:
                            self.logger.error(f"Error evaluating criterion {criterion['name']} for index {index}: {str(e)}")
                            response_scores.append(1)  # Default low score for failed evaluations
                            detailed_scores[index][criterion["name"]] = {
                                "score": 1,
                                "reason": f"Evaluation failed: {str(e)}"
                            }

                    # Calculate average score for this response
                    if response_scores:
                        avg_score = sum(response_scores) / len(response_scores)
                        all_scores.append(avg_score)

            # Calculate overall metrics
            overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

            # Calculate domain-specific scores
            self.logger.info(f"About to calculate domain scores with {len(responses_data)} responses and {len(detailed_scores)} detailed scores")
            domain_scores = self._calculate_domain_scores(responses_data, detailed_scores)
            self.logger.info(f"Domain scores calculated: {domain_scores}")

            # Organize domain scores by hierarchy (like original WritingBench)
            organized_domain_scores = self._organize_domain_scores(domain_scores)
            self.logger.info(f"Organized domain scores: {organized_domain_scores}")

            # Update results with evaluation data (like HLE pattern)
            results.update({
                "overall_score": overall_score,
                "num_evaluated": len(all_scores),
                "benchmark_version": "writing_bench_v1",
                "detailed_scores": detailed_scores,
                "evaluator_model": self.config.evaluator_model,
                "evaluation_config": {
                    "gpt_model": self.config.gpt_model,
                    "temperature": self.config.eval_temperature,
                    "max_tokens": self.config.eval_max_tokens
                },
                # Include both flat and organized domain scores
                **domain_scores,
                "domain_performance": organized_domain_scores
            })

            self.logger.info(f"WritingBench evaluation completed. Overall score: {overall_score:.3f}")
            return results  # Return the complete results dict with input/output data

        except Exception as e:
            self.logger.error(f"Error in evaluate_responses: {str(e)}")
            return {"error": str(e)}



    def _calculate_domain_scores(self, responses_data: List[Dict], detailed_scores: Dict) -> Dict[str, float]:
        """Calculate domain-specific scores following original WritingBench methodology."""
        from collections import defaultdict

        self.logger.info(f"Calculating domain scores for {len(responses_data)} responses")
        self.logger.info(f"Detailed scores available for {len(detailed_scores)} items")

        # Debug: Show detailed_scores keys and responses_data sample
        self.logger.info(f"DEBUG: detailed_scores keys: {list(detailed_scores.keys())}")
        if responses_data:
            sample_response = responses_data[0]
            self.logger.info(f"DEBUG: Sample response_data keys: {sample_response.keys()}")
            self.logger.info(f"DEBUG: Sample response index: {sample_response.get('index')}, domain1: {sample_response.get('domain1')}, domain2: {sample_response.get('domain2')}")

        # Initialize score accumulators (like original calculate_scores.py)
        domain1_scores_sum = defaultdict(lambda: 0)
        domain1_count = defaultdict(lambda: 0)
        domain2_scores_sum = defaultdict(lambda: 0)
        domain2_count = defaultdict(lambda: 0)

        processed_count = 0
        matched_count = 0

        for response_data in responses_data:
            processed_count += 1
            index = response_data["index"]
            domain1 = response_data.get("domain1")  # 대분류
            domain2 = response_data.get("domain2")  # 소분류

            self.logger.info(f"Processing index {index}: domain1={domain1}, domain2={domain2}")

            # Try both string and integer keys for detailed_scores
            index_key = None
            if str(index) in detailed_scores:
                index_key = str(index)
            elif index in detailed_scores:
                index_key = index

            if index_key is not None:
                matched_count += 1
                # Calculate average score for this response (like original)
                criteria_scores = detailed_scores[index_key]
                self.logger.info(f"Found detailed scores for index {index}: {list(criteria_scores.keys()) if criteria_scores else 'None'}")

                if criteria_scores:
                    # Extract scores from different possible structures
                    scores = []
                    for criteria_name, criteria_data in criteria_scores.items():
                        if isinstance(criteria_data, dict) and "score" in criteria_data:
                            scores.append(criteria_data["score"])
                            self.logger.debug(f"Extracted score {criteria_data['score']} from criteria '{criteria_name}'")
                        elif isinstance(criteria_data, (int, float)):
                            scores.append(criteria_data)
                            self.logger.debug(f"Extracted direct score {criteria_data} from criteria '{criteria_name}'")

                    if scores:
                        avg_score = sum(scores) / len(scores)
                        self.logger.info(f"Index {index}: avg_score={avg_score:.2f}, domain1={domain1}, domain2={domain2}")

                        # Aggregate by domain1 (대분류)
                        if domain1 and domain1 != "Unknown":
                            domain1_scores_sum[domain1] += avg_score
                            domain1_count[domain1] += 1
                            self.logger.info(f"Added to domain1 '{domain1}': score={avg_score:.2f}")
                        else:
                            self.logger.warning(f"Skipped domain1 (domain1='{domain1}')")

                        # Aggregate by domain2 (소분류)
                        if domain2 and domain2 != "Unknown":
                            domain2_scores_sum[domain2] += avg_score
                            domain2_count[domain2] += 1
                            self.logger.info(f"Added to domain2 '{domain2}': score={avg_score:.2f}")
                        else:
                            self.logger.warning(f"Skipped domain2 (domain2='{domain2}')")
                    else:
                        self.logger.warning(f"No valid scores found for index {index}: {criteria_scores}")
                else:
                    self.logger.warning(f"No criteria scores found for index {index}")
            else:
                self.logger.warning(f"No detailed scores found for index {index} (tried keys: '{index}', '{str(index)}')")

        # Calculate averages (like original calculate_scores.py)
        domain1_avg_scores = {domain: domain1_scores_sum[domain] / domain1_count[domain]
                             for domain in domain1_scores_sum if domain1_count[domain] > 0}

        domain2_avg_scores = {domain: domain2_scores_sum[domain] / domain2_count[domain]
                             for domain in domain2_scores_sum if domain2_count[domain] > 0}

        # Combine results with prefixes (like original aggregate_scores)
        result = {}
        for domain, avg_score in domain1_avg_scores.items():
            result[f'Domain1_{domain}'] = avg_score

        for domain, avg_score in domain2_avg_scores.items():
            result[f'Domain2_{domain}'] = avg_score

        self.logger.info(f"Domain scores calculated: {len(result)} total scores")
        self.logger.info(f"Domain1 categories: {list(domain1_avg_scores.keys())}")
        self.logger.info(f"Domain2 categories: {list(domain2_avg_scores.keys())}")

        # Debug: Show the actual counts
        self.logger.info(f"DEBUG: Processed {processed_count} responses, matched {matched_count} with detailed_scores")
        self.logger.info(f"DEBUG: domain1_count: {dict(domain1_count)}")
        self.logger.info(f"DEBUG: domain2_count: {dict(domain2_count)}")
        self.logger.info(f"DEBUG: domain1_scores_sum: {dict(domain1_scores_sum)}")
        self.logger.info(f"DEBUG: domain2_scores_sum: {dict(domain2_scores_sum)}")

        return result

    def _organize_domain_scores(self, domain_scores: Dict[str, float]) -> Dict[str, Any]:
        """Organize domain scores by hierarchy following original WritingBench structure."""
        organized = {
            "domain1_scores": {},  # 대분류별 점수
            "domain2_scores": {},  # 소분류별 점수
            "domain_hierarchy": {},  # 대분류 -> 소분류 구조
            "domain_summary": {}  # 요약 정보
        }

        # Extract domain1 scores (대분류)
        for key, score in domain_scores.items():
            if key.startswith("Domain1_"):
                domain_name = key[8:]  # Remove "Domain1_" prefix
                organized["domain1_scores"][domain_name] = score

        # Extract domain2 scores (소분류)
        for key, score in domain_scores.items():
            if key.startswith("Domain2_"):
                domain_name = key[8:]  # Remove "Domain2_" prefix
                organized["domain2_scores"][domain_name] = score

        # Create hierarchy structure (like original WritingBench)
        # Build domain2 to domain1 mapping from actual data
        domain2_to_domain1 = self._build_domain_mapping()

        for domain1 in DOMAIN1_ORDER:
            if domain1 in organized["domain1_scores"]:
                organized["domain_hierarchy"][domain1] = {
                    "score": organized["domain1_scores"][domain1],
                    "subdomains": {}
                }

                # Find subdomains that belong to this domain1
                for domain2, score in organized["domain2_scores"].items():
                    if domain2_to_domain1.get(domain2) == domain1:
                        organized["domain_hierarchy"][domain1]["subdomains"][domain2] = score

        # Sort domain2 scores by performance (like original)
        organized["domain2_scores_sorted"] = dict(
            sorted(organized["domain2_scores"].items(), key=lambda x: x[1], reverse=True)
        )

        # Create summary
        organized["domain_summary"] = {
            "total_domain1_categories": len(organized["domain1_scores"]),
            "total_domain2_categories": len(organized["domain2_scores"]),
            "best_performing_domain1": max(organized["domain1_scores"].items(), key=lambda x: x[1]) if organized["domain1_scores"] else None,
            "best_performing_domain2": max(organized["domain2_scores"].items(), key=lambda x: x[1]) if organized["domain2_scores"] else None
        }

        return organized

    def _build_domain_mapping(self) -> Dict[str, str]:
        """Build mapping from domain2 (소분류) to domain1 (대분류) from dataset."""
        domain2_to_domain1 = {}

        try:
            # Load dataset to build mapping
            dataset_path = os.path.join(
                os.path.dirname(__file__),
                "data", "benchmark_all.jsonl"
            )

            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        domain1 = data.get("domain1")
                        domain2 = data.get("domain2")

                        if domain1 and domain2:
                            domain2_to_domain1[domain2] = domain1

            self.logger.info(f"Built domain mapping: {len(domain2_to_domain1)} domain2 -> domain1 mappings")

        except Exception as e:
            self.logger.warning(f"Could not build domain mapping: {str(e)}")

        return domain2_to_domain1

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete WritingBench evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation results, or None for non-primary ranks
        """
        self.logger.info("Starting WritingBench evaluation")
        try:
            generation_results = self.generate_responses(model)

            # If not rank 0, return None early
            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)

            # Add configuration info to results
            if evaluation_results and "error" not in evaluation_results:
                evaluation_results.update({
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_new_tokens,
                    "evaluator": self.config.evaluator_model,
                })

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running WritingBench benchmark: {str(e)}")
            return {"error": str(e)}