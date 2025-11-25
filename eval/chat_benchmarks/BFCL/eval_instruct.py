import logging
import torch
import datasets
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from eval.task import BaseBenchmark
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from bfcl_eval.utils import *
from bfcl_eval._llm_response_generation import build_handler
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.eval_runner import load_ground_truth_entry, format_sensitivity_runner, multi_turn_runner, agentic_runner, ast_file_runner, _subset_entries_by_model_ids, relevance_file_runner
from collections import defaultdict

class BFCLBenchmark(BaseBenchmark):
    """
    BFCL benchmark for evaluating language model responses on instruction following.
    """

    def __init__(
        self,
        dataset_name: str = "BFCL_v4_multiple",
        data_dir: str = "eval/chat_benchmarks/BFCL/bfcl_eval/data",
        test_category: List[str] = ['all'],  
        max_tokens: Optional[int] = 2048,
        temperature: float = 0.001,
        do_sample: bool = True,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize ArenaHard benchmark.

        Args:
            dataset_name: HuggingFace dataset name
            subset: Dataset subset name
            split: Dataset split to use
            max_tokens: Maximum number of tokens for generation
            temperature: Sampling temperature
            do_sample: Whether to use sampling for generation
            debug: debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)

        self.dataset_name = dataset_name
        self.test_category = test_category
        self.data_dir = data_dir
        self.generation_configs = {}
        self.max_tokens = max_tokens if max_tokens is not None else 1024
        self.temperature = temperature
        self.do_sample = do_sample
        self.debug = debug



    def get_involved_test_entries(self, test_category_args, run_ids):
        test_categories, all_test_entries_involved = [], {}
        if run_ids:
            test_categories, all_test_entries_involved = load_test_entries_from_id_file(
                TEST_IDS_TO_GENERATE_PATH
            )

        else:
            test_categories = parse_test_category_argument(test_category_args)
            for test_category in test_categories:
                all_test_entries_involved[test_category]= load_dataset_entry(test_category)

        return (
            test_categories,
            all_test_entries_involved,
        )


    def load_dataset(self) -> datasets.Dataset:
        """Load the evaluation dataset."""
        try:
            (
                test_categories,
                all_test_entries_involved,
            ) = self.get_involved_test_entries(self.test_category, False)
            if self.debug:
                all_test_entries_involved = {k: v[:2] for k, v in all_test_entries_involved.items()}
                self.logger.info(f"Debug mode: using 2 examples")
            return test_categories, all_test_entries_involved         
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate completions for instructions using the provided model.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing model outputs and identifier
        """
        def _parse_query_response_prompting(model_response: Any) -> dict:
            cleaned_response = model_response
            if "</think>" in model_response:
                parts = model_response.split("</think>")
                cleaned_response = parts[-1].lstrip("\n")

            return cleaned_response
        
        if hasattr(model, "pretrained"):
            model_name = model.pretrained
        else:
            raise ValueError("Model instance does not have 'pretrained' attribute. We should add support for other classes...")
        
        if model_name not in MODEL_CONFIG_MAPPING:
            raise ValueError(
                f"Unknown model_name '{model_name}'.\n"
                "• For officially supported models, please refer to `SUPPORTED_MODELS.md`.\n"
                "• For running new models, please refer to `README.md` and `CONTRIBUTING.md`."
            )
        
        test_categories, all_eval_set = self.load_dataset()
        if any(is_format_sensitivity(test_category) for test_category in test_categories):
            if MODEL_CONFIG_MAPPING[model_name].is_fc_model:
                tqdm.write(
                    "⚠️ Warning: Format sensitivity test cases are only supported for prompting (non-FC) models. "
                    f"Since {model_name} is a FC model based on its config, the format sensitivity test cases will be skipped."
                )
        
        all_instances = defaultdict(list)
        all_answers = {}
        for test_category in test_categories:
            eval_set = all_eval_set[test_category]  
            try:
                handler = build_handler(model_name, self.temperature)
                for idx, example in enumerate(eval_set):
                    try:
                        inference_data = handler._pre_query_processing_prompting(example)
                        inference_data = handler.add_first_turn_message_prompting(inference_data, example["question"][0])
                        inference_data = handler._format_prompt(messages=inference_data['message'], function=inference_data['function'])
                        all_instances[test_category].append(
                            Instance(
                                "generate_until",
                                example,
                                (
                                    inference_data,
                                    {
                                        "max_new_tokens": self.max_tokens,
                                        "do_sample": self.do_sample,
                                        "temperature": self.temperature,
                                    },
                                ),
                                idx,
                            )
                        )
                        inference_data = None
                    except Exception as e:
                        self.logger.error(f"Error preparing instance {idx}: {str(e)}")
                        continue
            except: # This is our model handler
                for idx, example in enumerate(eval_set):
                    question = example["question"][0]
                    messages = []
                    if len(question) > 1:
                        system_prompt = question[0]
                        user_prompt = question[1]
                        messages.extend(
                            [
                            system_prompt,
                            user_prompt,
                            ]
                        )   
                    else:
                        user_prompt = question[0]
                        messages.append(user_prompt)

                    inputs = model.apply_chat_template(messages)
                    all_instances[test_category].append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                inputs,
                                {
                                    "max_new_tokens": self.max_tokens,
                                    "do_sample": False,
                                },
                            ),
                            idx,
                        )
                    )

                raise NotImplementedError("Model handler generation not implemented yet.")
        
            with torch.no_grad():
                self.logger.info("Generating responses for BFCL...")
                answers = self.compute(model, all_instances[test_category])
                preprocessed_answers = []
                for answer in answers:
                    answer = _parse_query_response_prompting(answer)
                    preprocessed_answers.append(answer)
                answers = preprocessed_answers
                all_answers[test_category] = answers
                
        results = {"test_categories": test_categories, "all_answers": all_answers, "model_name": model_name, "handler": handler, "all_eval_set": all_eval_set}
        return results
    
    def evaluate_responses(
            self,
            results: Dict[str, Any]) -> Dict[str, float]: 
        """
        Evaluate the generated responses using ArenaHard evaluation metrics.

        Args:
            results: Dictionary containing model outputs and identifier

        Returns:
            Dictionary containing evaluation metrics
        """
        test_categories = results["test_categories"]
        all_answers = results["all_answers"]
        model_name = results["model_name"]
        handler = results["handler"]
        all_eval_set = results["all_eval_set"]
        total_metrics = {}
        total_acc, total_cnt = 0, 0
        for test_category in test_categories:
            prompt = all_eval_set[test_category]
            model_result = all_answers[test_category]
            model_result = [{"id": p["id"], "result": answer} for p, answer in zip(prompt, model_result)]
            # Find the corresponding possible answer entries
            cur_dir = Path(__file__).resolve().parent
            score_dir = cur_dir / "results"
            if is_relevance_or_irrelevance(test_category):
                prompt, _ = _subset_entries_by_model_ids(
                    model_result, prompt, None, allow_missing=False
                )

                accuracy, total_count = relevance_file_runner(
                    handler, model_result, prompt, model_name, test_category, score_dir
                )

            else:
                possible_answer = load_ground_truth_entry(test_category)
                if self.debug:
                    possible_answer = possible_answer[:2]
                prompt, possible_answer = _subset_entries_by_model_ids(
                    model_result, prompt, possible_answer, allow_missing=False
                )

                if is_format_sensitivity(test_category):
                    accuracy, total_count = format_sensitivity_runner(
                        handler,
                        model_result,
                        prompt,
                        possible_answer,
                        model_name,
                        test_category,
                        score_dir
                    )

                elif is_multi_turn(test_category):
                    accuracy, total_count = multi_turn_runner(
                        handler,
                        model_result,
                        prompt,
                        possible_answer,
                        model_name,
                        test_category,
                        score_dir,
                    )

                elif is_agentic(test_category):
                    accuracy, total_count = agentic_runner(
                        handler,
                        model_result,
                        prompt,
                        possible_answer,
                        model_name,
                        test_category,
                        score_dir,
                    )
                # Single turn test
                else:
                    accuracy, total_count = ast_file_runner(
                        handler,
                        model_result,
                        prompt,
                        possible_answer,
                        test_category,
                        model_name,
                        score_dir,
                    )
            metrics = {
                "accuracy": accuracy,
                "total_count": total_count,
            }
            total_acc += accuracy
            total_cnt += total_count
            total_metrics[test_category] = metrics
        
        total_metrics["mean_accuracy"] = total_acc / len(test_categories)
        total_metrics["total_count"] = total_cnt    
        return total_metrics


    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete ArenaHard benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        self.logger.info("Starting ArenaHard benchmark evaluation")
        evaluation_results = {}
        try:
            # test_categories, all_results, model_name, handler, all_eval_set = self.generate_responses(model)
            generation_results = self.generate_responses(model)

            # If not rank 0, return None early
            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
