from typing import Dict, List, Any, Optional
import logging
from sympy import subsets
import torch
import datasets
from tqdm import tqdm
import json
import pandas as pd
import os
import yaml

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark
from .gen_judgment import judgment as arena_eval_judgement
from .utils.completion import load_model_answers, load_id_to_model_answers
from .show_result import print_leaderboard

class ArenaHardBenchmark(BaseBenchmark):
    """
    ArenaHard benchmark for evaluating language model responses on instruction following.
    """

    def __init__(
        self,
        dataset_name: str = "CohereLabs/m-ArenaHard-v2.0",
        data_dir: str = "eval/chat_benchmarks/ArenaHard/data",
        config_dir: str = "eval/chat_benchmarks/ArenaHard/config",
        subsets: List[str] = ["en", "ko", "ja", "zh"],
        split: str = "test",
        version: str = "2.0",
        judge_model: str = "gpt-4.1",
        baseline_model_name: str = "o3-mini-2025-01-31",
        # eval_model_name: str = "llama-3.1",
        reference_model_name: str = None,
        # split: str = "test",
        max_tokens: Optional[int] = 1024,
        temperature: float = 0.6,
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
        # if reference_model_name:
        #     assert reference_model_name != eval_model_name, "ERROR: one of the models being evaluated is used as reference."
        
        # self.eval_model_name = eval_model_name
        self.dataset_name = dataset_name
        self.subsets = subsets
        self.split = split
        self.data_dir = data_dir
        self.generation_configs = {}
        self.max_tokens = max_tokens if max_tokens is not None else 1024
        self.temperature = temperature
        self.do_sample = do_sample
        self.debug = debug
        self.reference_model_name = reference_model_name
        self.baseline_model_name = baseline_model_name
        self.judge_model = judge_model

        if version == '2.0':
            self.version_file_name = "arena-hard-v2.0"
        elif version == "0.1":
            self.version_file_name = "arena-hard-v0.1"

        # Open yaml file
        with open(os.path.join(config_dir, "api_config.yaml"), "r", encoding="utf-8") as f:
            self.endpoint_settings = yaml.safe_load(f)[judge_model]

        with open(os.path.join(config_dir, self.version_file_name+".yaml"), "r", encoding="utf-8") as f:
            self.eval_config = yaml.safe_load(f)

    
    def load_dataset(self) -> datasets.Dataset:
        """Load the evaluation dataset."""
        try:
            dataset = datasets.load_dataset(self.dataset_name, self.subsets[0], trust_remote_code=True)[self.split]
            for subset in self.subsets[1:]:
                _dataset = datasets.load_dataset(self.dataset_name, subset, trust_remote_code=True)[self.split]
                dataset = datasets.concatenate_datasets([dataset, _dataset])
            dataset = dataset.rename_column("question_id", "uid")
            questions = [dataset[i] for i in range(len(dataset))]
            uids = [q["uid"] for q in questions]
            # question_path = os.path.join(self.data_dir, self.version_file_name, "question.jsonl")
            baseline_path = os.path.join(self.data_dir, self.version_file_name, "model_answer") 
            baselines = load_model_answers(baseline_path)[self.baseline_model_name]
            baselines = [baselines[uid] for uid in uids]
            
            if self.reference_model_name is None: 
                references = [None] * len(questions)
            else:
                reference_path = os.path.join(self.data_dir, self.version_file_name, "model_answer", self.reference_model_name) 
                with open(reference_path, "r", encoding="utf-8") as f:   
                    references = json.load(f)
                references = [r for r in references if r["uid"] in uids]
            # map questions and baselines based on "uid" column

            assert len(questions) == len(baselines) == len(references), "ERROR: number of questions, baselines, and references do not match."
            
            questions = sorted(questions, key=lambda x: x["uid"])
            baselines = sorted(baselines, key=lambda x: x["uid"])
            uids = sorted(uids)
            
            if self.debug:
                questions = questions[:2]
                baselines = baselines[:2]
                references = references[:2]
                self.logger.info(f"Debug mode: using 2 examples")

            dataset = [{"uid": uid, "question": q, "baseline": b, "reference": r} for uid, q, b, r in zip(uids, questions, baselines, references)]
            self.logger.info(f"Loaded {len(dataset)} examples for evaluation")
            return dataset

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
        try:
            eval_set = self.load_dataset()

            all_instances = []
            for idx, example in enumerate(eval_set):
                try:
                    question = example["question"]
                    formatted_instruction = self._prepare_messages([{"role": "user", "content": question}], model)

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                formatted_instruction,
                                {
                                    "max_new_tokens": self.max_tokens,
                                    "do_sample": self.do_sample,
                                    "temperature": self.temperature,
                                },
                            ),
                            idx,
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error preparing instance {idx}: {str(e)}")
                    continue

            with torch.no_grad():
                self.logger.info("Generating responses for ArenaHard...")
                answers = self.compute(model, all_instances)

            if model.rank != 0:
                return None
            
            model_outputs, questions, baselines, references = [], [], [], []
            
            for idx, (example, answer) in enumerate(zip(eval_set, answers)):
                try:
                    question = example["question"]
                    baseline = example["baseline"]
                    
                    instance = {
                        "question": question,
                        "generator": model.model_identifier,
                        "answer": answer,
                    }
                    model_outputs.append(instance)
                    questions.append(question)
                    baselines.append(baseline)
                    references.append(example["reference"])
                    
                except Exception as e:
                    self.logger.error(f"Error processing output {idx}: {str(e)}")
                    continue

            self.logger.info(f"Generated {len(model_outputs)} responses")
            output = {
                "question": questions,
                "baseline": baselines,
                "reference": references,
                "answer": answers, 
                "model_identifier": model.model_identifier,
                }
            
            return output

        except Exception as e:
            self.logger.error(f"Error in generate_responses: {str(e)}")
            raise

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated responses using ArenaHard evaluation metrics.

        Args:
            results: Dictionary containing model outputs and identifier

        Returns:
            Dictionary containing evaluation metrics
        """
        if results is None:
            return None

        eval_model_name = results["model_identifier"].split("=")
        if len(eval_model_name) > 1:
            eval_model_name = eval_model_name[1]
            if "," in eval_model_name:
                eval_model_name = eval_model_name.split(",")[0]
            
        questions = results["question"]
        baselines = results["baseline"]
        references = results["reference"]
        answers = results["answer"]
        # model_identifier = results["model_identifier"]
        uids = [q["uid"] for q in questions]
        if not questions:
            raise ValueError("No model outputs to evaluate")

        self.eval_config.update({
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "judge_model": self.judge_model
        })
        self.logger.info("Running ArenaHard evaluation...")
        outputs = []
        for uid, question, baseline, reference, answer in zip(uids, questions, baselines, references, answers):
            args = {
                    "question": question,
                    "baseline": baseline,
                    "reference": reference,
                    "answer": {"messages": [{"content": {"answer": answer}}], "model": eval_model_name},
                    "settings": self.endpoint_settings,
                    "configs": self.eval_config
            }
            output = arena_eval_judgement(args)
            output["uid"] = uid
            outputs.append(output)

        # metrics = leaderboard[0].loc[model_identifier].to_dict()
        metrics = {}
        output_keys = outputs[0].keys()
        for key in output_keys:
            metrics[key] = [output[key] for output in outputs]

        weight = 3
        metrics_df = pd.DataFrame(metrics)
        categories = metrics_df['category'].unique().tolist()
        null_indices = metrics_df.games.map(lambda x: x[0] is None or x[1] is None or x[0]['score'] is None or x[1]['score'] is None)
        _metrics_df = metrics_df[~null_indices].reset_index(drop=True)
        print(f"Number of null judgments found: {len(metrics_df) - len(_metrics_df)}")
        
        # map label to score
        label_to_score = {
            "A>B": [1],
            "A>>B": [1] * weight,
            "A=B": [0.5],
            "A<<B": [0] * weight,
            "A<B": [0],
            "B>A": [0],
            "B>>A": [0] * weight,
            "B=A": [0.5],
            "B<<A": [1] * weight,
            "B<A": [1],
        }

        _metrics_df['scores'] = _metrics_df.games.map(
            lambda x: label_to_score[x[1]['score']] + [1 - s for s in label_to_score[x[0]['score']]]
        )
        
        battles = _metrics_df[['uid', 'model', 'category', 'scores']].explode('scores').reset_index(drop=True)
        for category in categories:
            scores = print_leaderboard(battles, category).to_dict()
            metrics.update(scores)
        metrics.update({
                "num_examples": len(outputs),
                "outputs": outputs,
                "completion_rate": len(outputs) / len(answers),
            })

        self.logger.info("Evaluation complete")
        return metrics


    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete ArenaHard benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        self.logger.info("Starting ArenaHard benchmark evaluation")
        try:
            generation_results = self.generate_responses(model)

            # If not rank 0, return None early
            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)
            evaluation_results.update(
                {"benchmark_version": "arena_hard_eval", "temperature": self.temperature, "max_tokens": self.max_tokens}
            )
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
