"""
TauBench evaluation benchmark for evalchemy.

This module integrates the TauBench benchmark (https://github.com/sierra-research/tau-bench)
into the evalchemy evaluation framework. TauBench evaluates tool-agent-user interaction
in real-world domains (airline and retail).

Note: This implementation requires the tau-bench package to be installed:
    pip install tau-bench
"""

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from lm_eval.api.model import LM
except ImportError:
    LM = None

try:
    from eval.task import BaseBenchmark
except ImportError:
    BaseBenchmark = None

# Try to import tau-bench dependencies
try:
    import tau_bench
    from tau_bench.types import RunConfig
    from tau_bench.run import run as tau_run

    TAU_BENCH_AVAILABLE = True
except ImportError:
    tau_bench = None
    RunConfig = None
    tau_run = None
    TAU_BENCH_AVAILABLE = False


@dataclass
class TauBenchConfig:
    """Configuration for TauBench evaluation."""

    env: str = "retail"  # Environment: "retail" or "airline"
    agent_strategy: str = "tool-calling"  # Agent strategy: "tool-calling", "act", "react", "few-shot"
    user_strategy: str = "llm"  # User simulator strategy: "llm", "react", "verify", "reflection"
    user_model: str = "gpt-4o"  # Model for user simulator
    user_model_provider: str = "openai"  # Provider for user simulator
    temperature: float = 0.0  # Sampling temperature
    task_split: str = "test"  # Task split: "train", "test", "dev"
    start_index: int = 0  # Start index for tasks
    end_index: int = -1  # End index for tasks (-1 for all)
    task_ids: Optional[List[int]] = None  # Specific task IDs to run
    num_trials: int = 1  # Number of trials per task
    max_concurrency: int = 1  # Number of tasks to run in parallel
    seed: int = 10  # Random seed
    shuffle: int = 0  # Whether to shuffle tasks


class TauBenchBenchmark(BaseBenchmark):
    """
    TauBench benchmark for evaluating tool-agent-user interaction in real-world domains.

    TauBench emulates dynamic conversations between a user (simulated by language models)
    and a language agent provided with domain-specific API tools and policy guidelines.

    Supports two domains:
    - Retail: Customer service interactions for an e-commerce platform
    - Airline: Customer service interactions for an airline booking system

    Reference: https://github.com/sierra-research/tau-bench
    Paper: https://arxiv.org/abs/2406.12045
    """

    REQUIRES_OPENAI_ANNOTATOR = False  # TauBench uses its own evaluation

    def __init__(
        self,
        config: Optional[TauBenchConfig] = None,
        env: str = "retail",
        agent_strategy: str = "tool-calling",
        user_strategy: str = "llm",
        user_model: str = "gpt-4o",
        user_model_provider: str = "openai",
        temperature: float = 0.0,
        task_split: str = "test",
        start_index: int = 0,
        end_index: int = -1,
        task_ids: Optional[List[int]] = None,
        num_trials: int = 1,
        max_concurrency: int = 1,
        seed: int = 10,
        shuffle: int = 0,
        debug: bool = False,
        max_tokens: int = 4096,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize TauBench benchmark.

        Args:
            config: TauBenchConfig object (if provided, overrides other args)
            env: Environment to evaluate ("retail" or "airline")
            agent_strategy: Strategy for the agent ("tool-calling", "act", "react", "few-shot")
            user_strategy: Strategy for user simulator ("llm", "react", "verify", "reflection")
            user_model: Model name for user simulator
            user_model_provider: Provider for user simulator model
            temperature: Sampling temperature for generation
            task_split: Task split to use ("train", "test", "dev")
            start_index: Start index for task selection
            end_index: End index for task selection (-1 for all)
            task_ids: Specific task IDs to run (overrides start/end_index)
            num_trials: Number of trials per task
            max_concurrency: Number of parallel tasks
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle tasks (0 or 1)
            debug: If True, run on limited examples
            max_tokens: Maximum tokens for generation
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)

        # Use config if provided, otherwise use individual parameters
        if config is not None:
            self.config = config
        else:
            self.config = TauBenchConfig(
                env=env,
                agent_strategy=agent_strategy,
                user_strategy=user_strategy,
                user_model=user_model,
                user_model_provider=user_model_provider,
                temperature=temperature,
                task_split=task_split,
                start_index=start_index,
                end_index=end_index,
                task_ids=task_ids,
                num_trials=num_trials,
                max_concurrency=max_concurrency,
                seed=seed,
                shuffle=shuffle,
            )

        # Check for required dependencies
        missing_deps = []
        if not TAU_BENCH_AVAILABLE:
            missing_deps.append("tau-bench")
        if LM is None:
            missing_deps.append("lm_eval")
        if BaseBenchmark is None:
            missing_deps.append("lm_eval")

        if missing_deps:
            raise ImportError(
                f"Missing required dependencies for TauBench evaluation: {', '.join(missing_deps)}. "
                f"Install tau-bench with: pip install tau-bench"
            )

        super().__init__(logger=logger, system_instruction=system_instruction)

        self.debug = debug
        self.max_tokens = max_tokens

        # Validate environment
        if self.config.env not in ["retail", "airline"]:
            raise ValueError(f"Invalid environment: {self.config.env}. Must be 'retail' or 'airline'")

        # Store tau-bench references
        self.tau_bench = tau_bench
        self.RunConfig = RunConfig
        self.tau_run = tau_run

        self.logger.info(f"Initialized TauBench benchmark for {self.config.env} environment")

    def _create_tau_bench_config(self, model_name: str, model_provider: str, base_url: Optional[str] = None) -> Any:
        """
        Create a RunConfig for tau-bench from our configuration.

        Args:
            model_name: Name of the model to evaluate
            model_provider: Provider of the model
            base_url: Optional base URL for API endpoint (for local vLLM servers)

        Returns:
            RunConfig object for tau-bench
        """
        config_dict = {
            "model": model_name,
            "model_provider": model_provider,
            "user_model": self.config.user_model,
            "user_model_provider": self.config.user_model_provider,
            "num_trials": self.config.num_trials,
            "env": self.config.env,
            "agent_strategy": self.config.agent_strategy,
            "temperature": self.config.temperature,
            "task_split": self.config.task_split,
            "start_index": self.config.start_index,
            "end_index": 10 if self.debug else self.config.end_index,  # Limit to 10 tasks in debug mode
            "task_ids": self.config.task_ids,
            "log_dir": None,  # Will be set to temp directory
            "max_concurrency": self.config.max_concurrency,
            "seed": self.config.seed,
            "shuffle": self.config.shuffle,
            "user_strategy": self.config.user_strategy,
            "few_shot_displays_path": None,
        }

        # Add base_url if provided (for local vLLM servers)
        if base_url is not None:
            config_dict["base_url"] = base_url

        return self.RunConfig(**config_dict)

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses using tau-bench's evaluation framework.

        Note: TauBench requires API-based models via litellm.
        - For local vLLM models: Start a vLLM server manually, then use --model openai
        - For HuggingFace models: Not supported (use vLLM instead)
        - For OpenAI/Anthropic/etc: Use directly with --model openai

        Args:
            model: Language model instance

        Returns:
            Dictionary containing results file path and metadata,
            or None for non-primary ranks
        """
        # Only run on primary rank
        if model.rank != 0:
            return None

        # Determine model name and provider
        import lm_eval.models

        if isinstance(model, lm_eval.models.huggingface.HFLM):
            self.logger.error(
                "TauBench does not support HuggingFace models. "
                "TauBench uses litellm which requires API-based models. "
                "\n\nTo use a local model with TauBench:"
                "\n1. Start a vLLM server: vllm serve MODEL_PATH --port 8000"
                "\n2. Run evaluation: python -m eval.eval --model openai --model_args "
                "'model=MODEL_NAME,base_url=http://localhost:8000/v1' --tasks TauBench"
            )
            raise ValueError(
                "TauBench does not support HuggingFace models. "
                "Please start a vLLM server and use --model openai instead."
            )
        elif isinstance(model, lm_eval.models.vllm_causallms.VLLM):
            self.logger.error(
                "TauBench does not support vLLM models directly. "
                "TauBench uses litellm which requires API-based models. "
                "\n\nTo use your vLLM model with TauBench:"
                "\n1. Start a vLLM server: vllm serve MODEL_PATH --port 8000 [--tensor-parallel-size N ...]"
                "\n2. Run evaluation: python -m eval.eval --model openai --model_args "
                "'model=MODEL_NAME,base_url=http://localhost:8000/v1' --tasks TauBench"
            )
            raise ValueError(
                "TauBench does not support vLLM models directly. "
                "Please start a vLLM server and use --model openai instead."
            )
        elif isinstance(model, lm_eval.models.openai_completions.OpenAIChatCompletion):
            model_name = model.model
            model_provider = "openai"
            base_url = None  # Use default OpenAI endpoint
        else:
            # Try to extract from model_args
            model_name = getattr(model, "model", "unknown")
            model_provider = "unknown"
            base_url = None
            self.logger.warning(
                f"Unknown model type: {type(model)}. Using model_name={model_name}, provider={model_provider}"
            )

        self.logger.info(f"Running TauBench with model={model_name}, provider={model_provider}")
        if 'base_url' in locals() and base_url:
            self.logger.info(f"Using base_url: {base_url}")

        # Create temporary directory for results
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name

        # Create tau-bench config
        tau_config = self._create_tau_bench_config(
            model_name,
            model_provider,
            base_url=base_url if 'base_url' in locals() else None
        )
        tau_config.log_dir = temp_dir

        try:
            # Run tau-bench evaluation
            self.logger.info(f"Starting tau-bench evaluation for {self.config.env} environment...")
            self.tau_run(tau_config)

            # Find the results file
            results_files = list(Path(temp_dir).glob("*.json"))
            if not results_files:
                raise FileNotFoundError(f"No results file found in {temp_dir}")

            results_file = results_files[0]
            self.logger.info(f"Tau-bench evaluation completed. Results saved to {results_file}")

            return {
                "temp_dir_obj": temp_dir_obj,
                "results_file": str(results_file),
                "model_name": model_name,
                "model_provider": model_provider,
                "env": self.config.env,
            }

        except Exception as e:
            self.logger.error(f"Error during tau-bench evaluation: {str(e)}")
            # Cleanup vLLM server if it was started
            self._cleanup_vllm_server()
            temp_dir_obj.cleanup()
            raise

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated responses using tau-bench's metrics.

        Args:
            results: Dictionary containing results file path and metadata

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        try:
            # Load results from file
            results_file = results["results_file"]
            with open(results_file, "r") as f:
                tau_results = json.load(f)

            # Extract metrics from tau-bench results
            # Tau-bench typically reports Pass@k metrics
            metrics = {}

            # Extract overall metrics
            if "overall" in tau_results:
                overall = tau_results["overall"]
                for key, value in overall.items():
                    if isinstance(value, (int, float)):
                        metrics[key] = value

            # Extract per-task metrics if available
            if "tasks" in tau_results:
                tasks = tau_results["tasks"]
                metrics["num_tasks"] = len(tasks)
                metrics["num_completed"] = sum(1 for t in tasks if t.get("success", False))
                metrics["completion_rate"] = metrics["num_completed"] / metrics["num_tasks"] if metrics["num_tasks"] > 0 else 0

            # Add metadata
            metrics.update(
                {
                    "env": results["env"],
                    "model_name": results["model_name"],
                    "model_provider": results["model_provider"],
                    "agent_strategy": self.config.agent_strategy,
                    "user_strategy": self.config.user_strategy,
                }
            )

            # Cleanup temporary directory
            results["temp_dir_obj"].cleanup()

            self.logger.info(f"TauBench evaluation metrics: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error evaluating tau-bench results: {str(e)}")
            if "temp_dir_obj" in results:
                results["temp_dir_obj"].cleanup()
            raise

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete TauBench evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        self.logger.info(f"Starting TauBench evaluation for {self.config.env} environment")
        try:
            generation_results = self.generate_responses(model)

            # If not primary rank, return None early
            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running TauBench benchmark: {str(e)}")
            return {"error": str(e)}

