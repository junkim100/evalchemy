"""
TauBench evaluation benchmark for evalchemy.

This module integrates the TauBench benchmark (https://github.com/sierra-research/tau-bench)
into the evalchemy evaluation framework. TauBench evaluates tool-agent-user interaction
in real-world domains (airline and retail).

Note: This implementation requires the tau-bench package to be installed:
    pip install tau-bench
"""

import os
import logging
import tempfile
import json
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
    import sys
    sys.path.append('tau2-bench/src')
    from tau2.data_model.simulation import RunConfig
    from tau2.run import run_domain as tau_run
    from tau2.metrics.agent_metrics import compute_metrics
    TAU_BENCH_AVAILABLE = True
except ImportError:
    tau_bench = None
    RunConfig = None
    tau_run = None
    TAU_BENCH_AVAILABLE = False


# @dataclass
# class TauBenchConfig:
#     """Configuration for TauBench evaluation."""

#     domain: str = "retail"  # Environment: "retail" or "airline"
#     llm_agent: str = "tool-calling"  # Agent strategy: "tool-calling", "act", "react", "few-shot"
#     llm_user: str = "llm"  # User simulator strategy: "llm", "react", "verify", "reflection"
#     user_model: str = "gpt-4o"  # Model for user simulator
#     user_model_provider: str = "openai"  # Provider for user simulator
#     temperature: float = 0.0  # Sampling temperature
#     task_split: str = "test"  # Task split: "train", "test", "dev"
#     start_index: int = 0  # Start index for tasks
#     end_index: int = -1  # End index for tasks (-1 for all)
#     task_ids: Optional[List[int]] = None  # Specific task IDs to run
#     num_trials: int = 1  # Number of trials per task
#     max_concurrency: int = 1  # Number of tasks to run in parallel
#     seed: int = 10  # Random seed
#     shuffle: int = 0  # Whether to shuffle tasks


class TauBench2Benchmark(BaseBenchmark):
    """
    TauBench2 benchmark for evaluating tool-agent-user interaction in real-world domains.

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
        domain: str = "airline",
        llm_agent: str = "hosted_vllm/test",      # TODO: update default model. 
        llm_user: str = "gpt-4o-mini",
        base_url: str = "http://ray-serve-group1.p-ncai-wbl.svc.cluster.local:8000/v1",
        api_key: Optional[str] = "EMPTY",
        task_split: str = "base",
        max_steps: int = 50,
        debug: bool = False,
        seed: int = 10,
        max_tokens: int = 4096,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        n_repeat: str = 1,
    ):
        """
        Initialize TauBench benchmark.

        Args:
            domain: Domain to evaluate ("retail" or "airline")
            llm_agent: Model name for agent
            llm_user: Model name for user
            task_split: Task split to use ("train", "test", "dev")
            debug: If True, run on limited examples
            seed: Random seed for reproducibility
            n_repeat: Number of times to repeat the evaluation
            max_tokens: Maximum tokens for generation
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)

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

        # Validate domain
        if domain not in ["mock", "airline", "retail", "telecom"]:
            raise ValueError(f"Invalid domain: {domain}. Must be 'mock', 'airline', 'retail', or 'telecom'")
        
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.llm_agent = llm_agent
        self.prev_openai_api_key = os.environ.get("OPENAI_API_KEY", None)
        
        os.environ["TAU_BENCH_BASE_URL"] = base_url
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Create temporary directory for results
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        
        if debug:
            num_tasks = 10
        else:
            num_tasks = None  # Use all tasks
            
        self.tau_config = RunConfig(
            domain=domain,
            llm_agent=self.llm_agent,
            llm_user=llm_user,
            task_split=task_split,
            save_to=self.temp_dir_obj.name,
            num_trials=n_repeat,
            num_tasks=num_tasks,
            max_steps=max_steps,
        )
        self.seed = seed
        self.max_tokens = max_tokens

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
            self.logger.warning(
                "TauBench does not support HuggingFace models. "
                "TauBench uses litellm which requires API-based models. "
                "\n\nTo use a local model with TauBench:"
                "\n1. Start a vLLM server: vllm serve MODEL_PATH --port 8000"
                "\n2. Run evaluation: python -m eval.eval --model openai --model_args "
                "'model=MODEL_NAME,base_url=http://localhost:8000/v1' --tasks TauBench"
            )

        elif isinstance(model, lm_eval.models.vllm_causallms.VLLM):
            self.logger.warning(
                "TauBench does not support vLLM models directly. "
                "TauBench uses litellm which requires API-based models. "
                "\n\nTo use your vLLM model with TauBench:"
                "\n1. Start a vLLM server: vllm serve MODEL_PATH --port 8000 [--tensor-parallel-size N ...]"
                "\n2. Run evaluation: python -m eval.eval --model openai --model_args "
                "'model=MODEL_NAME,base_url=http://localhost:8000/v1' --tasks TauBench"
            )

        model_provider = "openai"
        base_url = None  # Use default OpenAI endpoint
        self.logger.info(f"Running TauBench2 with model={self.llm_agent}, provider={model_provider}")
        
        if 'base_url' in locals() and base_url:
            self.logger.info(f"Using base_url: {base_url}")

        try:
            # Run tau-bench evaluation
            self.logger.info(f"Starting tau-bench-2 evaluation for {self.tau_config.domain} environment...")
            results = tau_run(self.tau_config)
            self.logger.info(f"Taubench2 evaluation completed.")

            return {
                "results": results,
                "model_name": self.llm_agent,
                "model_provider": model_provider,
                "domain": self.tau_config.domain,
            }

        except Exception as e:
            self.logger.error(f"Error during tau-bench evaluation: {str(e)}")
            # Cleanup vLLM server if it was started
            self.temp_dir_obj.cleanup()
            os.environ["OPENAI_API_KEY"] = self.prev_openai_api_key
            raise 

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated responses using tau-bench's metrics.

        Args:
            results: Dictionary containing results file path and metadata

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        self.temp_dir_obj.cleanup()
        os.environ["OPENAI_API_KEY"] = self.prev_openai_api_key
        # Handle None result from non-primary ranks
        if results is None:
            return None

        run_results = results["results"]
        avg_metrics = compute_metrics(run_results)
        run_results = json.loads(run_results.model_dump_json())
        run_results["run_stats"] = json.loads(avg_metrics.model_dump_json())
        return run_results