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
import subprocess
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import requests

try:
    from lm_eval.api.model import LM
except ImportError:
    LM = None

try:
    from eval.task import BaseBenchmark
except ImportError:
    BaseBenchmark = None

# Try to import tau-bench dependencies
import sys
# Use absolute path based on this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TAU_BENCH_SRC = os.path.join(_SCRIPT_DIR, 'tau2-bench', 'src')
if _TAU_BENCH_SRC not in sys.path:
    sys.path.insert(0, _TAU_BENCH_SRC)
from tau2.data_model.simulation import RunConfig
from tau2.run import run_domain as tau_run
from tau2.metrics.agent_metrics import compute_metrics

class TauBench2Benchmark(BaseBenchmark):
    """
    TauBench2 benchmark for evaluating tool-agent-user interaction in real-world domains.

    TauBench emulates dynamic conversations between a user (simulated by language models)
    and a language agent provided with domain-specific API tools and policy guidelines.

    Supports two domains:
    - Retail: Customer service interactions for an e-commerce platform
    - Airline: Customer service interactions for an airline booking system

    Reference: https://github.com/sierra-research/tau2-bench
    Paper: https://arxiv.org/abs/2406.12045
    """

    REQUIRES_OPENAI_ANNOTATOR = False  # TauBench uses its own evaluation

    def __init__(
        self,
        domain: str = "airline",
        llm_user: str = "gpt-4o-mini",
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

        # Validate domain
        if domain not in ["mock", "airline", "retail", "telecom"]:
            raise ValueError(f"Invalid domain: {domain}. Must be 'mock', 'airline', 'retail', or 'telecom'")
        
        super().__init__(logger=logger, system_instruction=system_instruction)
        # Create temporary directory for results
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        
        if debug:
            num_tasks = 1
        else:
            num_tasks = None  # Use all tasks
            
        self.tau_config = RunConfig(
            domain=domain,
            llm_user=llm_user,
            task_split=task_split,
            save_to=self.temp_dir_obj.name,
            num_trials=n_repeat,
            num_tasks=num_tasks,
            max_steps=max_steps,
        )
        self.seed = seed
        self.max_tokens = max_tokens
        self._server_process = None  # Track vLLM server process
        self._model_config = None  # Store model config for reloading
        self._original_model = None  # Store reference to original model for restoration

    def _start_vllm_server(self, model: LM, port: int = 8000) -> subprocess.Popen:
        """
        Free GPU memory from loaded model and start a vLLM server.
        
        Args:
            model: The loaded lm_eval model to extract config from
            port: Port for vLLM server
            
        Returns:
            subprocess.Popen object for the server process
        """
        import lm_eval.models
        
        # Extract model path/name from the loaded model
        if isinstance(model, lm_eval.models.vllm_causallms.VLLM):
            llm_engine = model.model.llm_engine
            model_name = llm_engine.model_config.model
            
            # Handle different vLLM versions - tensor_parallel_size location varies
            tensor_parallel_size = 1  # Default
            # First check if lm_eval wrapper stores it directly
            if hasattr(model, 'tensor_parallel_size'):
                tensor_parallel_size = model.tensor_parallel_size
            elif hasattr(llm_engine, 'parallel_config') and llm_engine.parallel_config is not None:
                tensor_parallel_size = llm_engine.parallel_config.tensor_parallel_size
            elif hasattr(llm_engine, 'vllm_config') and hasattr(llm_engine.vllm_config, 'parallel_config'):
                tensor_parallel_size = llm_engine.vllm_config.parallel_config.tensor_parallel_size
            elif hasattr(llm_engine.model_config, 'tensor_parallel_size'):
                tensor_parallel_size = llm_engine.model_config.tensor_parallel_size
            # Also check via world_size from model config
            elif hasattr(llm_engine.model_config, 'world_size'):
                tensor_parallel_size = llm_engine.model_config.world_size
            
            # Handle different vLLM versions - gpu_memory_utilization location varies
            gpu_memory_utilization = 0.9  # Default
            # First check if lm_eval wrapper stores it directly
            if hasattr(model, 'gpu_memory_utilization'):
                gpu_memory_utilization = model.gpu_memory_utilization
            elif hasattr(llm_engine, 'cache_config') and llm_engine.cache_config is not None:
                gpu_memory_utilization = llm_engine.cache_config.gpu_memory_utilization
            elif hasattr(llm_engine, 'vllm_config') and hasattr(llm_engine.vllm_config, 'cache_config'):
                gpu_memory_utilization = llm_engine.vllm_config.cache_config.gpu_memory_utilization
            elif hasattr(llm_engine.model_config, 'gpu_memory_utilization'):
                gpu_memory_utilization = llm_engine.model_config.gpu_memory_utilization
                
        elif isinstance(model, lm_eval.models.huggingface.HFLM):
            model_name = model.pretrained
            tensor_parallel_size = 1
            gpu_memory_utilization = 0.9
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        
        self.logger.info(f"Extracted model config: {model_name}, TP={tensor_parallel_size}, GPU mem={gpu_memory_utilization}")
        
        # Store model config for reloading after evaluation
        self._model_config = {
            "model_name": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "model_type": type(model).__name__,
        }
        
        # Store model name for tau-bench
        self._model_name = model_name
        
        # Store reference to original model for restoration later
        self._original_model = model
        
        # Free GPU memory from the loaded model
        self.logger.info("Freeing GPU memory from loaded model...")
        if hasattr(model, 'model'):
            # For vLLM models
            if hasattr(model.model, 'llm_engine'):
                del model.model.llm_engine
            del model.model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Give GPU memory time to be released
        time.sleep(10)
        self.logger.info("GPU memory freed.")
        
        # Start vLLM server
        self.logger.info(f"Starting vLLM server on port {port}...")
        cmd = [
            "vllm", "serve", model_name,
            "--port", str(port),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--trust_remote_code"
        ]
        
        # Set environment for spawn method (needed for multi-GPU)
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        
        # Wait for server to be ready
        base_url = f"http://localhost:{port}"
        max_retries = 1200  # Wait up to 120 seconds for larger models
        self.logger.info(f"Waiting for vLLM server to be ready at {base_url}...")
        
        for i in range(max_retries):
            # Check if process died
            if server_process.poll() is not None:
                stdout, stderr = server_process.communicate()
                raise RuntimeError(
                    f"vLLM server process died unexpectedly.\n"
                    f"stdout: {stdout.decode()}\n"
                    f"stderr: {stderr.decode()}"
                )
            
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.status_code == 200:
                    self.logger.info(f"vLLM server is ready at {base_url}")
                    break
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
            if i % 10 == 0 and i > 0:
                self.logger.info(f"Still waiting for vLLM server... ({i}/{max_retries}s)")
        else:
            server_process.terminate()
            stdout, stderr = server_process.communicate()
            raise RuntimeError(
                f"vLLM server failed to start within {max_retries} seconds.\n"
                f"stdout: {stdout.decode()}\n"
                f"stderr: {stderr.decode()}"
            )
        
        # Update TauBench config to use this server
        os.environ["TAU_BENCH_BASE_URL"] = os.environ["HOSTED_VLLM_API_BASE"]=f"{base_url}/v1"
        self.llm_agent = f"hosted_vllm/{model_name}"
        
        # Update tau_config with new llm_agent
        self.tau_config = self.tau_config.model_copy(update={"llm_agent": self.llm_agent})
        
        return server_process

    def _stop_vllm_server(self):
        """Stop the vLLM server process if running."""
        if self._server_process is not None:
            self.logger.info("Stopping vLLM server...")
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning("vLLM server did not terminate gracefully, killing...")
                self._server_process.kill()
                self._server_process.wait()
            self._server_process = None
            self.logger.info("vLLM server stopped.")

    def _reload_vllm_model(self) -> bool:
        """
        Reload the vLLM model after evaluation using stored config.
        
        This restores the internal vLLM model on the original lm_eval wrapper
        so that subsequent benchmarks can use the same model object.
        
        Returns:
            True if model was successfully reloaded, False otherwise
        """
        if self._model_config is None:
            self.logger.warning("No model config stored, cannot reload model")
            return False
        
        if self._original_model is None:
            self.logger.warning("No original model reference stored, cannot reload model")
            return False
        
        from vllm import LLM
        
        config = self._model_config
        self.logger.info(f"Reloading vLLM model: {config}")
        
        # Clear GPU cache before reloading
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)
        
        try:
            # Create new vLLM LLM instance directly
            new_llm = LLM(
                model=config["model_name"],
                tokenizer=config["model_name"],
                tensor_parallel_size=config["tensor_parallel_size"],
                gpu_memory_utilization=config["gpu_memory_utilization"],
                trust_remote_code=True,
                # trust_remote_code=config["trust_remote_code"],
                # dtype=config["dtype"],
                # max_model_len=config["max_model_len"],
                # disable_log_stats=config["disable_log_stats"],
                # max_num_seqs=config["max_num_seqs"],
            )
            
            # Restore the model attribute on the original lm_eval wrapper
            self._original_model.model = new_llm
            
            self.logger.info(f"Successfully reloaded vLLM model: {config['model_name']}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reload vLLM model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate responses using tau-bench's evaluation framework.

        This method automatically handles vLLM and HuggingFace models by:
        1. Extracting model configuration from the loaded model
        2. Freeing GPU memory from the loaded model
        3. Starting a vLLM server with the same configuration
        4. Running TauBench evaluation against the server

        Args:
            model: Language model instance

        Returns:
            Dictionary containing results file path and metadata,
            or None for non-primary ranks
        """
        # Only run on primary rank
        if model.rank != 0:
            return None

        import lm_eval.models

        try:
            # Check if we need to start a vLLM server
            if isinstance(model, (lm_eval.models.vllm_causallms.VLLM, 
                                  lm_eval.models.huggingface.HFLM)):
                self.logger.info(
                    "Detected local model. Will free GPU memory and start vLLM server..."
                )
                self._server_process = self._start_vllm_server(model, port=8000)
                model_provider = "openai"
            else:
                # For OpenAI-compatible models, use directly
                model_provider = "openai"
                self.logger.info(f"Using model directly: {self.llm_agent}")

            self.logger.info(f"Running TauBench2 with model={self.llm_agent}, provider={model_provider}")

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
            self.temp_dir_obj.cleanup()
            raise 

        finally:
            # Stop vLLM server if it was started
            self._stop_vllm_server()
            # Reload the vLLM model so it's available for subsequent benchmarks
            if self._model_config is not None:
                self._reload_vllm_model()

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated responses using tau-bench's metrics.

        Args:
            results: Dictionary containing results file path and metadata

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        
        # Cleanup
        self.temp_dir_obj.cleanup()
        
        # Handle None result from non-primary ranks
        if results is None:
            return None

        run_results = results["results"]
        avg_metrics = compute_metrics(run_results)
        run_results = json.loads(run_results.model_dump_json())
        run_results["run_stats"] = json.loads(avg_metrics.model_dump_json())
        return run_results

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        self.logger.info("Starting TauBench2 evaluation")
        try:
            generation_results = self.generate_responses(model)

            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
