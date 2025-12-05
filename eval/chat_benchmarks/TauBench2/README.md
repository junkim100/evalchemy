# TauBench Evaluation

This directory contains the integration of [TauBench](https://github.com/sierra-research/tau-bench) into the evalchemy evaluation framework.

## Overview

TauBench (τ-bench) is a benchmark for evaluating Tool-Agent-User interaction in real-world domains. It emulates dynamic conversations between a user (simulated by language models) and a language agent provided with domain-specific API tools and policy guidelines.

**Paper**: [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)

**Note**: The TauBench team has released [τ²-bench](https://github.com/sierra-research/tau2-bench) as an extension with code fixes and an additional telecom domain. Consider using τ²-bench for the latest version.

## Supported Domains

- **Retail**: Customer service interactions for an e-commerce platform
- **Airline**: Customer service interactions for an airline booking system

## Installation

**TauBench is automatically installed when you install evalchemy!** It's included in the `pyproject.toml` dependencies.

If you've already installed evalchemy, tau-bench should be available. If not, reinstall evalchemy:

```bash
pip install -e .
```

### API Keys Setup

You need to set up API keys for the models you want to use:

```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
# Add other API keys as needed
```

### Manual Installation (Optional)

If you need to install tau-bench separately for development:

```bash
# Install from GitHub
pip install git+https://github.com/sierra-research/tau-bench

# Or clone and install in editable mode
git clone https://github.com/sierra-research/tau-bench
cd tau-bench
pip install -e .
```

### Verify Installation

You can verify that TauBench is properly installed and integrated using the test script:

```bash
# Check if all dependencies are available
python eval/chat_benchmarks/TauBench/test_integration.py --check-dependencies

# Test loading the benchmark
python eval/chat_benchmarks/TauBench/test_integration.py --test-load

# Test TaskManager integration
python eval/chat_benchmarks/TauBench/test_integration.py --test-task-manager

# Run all tests
python eval/chat_benchmarks/TauBench/test_integration.py --all
```

## Important: API-Based Models Only

**TauBench uses litellm which only supports API-based models.**

- ✅ **Supported**: OpenAI, Anthropic, Google, Mistral, etc. (via API)
- ✅ **Supported**: Local vLLM models (via OpenAI-compatible server - see below)
- ❌ **Not Supported**: HuggingFace models directly, vLLM models directly

### Two Models Required

TauBench requires **two models**:
1. **Agent Model**: The model being evaluated (can be local via vLLM server)
2. **User Simulator**: Simulates customer behavior (requires API key, default: GPT-4o)

**Cost Note**: Only the user simulator API calls are charged. The agent model can run locally for free.

## Usage

### Option 1: Local vLLM Model (Recommended for Local Models)

To use a local model with TauBench, you need to start a vLLM OpenAI-compatible server first.

**Step 1: Start vLLM Server**

```bash
# Start vLLM server on port 8000
CUDA_VISIBLE_DEVICES=0,1,3,4 vllm serve /data_x/junkim100/models/Llama-3.1-8B-Instruct/ \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
```

**Step 2: Run TauBench Evaluation**

```bash
# Set your OpenAI API key (for user simulator)
export OPENAI_API_KEY=your_key_here

# Run evaluation using the local vLLM server
python -m eval.eval \
    --model openai \
    --model_args "model=Llama-3.1-8B-Instruct,base_url=http://localhost:8000/v1,env=retail" \
    --tasks TauBench \
    --batch_size auto
```

**Note**: Replace the model name with any identifier. The `base_url` points to your local vLLM server.

### Option 2: API-Based Models (OpenAI, Anthropic, etc.)

Run TauBench evaluation with GPT-4o:

```bash
python -m eval.eval \
    --model openai \
    --tasks TauBench \
    --model_args "model=gpt-4o,env=retail"
```

Run with Claude:

```bash
python -m eval.eval \
    --model anthropic \
    --tasks TauBench \
    --model_args "model=claude-3-5-sonnet-20240620,env=airline"
```

### Configuration Options

The TauBench benchmark supports extensive configuration through model_args:

```bash
# Example with local vLLM server
python -m eval.eval \
    --model openai \
    --tasks TauBench \
    --model_args "model=MODEL_NAME,base_url=http://localhost:8000/v1,env=retail,agent_strategy=tool-calling,user_model=gpt-4o,user_model_provider=openai,user_strategy=llm,temperature=0.0,task_split=test,max_concurrency=10"
```

#### Available Parameters

- **env** (str, default: "retail"): Environment to evaluate
  - Options: "retail", "airline"

- **agent_strategy** (str, default: "tool-calling"): Strategy for the agent
  - Options: "tool-calling", "act", "react", "few-shot"

- **user_strategy** (str, default: "llm"): Strategy for user simulator
  - Options: "llm", "react", "verify", "reflection"

- **user_model** (str, default: "gpt-4o"): Model name for user simulator

- **user_model_provider** (str, default: "openai"): Provider for user simulator
  - Options: "openai", "anthropic", "google", "mistral", etc.

- **temperature** (float, default: 0.0): Sampling temperature for generation

- **task_split** (str, default: "test"): Task split to use
  - Options: "train", "test", "dev"

- **start_index** (int, default: 0): Start index for task selection

- **end_index** (int, default: -1): End index for task selection (-1 for all tasks)

- **task_ids** (list, optional): Specific task IDs to run (comma-separated)
  - Example: "task_ids=2,4,6"

- **num_trials** (int, default: 1): Number of trials per task

- **max_concurrency** (int, default: 1): Number of tasks to run in parallel

- **seed** (int, default: 10): Random seed for reproducibility

- **shuffle** (int, default: 0): Whether to shuffle tasks (0 or 1)

### Debug Mode

Run in debug mode to test on a limited number of examples:

```bash
python -m eval.eval \
    --model vllm \
    --tasks TauBench \
    --model_args "pretrained=MODEL_NAME" \
    --debug
```

### Advanced: Manual vLLM Server (Optional)

If you prefer to manage the vLLM server yourself (e.g., for multiple evaluations), you can start it manually:

#### Terminal 1: Start vLLM server

```bash
conda activate evalchemy

# Serve your local model
vllm serve /data_x/junkim100/models/Llama-3.1-8B-Instruct/ \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192
```

Wait for the server to start (you'll see "Application startup complete").

#### Terminal 2: Run TauBench evaluation

```bash
conda activate evalchemy

# Set your OpenAI API key for the user simulator
export OPENAI_API_KEY=your_key_here

# Run evaluation pointing to local vLLM server
python -m eval.eval \
    --model openai \
    --tasks TauBench \
    --model_args "model=meta-llama/Llama-3.1-8B-Instruct,base_url=http://localhost:8000/v1,env=retail,user_model=gpt-4o,user_model_provider=openai,max_concurrency=5"
```

**Important Notes**:
- The agent model runs locally via vLLM server
- The user simulator still requires an API key (default: GPT-4o via OpenAI)
- You can use a different user simulator by changing `user_model` and `user_model_provider`
- Adjust `max_concurrency` based on your hardware and API limits

### Advanced Examples

#### Evaluate with different agent strategies

```bash
# Tool-calling strategy (default)
python -m eval.eval --model openai --tasks TauBench \
    --model_args "model=gpt-4o,agent_strategy=tool-calling"

# ReAct strategy
python -m eval.eval --model openai --tasks TauBench \
    --model_args "model=gpt-4o,agent_strategy=react"

# Act strategy
python -m eval.eval --model openai --tasks TauBench \
    --model_args "model=gpt-4o,agent_strategy=act"
```

#### Evaluate with different user simulators

```bash
# GPT-4o user simulator (default)
python -m eval.eval --model openai --tasks TauBench \
    --model_args "model=gpt-4o,user_model=gpt-4o,user_model_provider=openai"

# Claude user simulator
python -m eval.eval --model openai --tasks TauBench \
    --model_args "model=gpt-4o,user_model=claude-3-5-sonnet-20240620,user_model_provider=anthropic"

# Use local model for agent, GPT-4o for user (automatic vLLM server)
python -m eval.eval --model vllm --tasks TauBench \
    --model_args "pretrained=/path/to/model,user_model=gpt-4o,user_model_provider=openai"
```

#### Run specific tasks

```bash
# Run tasks 2, 4, and 6
python -m eval.eval --model openai --tasks TauBench \
    --model_args "model=gpt-4o,task_ids=2,4,6"

# Run tasks from index 10 to 20
python -m eval.eval --model openai --tasks TauBench \
    --model_args "model=gpt-4o,start_index=10,end_index=20"
```

#### Parallel evaluation

```bash
# Run 10 tasks in parallel (adjust based on your API limits)
python -m eval.eval --model openai --tasks TauBench \
    --model_args "model=gpt-4o,max_concurrency=10"
```

## Evaluation Metrics

TauBench reports the following metrics:

- **Pass@k**: Success rate at k attempts (e.g., Pass@1, Pass@2, Pass@3, Pass@4)
- **completion_rate**: Percentage of tasks completed successfully
- **num_tasks**: Total number of tasks evaluated
- **num_completed**: Number of tasks completed successfully

The benchmark evaluates whether the agent successfully completes the user's goal while following policy guidelines and using the provided tools correctly.

## User Simulator Strategies

TauBench uses language models to simulate users. Different strategies are available:

1. **llm** (default): Standard LLM-based user simulation
2. **react**: User simulator with reasoning traces (Thought + User Response)
3. **verify**: Uses LLM verification to check if responses are satisfactory
4. **reflection**: Uses LLM reflection to improve responses

## Agent Strategies

Different strategies for the agent being evaluated:

1. **tool-calling** (default): Uses function calling / tool calling APIs
2. **act**: Action-only prompting
3. **react**: Reasoning + Action prompting
4. **few-shot**: Few-shot prompting with examples

## Leaderboard Results

### Airline Domain

| Strategy | Pass^1 | Pass^2 | Pass^3 | Pass^4 |
|----------|--------|--------|--------|--------|
| TC (claude-3-5-sonnet-20241022) | 0.460 | 0.326 | 0.263 | 0.225 |
| TC (gpt-4o) | 0.420 | 0.273 | 0.220 | 0.200 |
| TC (claude-3-5-sonnet-20240620) | 0.360 | 0.224 | 0.169 | 0.139 |

### Retail Domain

| Strategy | Pass^1 | Pass^2 | Pass^3 | Pass^4 |
|----------|--------|--------|--------|--------|
| TC (claude-3-5-sonnet-20241022) | 0.692 | 0.576 | 0.509 | 0.462 |
| TC (gpt-4o) | 0.604 | 0.491 | 0.430 | 0.383 |
| TC (claude-3-5-sonnet-20240620) | 0.626 | 0.506 | 0.435 | 0.387 |

*TC = tool-calling strategy

## Troubleshooting

### Error: "TauBench requires API-based models"

If you see this error when using `--model vllm` or `--model hf`, it means you're trying to use a local model directly. TauBench only supports API-based models through litellm.

**Solution**: Serve your local model via vLLM's OpenAI-compatible server (see "Using Local Models" section above).

### Error: "Invalid model provider"

This means the model provider is not recognized by litellm.

**Solution**: Use one of the supported providers: `openai`, `anthropic`, `google`, `mistral`, `cohere`, etc.

### vLLM Server Connection Issues

If the evaluation can't connect to your local vLLM server:

1. Make sure the vLLM server is running: `curl http://localhost:8000/v1/models`
2. Check the port number matches in both server and client
3. Ensure `base_url` includes `/v1`: `http://localhost:8000/v1`

### User Simulator API Key Issues

The user simulator requires an API key even when using a local model for the agent:

```bash
export OPENAI_API_KEY=your_key_here  # For GPT-4o user simulator
export ANTHROPIC_API_KEY=your_key_here  # For Claude user simulator
```

## Notes

- TauBench requires API access to language models for both the agent and user simulator
- **Local models must be served via an API endpoint** (e.g., vLLM's OpenAI-compatible server)
- The benchmark can be expensive to run due to the interactive nature of the evaluation
- Set `max_concurrency` according to your API rate limits
- The user simulator typically uses GPT-4o by default, which requires an OpenAI API key
- Even when using a local model for the agent, the user simulator still requires an API key

## References

- [TauBench GitHub Repository](https://github.com/sierra-research/tau-bench)
- [TauBench Paper](https://arxiv.org/abs/2406.12045)
- [τ²-bench (Extended Version)](https://github.com/sierra-research/tau2-bench)
- [τ²-bench Paper](https://arxiv.org/abs/2506.07982)

## Citation

```bibtex
@misc{yao2024tau,
    title={$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains},
    author={Shunyu Yao and Noah Shinn and Pedram Razavi and Karthik Narasimhan},
    year={2024},
    eprint={2406.12045},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2406.12045},
}
```

