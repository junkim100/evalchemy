# HarmBench Safety Evaluation Benchmark

This benchmark evaluates language models on their safety and robustness against harmful prompts using the HarmBench framework. It measures how well models refuse to generate harmful content across multiple categories.

## Overview

The HarmBench benchmark tests a model's ability to refuse harmful requests across various categories including:
- **Illegal Activities**: Instructions for illegal actions
- **Harmful Content**: Content that could cause harm
- **Misinformation/Disinformation**: False or misleading information
- **Chemical/Biological**: Dangerous chemical or biological instructions
- **Cybercrime**: Hacking, intrusion, and cybercrime instructions
- **Copyright**: Reproduction of copyrighted material

## Evaluation Methodology

The benchmark uses a specialized classifier to determine whether model outputs constitute successful "attacks" (i.e., the model generated harmful content instead of refusing). The primary metric is **Attack Success Rate (ASR)** - the percentage of harmful prompts that resulted in harmful outputs.

## Supported Behavior Categories

### Functional Categories
- `standard`: Standard harmful behaviors
- `contextual`: Behaviors that require context to be harmful
- `copyright`: Copyright infringement behaviors

### Semantic Categories
- `illegal`: Illegal activities
- `harmful`: Generally harmful content
- `misinformation_disinformation`: False information
- `chemical_biological`: Dangerous substances/processes
- `cybercrime_intrusion`: Cybercrime and hacking
- `copyright`: Copyrighted material reproduction

## Dependencies

Install the required dependencies:

```bash
pip install pandas numpy transformers vllm torch spacy datasketch accelerate sentence-transformers
```

For full functionality, you may also need:
```bash
pip install ray openai anthropic mistralai
```

## Usage

```python
from eval.chat_benchmarks.HarmBench.eval_instruct import HarmBenchBenchmark

# Initialize benchmark
benchmark = HarmBenchBenchmark(
    behaviors_path="eval/chat_benchmarks/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv",
    classifier_path="cais/HarmBench-Llama-2-13b-cls",
    debug=False,                    # Set to True for quick testing (10 examples)
    max_examples=100,               # Limit number of examples (None for all)
    functional_categories=["standard"],  # Filter by functional category
    semantic_categories=None,       # Filter by semantic category (None for all)
    include_advbench_metric=True    # Include AdvBench refusal metric
)

# Run evaluation
results = benchmark.run_benchmark(model)
```

## Configuration Options

- `behaviors_path`: Path to the behaviors CSV file (default: test set)
- `classifier_path`: HuggingFace model path for the HarmBench classifier
- `max_examples`: Maximum number of examples to evaluate (default: None for all)
- `debug`: If True, only evaluate 10 examples for quick testing (default: False)
- `seed`: Random seed for reproducibility (default: [0, 1234, 1234, 1234])
- `max_tokens`: Maximum tokens to generate (default: 512)
- `temperature`: Temperature for generation (default: 0.0)
- `include_advbench_metric`: Include AdvBench refusal metric (default: True)
- `functional_categories`: List of functional categories to include (default: None for all)
- `semantic_categories`: List of semantic categories to include (default: None for all)
- `logger`: Optional logger instance
- `system_instruction`: Optional system instruction for the model

## Evaluation Metrics

The benchmark reports:

- **Attack Success Rate (ASR)**: Primary metric - percentage of harmful prompts that resulted in harmful outputs
- **Category-specific ASR**: ASR broken down by functional and semantic categories
- **Total Examples**: Number of behaviors evaluated
- **Harmful/Safe Responses**: Count of harmful vs safe responses
- **AdvBench ASR**: Additional refusal metric based on AdvBench patterns (if enabled)

## Dataset Splits

The benchmark includes multiple dataset splits:

- `harmbench_behaviors_text_test.csv`: Test set (1,216 behaviors)
- `harmbench_behaviors_text_val.csv`: Validation set (400 behaviors)  
- `harmbench_behaviors_text_all.csv`: Full dataset (1,616 behaviors)

## Classifier Models

The benchmark uses specialized classifier models to evaluate outputs:

- `cais/HarmBench-Llama-2-13b-cls`: Primary classifier for standard evaluation
- `cais/HarmBench-Mistral-7b-val-cls`: Alternative classifier supporting multimodal behaviors

## Implementation Details

- Uses a specialized HarmBench classifier to determine if outputs are harmful
- Handles different behavior types (standard, contextual, copyright) appropriately
- Supports copyright detection using MinHash-based similarity matching
- Provides robust output extraction from different model formats
- Follows the same architectural patterns as other benchmarks in the framework
- Includes comprehensive error handling and logging

## Safety Considerations

This benchmark contains harmful prompts by design. It should only be used for:
- Academic research on AI safety
- Responsible evaluation of model safety measures
- Development of better safety techniques

**Do not use this benchmark to actually generate harmful content for malicious purposes.**

## Citation

If you use this benchmark, please cite the HarmBench paper:

```bibtex
@article{mazeika2024harmbench,
  title={HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author={Mantas Mazeika and Long Phan and Xuwang Yin and Andy Zou and Zifan Wang and Norman Mu and Elham Sakhaee and Nathaniel Li and Steven Basart and Bo Li and David Forsyth and Dan Hendrycks},
  year={2024},
  eprint={2402.04249},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## Links

- [HarmBench Paper](https://arxiv.org/abs/2402.04249)
- [HarmBench GitHub](https://github.com/centerforaisafety/HarmBench)
- [HarmBench Classifiers on HuggingFace](https://huggingface.co/cais)
