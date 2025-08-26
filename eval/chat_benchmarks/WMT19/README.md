# WMT19 Machine Translation Benchmark

This benchmark evaluates language models on machine translation tasks using the WMT19 (Workshop on Machine Translation 2019) dataset.

## Overview

The WMT19 benchmark tests a model's ability to translate text between different language pairs. It uses the validation split of the WMT19 dataset and evaluates translations using BLEU scores.

## Supported Language Pairs

- `cs-en`: Czech to English
- `de-en`: German to English  
- `fi-en`: Finnish to English
- `fr-de`: French to German
- `gu-en`: Gujarati to English
- `kk-en`: Kazakh to English
- `lt-en`: Lithuanian to English
- `ru-en`: Russian to English
- `zh-en`: Chinese to English

## Dependencies

Install the required dependencies:

```bash
pip install datasets sacrebleu numpy
```

## Usage

```python
from eval.chat_benchmarks.WMT19.eval_instruct import WMT19Benchmark

# Initialize benchmark
benchmark = WMT19Benchmark(
    language_pair="de-en",  # German to English
    debug=False,            # Set to True for quick testing (10 examples)
    max_examples=100        # Limit number of examples (None for all)
)

# Run evaluation
results = benchmark.run_benchmark(model)
```

## Configuration Options

- `language_pair`: Language pair to evaluate (default: "de-en")
- `max_examples`: Maximum number of examples to evaluate (default: None for all)
- `debug`: If True, only evaluate 10 examples for quick testing (default: False)
- `seed`: Random seed for reproducibility (default: [0, 1234, 1234, 1234])
- `max_tokens`: Maximum tokens to generate (default: 1024)
- `logger`: Optional logger instance
- `system_instruction`: Optional system instruction for the model

## Evaluation Metrics

The benchmark reports:

- **BLEU Score**: Primary metric using sacrebleu implementation
- **Coverage**: Percentage of examples with valid translations
- **Average Lengths**: Average word counts for references and hypotheses
- **Valid Translations**: Number of successfully generated translations

## Dataset

- **Source**: [WMT19 Dataset on Hugging Face](https://huggingface.co/datasets/wmt/wmt19)
- **Split Used**: Validation split
- **Format**: Translation pairs with source and target text

## Implementation Details

- Uses sacrebleu for BLEU score calculation (standard in MT evaluation)
- Handles multiple output formats from different model types
- Provides robust translation extraction from model outputs
- Follows the same architectural patterns as other benchmarks in the framework
