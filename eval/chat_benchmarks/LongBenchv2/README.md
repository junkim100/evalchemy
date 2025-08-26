# LongBench v2 Benchmark

LongBench v2 is a comprehensive benchmark for evaluating long context understanding capabilities of large language models, featuring contexts ranging from 8k to 2M words.

## Overview

- **Paper**: [LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](https://arxiv.org/abs/2412.15204)
- **Dataset**: `THUDM/LongBench-v2` (HuggingFace)
- **Task Type**: Multiple choice questions (A, B, C, D)
- **Size**: 503 challenging examples
- **Context Length**: 8k to 2M words

## Key Features

- **Long Context**: Extremely long contexts requiring deep understanding
- **Multiple Choice**: Reliable A/B/C/D format for consistent evaluation
- **Six Task Categories**: Comprehensive coverage of long-context scenarios
- **Difficulty Levels**: Easy and hard examples
- **Length Categories**: Short, medium, and long contexts
- **Domain Diversity**: Multiple domains and sub-domains

## Task Categories

1. **Single-Document QA**: Understanding within a single long document
2. **Multi-Document QA**: Reasoning across multiple documents
3. **Long In-context Learning**: Learning from extensive examples
4. **Long-Dialogue History Understanding**: Processing extended conversations
5. **Code Repository Understanding**: Comprehending large codebases
6. **Long Structured Data Understanding**: Analyzing complex structured data

## Dataset Structure

Each example contains:
- `_id`: Unique identifier
- `domain`: Primary domain category
- `sub_domain`: Specific sub-domain
- `difficulty`: "easy" or "hard"
- `length`: "short", "medium", or "long"
- `question`: The question to answer
- `choice_A`, `choice_B`, `choice_C`, `choice_D`: Multiple choice options
- `answer`: Ground truth answer (A, B, C, or D)
- `context`: The long context (documents, code, conversations, etc.)

## Usage

### Basic Usage

```bash
python -m eval.eval --model vllm --tasks LongBenchv2 --model_args "pretrained=MODEL_NAME" --batch_size auto --output_path logs
```

### Configuration Options

The `LongBenchv2Benchmark` class supports the following parameters:

#### Dataset Options
- `dataset_name` (str): HuggingFace dataset name (default: `"THUDM/LongBench-v2"`)

#### Generation Options
- `max_tokens` (int): Maximum tokens for model generation (default: `32768`)
- `seed` (List[int]): Random seed for reproducibility (default: `[0, 1234, 1234, 1234]`)
- `enable_cot` (bool): Enable Chain-of-Thought prompting (default: `False`)

#### Filtering Options
- `filter_domain` (str): Filter by specific domain (optional)
- `filter_difficulty` (str): Filter by difficulty level - `"easy"` or `"hard"` (optional)
- `filter_length` (str): Filter by length category - `"short"`, `"medium"`, or `"long"` (optional)

#### Evaluation Options
- `debug` (bool): If True, only evaluate on 5 examples (default: `False`)
- `logger` (logging.Logger): Optional logger instance
- `system_instruction` (str): Optional system instruction for the model

### Example Configurations

#### Full Dataset
```python
benchmark = LongBenchv2Benchmark(
    debug=False,
    max_tokens=32768
)
```

#### Hard Questions Only
```python
benchmark = LongBenchv2Benchmark(
    filter_difficulty="hard",
    debug=False
)
```

#### Single Domain
```python
benchmark = LongBenchv2Benchmark(
    filter_domain="single_document_qa",
    debug=False
)
```

#### Chain-of-Thought Mode
```python
benchmark = LongBenchv2Benchmark(
    enable_cot=True,
    max_tokens=32768
)
```

#### Debug Mode
```python
benchmark = LongBenchv2Benchmark(
    debug=True,  # Only 5 examples
    filter_difficulty="easy"
)
```

#### Combined Filters
```python
benchmark = LongBenchv2Benchmark(
    filter_domain="multi_document_qa",
    filter_difficulty="hard",
    filter_length="long",
    enable_cot=True
)
```

## Available Domains

Based on the dataset, common domains include:
- `single_document_qa`
- `multi_document_qa` 
- `long_in_context_learning`
- `long_dialogue_history_understanding`
- `code_repo_understanding`
- `long_structured_data_understanding`

## Prompting Modes

### Direct Prompting (Default)
```
Please read the following context carefully and answer the question.

Context: [LONG_CONTEXT]

Question: [QUESTION]

A. [CHOICE_A]
B. [CHOICE_B] 
C. [CHOICE_C]
D. [CHOICE_D]

Please provide your answer as one of A, B, C, or D.
```

### Chain-of-Thought Prompting
```
Please read the following context carefully and answer the question.

Context: [LONG_CONTEXT]

Question: [QUESTION]

A. [CHOICE_A]
B. [CHOICE_B]
C. [CHOICE_C] 
D. [CHOICE_D]

Please think step by step and then provide your answer. Your final answer should be one of A, B, C, or D.

Let me think step by step:
```

## Evaluation Metrics

- **Overall Accuracy**: Percentage of correct answers
- **Domain-specific Accuracy**: Accuracy per domain category
- **Difficulty-specific Accuracy**: Accuracy for easy vs hard questions
- **Length-specific Accuracy**: Accuracy for short/medium/long contexts
- **Standard Error**: Statistical confidence measure

## Answer Extraction

The benchmark uses robust answer extraction that handles various response formats:
- "Answer: A" or "The answer is A"
- "A" at the end of response
- "Option A" or "Choice A"
- First occurrence of A/B/C/D in response

## Performance Considerations

- **Memory**: Long contexts require significant GPU memory
- **Time**: Processing 2M word contexts takes considerable time
- **Batch Size**: Use `auto` batch size for optimal memory usage
- **Context Length**: Ensure model supports the required context length

## Notes

- The benchmark automatically handles dataset loading from HuggingFace
- Filtering reduces the dataset size for focused evaluation
- Debug mode is recommended for initial testing
- Chain-of-Thought prompting may improve performance on complex reasoning tasks
- The framework provides detailed statistics for analysis
