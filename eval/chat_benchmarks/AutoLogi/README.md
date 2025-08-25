# AutoLogi Benchmark

AutoLogi is a logical reasoning benchmark that generates open-ended logic puzzles with code-based verification, avoiding the random guessing problem of multiple-choice questions.

## Overview

- **Paper**: [AutoLogi: Automated Logical Reasoning Benchmark Generation](https://arxiv.org/abs/2502.16906)
- **Dataset**: `8188zq/AutoLogi` (HuggingFace) or local JSONL files
- **Task Type**: Logical reasoning with code-based evaluation
- **Languages**: English (`en`) and Chinese (`zh`)
- **Evaluation**: Code execution for format and constraint verification

## Key Features

- **Open-ended puzzles**: No multiple choice, requires generating valid solutions
- **Code-based verification**: Uses Python functions to verify solution correctness
- **Format checking**: Validates JSON/dict structure of solutions
- **Constraint checking**: Verifies logical constraints are satisfied
- **Dual language support**: English and Chinese datasets

## Dataset Structure

Each example contains:
- `prompt`: Complete prompt for the model
- `question`: Background information
- `logi_constraints`: Logical constraints to satisfy
- `input_format`: Required solution format
- `example`: Example solution (may not always be correct)
- `code`: Verification functions
  - `Inputs_Check_code`: Format verification function
  - `Constraint_List_code`: Constraint verification functions

## Usage

### Basic Usage

```bash
python -m eval.eval --model vllm --tasks AutoLogi --model_args "pretrained=MODEL_NAME" --batch_size auto --output_path logs
```

### Configuration Options

The `AutoLogiBenchmark` class supports the following parameters:

#### Dataset Options
- `dataset_name` (str): HuggingFace dataset name (default: `"8188zq/AutoLogi"`)
- `language` (str): Language version - `"en"` for English, `"zh"` for Chinese (default: `"en"`)

#### Generation Options
- `max_tokens` (int): Maximum tokens for model generation (default: `32768`)
- `seed` (List[int]): Random seed for reproducibility (default: `[0, 1234, 1234, 1234]`)

#### Evaluation Options
- `debug` (bool): If True, only evaluate on 2 examples (default: `False`)
- `logger` (logging.Logger): Optional logger instance
- `system_instruction` (str): Optional system instruction for the model

### Example Configurations

#### English Dataset (Default)
```python
benchmark = AutoLogiBenchmark(
    language="en",
    debug=False,
    max_tokens=32768
)
```

#### Chinese Dataset
```python
benchmark = AutoLogiBenchmark(
    language="zh",
    debug=False,
    max_tokens=32768
)
```

#### Debug Mode
```python
benchmark = AutoLogiBenchmark(
    language="en",
    debug=True,  # Only 2 examples
    max_tokens=16384
)
```

## Data Format

### English Dataset
Solutions are typically Python dictionaries:
```python
{
    'Monday': {'morning': 'Helen', 'afternoon': 'Irving'}, 
    'Tuesday': {'morning': 'George', 'afternoon': 'Lenore'}, 
    'Wednesday': {'morning': 'Kyle', 'afternoon': 'Nina'}
}
```

### Chinese Dataset
Solutions are typically Python lists:
```python
['银岭站', '灏韵站', '韮上站', '扶夷站', '胡瑶站']
```

## Evaluation Metrics

- **Format Accuracy**: Percentage of solutions with correct format
- **Constraint Accuracy**: Percentage of solutions satisfying logical constraints
- **Overall Accuracy**: Percentage of completely correct solutions (format + constraints)
- **Standard Error**: Statistical confidence measure

## Local Data Setup

To use local data files instead of HuggingFace:

1. Create directory: `eval/chat_benchmarks/AutoLogi/data/`
2. Download data files:
   ```bash
   wget https://raw.githubusercontent.com/8188zq/AutoLogi/main/testing-data/AutoLogi_en.jsonl
   wget https://raw.githubusercontent.com/8188zq/AutoLogi/main/testing-data/AutoLogi_cn.jsonl
   ```

## Code Execution Safety

The benchmark safely executes verification code with:
- Restricted execution environment
- Limited built-in functions
- Proper exception handling
- Timeout protection

## Notes

- The benchmark uses `eval()` for parsing Python dict/list formats
- Code verification functions are executed in a sandboxed environment
- Some example solutions in the dataset may not satisfy all constraints (this is expected and tests the verification system)
- The framework automatically falls back to test data if the dataset cannot be loaded
